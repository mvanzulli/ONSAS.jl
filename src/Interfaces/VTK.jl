"""
Module to export ONSAS data structures to `.vtk` files suitable for [Paraview](https://www.paraview.org/).
"""
module VTK

using Reexport
using WriteVTK

using ..Meshes
using ..Nodes
using ..Utils
using ..Entities
using ..Trusses
using ..Frames
using ..Tetrahedrons
using ..TriangularFaces
using ..Solutions
using ..Structures
using ..StructuralAnalyses

export VTKMeshFile, create_vtk_grid, write_node_data, write_cell_data, write_vtk, to_vtk

"""
Represents a VTK file for mesh data export.
This structure is compatible with the `WriteVTK` library.
"""
struct VTKMeshFile{VTK <: WriteVTK.DatasetFile}
    "`WriteVTK` `VTK` native file."
    vtk::VTK
end
function VTKMeshFile(filename::String, Mesh::AbstractMesh; kwargs...)
    # Append `true` produces a bug when opening with PareView 5.11
    vtk = create_vtk_grid(filename, Mesh; kwargs...)
    VTKMeshFile(vtk)
end
# Functor handler allowing do syntax
function VTKMeshFile(f::Function, args...; kwargs...)
    vtk = VTKMeshFile(args...; kwargs...)
    try
        f(vtk)
    finally
        close(vtk)
    end
    vtk
end
function Base.show(io::IO, ::MIME"text/plain", (; vtk)::VTKMeshFile)
    open_str = isopen(vtk) ? "open" : "closed"
    filename = vtk.path
    ncells = vtk.Ncls
    nnodes = vtk.Npts
    print(io,
        "• VTKMeshFile file \"$(filename)\" is $open_str with $nnodes nodes and $ncells cells.")
end

"""
Create a VTK mesh grid from an `AbstractMesh`.
Initializes the VTK grid with node coordinates and cell connectivity.
"""
function create_vtk_grid(filename::String, mesh::AbstractMesh{dim}; kwargs...) where {dim}
    cls = Vector{WriteVTK.MeshCell}(undef, num_elements(mesh))
    for (i, elem) in enumerate(elements(mesh))
        VTK_elem_type = to_vtkcell_type(elem)
        cls[i] = WriteVTK.MeshCell(VTK_elem_type, to_vtk_cell_nodes(elem, mesh))
    end
    coords = node_matrix(mesh)

    WriteVTK.vtk_grid(filename, coords, cls; kwargs...)
end
function Base.close(vtk::VTKMeshFile)
    WriteVTK.vtk_save(vtk.vtk)
end

"""
Converts element types from `ONSAS` to VTK-compatible cell types for tetrahedral elements.
"""
to_vtkcell_type(::Tetrahedron) = VTKCellTypes.VTK_TETRA
# Fallback for generic cross-section
to_vtkcell_type(::Truss) = VTKCellTypes.VTK_LINE
to_vtkcell_type(::Frame) = VTKCellTypes.VTK_POLY_LINE

"""
Maps the nodes of an element within a mesh to its VTK cell node indices.
Returns connectivity indices for the specified element in the mesh.
"""
function to_vtk_cell_nodes(e::AbstractEntity, msh::AbstractMesh)
    [findfirst(==(n), nodes(msh)) for n in nodes(e)]
end

function _vtk_write_node_data(vtk::WriteVTK.DatasetFile,
        nodal_data::Vector{<:Real},
        name::AbstractString;
        kwargs...)
    WriteVTK.vtk_point_data(vtk, nodal_data, name; kwargs...)
end
function _vtk_write_node_data(vtk::WriteVTK.DatasetFile,
        nodal_data::Matrix{<:Real},
        name::AbstractString;
        kwargs...)
    WriteVTK.vtk_point_data(vtk, nodal_data, name; kwargs...)
end
"""
Write nodal data to a VTK file.
Handles scalar or matrix data for each node in a mesh.
"""
function write_node_data(vtk::VTKMeshFile, nodedata, name; kwargs...)
    _vtk_write_node_data(vtk.vtk, nodedata, name; kwargs...)
    vtk
end

"""
Write cell data to a VTK file.
Handles scalar or matrix data for each cell in a mesh.
"""
function write_cell_data(vtk::VTKMeshFile, celldata, name; kwargs...)
    WriteVTK.vtk_cell_data(vtk.vtk, celldata, name; kwargs...)
    vtk
end

INDEX_MAP = Dict("xx" => (1, 1), "yy" => (2, 2), "zz" => (3, 3),
    "xy" => (1, 2), "yz" => (2, 3), "zx" => (3, 1),
    "yx" => (2, 1), "zy" => (3, 2), "xz" => (1, 3))

to_vtk(x::Vector) = x
# const VTX_VOIGT_ORDER = [1, 6, 5, 9, 2, 4, 8, 7, 3]
# to_vtk(x::AbstractMatrix) = view(x, VTX_VOIGT_ORDER)
to_vtk(x::AbstractMatrix) = x

function write_cell_data(vtk::VTKMeshFile, celldata::Vector{<:Matrix{<:Real}}, name;
        component_names,
        kwargs...)
    for label in component_names
        found = false
        for (key, (i, j)) in INDEX_MAP
            if occursin(key, label)
                found = true
                component_data = [celldata[k][i, j] for k in 1:length(celldata)]
                @assert !any(isnan, component_data)
                @assert !any(isinf, component_data)
                write_cell_data(vtk, reshape(component_data, length(component_data), 1, 1),
                    "$label")
            end
        end
        !found && throw(ArgumentError("Unexpected label $label didnt match INDEX_MAP"))
    end
    vtk
end

function default_dof_fields(sol::AbstractSolution)
    ns = nodes(mesh(structure(analysis(sol))))
    nf = collect(keys(dofs(first(ns))))
    push!(nf, :σ)
    push!(nf, :ϵ)
    nf
end

const POINT_FIELDS = [:u, :θ]
const CELL_FIELDS = Dict(:σ => (3, 3), :ϵ => (3, 3))
const FIELD_NAMES = Dict(:u => "Displacement" => ["ux", "uy", "uz"],
    :θ => "Rotation" => ["θx", "θy", "θz"],
    :σ => "Stress" => ["σxx", "σyy", "σzz", "τyz", "τxz", "τxy", "τzy", "τzx",
        "τyx"],
    :ϵ => "Strain" => ["ϵxx", "ϵyy", "ϵzz", "γyz", "γxz", "γxy", "γzy", "γzx",
        "γyx"])

function _extract_node_data(
        sol::AbstractSolution, fields::Vector{Field}, msh, time_index::Integer)
    # TODO: Initialize empty vectors for each field
    nodal_data = Dict(field => Float64[]
    for field in fields if field ∈ POINT_FIELDS)

    for node in nodes(msh)
        for field in fields
            if field ∈ POINT_FIELDS
                ndofs = dofs(node, field)
                node_dof_data = displacements(sol, ndofs, time_index)
                vtk_data = to_vtk(node_dof_data)
                append!(nodal_data[field], vtk_data)
            end
        end
    end
    nodal_data
end
function _extract_cell_data(sol::AbstractSolution, fields::Vector{Field},
        msh::AbstractMesh, time_index::Integer)
    cell_data = Dict(field => [zeros(CELL_FIELDS[field]...) for _ in 1:num_elements(msh)]
    for field in fields if field ∈ keys(CELL_FIELDS))
    for (i, elem) in enumerate(elements(msh))
        for field in fields
            if haskey(CELL_FIELDS, field)
                if field == :σ
                    σ = stress(sol, elem, time_index)
                    cell_data[field][i] .= to_vtk(σ)
                elseif field == :ϵ
                    ϵ = strain(sol, elem, time_index)
                    cell_data[field][i] .= to_vtk(ϵ)
                end
            end
        end
    end
    cell_data
end
"""
Generate a VTK file from a solution structure, exporting simulation data such as displacements, strain, and stress.

The `write_vtk` function creates a VTK file that captures nodal and elemental data at a specified time step, making it suitable for visualization in ParaView or other compatible tools. This function is part of the export pipeline for `ONSAS` solutions, allowing users to analyze and visualize time-dependent results in a structured format.

The exported data typically includes:
- **Displacements** (`:u`): Vector field data representing nodal displacements, with components labeled `ux`, `uy`, `uz`.
- **Strain** (`:ϵ`): Second-order tensor field data for each element, representing deformation with components such as `ϵxx`, `ϵyy`, `ϵzz`, and shear components.
- **Stress** (`:σ`): Second-order tensor field data for each element, representing internal forces with components such as `σxx`, `σyy`, `σzz`, and shear components.

**DISCLAIMER:** The user is responsible for inspecting which strain and stress tensors are exported for each element formulation. It is essential to verify compatibility with the desired output before proceeding.
"""
function write_vtk(sol::AbstractSolution, filename::String,
        time_index::Integer;
        fields::Vector{Field} = default_dof_fields(sol),
        kwargs...)
    msh = mesh(structure(analysis(sol)))

    # Extract node and cell data using helper functions
    nodal_data = _extract_node_data(sol, fields, msh, time_index)
    cell_data = _extract_cell_data(sol, fields, msh, time_index)

    VTKMeshFile(filename, msh; kwargs...) do vtx
        for (field, data) in nodal_data
            field_name, component_names = FIELD_NAMES[field]
            write_node_data(vtx, data, field_name; component_names)
        end
        for (field, data) in cell_data
            field_name, component_names = FIELD_NAMES[field]
            write_cell_data(vtx, data, field_name; component_names)
        end
    end
end

function WriteVTK.collection_add_timestep(
        pvd::WriteVTK.CollectionFile, datfile::VTKMeshFile,
        time::Real)
    WriteVTK.collection_add_timestep(pvd, datfile.vtk, time)
end
function Base.setindex!(pvd::WriteVTK.CollectionFile, datfile::VTKMeshFile, time::Real)
    WriteVTK.collection_add_timestep(pvd, datfile, time)
end

"""
Generate a time-series of VTK files for a solution struct and organize them into a ParaView collection.
Each time step's VTK file is added to the collection for seamless visualization of the simulation over time.
`states(sol)[1]` (initial state) is written at time 0.0; subsequent states at their respective times.
Returns the path to the generated ParaView collection file.
"""
function write_vtk(sol::AbstractSolution, base_filename::String;
        fields::Vector{Field} = default_dof_fields(sol),
        append = false)
    times_vector = times(analysis(sol))  # Extract the times from the solution analysis
    pvd = paraview_collection(base_filename)

    for (step, time) in enumerate(times_vector)
        vtk_filename = "$(base_filename)_timestep_$(step - 1).vtu"
        vtk = write_vtk(sol, vtk_filename, step; fields, append)
        pvd[time] = vtk
    end

    close(pvd)
    @info "VTK file collection written to $base_filename.pvd"
    "$base_filename.pvd"
end

end
