"""
Module defining the methods and types to perform searches in meshes.
"""
module Searches

using LazySets

using ..Entities
using ..Nodes
using ..Meshes

export AbstractSearchAlgorithm, Serial, Threaded, Partition, PartitionThreaded,
       evaluate_points_in_mesh, evaluate_points_in_elements

"""An `AbstractSearchAlgorithm ` is a functor to use different searching algorithms.

** Common methods:**
[`evaluate_points_in_elements`](@ref)
"""
abstract type AbstractSearchAlgorithm end

""" Serial search algorithm. """
struct Serial <: AbstractSearchAlgorithm end

""" Multithread search algorithm. """
struct Threaded <: AbstractSearchAlgorithm end

""" Serial search algorithm. """
Base.@kwdef struct Partition <: AbstractSearchAlgorithm
    "Partition in x direction."
    Nx::Int64 = 5
    "Partition in y direction."
    Ny::Int64 = 5
    "Partition in z direction."
    Nz::Int64 = 5
end

""" Partitioned and threaded search algorithm. """
Base.@kwdef struct PartitionThreaded <: AbstractSearchAlgorithm
    "Partition in x direction."
    Nx::Int64 = 5
    "Partition in y direction."
    Ny::Int64 = 5
    "Partition in z direction."
    Nz::Int64 = 5
end

"Return true if a point is inside a mesh."
function Base.:∈(p::Point{dim}, mesh::AbstractMesh{dim}) where {dim}
    !isempty(first(evaluate_points_in_mesh(mesh, [p])))
end

"Return indexes checking if points are inside the mesh."
function evaluate_points_in_mesh(mesh::AbstractMesh{dim}, vec_points::Vector{P},
        alg::A = Serial()) where {dim, P <: Point{dim},
        A <: Union{Serial, Threaded}}
    evaluate_points_in_elements(elements(mesh), vec_points, alg)
end

"Return indexes checking if  a vector of points is inside a vector of
elements with a serial searching algorithm. Two integer vectors are returned,
the first one contains the indexes of the points inside the mesh, the second one
contains the indexes of the elements containing the points."
function evaluate_points_in_elements(elements::Vector{E}, vec_points::Vector{P},
        ::Serial) where {dim, T, E <: AbstractElement, P <: Point{dim, T}}
    in_mesh_points_idx = Vector{Int64}()
    in_mesh_elements_idx = Vector{Int64}()
    @inbounds for (point_idx, point) in enumerate(vec_points)
        @inbounds for (elem_idx, elem) in enumerate(elements)
            if point ∈ elem
                push!(in_mesh_points_idx, point_idx)
                push!(in_mesh_elements_idx, elem_idx)
                # Continue with next point, without checking the following elements.
                break
            end
        end
    end
    in_mesh_points_idx, in_mesh_elements_idx
end

"Return indexes checking if a vector of points is inside a vector of
elements with a serial searching algorithm. Two vector of integers are returned,
the first one contains the indexes of the points inside the mesh, the second one
contains the indexes of the elements containing the points."
function evaluate_points_in_elements(elements::Vector{E}, vec_points::Vector{P},
        ::Threaded) where {dim, E <: AbstractElement{dim},
        P <: Point{dim}}
    num_points = length(vec_points)
    # Store per-point results: 0 means not found, >0 is the element index.
    result_elem_idx = zeros(Int64, num_points)
    Threads.@threads for point_idx in 1:num_points
        point = vec_points[point_idx]
        for (elem_idx, elem) in enumerate(elements)
            if point ∈ elem
                result_elem_idx[point_idx] = elem_idx
                break
            end
        end
    end
    # Collect results
    in_mesh_points_idx = Vector{Int64}()
    in_mesh_elements_idx = Vector{Int64}()
    for point_idx in 1:num_points
        if result_elem_idx[point_idx] != 0
            push!(in_mesh_points_idx, point_idx)
            push!(in_mesh_elements_idx, result_elem_idx[point_idx])
        end
    end
    in_mesh_points_idx, in_mesh_elements_idx
end

# "Return the ordered indexes following the the same of the first vector."
# function _sort_indexes!(in_mesh_points_idx::Vector{Int}, in_mesh_elements_idx::Vector{Int})
#     in_mesh_to_elements = in_mesh_points_idx .=> in_mesh_elements_idx
#     sort!(in_mesh_to_elements; by=x -> x[1])
#     @inbounds for (i, (point_idx, elem_idx)) in enumerate(in_mesh_to_elements)
#         in_mesh_points_idx[i] = point_idx
#         in_mesh_elements_idx[i] = elem_idx
#     end
# end

"Return a box approximation of a node's vector transforming them to LazySets singletons."
function bounding_box(nodes::Vector{N}) where {N <: AbstractNode}
    # Allocates, but sufficiently fast for now.
    box_approximation(UnionSetArray([Singleton(n) for n in nodes]))
end

"Return indexes checking if a vector of points is inside a mesh
with a partitioned searching algorithm. Two integers vectors are returned,
the first one contains the indexes of the points inside the mesh, the second one
contains the indexes of the elements containing the points."
function evaluate_points_in_mesh(mesh::AbstractMesh{dim},
        vec_points::Vector{P},
        alg::Partition) where {dim, T, P <: Point{dim, T}}
    # Array of hyperrectangles inside the bounding box.
    H = bounding_box(nodes(mesh))
    (; Nx, Ny, Nz) = alg
    Hpart = split(H, [Nx, Ny, Nz])
    Tboxes = [box_approximation(convert(LazySets.Tetrahedron, e)) for e in elements(mesh)]

    # Mapping of elements with non-empty intersection with each hyperrectangle.
    elems_idx_in_box = [Vector{Int64}() for _ in 1:length(Hpart)]
    @inbounds for (elem_idx, Tb) in enumerate(Tboxes)
        @inbounds for (box_idx, Hi) in enumerate(Hpart)
            if !is_intersection_empty(Tb, Hi)
                push!(elems_idx_in_box[box_idx], elem_idx)
            end
        end
    end

    in_mesh_points_idx = Vector{Int64}()
    in_mesh_elements_idx = Vector{Int64}()
    @inbounds for (point_idx, point) in enumerate(vec_points)
        @inbounds for (box_idx, Hi) in enumerate(Hpart)
            if point ∈ Hi
                # Loop over admissible tetrahedra inside the box.
                found = false
                @inbounds for elem_idx in elems_idx_in_box[box_idx]
                    if point ∈ Meshes.element(mesh, elem_idx)
                        push!(in_mesh_points_idx, point_idx)
                        push!(in_mesh_elements_idx, elem_idx)
                        found = true
                        break
                    end
                end
                if found
                    # Continue with next point.
                    break
                end
            end
        end
    end
    in_mesh_points_idx, in_mesh_elements_idx
end

"Return indexes checking if a vector of points is inside a mesh
with a threaded partitioned searching algorithm. Two integers vectors are returned,
the first one contains the indexes of the points inside the mesh, the second one
contains the indexes of the elements containing the points."
function evaluate_points_in_mesh(mesh::AbstractMesh,
        vec_points::Vector{P},
        alg::PartitionThreaded) where {dim, T, P <: Point{dim, T}}
    # Array of hyperrectangles inside the bounding box.
    H = bounding_box(nodes(mesh))
    (; Nx, Ny, Nz) = alg
    Hpart = split(H, [Nx, Ny, Nz])
    Tboxes = [box_approximation(convert(LazySets.Tetrahedron, elem))
              for elem in elements(mesh)]

    # Mapping of elements with non-empty intersection with each hyperrectangle.
    elems_idx_in_box = [Vector{Int64}() for _ in 1:length(Hpart)]
    @inbounds for (elem_idx, Tb) in enumerate(Tboxes)
        @inbounds for (box_idx, Hi) in enumerate(Hpart)
            if !is_intersection_empty(Tb, Hi)
                push!(elems_idx_in_box[box_idx], elem_idx)
            end
        end
    end

    num_points = length(vec_points)
    result_elem_idx = zeros(Int64, num_points)
    Threads.@threads for point_idx in 1:num_points
        point = vec_points[point_idx]
        @inbounds for (box_idx, Hi) in enumerate(Hpart)
            if point ∈ Hi
                # Loop over admissible tetrahedra inside the box.
                found = false
                @inbounds for elem_idx in elems_idx_in_box[box_idx]
                    if point ∈ Meshes.element(mesh, elem_idx)
                        result_elem_idx[point_idx] = elem_idx
                        found = true
                        break
                    end
                end
                if found
                    break
                end
            end
        end
    end
    in_mesh_points_idx = Vector{Int64}()
    in_mesh_elements_idx = Vector{Int64}()
    for point_idx in 1:num_points
        if result_elem_idx[point_idx] != 0
            push!(in_mesh_points_idx, point_idx)
            push!(in_mesh_elements_idx, result_elem_idx[point_idx])
        end
    end
    in_mesh_points_idx, in_mesh_elements_idx
end

end
