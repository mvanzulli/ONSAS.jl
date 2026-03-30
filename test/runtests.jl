using Test
using SafeTestsets: @safetestset

UTILS = ["aqua.jl", "utils.jl"]

INTERFACES = ["interfaces/gmsh.jl",
    "interfaces/vtk.jl"]

MODULES = ["boundary_conditions/boundary_conditions.jl",
    "cross_sections/cross_sections.jl",
    "entities/frames.jl",
    "entities/nodes.jl",
    "entities/tetrahedrons.jl",
    "entities/triangular_faces.jl",
    "entities/trusses.jl",
    "materials/materials.jl",
    "meshes/handlers.jl",
    "meshes/interpolators.jl",
    "meshes/meshes.jl",
    "meshes/searches.jl",
    "structural_analyses/dynamic_analyses.jl",
    "structural_analyses/static_analyses.jl",
    "structural_model/structures.jl",
    "structural_solvers/structural_solvers.jl"
]

EXAMPLES_FOLDER = joinpath("..", "examples")

EXAMPLE_NAMES = ["von_misses_truss",
    "linear_extension",
    "uniaxial_extension",
    "uniaxial_compression",
    "cylinder_internal_pressure",
    "clamped_truss",
    "cantilever",
    "self_weight_clamped_truss"]

EXAMPLES = [joinpath(EXAMPLES_FOLDER, name, name * ".jl") for name in EXAMPLE_NAMES]

function test(files::Vector)
    @testset "ONSAS.jl" begin
        foreach(test, files)
    end
end

function test(file::String)
    @info "Testing $file..."
    path = joinpath(@__DIR__, file)
    @eval @time @safetestset $file begin
        include($path)
    end
end

test([UTILS, MODULES, EXAMPLES, INTERFACES]);
