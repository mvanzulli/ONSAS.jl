module ONSAS

using Reexport: @reexport
using CommonSolve

FILES = ["Utils.jl",
    # Materials
    "Materials/Materials.jl",
    "Materials/LinearElasticMaterials.jl",
    "Materials/HyperElasticMaterials.jl",
    "Materials/IsotropicLinearElasticMaterial.jl",
    "Materials/SVKMaterial.jl",
    "Materials/NeoHookeanMaterial.jl",
    "Materials/HyperElasticMaterial.jl",
    # Cross-sections
    "CrossSections/CrossSections.jl",
    "CrossSections/Circles.jl",
    "CrossSections/Rectangles.jl",
    "CrossSections/Squares.jl",
    "CrossSections/GenericCrossSections.jl",
    # Entities
    "Entities/Nodes.jl",
    "Entities/Entities.jl",
    "Entities/Trusses.jl",
    "Entities/Tetrahedrons.jl",
    "Entities/TriangularFaces.jl",
    "Entities/Frames.jl",
    # Boundary conditions
    "BoundaryConditions/BoundaryConditions.jl",
    "BoundaryConditions/FixedFieldBoundaryConditions.jl",
    "BoundaryConditions/DirichletBoundaryConditions.jl",
    "BoundaryConditions/GlobalLoadBoundaryConditions.jl",
    "BoundaryConditions/LocalLoadBoundaryConditions.jl",
    # Meshes
    "Meshes/Meshes.jl",
    "Meshes/Searches.jl",
    "Meshes/Interpolators.jl",
    "Meshes/Handlers.jl",
    # Interface Gmsh
    "Interfaces/Gmsh.jl",
    # Structural Model
    "StructuralModel/StructuralEntities.jl",
    "StructuralModel/StructuralBoundaryConditions.jl",
    "StructuralModel/StructuralMaterials.jl",
    "StructuralModel/Structures.jl",
    # Structural Solvers
    "StructuralSolvers/Assemblers.jl",
    "StructuralAnalyses/StructuralAnalyses.jl",
    "StructuralSolvers/SolversConfig.jl",
    "StructuralSolvers/StructuralSolvers.jl",
    "StructuralSolvers/Solvers.jl",
    # Structural States
    "StructuralAnalyses/StaticStates.jl",
    "StructuralAnalyses/DynamicStates.jl",
    # Structural Solutions
    "StructuralSolvers/Solutions.jl",
    # Structural Analyses
    "StructuralAnalyses/StaticAnalyses.jl",
    "StructuralAnalyses/LinearStaticAnalyses.jl",
    "StructuralAnalyses/NonLinearStaticAnalyses.jl",
    # Interface VTK
    "Interfaces/VTK.jl"]

foreach(FILES) do m
    include(m)
end

# Utility methods
@reexport using .Utils

# Physical models
## Materials
@reexport using .Materials
@reexport using .LinearElasticMaterials
@reexport using .IsotropicLinearElasticMaterial
@reexport using .HyperElasticMaterials
@reexport using .SVKMaterial
@reexport using .NeoHookeanMaterial
@reexport using .HyperElasticMaterial

## Cross-sections
@reexport using .CrossSections
@reexport using .Circles
@reexport using .Rectangles
@reexport using .Squares
@reexport using .GenericCrossSections

## Entities
@reexport using .Nodes
@reexport using .Entities
@reexport using .Trusses
@reexport using .Tetrahedrons
@reexport using .TriangularFaces
@reexport using .Frames

## Boundary conditions
@reexport using .BoundaryConditions
@reexport using .FixedFieldBoundaryConditions
@reexport using .DirichletBoundaryConditions
@reexport using .LocalLoadBoundaryConditions
@reexport using .GlobalLoadBoundaryConditions

# Geometric entities and interpolation
@reexport using .Meshes
@reexport using .Searches
@reexport using .Interpolators
@reexport using .Handlers

# Interface Gmsh
@reexport using .Gmsh

# Structural model
@reexport using .StructuralEntities
@reexport using .StructuralBoundaryConditions
@reexport using .StructuralMaterials
@reexport using .Structures

# Finite element solvers
@reexport using .Assemblers
@reexport using .StructuralAnalyses
@reexport using .SolversConfig
@reexport using .StructuralSolvers
@reexport using .Solvers
@reexport using .Solutions

# Structural analyses
@reexport using .StaticStates
@reexport using .DynamicStates
@reexport using .StaticAnalyses
@reexport using .LinearStaticAnalyses
@reexport using .NonLinearStaticAnalyses

# Interface VTK
@reexport using .VTK

end
