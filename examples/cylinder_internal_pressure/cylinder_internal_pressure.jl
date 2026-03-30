# --------------------------------------------------
# Cylinder submitted to an Internal Pressure Example
#----------------------------------------------------
using LinearAlgebra, Test, Suppressor
using ONSAS
using ONSAS.LinearStaticAnalyses: LinearResidualsIterationStep
import Random
Random.seed!(1234)

# Mesh with Gmsh.jl
include("cylinder_mesh.jl")

function parameters()
    ## scalar parameters (dimensions in mm an MPa)
    Lz = 30                 # cylinder length in z mm
    Ri = 100                # inner radius in mm
    Re = 200                # outer radius in mm
    p = 10                  # internal pressure in MPa
    pressure(t::Real) = p * t
    material_label = "mat"  # Material label used to replace the material structure
    E = 210                 # Young modulus in MPa
    ν = 0.3                 # Poisson ratio
    NSTEPS = 9              # Number of load steps
    ATOL = 1e-2 * (Re - Ri) # Absolut tolerances for testing
    ms = 1.0                # ms = 2.5 for much more refined mesh (approx 200.000 elems)
    PLOT_RESULTS = false    # boolean to plot results
    (; Lz, Ri, Re, p, ν, E, pressure, ATOL, NSTEPS, PLOT_RESULTS, ms, material_label)
end;

#= -----------------------------------------------------------
Two cases are considered:
Case 1 - Linear Elastic Analysis with Analytic Soltuion
Case 2 - Hyper Elastic Analysis to check the solution's symmetry
-------------------------------------------------------------=#
abstract type AbstractCase end
struct FirstCase <: AbstractCase end
struct SecondCase <: AbstractCase end

"Return the problem structural model"
function structure(material::AbstractMaterial)
    (; Lz, Ri, Re, pressure, ms) = parameters()
    # -------------------------------
    # Materials
    # -------------------------------
    mat_label = label(material)
    materials = StructuralMaterial(material)
    # -------------------------------
    # Boundary conditions
    # -------------------------------
    # Dirichlet boundary conditions
    bc1_label = "fixed-ui"
    bc1 = FixedField(:u, [1], bc1_label)
    bc2_label = "fixed-uj"
    bc2 = FixedField(:u, [2], bc2_label)
    bc3_label = "fixed-uk"
    bc3 = FixedField(:u, [3], bc3_label)
    # Neumann boundary conditions
    bc4_label = "pressure"
    bc4 = Pressure(:u, pressure, bc4_label)
    boundary_conditions = StructuralBoundaryCondition(bc1, bc2, bc3, bc4)
    bc_labels = [bc1_label, bc2_label, bc3_label, bc4_label]
    # -------------------------------
    # Entities
    # -------------------------------
    # Entities types without assigned nodes, faces and elements
    node_label = "node"
    faces_label = "triangle"
    elements_label = "tetrahedron"
    vfaces = [TriangularFace(faces_label)]
    velems = [Tetrahedron(elements_label)]
    entities_labels = [node_label, faces_label, elements_label]
    entities = StructuralEntity(velems, vfaces)
    # -------------------------------
    # Mesh
    # -------------------------------
    labels = [mat_label, entities_labels, bc_labels]
    filename = "cylinder"
    local msh_path
    out = @capture_out begin
        msh_path = create_cylinder_mesh(Ri, Re, Lz, labels, filename, ms)
    end
    gmsh_println(out)
    msh_mesh = MshFile(msh_path)
    mesh = Mesh(msh_mesh, entities)
    #--------------------------------
    # Dofs
    #--------------------------------
    dof_dim = 3
    dof_u_symbol = :u
    set_dofs!(mesh, dof_u_symbol, dof_dim)
    # -------------------------------
    # Structure
    # -------------------------------
    apply!(materials, mesh)
    apply!(boundary_conditions, mesh)

    Structure(mesh, materials, boundary_conditions)
end;

"Return the problem solution"
function solve(::FirstCase)
    (; NSTEPS, E, ν, material_label) = parameters()
    liner_elastic = IsotropicLinearElastic(E, ν, material_label)
    # -------------------------------
    # Structural Analysis
    # -------------------------------
    s = structure(liner_elastic)
    # TODO: Revise possible singular K matrix influenced by LinearSolver initialization.
    linear_solver = nothing
    initial_state = FullStaticState(s, LinearResidualsIterationStep, linear_solver)
    sa = LinearStaticAnalysis(s; NSTEPS, initial_state)
    # -------------------------------
    # Numerical solution
    # -------------------------------
    ONSAS.solve(sa)
end;

"Return the problem solution"
function solve(::SecondCase)
    (; NSTEPS, E, ν, material_label) = parameters()
    # The material is replaced just to test the replace method
    liner_elastic = IsotropicLinearElastic(E, ν, material_label)
    s = structure(liner_elastic)
    svk_material = SVK(; E = E, ν = ν, label = material_label)
    replace!(s, svk_material)
    # -------------------------------
    # Structural Analysis
    # -------------------------------
    sa = NonLinearStaticAnalysis(s; NSTEPS)
    # -------------------------------
    # Solver
    # -------------------------------
    tol_f = 1e-10
    tol_u = 1e-10
    max_iter = 30
    tols = ConvergenceSettings(tol_u, tol_f, max_iter)
    nr = NewtonRaphson(tols)
    # -------------------------------
    # Numerical solution
    # -------------------------------
    ONSAS.solve(sa, OnsasConfig(solver = nr))
end;

"Return a rand point in the cylinder (R, θ, L)."
function rand_point_cylinder()
    (; Ri, Re, Lz) = parameters()
    [rand() * (Re - Ri) + Ri, rand() * 2 * π, rand() * Lz]
end;

"Run the example"
function run()
    for case in (FirstCase(), SecondCase())
        sol = solve(case)
        test(sol)
    end
end;

"Return booleans testing the solution symmetry at a random z section"
function booleans_solution_at_slice(sol::AbstractSolution)
    (; Lz, ATOL) = parameters()

    structure = ONSAS.structure(analysis(sol))
    # Generic surface s at z = Lz
    rand_R, rand_θ1, Lz = rand_point_cylinder()
    # Set by force Lz
    rand_θ2 = rand() * 2 * π
    # Random point ∈ axis x
    p_rand_in_axis_x = [rand_R, 0.0, Lz]
    # Random point ∈ axis y
    p_rand_in_axis_y = [0.0, rand_R, Lz]
    # Random point between the internal and external surface
    p_rand1 = [rand_R * cos(rand_θ1), rand_R * sin(rand_θ1), Lz]
    p_rand2 = [rand_R * cos(rand_θ2), rand_R * sin(rand_θ2), Lz]
    # Vector of points to test
    vec_points = [p_rand_in_axis_x, p_rand_in_axis_y, p_rand1, p_rand2]
    #
    point_evaluator = PointEvalHandler(mesh(structure), vec_points)
    U = displacements(sol, point_evaluator)
    # Check uₖ = 0 ∀ p ∈ s
    zero_uz = all([≈(norm(u[3]), 0.0; atol = ATOL) for u in U])
    # Check uᵢ = 0 ∀ p ∈ s & ∈ axis y
    index_p_rand_in_axis_y = findall([p == p_rand_in_axis_y for p in vec_points])
    Uᵢ_in_axis_y = getindex(displacements(sol, point_evaluator, 1), index_p_rand_in_axis_y)
    zero_ui_axis_y = all([≈(norm(ui_p_in_axis_y), 0.0; atol = ATOL)
                          for ui_p_in_axis_y in Uᵢ_in_axis_y])
    # Check uⱼ = 0 ∀ p ∈ s & ∈ axis x
    index_p_rand_in_axis_x = findall([p == p_rand_in_axis_x for p in vec_points])
    Uⱼ_in_axis_x = getindex(displacements(sol, point_evaluator, 2), index_p_rand_in_axis_x)
    zero_uj_axis_x = all([≈(norm(uj_p_in_axis_y), 0.0; atol = ATOL)
                          for uj_p_in_axis_y in Uⱼ_in_axis_x])
    # Check ur(r,θ₁) =  ur(r,θ₁)  at last time
    rand_index_1 = 3
    u_rand_1 = sum(last.(U[rand_index_1][1:2]) .^ 2)
    rand_index_2 = 4
    u_rand_2 = sum(last.(U[rand_index_2][1:2]) .^ 2)
    ur_not_depends_on_θ = ≈(u_rand_1, u_rand_2; atol = ATOL)

    ur_not_depends_on_θ, zero_uz, zero_ui_axis_y, zero_uj_axis_x
end;

"Analytic radial displacements ur at radius `r` and time `t`"
function ur(r::Real, t::Real)
    (; Ri, Re, pressure, E, ν) = parameters()
    "Constant A for the analytic solution."
    function A(t::Real, Ri::Real, Re::Real, E::Real, ν::Real, p::Function)
        (1 + ν) * (1 - 2 * ν) * Ri^2 * p(t) / (E * (Re^2 - Ri^2))
    end
    "Constant B for the analytic solution."
    function B(t::Real, Ri::Real, Re::Real, E::Real, ν::Real, p::Function)
        (1 + ν) * Ri^2 * Re^2 * p(t) / (E * (Re^2 - Ri^2))
    end
    A(t, Ri, Re, E, ν, pressure) * r + B(t, Ri, Re, E, ν, pressure) / r
end;

"Return analytic solution"
function analytic_solution(sol::AbstractSolution, p::Point{3, <:Real}, ur::Function = ur)
    rand_R, _, _ = p
    (; Ri, Re) = parameters()
    λvec = load_factors(analysis(sol))
    rand_R, _ = rand_point_cylinder()
    ur_analytic_ni = [ur(Ri, λi) for λi in λvec]
    ur_analytic_ne = [ur(Re, λi) for λi in λvec]
    ur_analytic_p_rand = [ur(rand_R, λi) for λi in λvec]
    ur_analytic_ni, ur_analytic_ne, ur_analytic_p_rand
end;

"Return numercial solution"
function numerical_solution(sol::AbstractSolution, p::Point{3, <:Real}, ::FirstCase)
    # Unwarp a random point
    rand_R, rand_θ, rand_z = p
    s = ONSAS.structure(analysis(sol))
    # Get the internal radial displacement at p = (0, Ri, 0)
    m = mesh(s)
    ni = nodes(m)[4]
    ur_numeric_ni = displacements(sol, ni, 2)
    # Get the external radial displacement at p = (-Re, 0, Lz)
    ne = nodes(m)[15]
    ur_numeric_ne = displacements(sol, ne, 1)
    p_rand = Point(rand_R * cos(rand_θ), rand_R * sin(rand_θ), rand_z)
    # Displacements at p
    point_evaluator = PointEvalHandler(m, p_rand)
    ui_numeric_p_rand = displacements(sol, point_evaluator, 1)
    uj_numeric_p_rand = displacements(sol, point_evaluator, 2)
    ur_numeric_p_rand = sqrt.(@. ui_numeric_p_rand^2 + uj_numeric_p_rand^2)

    ur_numeric_ni, ur_numeric_ne, ur_numeric_p_rand
end;

"Test the solution"
function test(sol::AbstractSolution, case::FirstCase)
    (; ATOL) = parameters()
    p = Point(rand_point_cylinder()...)
    ur_analytic_ni, ur_analytic_ne, ur_analytic_p_rand = analytic_solution(sol, p)
    ur_numeric_ni, ur_numeric_ne, ur_numeric_p_rand = numerical_solution(sol, p, case)
    ur_not_depends_on_θ, zero_uz, zero_ui_axis_y,
    zero_uj_axis_x = booleans_solution_at_slice(sol)
    @testset "Solution symmetry case: $case" begin
        @test ur_not_depends_on_θ
        @test zero_uz
        @test zero_ui_axis_y
        @test zero_uj_axis_x
    end
    @testset "Radial displacement analytic solution case: $case" begin
        # A relaxed tolerace is defined, is normal to have a greater error interpolation
        @test ur_numeric_p_rand≈ur_analytic_p_rand atol=ATOL
        @test ur_numeric_ni≈ur_analytic_ni atol=ATOL
        @test ur_numeric_ne≈-ur_analytic_ne atol=ATOL
    end
end;

function test(sol::AbstractSolution, case::SecondCase)
    ur_not_depends_on_θ, zero_uz, zero_ui_axis_y,
    zero_uj_axis_x = booleans_solution_at_slice(sol)
    @testset "Solution symmetry case: $case" begin
        @test ur_not_depends_on_θ
        @test zero_uz
        @test zero_ui_axis_y
        @test zero_uj_axis_x
    end
end;

"Run the example"
function run()
    for case in (FirstCase(), SecondCase())
        sol = solve(case)
        test(sol, case)
    end
end;

run()
