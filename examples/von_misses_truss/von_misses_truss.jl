# -------------------------------------------------------------
# Von Misses Truss Example from (Zerpa, Bazzano 2017 ) - 2.5.4
# -------------------------------------------------------------
using Test, LinearAlgebra
using ONSAS

"Return the problem parameters"
function parameters()
    ## scalar parameters
    E = 210e9                 # Young modulus in Pa
    ν = 0.0                   # Poisson's modulus
    A = 2.5e-3                # Cross-section area in m²
    d = sqrt(4 * A / pi)      # Diameter in m
    a = sqrt(A)               # Side in m
    θ = 65                    # Truss angle in degrees
    L = 2                     # Length in m
    V = L * cos(deg2rad(θ))   # vertical distance in m
    H = L * sin(deg2rad(θ))   # horizontal distance in m
    Fk = -1e8                 # Vertical load in N
    RTOL = 1e-4               # Relative tolerance for tests
    NSTEPS = 5               # Number of steps for the analysis
    (; E, ν, A, a, d, θ, L, V, H, Fk, RTOL, NSTEPS)
end;

"Return the problem structural model"
function structure(strain_model::Type{<:AbstractStrainModel} = GreenStrain)
    (; H, a, d, V, E, ν, Fk) = parameters()
    # -------------
    # Mesh
    # -------------
    n1 = Node(0.0, 0.0, 0.0)
    n2 = Node(V, 0.0, H)
    n3 = Node(2V, 0.0, 0.0)
    nodes = [n1, n2, n3]
    s1 = Circle(d)
    s2 = Square(a)
    truss_left = Truss(n1, n2, s1, strain_model, "left_truss")
    truss_right = Truss(n2, n3, s2, strain_model, "right_truss")
    elements = [truss_left, truss_right]
    mesh = Mesh(; nodes, elements)
    set_dofs!(mesh, :u, 3)
    # -------------------------------
    # Materials
    # -------------------------------
    steel = SVK(; E = E, ν = ν, label = "steel")
    materials = StructuralMaterial(steel => [truss_left, truss_right])
    # -------------------------------
    # Boundary conditions
    # -------------------------------
    bc_fixed = FixedField(:u, [1, 2, 3], "all_u_fixed")
    bc_fixed_y = FixedField(:u, [2], "fixed_uy")
    bc_load = GlobalLoad(:u, t -> [0, 0, Fk * t], "load in j")
    s_boundary_conditions = StructuralBoundaryCondition(
        bc_fixed => [n1, n3], bc_fixed_y => [n2],
        bc_load => [n2])
    Structure(mesh, materials, s_boundary_conditions)
end;

"Return the problem solution"
function solve(strain_model::Type{<:AbstractStrainModel} = GreenStrain)
    (; NSTEPS) = parameters()
    # -------------------------------
    # Structural Analysis
    # -------------------------------
    s = structure(strain_model)
    sa = NonLinearStaticAnalysis(s; NSTEPS)
    # -------------------------------
    # Solver
    # -------------------------------
    tol_f = 1e-10
    tol_u = 1e-10
    max_iter = 10
    tols = ConvergenceSettings(tol_u, tol_f, max_iter)
    nr = NewtonRaphson(tols)
    # -------------------------------
    # Numerical solution
    # -------------------------------
    ONSAS.solve(sa, OnsasConfig(solver = nr))
end;

"Test problem solution"
function test(
        sol::AbstractSolution, strain_model::Type{<:AbstractStrainModel} = GreenStrain)
    (; V, H, E, A, L, Fk, RTOL) = parameters()
    sa = analysis(sol)
    mesh = ONSAS.mesh(ONSAS.structure(sa))
    elements = ONSAS.elements(mesh)
    right_truss = elements[2]
    n2 = nodes(mesh)[2]
    n2_displacements = displacements(sol, n2)
    numerical_ui = n2_displacements[1]
    numerical_uj = n2_displacements[2]
    numerical_uk = n2_displacements[3]
    numerical_λ = -load_factors(sa) * Fk
    # Test null displacements due tu symmetry and 2D reasons
    @testset "Null displacements 0 case: $strain_model" begin
        @test norm(numerical_ui) ≤ 100 * eps()
        @test norm(numerical_uj) ≤ eps()
    end
    σ_right_truss = stress(sol, right_truss)
    ϵ_right_truss = strain(sol, right_truss)
    # Test stress and strain
    @testset "Stress and strain case: $strain_model" begin
        @test σ_right_truss[1, 1]≈E*ϵ_right_truss[1, 1] rtol=RTOL skip=true
    end

    # Analytic solution
    #-----------------------------
    "Analytic load factor solution for the displacement `uk` towards z axis at node `n2` `3otatedEngineeringStrain` "
    function load_factors_analytic(uk::Real, ::Type{RotatedEngineeringStrain},
            E::Real = E, A::Real = A,
            H::Real = H, V::Real = V, l0 = L)
        -2 * E * A *
        ((H + uk)^2 + V^2 - l0^2) /
        (l0 * (l0 + sqrt((H + uk)^2 + V^2))) *
        (H + uk) / sqrt((H + uk)^2 + V^2)
    end
    "Analytic load factor solution for the displacement `uk` towards z axis at node `n2` `3otatedEngineeringStrain` "
    function load_factors_analytic(uk::Real, ::Type{GreenStrain},
            E::Real = E, A::Real = A,
            H::Real = H, V::Real = V, l0 = L)
        -2 * E * A * ((H + uk) * (2 * H * uk + uk^2)) / (2.0 * L^3)
    end
    analytics_λ = load_factors_analytic.(numerical_uk, strain_model)

    @testset "Analytic and numeric load factors case: $strain_model" begin
        @test analytics_λ≈numerical_λ rtol=RTOL
    end
end;

"Run the example"
function run()
    for strain_model in (RotatedEngineeringStrain, GreenStrain)
        sol = solve(strain_model)
        write_vtk(sol, "von_misses")
        test(sol, strain_model)
    end
end;

run()
