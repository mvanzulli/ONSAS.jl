using ONSAS, Test
using Dictionaries: Dictionary, dictionary

# ---------------------------------
# Self weight clamped truss example
# --------------------------------
#=
This model consist of a clamped truss with its self weight. First the
weight is applied and consecutive the load at the time starting from the self
weight deformed configuration.

# Analysis 1: Self weight

## Forces
-> g
--------> F

# Analysis case 1: Self weight
#-----------------------------
# ->  ->  ->  ->  ->  ->  -> -
#-----------------------------

# Analysis case 2: Self weight + load
#-----------------------------
# ->  ->  ->  ->  ->  ->  -> - -----> F
#-----------------------------
=#
abstract type AbstractCase end
struct FirstCase <: AbstractCase end
struct SecondCase <: AbstractCase end

function parameters()
    N = 4       # Number of elements.
    E = 30e6    # Young's modulus.
    ν = 0.3     # Poisson's ratio.
    ρ = 400.0   # Density.
    L = 200     # Element length.
    A = 1       # Cross section area.
    F = 10e6    # Force at the tip
    g = 9.81    # Gravity
    ϵ_model = RotatedEngineeringStrain
    NSTEPS = 10 # Number of load steps.
    ATOL = 1e-6 # Absolute tolerance for testing
    (; g, N, E, ν, ρ, L, A, F, ϵ_model, NSTEPS, ATOL)
end

"Analytic displacement uᵢ at the tip"
function analytic_u(x::Real; F::Real, E::Real,
        A::Real, L::Real, g::Real, ρ::Real)
    b = ρ * g
    C = F / (E * A) + b * L / (E * A)
    -b / (2 * E * A) * x^2 + C * x
end

"Compute the analytical external force for a given node index `i`"
function analytical_force(c::AbstractCase, i::Int;
        F::Real, N::Int, ρ::Real, g::Real, A::Real, ΔL::Real)
    body_load = if i == 1 || i == N + 1
        ρ * g * A * ΔL / 2
    else
        ρ * g * A * ΔL
    end
    if i == N + 1 && c isa SecondCase
        body_load + F
    else
        body_load
    end
end

"Compute the analytical external forces due to gravity"
function analytical_force(c::AbstractCase, s::Structure)
    (; ρ, g, A, L, N, F) = parameters()
    ΔL = L / N
    [analytical_force(c, i; F, N, ρ, g, A, ΔL) for i in 1:length(nodes(s))]
end

"Return mesh"
function create_mesh(N::Int, L::Real, A::Real, ϵ_model)
    nodes = [Node(l) for l in LinRange(0, L, N + 1)]
    elements = [Truss(nodes[i], nodes[i + 1], Square(sqrt(A)), ϵ_model)
                for i in 1:N]
    mesh = Mesh(; nodes, elements)
    dof_dim = 1
    set_dofs!(mesh, :u, dof_dim)
    mesh
end;

"Return structure"
function structure(N::Int;
        E::Real, ν::Real, ρ::Real, L::Real, A::Real, g::Real, ϵ_model)
    # -------------
    # Mesh
    # -------------
    m = create_mesh(N, L, A, ϵ_model)
    # -------------------------------
    # Materials
    # -------------------------------
    material = SVK(; E, ν, ρ, label = "material")
    materials = StructuralMaterial(material => elements(m))
    # -------------------------------
    # Boundary conditions
    # -------------------------------
    fixed_bc = FixedField(:u, [1], "fixed")
    gravity_bc_ramp = GlobalLoad(:u, t -> t * [density(material) * g], "gravity")
    node_bcs = Dictionary{AbstractBoundaryCondition, Vector{Node}}()
    insert!(node_bcs, fixed_bc, [first(nodes(m))])
    element_bcs = dictionary([gravity_bc_ramp => elements(m)])
    boundary_conditions = StructuralBoundaryCondition(; node_bcs, element_bcs)
    # -------------------------------
    # Structure
    # -------------------------------
    Structure(m, materials, boundary_conditions)
end;

function solve(::FirstCase)
    (; g, N, E, ν, ρ, L, A, F, ϵ_model) = parameters()
    s = structure(N; E, ν, ρ, L, A, g, ϵ_model)
    a = LinearStaticAnalysis(s; NSTEPS = 10)
    ONSAS.solve(a)
end;

function test(c::FirstCase, sol::AbstractSolution)
    (; ρ, A, g, L, N, ATOL, E) = parameters()
    s = ONSAS.structure(analysis(sol))
    # Check forces at the final state
    st = current_state(analysis(sol))
    numerical_Fext = external_forces(st)
    analytical_Fext = [ρ * g * A * L / N for n in nodes(s)]
    analytical_Fext[1] = analytical_Fext[end] = ρ * g * A / 2 * L / N
    # Displacement
    numerical_disp = last(displacements(sol, last(nodes(s)), 1))
    analytic_disp = analytic_u(L; F = 0.0, E, A, L, g, ρ)

    @testset "Displacements and external forces test $c" begin
        @test numerical_Fext≈analytical_Fext atol=ATOL
        @test numerical_disp≈analytic_disp atol=ATOL
    end
end

function solve(::SecondCase)
    (; NSTEPS, F, ρ, g) = parameters()
    c1 = FirstCase()
    s1 = solve(c1)
    a1 = analysis(s1)
    s = ONSAS.structure(a1)
    last_state_analysis_self_weight = current_state(a1)
    # Add tip and gravity forces
    constant_gravity = GlobalLoad(:u, t -> [ρ * g], "gravity")
    replace!(boundary_conditions(s), constant_gravity)
    tip_load_bc = GlobalLoad(:u, t -> t * [F], "gravity")
    insert!(boundary_conditions(s), tip_load_bc, last(nodes(s)))
    # Analysis
    load_analysis = LinearStaticAnalysis(s;
        NSTEPS,
        initial_state = last_state_analysis_self_weight)
    solve!(load_analysis)
end;

function test(c::SecondCase, sol::AbstractSolution)
    (; ρ, A, g, L, F, ATOL, E) = parameters()
    a = analysis(sol)
    s = ONSAS.structure(a)
    initial_state = first(sol.states)
    # Check displacement at the initial state
    analytic_initial_disp = analytic_u(L; F = 0.0, E, A, L, g, ρ)
    numeric_initial_disp = last(displacements(initial_state))
    # TODO: The first state of the second analysis is not the initial state as it is mutated
    # along the resolution process. Hence, we need to store also the initial state of the
    # analysis considering the fact that non-homogenous initial conditions are supported.
    # Check forces at the final state
    st = current_state(analysis(sol))
    numerical_Fext = external_forces(st)
    analytical_Fext = analytical_force(c, s)
    # Check displacements at the final state
    numerical_disp = last(displacements(sol, last(nodes(s)), 1))
    analytic_disp = analytic_u(L; F, E, A, L, g, ρ)

    @testset "Displacements and external forces test $c" begin
        @test_broken numeric_initial_disp≈analytic_initial_disp atol=ATOL
        @test numerical_Fext≈analytical_Fext atol=ATOL
        @test numerical_disp≈analytic_disp atol=ATOL
    end
end

"Run the example."
function run()
    for c in (FirstCase(), SecondCase())
        sol = solve(c)
        test(c, sol)
    end
end
