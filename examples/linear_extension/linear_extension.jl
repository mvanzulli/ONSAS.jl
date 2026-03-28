# -----------------------------------------------------------------------
# Linear Elastic Extension Example 1  from (Zerpa et. Al., 2019, CMAME).
# -----------------------------------------------------------------------
using Test, LinearAlgebra, Suppressor
using ONSAS

# Mesh with Gmsh.jl (see linear_extension_sketch)
include("linear_extension_mesh.jl")

function parameters()
    E = 2.0                             # Young modulus in Pa
    ν = 0.4                             # Poisson's ratio
    λ = E * ν / ((1 + ν) * (1 - 2 * ν)) # First Lamé parameter
    G = E / (2 * (1 + ν))               # Second Lamé parameter
    tension_load(t) = p * t             # Tension load function
    p = 3                               # Tension load in Pa
    Lx = 2.0                            # Dimension in x of the box in m
    Ly = 1.0                            # Dimension in y of the box in m
    Lz = 1.0                            # Dimension in z of the box in m
    RTOL = 1e-4                         # Relative tolerance for tests
    ATOL = 1e-6                         # Absolute tolerance for tests
    NSTEPS = 9                          # number of steps for the test
    ms = 0.5                            # Refinement factor
    (; Lx, Ly, Lz, E, ν, λ, G, tension_load, RTOL, ATOL, NSTEPS, ms)
end;

"Return the problem structural model"
function structure()
    (; Lx, Ly, Lz, E, ν, tension_load, ms) = parameters()
    # -------------------------------
    # Materials
    # -------------------------------
    mat_label = "mat"
    mat = IsotropicLinearElastic(E, ν, mat_label)
    materials = StructuralMaterial(mat)
    # -------------------------------
    # Boundary conditions
    # -------------------------------
    # Fixed dofs
    bc1_label = "fixed-ux"
    bc1 = FixedField(:u, [1], bc1_label)
    bc2_label = "fixed-uy"
    bc2 = FixedField(:u, [2], bc2_label)
    bc3_label = "fixed-uz"
    bc3 = FixedField(:u, [3], bc3_label)
    # Load
    bc4_label = "tension"
    bc4 = GlobalLoad(:u, t -> [tension_load(t), 0, 0], bc4_label)
    # Get bc labels for the mesh
    bc_labels = [bc1_label, bc2_label, bc3_label, bc4_label]
    boundary_conditions = StructuralBoundaryCondition(bc1, bc2, bc3, bc4)
    # -------------------------------
    # Entities
    # -------------------------------
    # Entities types without assigned nodes, faces and elements
    faces_label = "triangle"
    elems_label = "tetrahedron"
    vfaces = [TriangularFace(faces_label)]
    velems = [Tetrahedron(elems_label)]
    entities_labels = [faces_label, elems_label]
    entities = StructuralEntity(velems, vfaces)
    # -------------------------------
    # Mesh
    # -------------------------------
    filename = "linear_extension"
    labels = [mat_label, entities_labels, bc_labels]
    output = @capture_out begin
        global mesh_path = create_linear_extension_mesh(Lx, Ly, Lz, labels, filename, ms)
    end
    gmsh_println(output)
    msh_file = MshFile(mesh_path)
    # -------------------------------
    # Structure
    # -------------------------------
    Structure(msh_file, materials, boundary_conditions, entities)
end;

"Return the problem solution"
function solve()
    (; NSTEPS) = parameters()
    s = structure()
    # -------------------------------
    # Structural Analysis
    # -------------------------------
    sa = LinearStaticAnalysis(s; NSTEPS)
    # -------------------------------
    # Numerical solution
    # -------------------------------
    ONSAS.solve(sa)
end;

"Return random points to evaluate the solution"
function test_points(sol::AbstractSolution)
    (; Lx, Ly, Lz) = parameters()
    s = ONSAS.structure(analysis(sol))
    e = rand(elements(s))
    x0_rand = Lx * rand(2)
    y0_rand = Ly * rand(2)
    z0_rand = Lz * rand(2)
    p1 = Point(x0_rand[1], y0_rand[1], z0_rand[1])
    p2 = Point(x0_rand[2], y0_rand[2], z0_rand[2])
    return p1, p2, e
end;

"Return numerical results for testing"
function numerical_solution(sol::AbstractSolution,
        p1::Point{dim}, p2::Point{dim},
        e::ONSAS.AbstractElement{dim}) where {dim}
    s = ONSAS.structure(analysis(sol))
    ## Displacements
    # Evaluate the solution at p1, p2
    eval_handler_rand = PointEvalHandler(ONSAS.mesh(s), [p1, p2])
    # rand points displacements
    # point 1
    ui_1 = displacements(sol, eval_handler_rand, 1)[1]
    uj_1 = displacements(sol, eval_handler_rand, 2)[1]
    uk_1 = displacements(sol, eval_handler_rand, 3)[1]
    # point 2
    ui_2 = displacements(sol, eval_handler_rand, 1)[2]
    uj_2 = displacements(sol, eval_handler_rand, 2)[2]
    uk_2 = displacements(sol, eval_handler_rand, 3)[2]
    ## Strain and stresses
    # Evaluate the solution at a random element
    ϵ_e = strain(sol, e)
    ϵi = getindex.(ϵ_e, 1)
    ϵj = getindex.(ϵ_e, 2)
    ϵk = getindex.(ϵ_e, 3)
    σ_e = stress(sol, e)
    σi = getindex.(σ_e, 1)
    σj = getindex.(σ_e, 2)
    σk = getindex.(σ_e, 3)

    ui_1, uj_1, uk_1, ui_2, uj_2, uk_2,
    ϵi, ϵj, ϵk,
    σi, σj, σk
end;

"Return analytical results for testing"
function analytic_solution(sol::AbstractSolution, p1::Point{dim}, p2::Point{dim},
        e::AbstractElement{dim}) where {dim}
    (; E, ν, λ, G, tension_load) = parameters()
    sa = analysis(sol)
    ## Displacements
    "Compute displacements νmeric solution ui, uj and uk for analytic validation."
    function u_ijk_analytic(λv::Vector{<:Real},
            x0::Real, y0::Real, z0::Real,
            ν::Real = ν, E::Real = E)
        C(t) = tension_load(t) * (1 - ν - 2ν^2) / (1 - ν)

        ui(t) = C(t) / E * x0
        uj(t) = 0.0
        uk(t) = 0.0

        [[ui(t) for t in λv], [uj(t) for t in λv], [uk(t) for t in λv]]
    end
    # point 1
    u_1 = u_ijk_analytic(load_factors(sa), p1[1], p1[2], p1[3])
    ui_1 = u_1[1]
    uj_1 = u_1[2]
    uk_1 = u_1[3]
    # point 2
    u_2 = u_ijk_analytic(load_factors(sa), p2[1], p2[2], p2[3])
    ui_2 = u_2[1]
    uj_2 = u_2[2]
    uk_2 = u_2[3]
    ## Strains
    "Compute strains νmeric solution ϵi, ϵj and ϵk for analytic validation."
    function ϵ_ijk_analytic(λv::Vector{<:Real}, x0::Real, y0::Real, z0::Real, ν::Real = ν,
            E::Real = E)
        C(t) = tension_load(t) * (1 - ν - 2ν^2) / (1 - ν)

        ϵi(t) = C(t) / E
        ϵj(t) = 0.0
        ϵk(t) = 0.0

        [[ϵi(t) for t in λv], [ϵj(t) for t in λv], [ϵk(t) for t in λv]]
    end
    ## Stresses
    "Compute strains νmeric solution ϵi, ϵj and ϵk for analytic validation."
    function σ_ijk_analytic(
            λv::Vector{<:Real}, x0::Real, y0::Real, z0::Real, λ::Real, G::Real)
        C(t) = tension_load(t) * (1 - ν - 2ν^2) / (1 - ν)

        ϵi(t) = C(t) / E
        ϵj(t) = 0.0
        ϵk(t) = 0.0

        σi(t) = (λ + 2G) * ϵi(t) + λ * ϵj(t) + λ * ϵk(t)
        σj(t) = λ * ϵi(t) + (λ + 2G) * ϵj(t) + λ * ϵk(t)
        σk(t) = λ * ϵi(t) + λ * ϵj(t) + (λ + 2G) * ϵk(t)

        [[σi(t) for t in λv], [σj(t) for t in λv], [σk(t) for t in λv]]
    end
    # point in the rand element selected
    p = rand(coordinates(e))
    # strain
    λv = load_factors(sa)
    ϵ = ϵ_ijk_analytic(λv, p[1], p[2], p[3])
    ϵ_1 = ϵ[1]
    ϵ_2 = ϵ[2]
    ϵ_3 = ϵ[3]
    # stress
    σ = σ_ijk_analytic(λv, p[1], p[2], p[3], λ, G)
    σ_1 = σ[1]
    σ_2 = σ[2]
    σ_3 = σ[3]

    ui_1, uj_1, uk_1, ui_2, uj_2, uk_2, ϵ_1, ϵ_2, ϵ_3, σ_1, σ_2, σ_3
end;

function test(sol::AbstractSolution)
    (; RTOL, ATOL) = parameters()
    p1, p2, e = test_points(sol)

    ui_1_num, uj_1_num, uk_1_num, ui_2_num, uj_2_num, uk_2_num,
    ϵi_num, ϵj_num, ϵk_num,
    σi_num, σj_num, σk_num = numerical_solution(sol, p1, p2, e)

    ui_1_analy, uj_1_analy, uk_1_analy, ui_2_analy, uj_2_analy, uk_2_analy,
    ϵi_analy, ϵj_analy, ϵk_analy,
    σi_analy, σj_analy, σk_analy = analytic_solution(sol, p1, p2, e)

    #-----------------------------
    # Test booleans for CI
    #-----------------------------
    @testset "Linear Extension example displacements Point 1" begin
        @test ui_1_num≈ui_1_analy rtol=RTOL
        @test uj_1_num≈uj_1_analy atol=ATOL
        @test norm(uj_1_analy)≈0 atol=ATOL
        @test uk_1_num≈uk_1_analy atol=ATOL
        @test norm(uk_1_analy)≈0 atol=ATOL
    end

    @testset "Linear Extension example displacements Point 2" begin
        @test ui_2_num≈ui_2_analy rtol=RTOL
        @test uj_2_num≈uj_2_analy atol=ATOL
        @test norm(uj_2_analy)≈0 atol=ATOL
        @test uk_2_num≈uk_2_analy atol=ATOL
        @test norm(uk_2_analy)≈0 atol=ATOL
    end

    @testset "Linear Extension example strains random Point " begin
        @test ϵi_num≈ϵi_analy rtol=RTOL
        @test ϵj_num≈ϵj_analy atol=ATOL
        @test norm(ϵj_analy)≈0 atol=ATOL
        @test ϵk_num≈ϵk_analy atol=ATOL
        @test norm(ϵk_num)≈0 atol=ATOL
    end

    @testset "Linear Extension example stress random Point " begin
        @test σi_num≈σi_analy rtol=RTOL
        @test σj_num≈σj_analy rtol=RTOL skip=true
        @test σk_num≈σk_analy rtol=RTOL skip=true
    end
end;

"Run the example"
function run()
    sol = solve()
    test(sol)
end;

run()
