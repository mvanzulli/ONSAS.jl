# --------------------------------------------------------------------------
# Uniaxial Extension ExampleExercise 4 from section 6.5 in (Holzapfel,2000).
# For notation see: https://onsas.github.io/ONSAS.m/dev/examples/uniaxialExtension/
# --------------------------------------------------------------------------
using Test, LinearAlgebra, Suppressor, Roots
using ONSAS

include("uniaxial_mesh.jl") # Mesh Cube with Gmsh.jl

"Return the problem parameters"
function parameters()
    E = 1.0      # Young modulus in Pa
    ν = 0.3      # Poisson's ratio
    λ = E * ν / ((1 + ν) * (1 - 2 * ν))
    G = E / (2 * (1 + ν))
    p = 3        # Tension load in Pa
    Lx = 2.0     # Dimension in x of the box in m
    Ly = 1.0     # Dimension in y of the box in m
    Lz = 1.0     # Dimension in z of the box in m
    ms = 0.5     # Refinement factor for the mesh
    RTOL = 1e-4  # Relative tolerance for tests
    NSTEPS = 8   # Newton-Raphson load steps
    (; p, ν, E, λ, G, Lx, Ly, Lz, ms, RTOL, NSTEPS)
end;

#= -----------------------------------------------------------
Two cases are considered:
Case 1 - Non linear static analysis with a manufactured mesh and `SVK` material.
Case 2 - Non linear static analysis with a GMSH mesh and `HyperElastic` material defined
        from its strain energy function.
-------------------------------------------------------------=#
abstract type AbstractCase end
struct FirstCase <: AbstractCase end
struct SecondCase <: AbstractCase end

"Return the problem structural model"
function structure(::FirstCase)
    (; p, ν, E, Lx, Ly, Lz) = parameters()
    # -----------------------------------------------
    # Case 1 - Manufactured mesh and `SVK` material
    #------------------------------------------------
    # -------------------------------
    # Mesh
    #--------------------------------
    n1 = Node(0.0, 0.0, 0.0)
    n2 = Node(0.0, 0.0, Lz)
    n3 = Node(0.0, Ly, Lz)
    n4 = Node(0.0, Ly, 0.0)
    n5 = Node(Lx, 0.0, 0.0)
    n6 = Node(Lx, 0.0, Lz)
    n7 = Node(Lx, Ly, Lz)
    n8 = Node(Lx, Ly, 0.0)
    vec_nodes = [n1, n2, n3, n4, n5, n6, n7, n8]
    m = Mesh(; nodes = vec_nodes)
    ## Faces
    f1 = TriangularFace(n5, n8, n6, "loaded_face_1")
    f2 = TriangularFace(n6, n8, n7, "loaded_face_2")
    f3 = TriangularFace(n4, n1, n2, "x=0_face_1")
    f4 = TriangularFace(n4, n2, n3, "x=0_face_2")
    f5 = TriangularFace(n6, n2, n1, "y=0_face_1")
    f6 = TriangularFace(n6, n1, n5, "y=0_face_2")
    f7 = TriangularFace(n1, n4, n5, "z=0_face_1")
    f8 = TriangularFace(n4, n8, n5, "z=0_face_2")
    vec_faces = [f1, f2, f3, f4, f5, f6, f7, f8]
    append!(faces(m), vec_faces)
    ## Entities
    t1 = Tetrahedron(n1, n4, n2, n6, "tetra_1")
    t2 = Tetrahedron(n6, n2, n3, n4, "tetra_2")
    t3 = Tetrahedron(n4, n3, n6, n7, "tetra_3")
    t4 = Tetrahedron(n4, n1, n5, n6, "tetra_4")
    t5 = Tetrahedron(n4, n6, n5, n8, "tetra_5")
    t6 = Tetrahedron(n4, n7, n6, n8, "tetra_6")
    vec_elems = [t1, t2, t3, t4, t5, t6]
    append!(elements(m), vec_elems)
    # -------------------------------
    # Dofs
    #--------------------------------
    dof_dim = 3
    dof_u_symbol = :u
    set_dofs!(m, dof_u_symbol, dof_dim)
    # -------------------------------
    # Materials
    # -------------------------------
    svk = SVK(; E = E, ν = ν, label = "svk")
    mat = StructuralMaterial(svk => [t1, t2, t3, t4, t5, t6])
    # -------------------------------
    # Boundary conditions
    # -------------------------------
    # Fixed dofs
    bc1 = FixedField(:u, [1])
    bc2 = FixedField(:u, [2])
    bc3 = FixedField(:u, [3])
    # Load
    bc4 = GlobalLoad(:u, t -> [p * t, 0, 0])
    # Crete boundary conditions struct
    bcs = StructuralBoundaryCondition(bc1 => [f3, f4], bc2 => [f5, f6],
        bc3 => [f7, f8], bc4 => [f1, f2])
    # -------------------------------
    # Structure
    # -------------------------------
    Structure(m, mat, bcs)
end;

"Return the problem solution"
function solve(c::AbstractCase)
    (; NSTEPS) = parameters()
    s = structure(c)
    # -------------------------------
    # Structural Analysis
    # -------------------------------
    # Final load factor
    nsa = NonLinearStaticAnalysis(s; NSTEPS)
    # -------------------------------
    # Algorithm
    # -------------------------------
    tol_f = 1e-8
    tol_u = 1e-8
    max_iter = 30
    tols = ConvergenceSettings(tol_u, tol_f, max_iter)
    nr = NewtonRaphson(tols)
    # -------------------------------
    # Numerical solution
    # -------------------------------
    ONSAS.solve(nsa, OnsasConfig(solver = nr))
end;

"Computes numeric solution α, β and γ for analytic validation."
function αβγ_numeric(sol::AbstractSolution)
    (; Lx, Ly, Lz) = parameters()
    s = ONSAS.structure(analysis(sol))
    # Node at (Lx, Ly, Lz)
    n7 = nodes(s)[7]
    ui = displacements(sol, n7, 1)
    α = 1 .+ ui / Lx
    # Displacements in the y (component 2) axis at node 7
    uj = displacements(sol, n7, 2)
    β = 1 .+ uj / Ly
    # Displacements in the z (component 3) axis at node 7
    uk = displacements(sol, n7, 3)
    γ = 1 .+ uk / Lz
    α, β, γ, ui, uj, uk
end;

"Computes displacements numeric solution uᵢ, uⱼ and uₖ for analytic validation."
function u_ijk_numeric(α::Vector{<:Real}, β::Vector{<:Real}, γ::Vector{<:Real},
        x::Real, y::Real, z::Real)
    x * (α .- 1), y * (β .- 1), z * (γ .- 1)
end;

"Analytic load factor solution for the displacement `uᵢ` towards `x` axis at node `n₆`."
function load_factors_analytic(ux::Real)
    (; E, p, Lx) = parameters()
    1 / p * E * 0.5 * ((1 + ux / Lx)^3 - (1 + ux / Lx))
end

"Test case 1 problem solution"
function test(::FirstCase, sol::AbstractSolution)
    (; E, p, ν, RTOL) = parameters()
    a = analysis(sol)
    svk = materials(ONSAS.structure(a))[:svk]
    @testset "Verify VTK is written" begin
        write_vtk(sol, joinpath(@__DIR__, "uniaxial_extension"))
    end
    # -------------------------------
    # Numerical solution
    # -------------------------------
    α_numeric, β_numeric, _, ui, _, _ = αβγ_numeric(sol)
    # Extract ℙ and ℂ from the last state using a random element
    e = rand(elements(ONSAS.structure(a)))
    # Cosserat or second Piola-Kirchhoff stress tensor
    P_numeric = last(stress(sol, e))
    # Right hand Cauchy strain tensor
    C_numeric = last(strain(sol, e))
    # Load factors
    λ_numeric = load_factors(a)
    # -------------------------------
    # Analytic solution
    # -------------------------------
    λ_analytic = load_factors_analytic.(ui)
    α_analytic = find_zero(α -> E / 2 * α * (α^2 - 1) - p * last(λ_numeric), 1e-2)
    β_analytic = sqrt(-ν * (α_analytic^2 - 1) + 1)
    # Gradient tensor
    # 𝑢 = (αx, βy, γz)
    F_analytic = [α_analytic 0 0; 0 β_analytic 0; 0 0 β_analytic]
    # Right hand Cauchy tensor
    C_analytic = F_analytic' * F_analytic
    J = det(C_analytic)
    # Green-Lagrange strain tensor
    I = eye(3)
    E_analytic = 1 / 2 * (C_analytic - I)
    # Cosserat or second Piola-Kirchhoff stress tensor
    p₁, p₂ = lame_parameters(svk)
    S_analytic = p₁ * tr(E_analytic) * eye(3) + 2 * p₂ * E_analytic
    # First Piola-Kirchhoff stress tensor
    P_analytic = F_analytic * S_analytic
    # -------------------------------
    # Test solution
    # -------------------------------
    @testset "Case 1 Uniaxial Extension Example" begin
        @test λ_analytic≈λ_numeric rtol=RTOL
        @test α_analytic≈last(α_numeric) rtol=RTOL
        @test β_analytic≈last(β_numeric) rtol=RTOL
        @test C_analytic≈C_numeric rtol=RTOL
        @test P_analytic≈P_numeric rtol=RTOL
    end
end

"Return the problem structural model"
function structure(::SecondCase)
    (; λ, G, Lx, Ly, Lz, p, ms) = parameters()
    # -------------------------------
    # Materials
    # -------------------------------
    # Define a new HyperElastic material from the strain energy function
    ψ_svk(𝔼::AbstractMatrix, λ::Real, G::Real) = (λ / 2) * tr(𝔼)^2 + G * tr(𝔼^2)
    # The order must be the same defined in the strain energy beacuse we splat internally
    params = [λ, G]
    mat_label = "svkHyper"
    svkh = HyperElastic(params, ψ_svk, mat_label)
    # Material types without assigned elements
    mats = StructuralMaterial(svkh)
    # -------------------------------
    # Boundary Conditions
    # -------------------------------
    # Fixed dofs
    bc1_label = "fixed-ux"
    bc1 = FixedField(:u, [1], bc1_label)
    bc2_label = "fixed-uy"
    bc2 = FixedField(:u, [2], bc2_label)
    bc3_label = "fixed-uz"
    bc3 = FixedField(:u, [3], bc3_label)
    # Redefine the load boundary condition
    bc4_label = "tension"
    bc4 = Pressure(:u, t -> -p * t, bc4_label)
    # BoundaryConditions types without assigned node, feces and elements
    bc_labels = [bc1_label, bc2_label, bc3_label, bc4_label]
    bcs = StructuralBoundaryCondition(bc1, bc2, bc3, bc4)

    # -------------------------------
    # Entities
    # -------------------------------
    # Entities types without assigned nodes, faces and elements
    faces_label = "triangle"
    elems_label = "tetrahedron"
    vfaces = [TriangularFace(faces_label)]
    velems = [Tetrahedron(elems_label)]
    ents = StructuralEntity(velems, vfaces)
    entities_labels = [faces_label, elems_label]
    # -------------------------------
    # Mesh
    # -------------------------------
    filename = "uniaxial_extension"
    labels = [mat_label, entities_labels, bc_labels]
    local mesh_path
    output = @capture_out begin
        mesh_path = create_uniaxial_mesh(Lx, Ly, Lz, labels, filename, ms)
    end
    gmsh_println(output)
    msh_file = MshFile(mesh_path)
    # -------------------------------
    # Structure
    # -------------------------------
    Structure(msh_file, mats, bcs, ents)
end;

"Test case 2 problem solution"
function test(::SecondCase, sol::AbstractSolution)
    (; Lx, Ly, Lz, E, p, ν, λ, G, RTOL) = parameters()
    a = analysis(sol)
    s = ONSAS.structure(a)
    # -------------------------------
    # Numerical solution
    # -------------------------------
    α_numeric, β_numeric, γ_numeric, ui, _, _ = αβγ_numeric(sol)
    # Extract ℙ and ℂ from the last state using a random element
    e = rand(elements(s))
    # Cosserat or second Piola-Kirchhoff stress tensor
    P_numeric = last(stress(sol, e))
    # Right hand Cauchy strain tensor
    C_numeric = last(strain(sol, e))
    # Load factors
    λ_numeric = load_factors(a)
    # -------------------------------
    # Analytic solution
    # -------------------------------
    λ_analytic = load_factors_analytic.(ui)
    α_analytic = find_zero(α -> E / 2 * α * (α^2 - 1) - p * last(λ_numeric), 1e-2)
    β_analytic = sqrt(-ν * (α_analytic^2 - 1) + 1)
    # Gradient tensor
    # 𝑢 = (αx, βy, γz)
    F_analytic = [α_analytic 0 0; 0 β_analytic 0; 0 0 β_analytic]
    # Right hand Cauchy tensor
    C_analytic = F_analytic' * F_analytic
    J = det(C_analytic)
    # Green-Lagrange strain tensor
    I = eye(3)
    E_analytic = 1 / 2 * (C_analytic - I)
    # Cosserat or second Piola-Kirchhoff stress tensor
    p₁, p₂ = λ, G
    S_analytic = p₁ * tr(E_analytic) * eye(3) + 2 * p₂ * E_analytic
    # First Piola-Kirchhoff stress tensor
    P_analytic = F_analytic * S_analytic
    # -------------------------------
    # Test solution
    # -------------------------------
    @testset "Case 2 Uniaxial Extension Example" begin
        @test λ_analytic≈λ_numeric rtol=RTOL
        @test J > 0
        @test α_analytic≈last(α_numeric) rtol=RTOL
        @test β_analytic≈last(β_numeric) rtol=RTOL
        @test C_analytic≈C_numeric rtol=RTOL
        @test P_analytic≈P_numeric rtol=RTOL
    end
    # -------------------------------
    # Interpolator tests
    #--------------------------------
    rand_point = [[rand() * Lx, rand() * Ly, rand() * Lz]]
    ph = PointEvalHandler(mesh(s), rand_point)
    # Compute analytic solution at a random point
    uie, uje, uke = u_ijk_numeric(α_numeric, β_numeric, γ_numeric,
        rand_point[]...)
    r_ui = displacements(sol, ph, 1)
    r_uj = displacements(sol, ph, 2)
    r_uk = displacements(sol, ph, 3)
    @testset "Case 2 Uniaxial Extension Example Interpolation" begin
        @test uie≈r_ui rtol=RTOL
        @test uje≈r_uj rtol=RTOL
        @test uke≈r_uk rtol=RTOL
    end
end

"Run the example"
function run()
    for case in (FirstCase(), SecondCase())
        sol = solve(case)
        test(case, sol)
    end
end;

run()
