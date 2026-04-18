using Test
using ONSAS

"Return the problem parameters"
function parameters()
    L = 3                 # Length
    b = 0.2               # Cross-section Width
    h = 0.3               # Cross-section Height
    ρ = 2400              # Material density
    q = b * h * ρ * 9.81  # Distributed load
    Px = 1e2              # Nodal load in x
    Py = 1e3              # Nodal load in y
    E = 210e9             # Elastic Young's modulus.
    RTOL = 1e-4           # Relative tolerance for tests
    N = 5                 # Number of elements
    NSTEPS = 10           # Number of load factors steps
    (; b, h, ρ, q, Px, Py, E, RTOL, N, NSTEPS, L)
end

"Return the problem structural model"
function structure()
    (; L, N, b, h, E, ρ, Px, Py) = parameters()
    # -------------
    # Mesh
    # -------------
    x_coords = range(0, L, N + 1)
    nodes = [Node(promote(xi, 0, 0.0)) for xi in x_coords]  # to be safe when arguments supplied to Node command are not of  same types
    S = Rectangle(h, b)
    frames = [Frame(nodes[j], nodes[j + 1], S) for j in 1:(length(nodes) - 1)]
    mesh = Mesh(; nodes = nodes, elements = frames)
    set_dofs!(mesh, :u, 3)
    set_dofs!(mesh, :θ, 3)
    # ------------------------
    # Materials
    # ------------------------
    i = IsotropicLinearElastic(E, 0.3)
    materials = StructuralMaterial(i => frames)
    # -------------------------------
    # Boundary conditions
    # -------------------------------
    bc1 = FixedField(:u, [1, 2, 3])
    bc2 = FixedField(:θ, [1, 2, 3])
    bc3 = GlobalLoad(:u, t -> [Px, -Py, 0])
    boundary_conditions = StructuralBoundaryCondition(bc1 => [nodes[1]], bc2 => [nodes[1]],
        bc3 => [nodes[end]])

    Structure(mesh, materials, boundary_conditions)
end

"Return the problem solution"
function solve()
    s = structure()
    # -------------------------------
    # Structural Analysis
    # -------------------------------
    (; NSTEPS) = parameters()
    sa = LinearStaticAnalysis(s; NSTEPS)
    # -------------------------------
    # Numerical solution
    # -------------------------------
    ONSAS.solve(sa)
end;

"Test problem solution"
function test(sol::AbstractSolution)
    (; Px, Py, b, h, E, L, RTOL) = parameters()
    sa = analysis(sol)
    right_most_node = last(ONSAS.nodes(mesh(ONSAS.structure(sa))))

    @testset "Displacements at the right-most node" begin
        Izz = b * h^3 / 12
        A = b * h
        @test displacements(sol, right_most_node, 2)[end]≈-Py * L^3 / (3 * E * Izz) rtol=RTOL
        @test displacements(sol, right_most_node, 6)[end]≈-Py * L^2 / (2 * E * Izz) rtol=RTOL
        @test displacements(sol, right_most_node, 1)[end]≈Px * L / (E * A) rtol=RTOL
    end
end

"Run the example."
function run()
    sol = solve()
    write_vtk(sol, "cantilever")
    test(sol)
end

run()
