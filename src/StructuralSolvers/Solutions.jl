"""
Module defining solutions interface, `AbstractSolution`.

Each solution can at least access to displacements and external forces. More complete solutions
can include different forces vector, stresses and strains.

Of course the solution contains on the analysis and solver used to solve the problem.
"""
module Solutions

using Reexport, PrettyTables, Dictionaries, LinearSolve

using ..Utils
using ..Entities
using ..Nodes
using ..Meshes
using ..Interpolators
using ..Handlers
using ..Meshes
using ..Structures
using ..StructuralAnalyses
using ..StaticStates
using ..StructuralSolvers

@reexport import ..StructuralAnalyses: displacements, external_forces, iteration_residuals
@reexport import ..StructuralSolvers: residual_forces_tol, displacement_tol, criterion,
                                      iterations
@reexport import ..Entities: internal_forces, inertial_forces, strain, stress

export AbstractSolution, Solution, analysis, solver, states, reference_state,
       displacements, external_forces, iteration_residuals, deformed_node_positions

"""
Abstract supertype for all structural analysis solutions.

**Abstract Methods**
* [`displacements`](@ref)
* [`external_forces`](@ref)
* [`internal_forces`](@ref)
* [`stress`](@ref)
* [`strain`](@ref)

**Abstract fields**
* analysis
* solver
"""
abstract type AbstractSolution end

"Return the analysis solved."
analysis(sol::AbstractSolution) = sol.analysis

"Return the solver used to solve the analysis."
solver(sol::AbstractSolution) = sol.solver

"""
Solution that stores all states of the analysis.

`states[1]` is the initial (unloaded) state at time 0, capturing any applied boundary conditions
before the first load step. `states[k+1]` is the converged state at load step `k`.
This convention matches standard FEA solvers (FEBio, Abaqus, OpenSees): the reference state
is always at t = 0, and each subsequent entry corresponds to a load/time increment.
"""
struct Solution{ST <: AbstractStaticState,
    A <: AbstractStructuralAnalysis,
    SS <: Union{AbstractSolver, LinearSolver}} <: AbstractSolution
    """
    Purely undeformed reference configuration (zero displacements, zero stress, zero strain).
    Corresponds to the continuum mechanics reference configuration — independent of any applied BCs.
    """
    reference_state::ST
    """
    Vector of structural states. `states[1]` is the initial state at t = 0, capturing any applied
    boundary conditions before the first load step. `states[k+1]` is the converged state at load
    step `k`. Length is `N + 1` for `N` load steps.
    """
    states::Vector{ST}
    "Analysis solved."
    analysis::A
    "Solver employed."
    solver::SS
end

"Constructor pre-allocating states with an initial state at index 1."
function Solution(analysis::A,
        solver::SS) where {A <: AbstractStructuralAnalysis,
        SS <: Union{AbstractSolver, LinearSolver}}

    # TODO Use concrete types.
    state = current_state(analysis)
    Uᵏ = displacements(state)
    ϵᵏ = strain(state)
    σᵏ = stress(state)

    # λᵥ starts at 0; the N load steps are λᵥ[2:end].
    nsteps = length(analysis.λᵥ) - 1
    # states[1] = initial state; states[2..N+1] = one per load step.
    states = Vector{StaticState}(undef, nsteps + 1)

    # Reference configuration: purely undeformed (zero U, zero σ, zero ε).
    ref_state = StaticState(
        zero(Uᵏ),
        dictionary([e => zero(ϵ) for (e, ϵ) in pairs(ϵᵏ)]),
        dictionary([e => zero(σ) for (e, σ) in pairs(σᵏ)]))

    # Capture the initial state from the analysis (reflects BCs and any initial conditions,
    # e.g. prescribed displacements, pre-stress, thermal initial conditions — FEBio step 0).
    states[1] = StaticState(
        copy(Uᵏ),
        dictionary([e => copy(ϵ) for (e, ϵ) in pairs(ϵᵏ)]),
        dictionary([e => copy(σ) for (e, σ) in pairs(σᵏ)]))

    # Pre-allocate the remaining states for each load step.
    for i in 2:(nsteps + 1)
        sol_Uᵏ = similar(Uᵏ)
        sol_σᵏ = dictionary([e => similar(σ) for (e, σ) in pairs(σᵏ)])
        sol_ϵᵏ = dictionary([e => similar(ϵ) for (e, ϵ) in pairs(ϵᵏ)])
        states[i] = StaticState(sol_Uᵏ, sol_ϵᵏ, sol_σᵏ)
    end

    Solution{StaticState, A, SS}(ref_state, states, analysis, solver)
end

"Generic minimal show method for a generic `solution`"
function Base.show(io::IO, ::MIME"text/plain",
        solution::Solution)
    println("Analysis solved:")
    println("----------------\n")
    show(io, analysis(solution))

    println("\nSolver employed:")
    println("----------------\n")
    show(io, solver(solution))
end

"Print the solution table"
function _print_table(solution::AbstractSolution)
    header = ["iter", "time", "||Uᵏ||", "||ΔUᵏ||/||Uᵏ||", "||ΔRᵏ||", "||ΔRᵏ||/||Fₑₓₜ||",
        "convergence criterion", "iterations"]

    # states[1] is the initial state (not a load step); skip it so lengths match λᵥ.
    ΔU_rel = getindex.(displacement_tol(solution), 1)[2:end]
    ΔU = getindex.(displacement_tol(solution), 2)[2:end]
    ΔR_rel = getindex.(residual_forces_tol(solution), 1)[2:end]
    ΔR = getindex.(residual_forces_tol(solution), 2)[2:end]
    t = analysis(solution).λᵥ
    criterions = [string(criterion_step)[1:(end - 2)]
                  for criterion_step in criterion(solution)][2:end]
    iters = iterations(solution)[2:end]
    num_times = length(t)
    data = hcat(collect(1:num_times), t, ΔU, ΔU_rel, ΔR, ΔR_rel, criterions, iters)

    hl = Highlighter(;
        f = (data, i, j) -> i % 2 == 0,
        crayon = Crayon(; foreground = :white, background = :black, bold = :true))

    pretty_table(data;
        highlighters = hl,
        header = header,
        display_size = (20, 1000),
        vcrop_mode = :middle,
        formatters = (ft_printf("%i", [1]),
            ft_printf("%.2f", [2]),
            ft_printf("%e", [3, 4, 5, 6])),
        alignment = :c)
end

"Return the reference (undeformed) configuration: zero displacements, zero stress, zero strain."
reference_state(sol::Solution) = sol.reference_state

"Return all states: states[1] is the initial state, states[k+1] is load step k."
states(sol::Solution) = sol.states

for f in [:displacements, :internal_forces, :external_forces]
    "Return the $f vector Uᵏ at every time steps."
    @eval $f(st_sol::Solution) = $f.(states(st_sol))

    "Return the $f at a certain dof for every time step."
    @eval $f(st_sol::Solution, dof::Dof) = getindex.($f(st_sol), Utils.index(dof))

    "Return the $f at a certain dof at `time_index` time step."
    @eval $f(st_sol::Solution, dof::Dof,
        time_index::Integer) = getindex(
        $f(st_sol)[time_index],
        Utils.index(dof))
    "Return the $f at a certain dof's vector at `time_index` time step."
    @eval $f(st_sol::Solution,
        vdof::Vector{Dof},
        time_index::Integer) = [getindex(
                                    $f(st_sol)[time_index],
                                    Utils.index(dof))
                                for dof in vdof]

    "Return the $f at a certain dof's vector for every time step."
    @eval $f(st_sol::Solution, vdof::Vector{Dof}) = [$f(st_sol, dof) for dof in vdof]

    "Return the $f at a certain node for every time step."
    @eval $f(st_sol::Solution, n::AbstractNode) = $f(st_sol, reduce(vcat, collect(dofs(n))))

    "Return the $f component at a certain node for every time step."
    @eval $f(st_sol::Solution, n::AbstractNode, component::Int) = $f(st_sol, n)[component]

    "Return the $f of an element for every time step."
    @eval $f(st_sol::Solution, e::AbstractElement) = [$f(st_sol, n) for n in nodes(e)]
end

for f in [:stress, :strain]
    "Return the $f for every time step."
    @eval $f(st_sol::Solution) = $f.(states(st_sol))

    "Return the $f at a certain element for every time step."
    @eval function $f(st_sol::Solution, e::AbstractElement)
        [getindex($f.(states(st_sol))[step], e) for step in 1:length(states(st_sol))]
    end

    "Return the $f at a certain element at `time_step`."
    @eval function $f(st_sol::Solution, e::AbstractElement, time_index::Integer)
        getindex($f.(states(st_sol))[time_index], e)
    end
end

"Return the residuals iteration object at every time step."
iteration_residuals(st_sol::Solution) = iteration_residuals.(states(st_sol))

"Return the final residuals forces at each time step."
residual_forces_tol(st_sol::Solution) = residual_forces_tol.(iteration_residuals(st_sol))

"Return the final residual displacement at each time step."
displacement_tol(st_sol::Solution) = displacement_tol.(iteration_residuals(st_sol))

"Return the convergence criterion at each time step."
criterion(st_sol::Solution) = criterion.(iteration_residuals(st_sol))

"Return the number of iterations at each time step."
iterations(st_sol::Solution) = iterations.(iteration_residuals(st_sol))

"Return the displacements solution at the points in the point evaluator handler."
function displacements(st_sol::Solution, peh::PointEvalHandler)
    interpolate(st_sol, displacements, interpolator(peh))
end

"Return the displacements component in the point evaluator handler."
function displacements(st_sol::Solution, peh::PointEvalHandler, i::Int)
    sol_points = getindex.(displacements(st_sol, peh), i)
    if length(sol_points) == 1
        getindex(sol_points)
    else
        sol_points
    end
end

"Return the internal forces solution at the points in the point eval handler."
function internal_forces(st_sol::Solution, peh::PointEvalHandler)
    interpolate(st_sol, internal_forces, interpolator(peh))
end

"Return the internal force component at the points in the point eval handler."
function internal_forces(st_sol::Solution, peh::PointEvalHandler, i::Int)
    getindex.(internal_forces(st_sol, peh), i)
end

"Return the external forces at the points in the point eval handler."
function external_forces(st_sol::Solution, peh::PointEvalHandler)
    interpolate(st_sol, external_forces, interpolator(peh))
end

"Return the external force component at the points in the point eval handler."
function external_forces(st_sol::Solution, peh::PointEvalHandler, i::Int)
    getindex.(external_forces(st_sol, peh), i)
end

"Return the stress at the elements where the points are located in the point eval handler."
function stress(st_sol::Solution, peh::PointEvalHandler)
    point_to_element_vec = points_to_element(interpolator(peh))
    vec_points = points(peh)
    num_points = length(vec_points)

    sol_points = Vector{Vector{<:AbstractMatrix{Float64}}}(undef, num_points)

    for index_p in 1:num_points
        element_p = point_to_element_vec[index_p]
        sol_points[index_p] = stress(st_sol, element_p)
    end
    sol_points
end

"Return the strains at the elements where the points are located in the point eval handler."
function strain(st_sol::Solution, peh::PointEvalHandler)
    point_to_element_vec = points_to_element(interpolator(peh))
    vec_points = points(peh)
    num_points = length(vec_points)

    sol_points = Vector{Vector{<:AbstractMatrix{Float64}}}(undef, num_points)

    for index_p in 1:num_points
        element_p = point_to_element_vec[index_p]
        sol_points[index_p] = strain(st_sol, element_p)
    end
    sol_points
end

"Return deformed nodes of the mesh at time index."
function deformed_node_positions(st_sol::Solution, t_i::Int)
    mesh_nodes = nodes(mesh(analysis(st_sol).s))
    # Get nodes coordinates type
    CT = typeof(coordinates(rand(mesh_nodes)))

    deformed_points = Vector{CT}(undef, length(mesh_nodes))
    u_node = Vector{Float64}(undef, dimension(first(mesh_nodes)))

    for (i, node) in enumerate(mesh_nodes)
        displacements_node = displacements(st_sol, node)
        for dim in 1:dimension(node)
            u_node[dim] = displacements_node[dim][t_i]
        end
        deformed_points[i] = coordinates(node) + u_node
    end
    deformed_points
end

end # module
