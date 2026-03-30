"""
Module defining non-linear static analysis.
This file inherits from src/StructuralAnalyses/StaticAnalyses.jl and extends the methods for
geometric non-linear static analysis.
"""
module NonLinearStaticAnalyses

using LinearAlgebra: norm
using LinearSolve
using Reexport

using ..Utils
using ..StructuralBoundaryConditions
using ..Structures
using ..Assemblers
using ..StaticStates
using ..StructuralAnalyses
using ..StaticAnalyses
using ..StructuralSolvers
using ..Solvers
using ..Solutions

@reexport import ..StructuralSolvers: _solve!, step!, default_solver

export NonLinearStaticAnalysis

"""
A non linear static analysis is a collection of parameters for defining the static analysis of the structure.
In the static analysis, the structure is analyzed at a given load factor (this variable is analog to time).
As this analysis is nonlinear the stiffness of the structure is updated at each iteration.
"""
mutable struct NonLinearStaticAnalysis{
    S <: AbstractStructure, R <: Real, LFV <: Vector{R}} <:
               AbstractStaticAnalysis
    "Structure to be analyzed."
    const s::S
    "Structural state."
    const state::FullStaticState
    "Load factors vector of the analysis."
    const λᵥ::LFV
    "Current load factor step."
    current_step::Int64
end

"Default solver for non-linear static analysis is Newton-Raphson with default tolerances."
default_solver(::NonLinearStaticAnalysis) = NewtonRaphson()

"Constructor for a non linear analysis with load factors, optional initial step and initial state."
function NonLinearStaticAnalysis(s::S, λᵥ::LFV;
        initial_state::FullStaticState = FullStaticState(s),
        initial_step::Int = 1) where {S <: AbstractStructure,
        LFV <: AbstractVector{<:Real}}
    !(1 ≤ initial_step ≤ length(λᵥ)) &&
        throw(ArgumentError("initial_step must be in [1, $(length(λᵥ))] but is: $initial_step."))
    NonLinearStaticAnalysis(s, initial_state, λᵥ, initial_step)
end

"Constructor for non linear static analysis given a final time (or load factor) and the number of steps."
function NonLinearStaticAnalysis(
        s::AbstractStructure, t₁::Real = 1.0; NSTEPS = 10, initial_step::Int = 1)
    t₀ = t₁ / NSTEPS
    λᵥ = collect(LinRange(t₀, t₁, NSTEPS))
    NonLinearStaticAnalysis(s, λᵥ; initial_step)
end

function Base.show(io::IO, sa::NonLinearStaticAnalysis)
    println("NonLinearStaticAnalysis for:")
    println("• Current load factor of $(sa.current_step).")
    show(io, sa.s)
    show(io, sa.state)
end

"Solves an non linear static analysis problem with a given solver."
function _solve!(sa::NonLinearStaticAnalysis, config::OnsasConfig)
    alg = config.solver
    linear_solver = config.linear_solver.algorithm
    linear_solve_inplace = config.linear_solver.inplace
    s = structure(sa)
    # Initialize solution.
    sol = Solution(sa, alg)

    # Load factors iteration.
    while !is_done(sa)
        step = sa.current_step

        # Reset assembled magnitudes
        reset!(current_iteration(sa))

        # Computes external forces
        external_forces(current_state(sa)) .= 0
        @debugtime "Assemble external forces" apply!(sa, load_bcs(boundary_conditions(s)))

        # Displacements iteration.
        while isconverged!(current_iteration(sa), tolerances(alg)) isa NotConvergedYet
            # Compute residual forces and tangent matrix.
            @debugtime "Assemble internal forces" assemble!(s, sa)

            # Increment structure displacements `U = U + ΔU`.
            @debugtime "Step" step!(sa, alg, linear_solver; linear_solve_inplace)
        end
        # Save current state.
        @debugtime "Save current state" store!(sol, current_state(sa), step)

        # Increment the time or load factor step.
        @debugtime "Next step" next!(sa)
    end
    sol
end

"Computes ΔU for solving the non linear static analysis with a Newton Raphson method."
function step!(sa::NonLinearStaticAnalysis, ::NewtonRaphson,
        linear_solver::SciMLBase.AbstractLinearAlgorithm;
        linear_solve_inplace::Bool)
    # Extract state info
    state = current_state(sa)
    free_dofs_idx = free_dofs(state)
    linear_system = state.linear_system

    # Compute residual forces r = Fext - Fint
    r = residual_forces!(state)
    linear_system.b .= state.res_forces

    # Update stiffness matrix K
    linear_system.A .= view(tangent_matrix(state), free_dofs_idx, free_dofs_idx)

    # Define tolerances
    abstol, reltol,
    maxiter = StructuralSolvers._default_linear_solver_tolerances(
        linear_system.A,
        linear_system.b)

    # Compute ΔU
    sol = if linear_solve_inplace
        solve!(linear_system, linear_solver; abstol, reltol, maxiter)
    else
        linear_problem = LinearProblem(linear_system.A, linear_system.b)
        solve(linear_problem, linear_solver; abstol, reltol, maxiter)
    end
    ΔU = Δ_displacements!(state, sol.u)

    # Compute norms
    norm_ΔU = norm(ΔU)
    rel_norm_ΔU = norm_ΔU / norm(displacements(state))
    norm_r = norm(r)
    rel_norm_r = norm_r / norm(external_forces(state))

    # Update displacements into the state.
    state.Uᵏ[free_dofs_idx] .+= ΔU

    # Update iteration
    update!(current_iteration(sa), norm_ΔU, rel_norm_ΔU, norm_r, rel_norm_r)
end

"Show the solution when solved with an in-house algorithm."
function Base.show(io::IO, ::MIME"text/plain",
        solution::Solution{<:FullStaticState})
    show(io, solution)

    println("\nStats:")
    println("----------")
    # Check convergence
    is_any_step_not_converged = any([criterion_step isa
                                     Union{NotConvergedYet, MaxIterCriterion}
                                     for criterion_step in criterion(solution)])

    num_iterations = reduce(+, iterations(solution))
    avg_iterations = round(num_iterations / length(states(solution)); digits = 1)
    println("• Number of linear systems solved: $num_iterations")
    println("• Average of iterations per step : $avg_iterations")
    println("• Convergence success            : $(!is_any_step_not_converged)")
    _print_table(solution)
end

end # module
