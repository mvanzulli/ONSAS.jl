"""
Module defining structural solvers that can be used to solved different analyses.
Each solver consists of a data type with a convergence criterion and the iteration status.
A step! method is used to perform a single iteration step.
"""
module StructuralSolvers

using Reexport

using ..StructuralAnalyses
using ..Entities
using ..SolversConfig

@reexport import ..Assemblers: reset!
@reexport import ..SolversConfig: reset!
@reexport import ..StructuralAnalyses: tangent_matrix
@reexport import ..CommonSolve: solve, solve!
@reexport using ..SolversConfig

export step!, next!, tangent_matrix, solve, solve!, _solve!, default_solver, default_config

"Computes a step in time on the `analysis` considering the numerical `AbstractSolver` `solver`."
function step!(solver::AbstractSolver,
        analysis::AbstractStructuralAnalysis) end

"Increment the time step given of a structural analysis. Dispatch is done for different
solvers."
next!(a::AbstractStructuralAnalysis, solver::AbstractSolver) = a.t += time_step(a) # TODO Define `time_step` fallback.

"Return system tangent matrix in the structural state given a solver."
function tangent_matrix(st::AbstractStructuralState, alg::AbstractSolver) end

"""
Return the default [`AbstractSolver`](@ref) for a given analysis type.
Returns `nothing` for linear analyses (no outer iteration required).
Overloaded by each analysis module (e.g. `NonLinearStaticAnalyses`).
"""
function default_solver(::AbstractStructuralAnalysis)
    nothing
end

"""
Build an [`OnsasConfig`](@ref) with the solver automatically chosen for `problem`.
`default_solver` is overloaded per analysis type; the fallback returns `nothing`
(suitable for linear analyses that require no outer iteration loop).

This is the recommended way to get a fully-configured [`OnsasConfig`](@ref):
```julia
sol = solve(problem, default_config(problem))
```
"""
function default_config(problem::AbstractStructuralAnalysis;
        linear_solver::LinearSolverConfig = LinearSolverConfig())
    OnsasConfig(solver = default_solver(problem), linear_solver = linear_solver)
end

# ===============
# Solve function
# ===============

"""
Solve a structural analysis problem using an [`OnsasConfig`](@ref).
Mutates the problem state. Use [`solve`](@ref) to avoid mutation.
"""
function solve!(problem::AbstractStructuralAnalysis, config::OnsasConfig = default_config(problem))
    _solve!(problem, config)
end

"""
Non-mutating variant of [`solve!`](@ref): deepcopies the problem before solving.
"""
function solve(problem::AbstractStructuralAnalysis, config::OnsasConfig = default_config(problem))
    solve!(deepcopy(problem), config)
end

"Internal solve function to be overloaded by each analysis"
function _solve!(::AbstractStructuralAnalysis, ::OnsasConfig) end

function _default_linear_solver_tolerances(A::AbstractMatrix{<:Real}, b::Vector{<:Real})
    abstol = zero(real(eltype(b)))
    reltol = sqrt(eps(real(eltype(b))))
    maxiter = length(b)
    abstol, reltol, maxiter
end

end # module
