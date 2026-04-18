"""
Module defining configuration structs for structural solvers: convergence settings,
iteration tracking, convergence criteria, the abstract solver interface, and the
top-level `OnsasConfig` that bundles all parameters for `solve!` / `solve`.
"""
module SolversConfig

using LinearAlgebra: norm
using LinearSolve
using IterativeSolvers

import ..Assemblers: reset!

export DEFAULT_LINEAR_SOLVER, LinearSolver,
       ConvergenceSettings, residual_forces_tol, displacement_tol, max_iter_tol,
       AbstractConvergenceCriterion, ResidualForceCriterion, ΔUCriterion,
       ΔU_and_ResidualForce_Criteria, MaxIterCriterion, NotConvergedYet,
       ResidualsIterationStep, criterion, isconverged!, iterations, update!, reset!,
       AbstractSolver, step_size, tolerances,
       LinearSolverConfig, OnsasConfig

"Default LinearSolve.jl solver"
const DEFAULT_LINEAR_SOLVER = IterativeSolversJL_CG
"LinearSolve solver object. If is `nothing` default algorithm by `LinearSolve.jl` is used."
const LinearSolver = Union{SciMLBase.AbstractLinearAlgorithm, Nothing}

const INITIAL_Δ = 1e12

# ========================
# ConvergenceSettings
# ========================

"""
Facilitates the process of defining and checking numerical convergence.
"""
Base.@kwdef struct ConvergenceSettings
    "Relative displacement tolerance."
    rel_U_tol::Float64 = 1e-6
    "Relative residual force tolerance."
    rel_res_force_tol::Float64 = 1e-6
    "Maximum number of iterations."
    max_iter::Int = 20
end

"Show convergence settings."
function Base.show(io::IO, cs::ConvergenceSettings)
    println(io, "• ||ΔUᵏ||/||Uᵏ||  ≤ : $(displacement_tol(cs))")
    println(io, "• ||ΔRᵏ||/||Fₑₓₜ|| ≤ : $(residual_forces_tol(cs))")
    println(io, "• iter k            ≤ : $(max_iter_tol(cs))")
end

"Return residual forces tolerance set."
residual_forces_tol(tols::ConvergenceSettings) = tols.rel_res_force_tol

"Return displacements tolerance set."
displacement_tol(tols::ConvergenceSettings) = tols.rel_U_tol

"Return the maximum number of iterations set."
max_iter_tol(tols::ConvergenceSettings) = tols.max_iter

# ========================
# Convergence Criteria
# ========================

""" Abstract supertype for all convergence criterion."""
abstract type AbstractConvergenceCriterion end

""" `ResidualForceCriterion` convergence criterion. """
struct ResidualForceCriterion <: AbstractConvergenceCriterion end

""" `ΔUCriterion` indicates displacements increment convergence criterion. """
struct ΔUCriterion <: AbstractConvergenceCriterion end

"""
`ΔU_and_ResidualForce_Criteria` convergence criterion indicates that both
ΔU and residual forces converged.
"""
struct ΔU_and_ResidualForce_Criteria <: AbstractConvergenceCriterion end

""" `MaxIterCriterion` criteria indicates that the maximum number of iterations has been reached. """
struct MaxIterCriterion <: AbstractConvergenceCriterion end

""" `NotConvergedYet` indicates that the current iteration has not converged. """
struct NotConvergedYet <: AbstractConvergenceCriterion end

# ========================
# ResidualsIterationStep
# ========================

"""
Stores the convergence information at the current iteration step.
"""
Base.@kwdef mutable struct ResidualsIterationStep{T}
    "Norm of the displacement increment."
    ΔU_norm::T = INITIAL_Δ
    "Norm of the residual force increment."
    Δr_norm::T = INITIAL_Δ
    "Relative norm of the displacement increment."
    ΔU_rel::T = INITIAL_Δ
    "Relative norm of the residual force increment."
    Δr_rel::T = INITIAL_Δ
    "Current iteration number."
    iter::Int = 0
    criterion::AbstractConvergenceCriterion = NotConvergedYet()
end

"Increments a `ResidualsIterationStep` `i_step` by 1."
step!(i_step::ResidualsIterationStep) = i_step.iter += 1

"Return the iterations done so far."
iterations(i_step::ResidualsIterationStep) = i_step.iter

"Return the current convergence criterion."
criterion(ri_step::ResidualsIterationStep) = ri_step.criterion

"Return the current absolute and relative residual forces norm."
residual_forces_tol(ri_step::ResidualsIterationStep) = (ri_step.Δr_rel, ri_step.Δr_norm)

"Return the current absolute and relative displacement norm."
displacement_tol(ri_step::ResidualsIterationStep) = (ri_step.ΔU_rel, ri_step.ΔU_norm)

"Sets the iteration step to 0."
function reset!(ri_step::ResidualsIterationStep{T}) where {T <: Real}
    ri_step.ΔU_norm = ri_step.Δr_norm = ri_step.ΔU_rel = ri_step.Δr_rel = INITIAL_Δ *
                                                                          ones(T)[1]
    ri_step.iter = 0
    ri_step.criterion = NotConvergedYet()
    ri_step
end

"Sets the iteration step with nothing."
function reset!(ri_step::ResidualsIterationStep{<:Nothing})
    ri_step.iter = 0
    ri_step.criterion = NotConvergedYet()
    ri_step
end

"Updates the current convergence criterion."
function update!(ri_step::ResidualsIterationStep, criterion::AbstractConvergenceCriterion)
    ri_step.criterion = criterion
end

"Updates the iteration step with the current values of the displacement and forces residuals."
function update!(
        ri_step::ResidualsIterationStep, ΔU_norm::Real, ΔU_rel::Real, Δr_norm::Real,
        Δr_rel::Real)
    ri_step.ΔU_norm = ΔU_norm
    ri_step.ΔU_rel = ΔU_rel
    ri_step.Δr_norm = Δr_norm
    ri_step.Δr_rel = Δr_rel
    step!(ri_step)
    ri_step
end

"Updates the convergence criteria."
function isconverged!(ri_step::ResidualsIterationStep, cs::ConvergenceSettings)
    ΔU_relᵏ, ΔU_normᵏ = displacement_tol(ri_step)
    Δr_relᵏ, Δr_normᵏ = residual_forces_tol(ri_step)

    @assert ΔU_relᵏ>0 "Residual displacements norm must be greater than 0."
    @assert Δr_relᵏ>0 "Residual forces norm must be greater than 0."

    ΔU_rel_tol = displacement_tol(cs)
    Δr_rel_tol = residual_forces_tol(cs)
    max_iter = max_iter_tol(cs)

    criterion = if Δr_relᵏ ≤ Δr_rel_tol
        ResidualForceCriterion()
    elseif ΔU_relᵏ ≤ ΔU_rel_tol
        ΔUCriterion()
    elseif iterations(ri_step) > max_iter
        @warn "Maximum number of iterations was reached."
        MaxIterCriterion()
    else
        NotConvergedYet()
    end
    update!(ri_step, criterion)
end

# ========================
# AbstractSolver
# ========================

"""
Abstract supertype for all direct integration methods.
"""
abstract type AbstractSolver end

"Return the step size."
step_size(solver::AbstractSolver) = solver.Δt

"Return the numerical tolerances."
tolerances(solver::AbstractSolver) = solver.tol

# ========================
# LinearSolverConfig
# ========================

"""
Bundles the linear-system solver algorithm and its solve strategy.
"""
Base.@kwdef struct LinearSolverConfig
    "Any `LinearSolve.jl` algorithm (or `nothing` to use the default)"
    algorithm::SciMLBase.AbstractLinearAlgorithm = IterativeSolversJL_CG()

    """
    When `true` the linear system is solved in-place, reusing the
    factorisation across load steps; when `false` a fresh `LinearProblem` is
    constructed every step.
    """
    inplace::Bool = false
end

function Base.show(io::IO, lsc::LinearSolverConfig)
    println(io, "LinearSolverConfig:")
    println(io, "  algorithm: $(lsc.algorithm)")
    println(io, "  inplace  : $(lsc.inplace)")
end

# ========================
# OnsasConfig
# ========================

"""
Bundles all parameters for `solve!` / `solve`.

- `solver`: the nonlinear iteration algorithm (e.g. [`NewtonRaphson`](@ref)).
  Set to `nothing` for linear analyses, which require no outer iteration loop.
- `linear_solver`: settings for the inner linear-system solve; see
  [`LinearSolverConfig`](@ref).
"""
Base.@kwdef struct OnsasConfig
    solver::Union{AbstractSolver, Nothing} = nothing
    linear_solver::LinearSolverConfig = LinearSolverConfig()
end

function Base.show(io::IO, cfg::OnsasConfig)
    println(io, "OnsasConfig:")
    println(io, "  solver       : $(cfg.solver)")
    print(io, "  linear_solver: ")
    show(io, cfg.linear_solver)
end

end # module
