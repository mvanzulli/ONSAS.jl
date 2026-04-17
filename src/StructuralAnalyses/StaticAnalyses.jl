"""
Module defining static analyses.
Each static state analysis is defined by an structure and a load factors vector. During
the analysis the static state of the structure is mutated through the load factors vector.
"""
module StaticAnalyses

using Dictionaries: dictionary
using Reexport
using SparseArrays

using ..Materials
using ..Entities
using ..Meshes
using ..Structures
using ..Structures
using ..StructuralAnalyses
using ..StructuralSolvers
using ..Solvers
using ..Solutions
using ..StaticStates
using ..Assemblers

@reexport import ..StructuralAnalyses: initial_time, current_time, final_time, times,
                                       iteration_residuals, is_done
@reexport import ..StructuralSolvers: next!
@reexport import ..Assemblers: assemble!, reset!

export AbstractStaticAnalysis, load_factors, current_load_factor, store!

""" Abstract supertype for all structural analysis.

An `AbstractStaticAnalysis` object facilitates the process of defining an static analysis
to be solved. Time variable for static analysis is used to obtain a load factor value.
Of course this abstract type inherits from `AbstractStructuralAnalysis` type,
and extends the following methods:

**Abstract Methods**

* [`initial_time`](@ref)
* [`current_time`](@ref)
* [`final_time`](@ref)
* [`times`](@ref)
* [`load_factors`](@ref)
* [`current_load_factor`](@ref)
* [`next!`](@ref)
* [`is_done`](@ref)
* [`reset!`](@ref)
* [`_solve!`](@ref)
* [`step!`](@ref)

**Common fields**
- `λᵥ`           -- stores the load factors vector.
- `current_step` -- stores the current step of the analysis.
- `state`        -- stores the current state of the analysis.

"""
abstract type AbstractStaticAnalysis <: AbstractStructuralAnalysis end

"Return the initial load factor of an structural analysis (always 0)."
initial_time(sa::AbstractStaticAnalysis) = first(sa.λᵥ)

"Return load factors for all N+1 states, starting at 0."
load_factors(sa::AbstractStaticAnalysis) = sa.λᵥ
times(sa::AbstractStaticAnalysis) = load_factors(sa)

"Return the current load factor of an structural analysis."
current_time(sa::AbstractStaticAnalysis) = sa.λᵥ[sa.current_step[] + 1]

"Return the final load factor of an structural analysis."
final_time(sa::AbstractStaticAnalysis) = last(sa.λᵥ)

"Return true if the structural analysis is completed."
function is_done(sa::AbstractStaticAnalysis)
    if sa.current_step[] > length(sa.λᵥ) - 1
        sa.current_step -= 1
        true
    else
        false
    end
end

"Return the current load factor of an structural analysis."
current_load_factor(sa::AbstractStaticAnalysis) = current_time(sa)

"Jumps to the next current load factor defined in the structural analysis."
next!(sa::AbstractStaticAnalysis) = sa.current_step += 1

"Sets the current load factor of the structural analysis to the initial load factor.
Also Reset! the iteration and `AbstractStructuralState`."
function reset!(sa::AbstractStaticAnalysis)
    sa.current_step = 1
    reset!(current_state(sa))
    @info "The current time of analysis have been reset."
    sa
end

"Assembles the Structure `s` (internal forces) during the `StaticAnalysis` `sa`."
function assemble!(s::AbstractStructure, sa::AbstractStaticAnalysis)
    state = current_state(sa)

    # Reset assembled magnitudes
    reset_assemble!(state)

    for (mat, mat_elements) in pairs(materials(s))
        for e in mat_elements

            # Global dofs of the element (dofs where K must be added)
            u_e = view(displacements(state), local_dofs(e))
            cache = elements_cache(state, e)
            fᵢₙₜ_e, kₛ_e, σ_e, ϵ_e = internal_forces(mat, e, u_e, cache)

            # Assembles the element internal magnitudes
            assemble!(state, fᵢₙₜ_e, e)
            assemble!(state, kₛ_e, e)
            assemble!(state, σ_e, ϵ_e, e)
        end
    end

    # Insert values in the assembler objet into the sysyem tangent stiffness matrix
    end_assemble!(state)
end

"Reset the assembled magnitudes in the state."
function reset_assemble!(state::FullStaticState)
    reset!(assembler(state))
    internal_forces(state) .= 0.0
    K = tangent_matrix(state)
    I, J, V = findnz(tangent_matrix(state))
    K[I, J] .= zeros(eltype(V))
    nothing
end

"Stores the current state into the solution."
function Base.push!(st_sol::Solution{<:FullStaticState}, c_state::FullStaticState)
    # Copies TODO Need to store all these?
    fdofs = free_dofs(c_state)
    Uᵏ = deepcopy(displacements(c_state))
    ΔUᵏ = deepcopy(Δ_displacements(c_state))
    fₑₓₜᵏ = deepcopy(external_forces(c_state))
    fᵢₙₜᵏ = deepcopy(internal_forces(c_state))
    Kₛᵏ = deepcopy(tangent_matrix(c_state))
    res_forces = deepcopy(c_state.res_forces)
    σᵏ = dictionary([e => deepcopy(σ) for (e, σ) in pairs(stress(c_state))])
    ϵᵏ = dictionary([e => deepcopy(ϵ) for (e, ϵ) in pairs(strain(c_state))])
    iter_state = deepcopy(iteration_residuals(c_state))
    # Empty assembler since the info is stored in k
    assembler = c_state.assembler
    linear_system = c_state.linear_system

    state_copy = FullStaticState(
        fdofs, ΔUᵏ, Uᵏ, fₑₓₜᵏ, fᵢₙₜᵏ, Kₛᵏ, res_forces, ϵᵏ, σᵏ, assembler,
        iter_state, linear_system)
    push!(states(st_sol), state_copy)
end

function store!(sol::Solution{<:StaticState}, state::FullStaticState, step::Int)
    # sol.states[1] is the initial state; load step k is stored at sol.states[k+1].
    solution_state = states(sol)[step + 1]
    sol_Uᵏ = displacements(solution_state)
    sol_σᵏ = stress(solution_state)
    sol_ϵᵏ = strain(solution_state)

    Uᵏ = deepcopy(displacements(state))
    sol_Uᵏ .= Uᵏ

    state_σᵏ = stress(state)
    state_ϵᵏ = strain(state)
    for e in keys(state_σᵏ)
        state_σᵏ_e = getindex(state_σᵏ, e)
        sol_σᵏ[e] .= state_σᵏ_e
        state_ϵᵏ_e = getindex(state_ϵᵏ, e)
        sol_ϵᵏ[e] .= state_ϵᵏ_e
    end
end

end # module
