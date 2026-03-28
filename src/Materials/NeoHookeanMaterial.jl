"Module defining a Neo-Hookean hyper-elastic material."
module NeoHookeanMaterial

using ForwardDiff, LinearAlgebra, Reexport

using ..HyperElasticMaterials
using ..Utils

@reexport import ..LinearElasticMaterials: lame_parameters, elasticity_modulus,
                                           shear_modulus,
                                           bulk_modulus, poisson_ratio
@reexport import ..HyperElasticMaterials: cosserat_stress!, strain_energy

export NeoHookean

"""
Material with Neo-Hookean properties.
The strain energy `Ψ` is: `Ψ(𝔼)` = `G`/2 (tr(`ℂ`) -2 *log(`J`))^2 + `K`/2 (`J` - 1)^2

For context see the wikipedia article on [Neo-Hookean_solid](https://en.wikipedia.org/wiki/Neo-Hookean_solid).

It is also possible to construct a `NeoHookean` material given its elasticity and shear modulus `E`, `ν` respectively and its density `ρ`.
For context see the wikipedia article on [Lamé parameters](https://en.wikipedia.org/wiki/Lam%C3%A9_parameters).
"""
struct NeoHookean{T <: Real} <: AbstractHyperElasticMaterial
    "Bulk modulus."
    K::T
    "Shear modulus `G` or second Lamé parameter `μ`."
    G::T
    "Density (`nothing` for static cases)."
    ρ::Density
    "Material label."
    label::Label
    function NeoHookean(K::T, G::T, ρ::Density, label::Label = NO_LABEL) where {T <: Real}
        if ρ isa Real
            ρ > 0 || error("Density must be positive.")
        end
        @assert K≥0 "The bulk modulus `K` must be positive."
        @assert G≥0 "The shear modulus or second Lamé parameter `μ` must be positive."
        new{T}(K, G, ρ, Symbol(label))
    end
end

"Constructor for `NeoHookean` material with no density."
function NeoHookean(K::T, G::T, label::Label = NO_LABEL) where {T <: Real}
    NeoHookean(K, G, nothing, label)
end

"Constructor for `NeoHookean` material given its elasticity and shear modulus `E`, `ν` respectively and its density `ρ`."
function NeoHookean(; E::Real, ν::Real, ρ::Density = nothing, label::Label = NO_LABEL)
    # Compute λ, μ and K (μ = G) given E and ν.
    λ = E * ν / ((1 + ν) * (1 - 2 * ν))
    G = E / (2 * (1 + ν))
    K = λ + 2 * G / 3
    NeoHookean(K, G, ρ, label)
end

"Return the strain energy for a `NeoHookean` material `m` and the Green-Lagrange strain tensor `𝔼`."
function strain_energy(m::NeoHookean, 𝔼::AbstractMatrix)
    ℂ = Symmetric(2 * 𝔼 + eye(3))
    J = sqrt(det(ℂ))
    # First invariant
    I₁ = tr(ℂ)
    # Strain energy function
    Ψ = shear_modulus(m) / 2 * (I₁ - 2 * log(J)) + bulk_modulus(m) / 2 * (J - 1)^2
end

"Return Lamé parameters `λ` and `G` from a `NeoHookean` material `m`."
function lame_parameters(m::NeoHookean)
    G = shear_modulus(m)
    λ = bulk_modulus(m) - 2 * G / 3
    λ, G
end

"Return the shear modulus `G` from a `NeoHookean` material `m`."
shear_modulus(m::NeoHookean) = m.G

"Return the Poisson's ration `ν` form a `NeoHookean` material `m`."
function poisson_ratio(m::NeoHookean)
    λ, G = lame_parameters(m)
    λ / (2 * (λ + G))
end

"Return the elasticity modulus `E` form a `NeoHookean` material `m`."
function elasticity_modulus(m::NeoHookean)
    λ, G = lame_parameters(m)
    G * (3 * λ + 2 * G) / (λ + G)
end

"Return the bulk_modulus `K` for a `NeoHookean` material `m`."
bulk_modulus(m::NeoHookean) = m.K

"Return the Cosserat stress tensor `𝕊` given the Green-Lagrange `𝔼` strain tensor."
function _S_analytic(
        m::NeoHookean, E::AbstractMatrix; eye_cache::AbstractMatrix{<:Real} = eye(3))
    # Right hand Cauchy strain tensor
    C = Symmetric(2 * E + eye_cache)
    C⁻¹ = inv(C)
    J = sqrt(det(C))
    # Compute 𝕊
    shear_modulus(m) * (eye_cache - C⁻¹) + bulk_modulus(m) * (J * (J - 1) * C⁻¹)
end

"Return the Cosserat stress tensor `𝕊` given the Green-Lagrange `𝔼` strain tensor."
function _S_analytic!(S::AbstractMatrix, m::NeoHookean, E::AbstractMatrix;
        eye_cache::AbstractMatrix{<:Real} = eye(3))
    S .= Symmetric(_S_analytic(m, E; eye_cache))
end

const ∂S∂E_forward_diff = zeros(6, 6)
const aux_gradients = zeros(3, 3)

"Return the `∂𝕊∂𝔼` for a material `m`, the Gree-Lagrange strain tensor `𝔼` and a
function to compute 𝕊 analytically."
function _∂S∂E!(
        ∂S∂E::Matrix, m::NeoHookean, 𝔼::AbstractMatrix, S_analytic::Function = _S_analytic)
    row = 1
    for index in INDEXES_TO_VOIGT
        i, j = index
        ∂S∂E[row,
        :] .= voigt(
            ForwardDiff.gradient!(aux_gradients,
                E -> S_analytic(m, E)[i, j],
                collect(𝔼)),
            0.5)
        row += 1
    end
    ∂S∂E
end

"Return the Cosserat or Second-Piola Kirchoff stress tensor `𝕊`
considering a `SVK` material `m` and the Green-Lagrange
strain tensor `𝔼`.Also this function provides `∂𝕊∂𝔼` for the iterative method."
function cosserat_stress!(S::AbstractMatrix{<:Real}, ∂S∂E::Matrix{<:Real},
        m::NeoHookean, E::AbstractMatrix; eye_cache = eye(3)) # Is used in a different method
    _S_analytic!(S, m, E; eye_cache)
    _∂S∂E!(∂S∂E, m, E, _S_analytic)
end

end
