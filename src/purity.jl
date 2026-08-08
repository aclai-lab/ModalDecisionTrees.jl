"""
    function dishonor_min_purity_increase(
        ::Type{L},
        min_purity_increase,
        purity,
        best_purity_times_nt,
        nt
    ) where {L <: CLabel}

    function dishonor_min_purity_increase(
        ::Type{L},
        min_purity_increase,
        purity,
        best_purity_times_nt,
        nt
    ) where {L <: RLabel}

Returns `true` if the purity increase compared to the best does not exceed the minimum
threshold `min_purity_increase`.

# Arguments
- `L`: can be a subtype of `CLabel` or `RLabel`.
- `min_purity_increase`: Minimum required purity increase
- `purity`: Current purity
- `best_purity_times_nt`: Best purity times total count
- `nt`: Total count

# Returns
- `Bool`: `true` if the purity condition is not met

See also [`CLabel`](@ref), [`RLabel`](@ref).
"""
function dishonor_min_purity_increase(
    ::Type{L},
    min_purity_increase,
    purity,
    best_purity_times_nt,
    nt
) where {L <: CLabel}
    (best_purity_times_nt / nt - purity < min_purity_increase)
end
function dishonor_min_purity_increase(
    ::Type{L},
    min_purity_increase,
    purity,
    best_purity_times_nt,
    nt
) where {L <: RLabel}
    # (best_purity_times_nt - tsum * label <= min_purity_increase * nt) # ORIGINAL
    (best_purity_times_nt / nt - purity < min_purity_increase * nt)
end

# TODO fix
# function _compute_purity( # faster_version assuming L<:Integer and labels going from 1:n_classes
#     labels           ::AbstractVector{L},
#     n_classes        ::Integer,
#     weights          ::AbstractVector{U} = default_weights(labels);
#     loss_function    ::Union{Nothing,Loss} = default_loss_function(L),
# ) where {L<:CLabel,L<:Integer,U}
#     nc = fill(zero(U), n_classes)
#     @simd for i in 1:max(length(labels),length(weights))
#         nc[labels[i]] += weights[i]
#     end
#     nt = sum(nc)
#     return loss_function(nc, nt)::Float64
# end

"""
    compute_purity(
        labels::AbstractVector{L},
        weights::AbstractVector{U}=default_weights(labels);
        loss_function::Union{Nothing,Loss}=default_loss_function(L)
    ) where {L<:CLabel,U}

    compute_purity(
        labels::AbstractVector{L},
        weights::AbstractVector{U}=default_weights(labels);
        loss_function::Union{Nothing,Loss}=default_loss_function(L)
    ) where {L<:RLabel,U}

Computes the purity of a set of categorical labels, optionally weighted, using a loss
function.

# Arguments
- `labels::AbstractVector{L}`: Vector of categorical labels
- `weights::AbstractVector{U}`: Vector of weights (default: unit weights)
- `loss_function::Union{Nothing, Loss}`: Loss function (default: depends on label type)

# Returns
- `Float64`: Purity value

See also [`CLabel`](@ref), [`RLabel`](@ref).
"""

function compute_purity(
        labels::AbstractVector{L},
        weights::AbstractVector{U} = default_weights(labels);
        loss_function::Union{Nothing, Loss} = default_loss_function(L)
) where {L <: CLabel, U}
    nc = Dict{L, U}()
    @simd for i in 1:max(length(labels), length(weights))
        nc[labels[i]] = get(nc, labels[i], 0) + weights[i]
    end
    nc = collect(values(nc))
    nt = sum(nc)
    return loss_function(nc, nt)::Float64
end

# subroutine for the dispatch of compute_purity dealing with RLabel instead of CLabel
function _compute_purity(
        labels::AbstractVector{L},
        weights::AbstractVector{U} = default_weights(labels);
        loss_function::Union{Nothing, Loss} = default_loss_function(L)
) where {L <: RLabel, U}
    # sums = labels .* weights
    nt = sum(weights)
    return (loss_function(labels, weights, nt))::Float64
end

function compute_purity(
        labels::AbstractVector{L},
        weights::AbstractVector{U} = default_weights(labels);
        loss_function::Union{Nothing, Loss} = default_loss_function(L)
) where {L <: RLabel, U}
    _compute_purity(labels, weights; loss_function = loss_function)
end
