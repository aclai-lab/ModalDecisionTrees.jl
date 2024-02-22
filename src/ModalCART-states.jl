using Lazy

# TODO fix citation.
"""
Recursion state for ModalCART (see paper On The Foundations of Modal Decision Trees)
"""
abstract type MCARTState end

"""
TODO document
vector of current worlds for each instance and modality
"""
struct RestrictedMCARTState{WS<:AbstractVector{WST} where {WorldType,WST<:Vector{WorldType}}} <: MCARTState
    witnesses::WS
end

struct FullMCARTState{ANC<:Vector} <: MCARTState
    ancestors::ANC
end

@forward RestrictedMCARTState.witnesses (Base.getindex, Base.setindex!)
