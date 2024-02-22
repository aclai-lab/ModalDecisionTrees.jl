using Lazy

abstract type ModalCARTState end

"""
TODO document
vector of current worlds for each instance and modality
"""
struct RestrictedMCARTState{WS<:AbstractVector{WST} where {WorldType,WST<:Vector{WorldType}}} <: ModalCARTState
  witnesses::WS
end

@forward RestrictedMCARTState.witnesses (Base.getindex, Base.setindex!)
