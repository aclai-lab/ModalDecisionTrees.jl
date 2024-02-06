using SoleBase: movingwindow
using SoleData: AbstractDimensionalDataset

DOWNSIZE_MSG = "If this process gets killed, please downsize your dataset beforehand."

function make_downsizing_function(channelsize::NTuple)
    return function downsize(instance)
        return moving_average(instance, channelsize)
    end
end

function make_downsizing_function(::TreeModel)
    function downsize(instance)
        channelsize = MultiData.instance_channelsize(instance)
        nvariables = MultiData.instance_nvariables(instance)
        channelndims = length(channelsize)
        if channelndims == 1
            n_points = channelsize[1]
            if nvariables > 30 && n_points > 100
                # @warn "Downsizing series $(n_points) points to $(100) points ($(nvariables) variables). $DOWNSIZE_MSG"
                instance = moving_average(instance, 100)
            elseif n_points > 150
                # @warn "Downsizing series $(n_points) points to $(150) points ($(nvariables) variables). $DOWNSIZE_MSG"
                instance = moving_average(instance, 150)
            end
        elseif channelndims == 2
            if nvariables > 30 && prod(channelsize) > prod((7,7),)
                new_channelsize = min.(channelsize, (7,7))
                # @warn "Downsizing image of size $(channelsize) to $(new_channelsize) pixels ($(nvariables) variables). $DOWNSIZE_MSG"
                instance = moving_average(instance, new_channelsize)
            elseif prod(channelsize) > prod((10,10),)
                new_channelsize = min.(channelsize, (10,10))
                # @warn "Downsizing image of size $(channelsize) to $(new_channelsize) pixels ($(nvariables) variables). $DOWNSIZE_MSG"
                instance = moving_average(instance, new_channelsize)
            end
        end
        instance
    end
end

function make_downsizing_function(::ForestModel)
    function downsize(instance)
        channelsize = MultiData.instance_channelsize(instance)
        nvariables = MultiData.instance_nvariables(instance)
        channelndims = length(channelsize)
        if channelndims == 1
            n_points = channelsize[1]
            if nvariables > 30 && n_points > 100
                # @warn "Downsizing series $(n_points) points to $(100) points ($(nvariables) variables). $DOWNSIZE_MSG"
                instance = moving_average(instance, 100)
            elseif n_points > 150
                # @warn "Downsizing series $(n_points) points to $(150) points ($(nvariables) variables). $DOWNSIZE_MSG"
                instance = moving_average(instance, 150)
            end
        elseif channelndims == 2
            if nvariables > 30 && prod(channelsize) > prod((4,4),)
                new_channelsize = min.(channelsize, (4,4))
                # @warn "Downsizing image of size $(channelsize) to $(new_channelsize) pixels ($(nvariables) variables). $DOWNSIZE_MSG"
                instance = moving_average(instance, new_channelsize)
            elseif prod(channelsize) > prod((7,7),)
                new_channelsize = min.(channelsize, (7,7))
                # @warn "Downsizing image of size $(channelsize) to $(new_channelsize) pixels ($(nvariables) variables). $DOWNSIZE_MSG"
                instance = moving_average(instance, new_channelsize)
            end
        end
        instance
    end
end

_mean(::Type{T}, vals::AbstractArray{T}) where {T<:Number} = mean(vals)
_mean(::Type{T1}, vals::AbstractArray{T2}) where {T1<:AbstractFloat,T2<:Integer} = T1(mean(vals))
_mean(::Type{T1}, vals::AbstractArray{T2}) where {T1<:Integer,T2<:AbstractFloat} = round(T1, mean(vals))

# # 1D
# function moving_average(
#     instance::AbstractArray{T,1};
#     kwargs...
# ) where {T<:Union{Nothing,Number}}
#     npoints = length(instance)
#     return [_mean(T, instance[idxs]) for idxs in movingwindow(npoints; kwargs...)]
# end

# # 1D
# function moving_average(
#     instance::AbstractArray{T,1},
#     nwindows::Integer,
#     relative_overlap::AbstractFloat = .5,
# ) where {T<:Union{Nothing,Number}}
#     npoints = length(instance)
#     return [_mean(T, instance[idxs]) for idxs in movingwindow(npoints; nwindows = nwindows, relative_overlap = relative_overlap)]
# end

# 1D-instance
function moving_average(
    instance::AbstractArray{T,2},
    nwindows::Integer,
    relative_overlap::AbstractFloat = .5,
) where {T<:Union{Nothing,Number}}
    npoints, n_variables = size(instance)
    new_instance = similar(instance, (nwindows, n_variables))
    for i_variable in 1:n_variables
        new_instance[:, i_variable] .= [_mean(T, instance[idxs, i_variable]) for idxs in movingwindow(npoints; nwindows = nwindows, relative_overlap = relative_overlap)]
    end
    return new_instance
end

# 2D-instance
function moving_average(
    instance::AbstractArray{T,3},
    new_channelsize::Tuple{Integer,Integer},
    relative_overlap::AbstractFloat = .5,
) where {T<:Union{Nothing,Number}}
    n_instance, n_Y, n_variables = size(instance)
    windows_1 = movingwindow(n_instance; nwindows = new_channelsize[1], relative_overlap = relative_overlap)
    windows_2 = movingwindow(n_Y; nwindows = new_channelsize[2], relative_overlap = relative_overlap)
    new_instance = similar(instance, (new_channelsize..., n_variables))
    for i_variable in 1:n_variables
        new_instance[:, :, i_variable] .= [_mean(T, instance[idxs1, idxs2, i_variable]) for idxs1 in windows_1, idxs2 in windows_2]
    end
    return new_instance
end

function moving_average(dataset::AbstractDimensionalDataset, args...; kwargs...)
    return map(instance->moving_average(instance, args...; kwargs...), eachinstance(dataset))
end
