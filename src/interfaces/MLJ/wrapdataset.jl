
using SoleData
using SoleData: AbstractModalLogiset, SupportedLogiset

using MultiData
using MultiData: dataframe2dimensional

# UNI
# AbstractArray -> scalarlogiset -> supportedlogiset
# SupportedLogiset -> supportedlogiset
# AbstractModalLogiset -> supportedlogiset

# MULTI
# SoleData.MultiDataset -> multilogiset
# AbstractDataFrame -> naturalgrouping -> multilogiset
# MultiLogiset -> multilogiset

function wrapdataset(
    X,
    model,
    force_var_grouping::Union{Nothing,AbstractVector{<:AbstractVector}} = nothing;
    passive_mode = false
)
    if X isa MultiLogiset
        if !isnothing(force_var_grouping)
            @warn "Ignoring var_grouping $(force_var_grouping) (a MultiLogiset was provided)."
        end
        multimodal_X, var_grouping = X, nothing
        return multimodal_X, var_grouping
    end

    # Vector of instance values
    # Matrix instance x variable -> Matrix variable x instance
    if X isa AbstractVector
        X = collect(reshape(X, 1, length(X)))
    elseif X isa AbstractMatrix
        X = collect(X')
    end

    if X isa AbstractArray # Cube
        if !(X isa Union{AbstractVector,AbstractMatrix})
            @warn "AbstractArray of $(ndims(X)) dimensions and size $(size(X)) encountered. " *
                "This will be interpreted as a dataset of $(size(X)[end]) instances, " *
                "$(size(X)[end-1]) variables, and channel size $(size(X)[1:end-2])."
                # "datasets ($(typeof(X)) encountered)"
        end

        X = eachslice(X; dims=ndims(X))
    end

    X = begin
        if X isa AbstractDimensionalDataset
            X = model.downsize.(eachinstance(X))

            if !passive_mode
                @info "Precomputing logiset..."
                metaconditions = readconditions(model, X)
                features = unique(SoleData.feature.(metaconditions))
                scalarlogiset(X, features;
                    use_onestep_memoization = true,
                    conditions = metaconditions,
                    relations = readrelations(model, X),
                    print_progress = (ninstances(X) > 500)
                )
            else
                MultiData.dimensional2dataframe(X)
            end
        elseif X isa SupportedLogiset
            X
        elseif X isa AbstractModalLogiset
            SupportedLogiset(X;
                use_onestep_memoization = true,
                conditions = readconditions(model, X),
                relations = readrelations(model, X)
            )
        elseif X isa AbstractMultiDataset
            X
        elseif Tables.istable(X)
            DataFrame(X)
        else
            X
        end
    end

    # @show X
    # @show collect.(X)
    # readline()

    # DataFrame -> MultiDataset + variable grouping (needed for printing)
    X, var_grouping = begin
        if X isa AbstractDataFrame

            allowedcoltypes = Union{Real,AbstractArray{<:Real,0},AbstractVector{<:Real},AbstractMatrix{<:Real}}
            wrong_columns = filter(((colname,c),)->!(eltype(c) <: allowedcoltypes), collect(zip(names(X), eachcol(X))))
            @assert length(wrong_columns) == 0 "Invalid columns " *
                "encountered: `$(join(first.(wrong_columns), "`, `", "` and `"))`. $(MDT).jl only allows " *
                "variables that are `Real` and `AbstractArray{<:Real,N}` with N ∈ {0,1,2}. " *
                "Got: `$(join(eltype.(last.(wrong_columns)), "`, `", "` and `"))`" * (length(wrong_columns) > 1 ? ", respectively" : "") * "."

            var_grouping = begin
                if isnothing(force_var_grouping)
                    var_grouping = SoleData.naturalgrouping(X; allow_variable_drop = true)
                    if !(length(var_grouping) == 1 && length(var_grouping[1]) == ncol(X))
                        @info "Using variable grouping:\n" *
                            # join(map(((i_mod,variables),)->"[$i_mod] -> [$(join(string.(variables), ", "))]", enumerate(var_grouping)), "\n")
                            join(map(((i_mod,variables),)->"\t{$i_mod} => $(Tuple(variables))", enumerate(var_grouping)), "\n")
                    end
                    var_grouping
                else
                    @assert force_var_grouping isa AbstractVector{<:AbstractVector} "$(typeof(force_var_grouping))"
                    force_var_grouping
                end
            end

            md = MultiDataset(X, var_grouping)

            # Downsize
            md = MultiDataset([begin
                mod, varnames = dataframe2dimensional(mod)
                mod = model.downsize.(eachinstance(mod))
                SoleData.dimensional2dataframe(mod, varnames)
            end for mod in eachmodality(md)])

            md, var_grouping
        else
            X, nothing
        end
    end

    # println(X)
    # println(modality(X, 1))
    multimodal_X = begin
        if X isa SoleData.AbstractMultiDataset
            if !passive_mode || !SoleData.ismultilogiseed(X)
                @info "Precomputing logiset..."
                MultiLogiset([begin
                        _metaconditions = readconditions(model, mod)
                        features = unique(SoleData.feature.(_metaconditions))
                        # @show _metaconditions
                        # @show features
                        scalarlogiset(mod, features;
                            use_onestep_memoization = true,
                            conditions = _metaconditions,
                            relations = readrelations(model, mod),
                            print_progress = (ninstances(X) > 500)
                        )
                    end for mod in eachmodality(X)
                ])
            else
                X
            end
        elseif X isa AbstractModalLogiset
            MultiLogiset(X)
        elseif X isa MultiLogiset
            X
        else
            error("Unexpected dataset type: $(typeof(X)). Allowed dataset types are " *
                "AbstractArray, AbstractDataFrame, " *
                "SoleData.AbstractMultiDataset and SoleData.AbstractModalLogiset.")
        end
    end

    return (multimodal_X, var_grouping)
end
