
using ResumableFunctions
using SoleLogics: AbstractFrame
using SoleData: AbstractWorld, AbstractWorlds, AbstractFeature
using Logging: @logmsg
using SoleData: AbstractModalLogiset, SupportedLogiset

using SoleData: base, globmemoset
using SoleData: featchannel,
                    featchannel_onestep_aggregation,
                    onestep_aggregation

using SoleData: SupportedLogiset, ScalarOneStepMemoset, AbstractFullMemoset
using SoleData.DimensionalDatasets: UniformFullDimensionalLogiset

import SoleData: supports
import SoleData.DimensionalDatasets: nfeatures, features

using SoleData: Aggregator, TestOperator, ScalarMetaCondition
using SoleData: ScalarExistentialFormula

using DataStructures

import SoleData: AbstractScalarLogiset, nrelations, relations, nmetaconditions, metaconditions

"""
Perform the modal step, that is, evaluate an existential formula
 on a set of worlds, eventually computing the new world set.
"""
function modalstep(
    X, # ::AbstractScalarLogiset{W},
    i_instance::Integer,
    worlds::AbstractWorlds{W},
    decision::RestrictedDecision{<:ScalarExistentialFormula},
    return_worldmap::Union{Val{true},Val{false}} = Val(false)
) where {W<:AbstractWorld}
    @logmsg LogDetail "modalstep" worlds displaydecision(decision)

    # W = worldtype(frame(X, i_instance))

    φ = formula(decision)
    satisfied = false
    
    # TODO the's room for optimization here: with some relations (e.g. IA_A, IA_L) can be made smaller

    if return_worldmap isa Val{true}
        worlds_map = ThreadSafeDict{W,AbstractWorlds{W}}()
    end
    if length(worlds) == 0
        # If there are no neighboring worlds, then the modal decision is not met
        @logmsg LogDetail "   Empty worldset"
    else
        # Otherwise, check whether at least one of the accessible worlds witnesses truth of the decision.
        # TODO rewrite with new_worlds = map(...acc_worlds)
        # Initialize new worldset
        new_worlds = Worlds{W}()

        # List all accessible worlds
        acc_worlds = begin
            if return_worldmap isa Val{true}
                Threads.@threads for curr_w in worlds
                    acc = accessibles(frame(X, i_instance), curr_w, relation(φ)) |> collect
                    worlds_map[curr_w] = acc
                end
                unique(cat([ worlds_map[k] for k in keys(worlds_map) ]...; dims = 1))
            else
                accessibles(frame(X, i_instance), worlds, relation(φ))
            end
        end

        for w in acc_worlds
            if checkcondition(SoleLogics.value(atom(φ)), X, i_instance, w)
                # @logmsg LogDetail " Found world " w ch_readWorld ... ch_readWorld(w, channel)
                satisfied = true
                push!(new_worlds, w)
            end
        end

        if satisfied == true
            worlds = new_worlds
        else
            # If none of the neighboring worlds satisfies the decision, then 
            #  the new set is left unchanged
        end
    end
    if satisfied
        @logmsg LogDetail "   YES" worlds
    else
        @logmsg LogDetail "   NO"
    end
    if return_worldmap isa Val{true}
        return (satisfied, worlds, worlds_map)
    else
        return (satisfied, worlds)
    end
end

############################################################################################
############################################################################################
############################################################################################


Base.@propagate_inbounds @resumable function generate_decisions(
    X::AbstractScalarLogiset{W,U},
    i_instances::AbstractVector{<:Integer},
    Sf::AbstractVector{<:AbstractWorlds{W}},
    allow_propositional_decisions::Bool,
    allow_modal_decisions::Bool,
    allow_global_decisions::Bool,
    modal_relations_inds::AbstractVector,
    features_inds::AbstractVector,
    grouped_featsaggrsnops::AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:ScalarMetaCondition}}},
    grouped_featsnaggrs::AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}},
) where {W<:AbstractWorld,U}
    # Propositional splits
    if allow_propositional_decisions
        for decision in generate_propositional_decisions(X, i_instances, Sf, features_inds, grouped_featsaggrsnops, grouped_featsnaggrs)
            # @logmsg LogDebug " Testing decision: $(displaydecision(decision))"
            @yield decision
        end
    end
    # Global splits
    if allow_global_decisions
        for decision in generate_global_decisions(X, i_instances, Sf, features_inds, grouped_featsaggrsnops, grouped_featsnaggrs)
            # @logmsg LogDebug " Testing decision: $(displaydecision(decision))"
            @yield decision
        end
    end
    # Modal splits
    if allow_modal_decisions
        for decision in generate_modal_decisions(X, i_instances, Sf, modal_relations_inds, features_inds, grouped_featsaggrsnops, grouped_featsnaggrs)
            # @logmsg LogDebug " Testing decision: $(displaydecision(decision))"
            @yield decision
        end
    end
end

using StatsBase

function countmap_with_domain(v::AbstractVector, keyset::AbstractVector = unique(v), eltype = Float64)
    res = StatsBase.countmap(v)
    for el in keyset
        if !haskey(res, el)
            res[el] = zero(eltype)
        end
    end
    res = Dict(collect(zip(keys(res), eltype.(values(res)))))
    return res
end


"""
References:
- "Generalizing Boundary Points"
- "Multi-Interval Discretization of Continuous-Valued Attributes for Classification Learning"
"""
function limit_threshold_domain(
    aggr_thresholds::AbstractVector{T},
    Y::AbstractVector{L},
    W::AbstractVector{U},
    loss_function::Loss,
    test_op::TestOperator,
    min_samples_leaf::Integer,
    perform_domain_optimization::Bool;
    n_classes::Union{Nothing,Integer} = nothing,
    nc::Union{Nothing,AbstractVector{U}} = nothing,
    nt::Union{Nothing,U} = nothing,
) where {T,L<:Label,U}
    if allequal(aggr_thresholds) # Always zero entropy
        return T[], Nothing[]
    end
    if loss_function isa ShannonEntropy && test_op in [≥, <, ≤, >] && (W isa Ones) # TODO extendo to allequal(W) # TODO extend to Gini Index, Normalized Distance Measure, Info Gain, Gain Ratio (Ref. [Linear-Time Preprocessing in Optimal Numerical Range Partitioning])
        if !perform_domain_optimization
            thresh_domain = unique(aggr_thresholds)
            thresh_domain = begin
                if test_op in [≥, <] # Remove edge-case with zero entropy
                    _m = minimum(thresh_domain)
                    filter(x->x != _m, thresh_domain)
                elseif test_op in [≤, >] # Remove edge-case with zero entropy
                    _m = maximum(thresh_domain)
                    filter(x->x != _m, thresh_domain)
                else
                    thresh_domain
                end
            end
            return thresh_domain, fill(nothing, length(thresh_domain))
        else
            p = sortperm(aggr_thresholds)
            _ninstances = length(Y)
            _aggr_thresholds = aggr_thresholds[p]
            _Y = Y[p]

            # thresh_domain = unique(_aggr_thresholds)
            # sort!(thresh_domain)

            ps = pairs(SoleBase._groupby(first, zip(_aggr_thresholds, _Y) |> collect))
            groupedY = map(((k,v),)->begin
                Ys = map(last, v)
                # footprint = sort((Ys)) # associated with ==, it works
                # footprint = sort(unique(Ys)) # couldn't get it to work.
                # footprint = countmap(Ys; alg = :dict)
                footprint = countmap(Ys);
                footprint = collect(footprint); # footprint = map(((k,c),)->k=>c/sum(values(footprint)), collect(footprint)); # normalized
                sort!(footprint; by = first)
                k => (Ys, footprint)
                end, collect(ps))
            # groupedY = map(((k,v),)->(k=>sort(map(last, v))), collect(ps))
            # groupedY = map(((k,v),)->(k=>sort(unique(map(last, v)))), collect(ps))

            function is_same_footprint(f1, f2)
                if f1 == f2
                    return true
                end
                norm_f1 = map(((k,c),)->k=>c/sum(last.(f1)), f1)
                norm_f2 = map(((k,c),)->k=>c/sum(last.(f2)), f2)
                return norm_f1 == norm_f2
            end
            if test_op in [≥, <]
                sort!(groupedY; by=first, rev = true)
            elseif test_op in [≤, >]
                sort!(groupedY; by=first)
            else
                error("Unexpected test_op: $(test_op)")
            end

            thresh_domain, _thresh_Ys, _thresh_footprint = first.(groupedY), first.(last.(groupedY)), last.(last.(groupedY))

            # Filter out those that do not comply with min_samples_leaf
            n_left = 0
            is_boundary_point = map(__thresh_Ys->begin
                n_left = n_left + length(__thresh_Ys)
                ((n_left >= min_samples_leaf && _ninstances-n_left >= min_samples_leaf))
                end, _thresh_Ys)

            # Reference: ≤
            is_boundary_point = map(((i, honors_min_samples_leaf),)->begin
                # last = (i == length(is_boundary_point))
                (
                    (honors_min_samples_leaf &&
                    # (!last &&
                        !(is_boundary_point[i+1] && is_same_footprint(_thresh_footprint[i], _thresh_footprint[i+1])))
                        # !(is_boundary_point[i+1] && isapprox(_thresh_footprint[i], _thresh_footprint[i+1]))) # TODO better..?
                        # !(is_boundary_point[i+1] && issubset(_thresh_footprint[i+1], _thresh_footprint[i]))) # Probably doesn't work
                        # !(is_boundary_point[i+1] && issubset(_thresh_footprint[i], _thresh_footprint[i+1]))) # Probably doesn't work
                        # true)
                )
                end, enumerate(is_boundary_point))

            thresh_domain = thresh_domain[is_boundary_point]

            # NOTE: pretending that these are the right counts, when they are actually the left counts!!! It doesn't matter, it's symmetric.
            # cur_left_counts = countmap_with_domain(L[], UnitRange{L}(1:n_classes), U)
            cur_left_counts = fill(zero(U), n_classes)
            additional_info = map(Ys->begin
                # addcounts!(cur_left_counts, Ys)
                # f = collect(values(cur_left_counts))
                # weight = first(W) # when allequal(W)
                weight = one(U)
                [cur_left_counts[y] += weight for y in Ys]
                f = cur_left_counts
                if test_op in [≥, ≤]
                    # These are left counts
                    ncl, nl = copy(f), sum(f)
                    # ncr = Vector{U}(undef, n_classes)
                    # ncr .= nc .- ncl
                    ncr = nc .- ncl
                    nr = nt - nl
                else
                    # These are right counts
                    ncr, nr = copy(f), sum(f)
                    # ncl = Vector{U}(undef, n_classes)
                    # ncl .= nc .- ncr
                    ncl = nc .- ncr
                    nl = nt - nr
                end
                threshold_info = (ncr, nr, ncl, nl)
                threshold_info
            end, _thresh_Ys)[is_boundary_point]

            # @show typeof(additional_info)
            # @show typeof(additional_info[1])

            # @show test_op, min_samples_leaf
            # @show groupedY
            # @show sum(is_boundary_point), length(thresh_domain), sum(is_boundary_point)/length(thresh_domain)
            return thresh_domain, additional_info
        end
    else
        thresh_domain = unique(aggr_thresholds)
        return thresh_domain, fill(nothing, length(thresh_domain))
    end
end

# function limit_threshold_domain(loss_function::Loss, aggr_thresholds::AbstractVector{U}) where {U}
#     # @show aggr_thresholds
#     thresh_domain = begin
#         if U <: Bool
#             unique(aggr_thresholds)
#         else
#             setdiff(Set(aggr_thresholds),Set([typemin(U), typemax(U)]))
#         end
#     end
#     # @show thresh_domain
#     return thresh_domain
# end

############################################################################################

Base.@propagate_inbounds @resumable function generate_propositional_decisions(
    X::AbstractScalarLogiset{W,U,FT,FR},
    i_instances::AbstractVector{<:Integer},
    Sf::AbstractVector{<:AbstractWorlds{W}},
    features_inds::AbstractVector,
    grouped_featsaggrsnops::AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:ScalarMetaCondition}}},
    grouped_featsnaggrs::AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}},
) where {W<:AbstractWorld,U,FT<:AbstractFeature,N,FR<:FullDimensionalFrame{N,W}}
    relation = identityrel
    _ninstances = length(i_instances)

    _features = features(X)

    # For each feature
    @inbounds for i_feature in features_inds
        feature = _features[i_feature]
        @logmsg LogDebug "Feature $(i_feature): $(feature)"

        # operators for each aggregator
        aggrsnops = grouped_featsaggrsnops[i_feature]
        # Vector of aggregators
        aggregators = keys(aggrsnops) # Note: order-variant, but that's ok here
        
        # dict->vector
        # aggrsnops = [aggrsnops[i_aggregator] for i_aggregator in aggregators]

        # Initialize thresholds with the bottoms
        thresholds = Array{U,2}(undef, length(aggregators), _ninstances)
        for (i_aggregator,aggregator) in enumerate(aggregators)
            thresholds[i_aggregator,:] .= aggregator_bottom(aggregator, U)
        end

        # For each instance, compute thresholds by applying each aggregator to the set of existing values (from the worldset)
        for (instance_idx,i_instance) in enumerate(i_instances)
            # @logmsg LogDetail " Instance $(instance_idx)/$(_ninstances)"
            worlds = Sf[instance_idx]

            # TODO also try this instead
            # values = [X[i_instance, w, i_feature] for w in worlds]
            # thresholds[:,instance_idx] = map(aggregator->aggregator(values), aggregators)
            
            for w in worlds
                # gamma = featvalue(feature, X[i_instance, w) # TODO in general!
                gamma = featvalue(feature, X, i_instance, w, i_feature)
                for (i_aggregator,aggregator) in enumerate(aggregators)
                    thresholds[i_aggregator,instance_idx] = SoleData.aggregator_to_binary(aggregator)(gamma, thresholds[i_aggregator,instance_idx])
                end
            end
        end
        
        # tested_metacondition = TestOperator[]

        # @logmsg LogDebug "thresholds: " thresholds
        # For each aggregator
        for (i_aggregator,aggregator) in enumerate(aggregators)
            aggr_thresholds = thresholds[i_aggregator,:]

            for metacondition in aggrsnops[aggregator]
                test_op = SoleData.test_operator(metacondition)
                @yield relation, metacondition, test_op, aggr_thresholds
            end # for metacondition
        end # for aggregator
    end # for feature
end

############################################################################################

Base.@propagate_inbounds @resumable function generate_modal_decisions(
    X::AbstractScalarLogiset{W,U,FT,FR},
    i_instances::AbstractVector{<:Integer},
    Sf::AbstractVector{<:AbstractWorlds{W}},
    modal_relations_inds::AbstractVector,
    features_inds::AbstractVector,
    grouped_featsaggrsnops::AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:ScalarMetaCondition}}},
    grouped_featsnaggrs::AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}},
) where {W<:AbstractWorld,U,FT<:AbstractFeature,N,FR<:FullDimensionalFrame{N,W}}
    _ninstances = length(i_instances)

    _relations = relations(X)
    _features = features(X)
    
    # For each relational connective
    for i_relation in modal_relations_inds
        relation = _relations[i_relation]
        @logmsg LogDebug "Relation $(relation)..."

        # For each feature
        for i_feature in features_inds
            feature = _features[i_feature]
            # @logmsg LogDebug "Feature $(i_feature): $(feature)"

            # operators for each aggregator
            aggrsnops = grouped_featsaggrsnops[i_feature]
            # Vector of aggregators
            aggregators_with_ids = grouped_featsnaggrs[i_feature]

            # dict->vector?
            # aggrsnops = [aggrsnops[i_aggregator] for i_aggregator in aggregators]

            # Initialize thresholds with the bottoms
            thresholds = Array{U,2}(undef, length(aggregators_with_ids), _ninstances)
            for (i_aggregator,(_,aggregator)) in enumerate(aggregators_with_ids)
                thresholds[i_aggregator,:] .= aggregator_bottom(aggregator, U)
            end

            # For each instance, compute thresholds by applying each aggregator to the set of existing values (from the worldset)
            for (instance_id,i_instance) in enumerate(i_instances)
                # @logmsg LogDetail " Instance $(instance_id)/$(_ninstances)"
                worlds = Sf[instance_id]
                _featchannel = featchannel(base(X), i_instance, i_feature)
                for (i_aggregator,(i_metacond,aggregator)) in enumerate(aggregators_with_ids)
                    metacondition = metaconditions(X)[i_metacond]
                    for w in worlds
                        gamma = begin
                            if true
                                # _featchannel = featchannel(base(X), i_instance, i_feature)
                                # featchannel_onestep_aggregation(X, _featchannel, i_instance, w, relation, feature(metacondition), aggregator)
                                featchannel_onestep_aggregation(X, _featchannel, i_instance, w, relation, metacondition, i_metacond, i_relation)
                                # onestep_aggregation(X, i_instance, w, relation, feature, aggregator, i_metacond, i_relation)
                            # elseif X isa UniformFullDimensionalLogiset
                            #      onestep_aggregation(X, i_instance, w, relation, feature, aggregator, i_metacond, i_relation)
                            else
                                error("generate_global_decisions is broken.")
                            end
                        end
                        thresholds[i_aggregator,instance_id] = SoleData.aggregator_to_binary(aggregator)(gamma, thresholds[i_aggregator,instance_id])
                    end
                end
                
                # for (i_aggregator,(i_metacond,aggregator)) in enumerate(aggregators_with_ids)
                #     gammas = [onestep_aggregation(X, i_instance, w, relation, feature, aggregator, i_metacond, i_relation) for w in worlds]
                #     thresholds[i_aggregator,instance_id] = aggregator(gammas)
                # end
            end

            # @logmsg LogDebug "thresholds: " thresholds

            # For each aggregator
            for (i_aggregator,(_,aggregator)) in enumerate(aggregators_with_ids)

                aggr_thresholds = thresholds[i_aggregator,:]

                for metacondition in aggrsnops[aggregator]
                    # @logmsg LogDetail " Test operator $(metacondition)"
                    test_op = SoleData.test_operator(metacondition)
                    @yield relation, metacondition, test_op, aggr_thresholds
                end # for metacondition
            end # for aggregator
        end # for feature
    end # for relation
end

############################################################################################

Base.@propagate_inbounds @resumable function generate_global_decisions(
    X::AbstractScalarLogiset{W,U,FT,FR},
    i_instances::AbstractVector{<:Integer},
    Sf::AbstractVector{<:AbstractWorlds{W}},
    features_inds::AbstractVector,
    grouped_featsaggrsnops::AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:ScalarMetaCondition}}},
    grouped_featsnaggrs::AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}},
) where {W<:AbstractWorld,U,FT<:AbstractFeature,N,FR<:FullDimensionalFrame{N,W}}
    relation = globalrel
    _ninstances = length(i_instances)

    _features = features(X)

    # For each feature
    for i_feature in features_inds

        feature = _features[i_feature]
        @logmsg LogDebug "Feature $(i_feature): $(feature)"

        # operators for each aggregator
        aggrsnops = grouped_featsaggrsnops[i_feature]
        # println(aggrsnops)
        # Vector of aggregators
        aggregators_with_ids = grouped_featsnaggrs[i_feature]
        # println(aggregators_with_ids)
        # @show feature
        # @show aggrsnops
        # @show aggregators_with_ids
        # dict->vector
        # aggrsnops = [aggrsnops[i_aggregator] for i_aggregator in aggregators]

        # # TODO use this optimized version for SupportedLogiset:
        #   thresholds can in fact be directly given by slicing globmemoset and permuting the two dimensions
        # aggregators_ids = fst.(aggregators_with_ids)
        # thresholds = transpose(globmemoset(X)[i_instances, aggregators_ids])

        # Initialize thresholds with the bottoms
        # @show U
        thresholds = Array{U,2}(undef, length(aggregators_with_ids), _ninstances)
        # for (i_aggregator,(_,aggregator)) in enumerate(aggregators_with_ids)
        #     thresholds[i_aggregator,:] .= aggregator_bottom(aggregator, U)
        # end
        
        # For each instance, compute thresholds by applying each aggregator to the set of existing values (from the worldset)
        for (instance_id,i_instance) in enumerate(i_instances)
            # @logmsg LogDetail " Instance $(instance_id)/$(_ninstances)"
            _featchannel = featchannel(base(X), i_instance, i_feature)
            for (i_aggregator,(i_metacond,aggregator)) in enumerate(aggregators_with_ids)
                # TODO delegate this job to different flavors of `get_global_gamma`. Test whether the _featchannel assignment outside is faster!
                metacondition = metaconditions(X)[i_metacond]
                gamma = begin
                    if true
                        # _featchannel = featchannel(base(X), i_instance, i_feature)
                        featchannel_onestep_aggregation(X, _featchannel, i_instance, SoleLogics.emptyworld(frame(X, i_instance)), relation, metacondition, i_metacond)
                        # onestep_aggregation(X, i_instance, dummyworldTODO, relation, feature, aggregator, i_metacond)
                    # elseif X isa UniformFullDimensionalLogiset
                    #     onestep_aggregation(X, i_instance, dummyworldTODO, relation, feature, aggregator, i_metacond)
                    else
                        error("generate_global_decisions is broken.")
                    end
                end
                # @show gamma

                thresholds[i_aggregator,instance_id] = gamma
                # thresholds[i_aggregator,instance_id] = SoleData.aggregator_to_binary(aggregator)(gamma, thresholds[i_aggregator,instance_id])
                # println(gamma)
                # println(thresholds[i_aggregator,instance_id])
            end
        end

        # println(thresholds)
        @logmsg LogDetail "thresholds: " thresholds

        # For each aggregator
        for (i_aggregator,(_,aggregator)) in enumerate(aggregators_with_ids)

            # println(aggregator)
            # @show aggregator

            aggr_thresholds = thresholds[i_aggregator,:]

            for metacondition in aggrsnops[aggregator]
                test_op = SoleData.test_operator(metacondition)
                @yield relation, metacondition, test_op, aggr_thresholds
            end # for metacondition
        end # for aggregator
    end # for feature
end
