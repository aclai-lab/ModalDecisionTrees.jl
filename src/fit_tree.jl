# The code in this file for decision tree learning is inspired from:
# - Ben Sadeghi's DecisionTree.jl (released under the MIT license);
# - scikit-learn's and numpy's (released under the 3-Clause BSD license);

# Also thanks to Poom Chiarawongse <eight1911@gmail.com>

##############################################################################
##############################################################################
##############################################################################
##############################################################################

mutable struct NodeMeta{L<:Label,P} <: AbstractNode{L}
    region             :: UnitRange{Int}                   # a slice of the instances used to decide the split of the node
    depth              :: Int
    modaldepth         :: Int

    # worlds      :: AbstractVector{Worlds{W}}             # current set of worlds for each training instance

    purity             :: P                                # purity grade attained at training time
    prediction         :: L                                # most likely label
    is_leaf            :: Bool                             # whether this is a leaf node, or a split one
    # split node-only properties
    split_at           :: Int                              # index of instances

    parent             :: Union{Nothing,NodeMeta{L,P}}     # parent node
    l                  :: NodeMeta{L,P}                    # left child node
    r                  :: NodeMeta{L,P}                    # right child node
    
    purity_times_nt    :: P                                # purity grade attained at training time
    consistency        :: Any

    i_modality         :: ModalityId                       # modality id
    decision           :: AbstractDecision

    onlyallowglobal:: Vector{Bool}

    function NodeMeta{L,P}(
        region      :: UnitRange{Int},
        depth       :: Int,
        modaldepth  :: Int,
        oura        :: Vector{Bool},
    ) where {L,P}
        node = new{L,P}()
        node.region = region
        node.depth = depth
        node.modaldepth = modaldepth
        node.purity = P(NaN)
        node.is_leaf = false
        node.parent = nothing
        node.onlyallowglobal = oura
        node
    end
end

isleftchild(node::NodeMeta, parent::NodeMeta) = (parent.l == node)
isrightchild(node::NodeMeta, parent::NodeMeta) = (parent.r == node)
function lastrightancestor(node::NodeMeta)
    n = node
    while !isnothing(n.parent) && isrightchild(n, n.parent)
        n = n.parent
    end
    return n
end

function makeleaf!(node::NodeMeta)
    node.is_leaf = true
    # node.i_modality      = nothing
    # node.purity_times_nt = nothing
    # node.decision        = nothing
    # node.consistency     = nothing
end

# Conversion: NodeMeta (node + training info) -> DTNode (bare decision tree model)
function _convert(
    node          :: NodeMeta,
    labels        :: AbstractVector{L},
    class_names   :: AbstractVector{L},
    threshold_backmap :: Vector{<:Function}
) where {L<:CLabel}
    this_leaf = DTLeaf(class_names[node.prediction], labels[node.region])
    if node.is_leaf
        this_leaf
    else
        left  = _convert(node.l, labels, class_names, threshold_backmap)
        right = _convert(node.r, labels, class_names, threshold_backmap)
        DTInternal(node.i_modality, SimpleDecision(node.decision, threshold_backmap[node.i_modality]), this_leaf, left, right)
    end
end

# Conversion: NodeMeta (node + training info) -> DTNode (bare decision tree model)
function _convert(
    node   :: NodeMeta,
    labels :: AbstractVector{L},
    threshold_backmap :: Vector{<:Function}
) where {L<:RLabel}
    this_leaf = DTLeaf(node.prediction, labels[node.region])
    if node.is_leaf
        this_leaf
    else
        left  = _convert(node.l, labels, threshold_backmap)
        right = _convert(node.r, labels, threshold_backmap)
        DTInternal(node.i_modality, SimpleDecision(node.decision, threshold_backmap[node.i_modality]), this_leaf, left, right)
    end
end

##############################################################################
##############################################################################
##############################################################################
##############################################################################

# function optimize_tree_parameters!(
#       X               :: DimensionalLogise't{T,N},
#       initcond   :: InitialCondition,
#       allow_global_splits :: Bool,
#       test_operators  :: AbstractVector{<:TestOperator}
#   ) where {T,N}

#   # A dimensional ontological datasets:
#   #  flatten to adimensional case + strip of all relations from the ontology
#   if prod(maxchannelsize(X)) == 1
#       if (length(ontology(X).relations) > 0)
#           @warn "The DimensionalLogise't provided has degenerate maxchannelsize $(maxchannelsize(X)), and more than 0 relations: $(ontology(X).relations)."
#       end
#       # X = DimensionalLogise't{T,0}(DimensionalDatasets.strip_ontology(ontology(X)), @views DimensionalDatasets.strip_domain(domain(X)))
#   end

#   ontology_relations = deepcopy(ontology(X).relations)

#   # Fix test_operators order
#   test_operators = unique(test_operators)
#   DimensionalDatasets.sort_test_operators!(test_operators)

#   # Adimensional operators:
#   #  in the adimensional case, some pairs of operators (e.g. <= and >)
#   #  are complementary, and thus it is redundant to check both at the same node.
#   #  We avoid this by only keeping one of the two operators.
#   if prod(maxchannelsize(X)) == 1
#       # No ontological relation
#       ontology_relations = []
#       if test_operators ⊆ DimensionalDatasets.all_lowlevel_test_operators
#           test_operators = [canonical_geq]
#           # test_operators = filter(e->e ≠ canonical_geq,test_operators)
#       else
#           @warn "Test operators set includes non-lowlevel test operators. Update this part of the code accordingly."
#       end
#   end

#   # Softened operators:
#   #  when the largest world only has a few values, softened operators fallback
#   #  to being hard operators
#   # max_world_wratio = 1/prod(maxchannelsize(X))
#   # if canonical_geq in test_operators
#   #   test_operators = filter((e)->(!(e isa CanonicalConditionGeqSoft) || e.alpha < 1-max_world_wratio), test_operators)
#   # end
#   # if canonical_leq in test_operators
#   #   test_operators = filter((e)->(!(e isa CanonicalConditionLeqSoft) || e.alpha < 1-max_world_wratio), test_operators)
#   # end


#   # Binary relations (= unary modal connectives)
#   # Note: the identity relation is the first, and it is the one representing
#   #  propositional splits.

#   if identityrel in ontology_relations
#       error("Found identityrel in ontology provided. No need.")
#       # ontology_relations = filter(e->e ≠ identityrel, ontology_relations)
#   end

#   if globalrel in ontology_relations
#       error("Found globalrel in ontology provided. Use allow_global_splits = true instead.")
#       # ontology_relations = filter(e->e ≠ globalrel, ontology_relations)
#       # allow_global_splits = true
#   end

#   relations = [identityrel, globalrel, ontology_relations...]
#   relationId_id = 1
#   relationGlob_id = 2
#   ontology_relation_ids = map((x)->x+2, 1:length(ontology_relations))

#   compute_globmemoset = (allow_global_splits || (initcond == ModalDecisionTrees.start_without_world))

#   # Modal relations to compute gammas for
#   inUseRelation_ids = if compute_globmemoset
#       [relationGlob_id, ontology_relation_ids...]
#   else
#       ontology_relation_ids
#   end

#   # Relations to use at each split
#   availableRelation_ids = []

#   push!(availableRelation_ids, relationId_id)
#   if allow_global_splits
#       push!(availableRelation_ids, relationGlob_id)
#   end

#   availableRelation_ids = [availableRelation_ids..., ontology_relation_ids...]

#   (
#       test_operators, relations,
#       relationId_id, relationGlob_id,
#       inUseRelation_ids, availableRelation_ids
#   )
# end

# DEBUGprintln = println


############################################################################################
############################################################################################
############################################################################################

# TODO restore resumable. Unfortunately this yields "UndefRefError: access to undefined reference"
# Base.@propagate_inbounds @resumable function generate_relevant_decisions(
function generate_relevant_decisions(
    Xs,
    Sfs,
    n_subrelations,
    n_subfeatures,
    allow_global_splits,
    node,
    rng,
    max_modal_depth,
    idxs,
    region,
    grouped_featsaggrsnopss,
    grouped_featsnaggrss,
)
    out = []
    @inbounds for (i_modality,
        (X,
        modality_Sf,
        modality_n_subrelations::Function,
        modality_n_subfeatures,
        modality_allow_global_splits,
        modality_onlyallowglobal)
    ) in enumerate(zip(eachmodality(Xs), Sfs, n_subrelations, n_subfeatures, allow_global_splits, node.onlyallowglobal))

        @logmsg LogDetail "  Modality $(i_modality)/$(nmodalities(Xs))"

        allow_propositional_decisions, allow_modal_decisions, allow_global_decisions, modal_relations_inds, features_inds = begin

            # Derive subset of features to consider
            # Note: using "sample" function instead of "randperm" allows to insert weights for features which may be wanted in the future
            features_inds = StatsBase.sample(rng, 1:nfeatures(X), modality_n_subfeatures, replace = false)
            sort!(features_inds)

            # Derive all available relations
            allow_propositional_decisions, allow_modal_decisions, allow_global_decisions = begin
                if worldtype(X) == OneWorld
                    true, false, false
                elseif modality_onlyallowglobal
                    false, false, true
                else
                    true, true, modality_allow_global_splits
                end
            end

            if !isnothing(max_modal_depth) && max_modal_depth <= node.modaldepth
                allow_modal_decisions = false
            end

            n_tot_relations = 0
            if allow_modal_decisions
                n_tot_relations += length(relations(X))
            end
            if allow_global_decisions
                n_tot_relations += 1
            end

            # Derive subset of relations to consider
            n_subrel = Int(modality_n_subrelations(n_tot_relations))
            modal_relations_inds = StatsBase.sample(rng, 1:n_tot_relations, n_subrel, replace = false)
            sort!(modal_relations_inds)

            # Check whether the global relation survived
            if allow_global_decisions
                allow_global_decisions = (n_tot_relations in modal_relations_inds)
                modal_relations_inds = filter!(r->r≠n_tot_relations, modal_relations_inds)
                n_tot_relations = length(modal_relations_inds)
            end
            allow_propositional_decisions, allow_modal_decisions, allow_global_decisions, modal_relations_inds, features_inds
        end

        @inbounds for (relation, metacondition, test_op, aggr_thresholds) in generate_decisions(
            X,
            idxs[region],
            modality_Sf,
            allow_propositional_decisions,
            allow_modal_decisions,
            allow_global_decisions,
            modal_relations_inds,
            features_inds,
            grouped_featsaggrsnopss[i_modality],
            grouped_featsnaggrss[i_modality],
        )
            push!(out, (i_modality, relation, metacondition, test_op, aggr_thresholds))
            # @yield i_modality, relation, metacondition, test_op, aggr_thresholds
        end # END decisions
    end # END modality
    return out
end

############################################################################################
############################################################################################
############################################################################################

# Split a node
# Find an optimal local split satisfying the given constraints
#  (e.g. max_depth, min_samples_leaf, etc.)
Base.@propagate_inbounds @inline function optimize_node!(
    node                      :: NodeMeta{L,P},                                                                               # node to split
    Xs                        :: MultiLogiset,                                                                                # modal dataset
    Ss                        :: AbstractVector{<:AbstractVector{WST} where {WorldType,WST<:Vector{WorldType}}},              # vector of current worlds for each instance and modality
    Y                         :: AbstractVector{L},                                                                           # label vector
    initconditions            :: AbstractVector{<:InitialCondition},                                                          # world starting conditions
    W                         :: AbstractVector{U},                                                                           # weight vector
    grouped_featsaggrsnopss   :: AbstractVector{<:AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:ScalarMetaCondition}}}},
    grouped_featsnaggrss      :: AbstractVector{<:AbstractVector{<:AbstractVector{<:Tuple{<:Integer,<:Aggregator}}}},
    lookahead_depth           :: Integer,
    ##########################################################################
    _is_classification        :: Union{Val{true},Val{false}},
    _using_lookahead          :: Union{Val{true},Val{false}},
    _perform_consistency_check:: Union{Val{true},Val{false}},
    ##########################################################################
    ;
    # Logic-agnostic training parameters
    loss_function             :: Loss,
    lookahead                 :: Integer,                                                                                     # maximum depth of the tree to locally optimize for
    max_depth                 :: Union{Nothing,Int},                                                                          # maximum depth of the resultant tree
    min_samples_leaf          :: Int,                                                                                         # minimum number of instancs each leaf needs to have
    min_purity_increase       :: AbstractFloat,                                                                               # maximum purity allowed on a leaf
    max_purity_at_leaf        :: AbstractFloat,                                                                               # minimum purity increase needed for a split
    ##########################################################################
    # Modal parameters
    max_modal_depth           :: Union{Nothing,Int},                                                                          # maximum modal depth of the resultant tree
    n_subrelations            :: AbstractVector{NSubRelationsFunction},                                                       # relations used for the decisions
    n_subfeatures             :: AbstractVector{Int},                                                                         # number of features for the decisions
    allow_global_splits       :: AbstractVector{Bool},                                                                        # allow/disallow using globalrel at any decisional node
    ##########################################################################
    # Other
    idxs                      :: AbstractVector{Int},
    n_classes                 :: Int,
    rng                       :: Random.AbstractRNG,
) where{P,L<:_Label,U,NSubRelationsFunction<:Function}

    # Region of idxs to use to perform the split
    region = node.region
    _ninstances = length(region)
    r_start = region.start - 1

    # DEBUGprintln("optimize_node!"); readline()

    # Gather all values needed for the current set of instances
    # TODO also slice the dataset?

    @inbounds Yf = Y[idxs[region]]
    @inbounds Wf = W[idxs[region]]

    # Yf = Vector{L}(undef, _ninstances)
    # Wf = Vector{U}(undef, _ninstances)
    # @inbounds @simd for i in 1:_ninstances
    #   Yf[i] = Y[idxs[i + r_start]]
    #   Wf[i] = W[idxs[i + r_start]]
    # end

    ############################################################################
    # Prepare counts
    ############################################################################
    if isa(_is_classification, Val{true})
        (nc, nt), (node.purity, node.prediction) = begin
            nc = fill(zero(U), n_classes)
            @inbounds @simd for i in 1:_ninstances
                nc[Yf[i]] += Wf[i]
            end
            nt = sum(nc)
            # TODO use _compute_purity
            purity = loss_function(loss_function(nc, nt)::Float64)::Float64
            # Assign the most likely label before the split
            prediction = argmax(nc)
            # prediction = bestguess(Yf)
            (nc, nt), (purity, prediction)
        end
    else
        sums, (tsum, nt),
        (node.purity, node.prediction) = begin
            # sums = [Wf[i]*Yf[i]       for i in 1:_ninstances]
            sums = Yf
            # ssqs = [Wf[i]*Yf[i]*Yf[i] for i in 1:_ninstances]

            # tssq = zero(U)
            # tssq = sum(ssqs)
            # tsum = zero(U)
            tsum = sum(sums)
            # nt = zero(U)
            nt = sum(Wf)
            # @inbounds @simd for i in 1:_ninstances
            #   # tssq += Wf[i]*Yf[i]*Yf[i]
            #   # tsum += Wf[i]*Yf[i]
            #   nt += Wf[i]
            # end

            # purity = (tsum * prediction) # TODO use loss function
            # purity = tsum * tsum # TODO use loss function
            # tmean = tsum/nt
            # purity = -((tssq - 2*tmean*tsum + (tmean^2*nt)) / (nt-1)) # TODO use loss function
            # TODO use _compute_purity
            purity = begin
                if W isa Ones{Int}
                    loss_function(loss_function(sums, tsum, length(sums))::Float64)
                else
                    loss_function(loss_function(sums, Wf, nt)::Float64)
                end
            end
            # Assign the most likely label before the split
            prediction =  tsum / nt
            # prediction = bestguess(Yf)
            sums, (tsum, nt), (purity, prediction)
        end
    end

    ############################################################################
    ############################################################################
    ############################################################################

    @logmsg LogDebug "_split!(...) " _ninstances region nt

    ############################################################################
    # Preemptive leaf conditions
    ############################################################################
    if isa(_is_classification, Val{true})
        if (
            # If all instances belong to the same class, make this a leaf
                (nc[node.prediction]       == nt)
            # No binary split can honor min_samples_leaf if there are not as many as
            #  min_samples_leaf*2 instances in the first place
             || (min_samples_leaf * 2 >  _ninstances)
            # If the node is pure enough, avoid splitting # TODO rename purity to loss
             || (node.purity          > max_purity_at_leaf)
            # Honor maximum depth constraint
             || (!isnothing(max_depth) && max_depth <= node.depth))
            # DEBUGprintln("BEFORE LEAF!")
            # DEBUGprintln(nc[node.prediction])
            # DEBUGprintln(nt)
            # DEBUGprintln(min_samples_leaf)
            # DEBUGprintln(_ninstances)
            # DEBUGprintln(node.purity)
            # DEBUGprintln(max_purity_at_leaf)
            # DEBUGprintln(max_depth)
            # DEBUGprintln(node.depth)
            # readline()
            node.is_leaf = true
            # @logmsg LogDetail "leaf created: " (min_samples_leaf * 2 >  _ninstances) (nc[node.prediction] == nt) (node.purity  > max_purity_at_leaf) (max_depth <= node.depth)
            return
        end
    else
        if (
            # No binary split can honor min_samples_leaf if there are not as many as
            #  min_samples_leaf*2 instances in the first place
                (min_samples_leaf * 2 >  _ninstances)
          # equivalent to old_purity > -1e-7
             || (node.purity          > max_purity_at_leaf) # TODO
             # || (tsum * node.prediction    > -1e-7 * nt + tssq)
            # Honor maximum depth constraint
             || (!isnothing(max_depth) && max_depth            <= node.depth))
            node.is_leaf = true
            # @logmsg LogDetail "leaf created: " (min_samples_leaf * 2 >  _ninstances) (tsum * node.prediction    > -1e-7 * nt + tssq) (tsum * node.prediction) (-1e-7 * nt + tssq) (max_depth <= node.depth)
            return
        end
    end

    ########################################################################################
    ########################################################################################
    ########################################################################################

    # TODO try this solution for rsums and lsums (regression case)
    # rsums = Vector{U}(undef, _ninstances)
    # lsums = Vector{U}(undef, _ninstances)
    # @simd for i in 1:_ninstances
    #   rsums[i] = zero(U)
    #   lsums[i] = zero(U)
    # end

    Sfs = Vector{Vector{WST} where {WorldType,WST<:Vector{WorldType}}}(undef, nmodalities(Xs))
    for (i_modality,WT) in enumerate(worldtype.(eachmodality(Xs)))
        Sfs[i_modality] = Vector{Vector{WT}}(undef, _ninstances)
        @simd for i in 1:_ninstances
            Sfs[i_modality][i] = Ss[i_modality][idxs[i + r_start]]
        end
    end

    ########################################################################################
    ########################################################################################
    ########################################################################################

    is_lookahead_basecase = (isa(_using_lookahead, Val{true}) && lookahead_depth == lookahead)
    performing_consistency_check = (isa(_perform_consistency_check, Val{true}) || is_lookahead_basecase)

    function splitnode!(node, Ss, idxs)
        # TODO, actually, when using Shannon entropy, we must correct the purity:
        corrected_this_purity_times_nt = loss_function(node.purity_times_nt)::Float64

        # DEBUGprintln("corrected_this_purity_times_nt: $(corrected_this_purity_times_nt)")
        # DEBUGprintln(min_purity_increase)
        # DEBUGprintln(node.purity)
        # DEBUGprintln(corrected_this_purity_times_nt)
        # DEBUGprintln(nt)
        # DEBUGprintln(purity - node.purity_times_nt/nt)
        # DEBUGprintln("dishonor: $(dishonor_min_purity_increase(L, min_purity_increase, node.purity, corrected_this_purity_times_nt, nt))")
        # readline()

        # println("corrected_this_purity_times_nt = $(corrected_this_purity_times_nt)")
        # println("nt =  $(nt)")
        # println("node.purity =  $(node.purity)")
        # println("corrected_this_purity_times_nt / nt - node.purity = $(corrected_this_purity_times_nt / nt - node.purity)")
        # println("min_purity_increase * nt =  $(min_purity_increase) * $(nt) = $(min_purity_increase * nt)")

        # @logmsg LogOverview "purity_times_nt increase" corrected_this_purity_times_nt/nt node.purity (corrected_this_purity_times_nt/nt + node.purity) (node.purity_times_nt/nt - node.purity)
        # If the best split is good, partition and split accordingly
        @inbounds if ((
            corrected_this_purity_times_nt == typemin(P)) ||
            dishonor_min_purity_increase(L, min_purity_increase, node.purity, corrected_this_purity_times_nt, nt)
        )
            # if isa(_is_classification, Val{true})
            #     @logmsg LogDebug " Leaf" corrected_this_purity_times_nt min_purity_increase (corrected_this_purity_times_nt/nt) node.purity ((corrected_this_purity_times_nt/nt) - node.purity)
            # else
            #     @logmsg LogDebug " Leaf" corrected_this_purity_times_nt tsum node.prediction min_purity_increase nt (corrected_this_purity_times_nt / nt - tsum * node.prediction) (min_purity_increase * nt)
            # end
            makeleaf!(node)
            return false
        end

        # Compute new world sets (= take a modal step)

        # println(decision_str)
        decision_str = displaydecision(node.i_modality, node.decision)

        # TODO instead of using memory, here, just use two opposite indices and perform substitutions. indj = _ninstances
        post_unsatisfied = fill(1, _ninstances)
        if performing_consistency_check
            world_refs = []
        end
        for i_instance in 1:_ninstances
            # TODO perform step with an OntologicalModalDataset

            X = modality(Xs, node.i_modality)
            # instance = DimensionalDatasets.get_instance(X, idxs[i_instance + r_start])

            # println(instance)
            # println(Sfs[node.i_modality][i_instance])
            _sat, _ss = modalstep(X, idxs[i_instance + r_start], Sfs[node.i_modality][i_instance], node.decision)
            (issat,Ss[node.i_modality][idxs[i_instance + r_start]]) = _sat, _ss
            # @logmsg LogDetail " [$issat] Instance $(i_instance)/$(_ninstances)" Sfs[node.i_modality][i_instance] (if issat Ss[node.i_modality][idxs[i_instance + r_start]] end)
            # println(issat)
            # println(Ss[node.i_modality][idxs[i_instance + r_start]])
            # readline()

            # I'm using unsatisfied because sorting puts YES instances first, but TODO use the inverse sorting and use issat flag instead
            post_unsatisfied[i_instance] = !issat
            if performing_consistency_check
                push!(world_refs, _ss)
            end
        end

        @logmsg LogDetail " post_unsatisfied" post_unsatisfied

        # if length(unique(post_unsatisfied)) == 1
        #     @warn "An uninformative split was reached. Something's off\nPurity: $(node.purity)\nSplit: $(decision_str)\nUnsatisfied flags: $(post_unsatisfied)"
        #     makeleaf!(node)
        #     return false
        # end
        @logmsg LogDetail " Branch ($(sum(post_unsatisfied))+$(_ninstances-sum(post_unsatisfied))=$(_ninstances) instances) at modality $(node.i_modality) with decision: $(decision_str), purity $(node.purity)"

        # if sum(post_unsatisfied) >= min_samples_leaf && (_ninstances - sum(post_unsatisfied)) >= min_samples_leaf
            # DEBUGprintln("LEAF!")
        #     makeleaf!(node)
        #     return false
        # end


        ########################################################################################
        ########################################################################################
        ########################################################################################

        # Check consistency
        consistency = begin
            if performing_consistency_check
                post_unsatisfied
            else
                sum(Wf[BitVector(post_unsatisfied)])
            end
        end

        # @logmsg LogDetail " post_unsatisfied" post_unsatisfied

        # if !isapprox(node.consistency, consistency; atol=eps(Float16), rtol=eps(Float16))
        #     errStr = ""
        #     errStr *= "A low-level error occurred. Please open a pull request with the following info."
        #     errStr *= "Decision $(node.decision).\n"
        #     errStr *= "Possible causes:\n"
        #     errStr *= "- feature returning NaNs\n"
        #     errStr *= "- erroneous representatives for relation $(relation(node.decision)), aggregator $(existential_aggregator(test_operator(node.decision))) and feature $(feature(node.decision))\n"
        #     errStr *= "\n"
        #     errStr *= "Branch ($(sum(post_unsatisfied))+$(_ninstances-sum(post_unsatisfied))=$(_ninstances) instances) at modality $(node.i_modality) with decision: $(decision_str), purity $(node.purity)\n"
        #     errStr *= "$(length(idxs[region])) Instances: $(idxs[region])\n"
        #     errStr *= "Different partition was expected:\n"
        #     if performing_consistency_check
        #         errStr *= "Actual: $(consistency) ($(sum(consistency)))\n"
        #         errStr *= "Expected: $(node.consistency) ($(sum(node.consistency)))\n"
        #         diff = node.consistency.-consistency
        #         errStr *= "Difference: $(diff) ($(sum(abs.(diff))))\n"
        #     else
        #         errStr *= "Actual: $(consistency)\n"
        #         errStr *= "Expected: $(node.consistency)\n"
        #         diff = node.consistency-consistency
        #         errStr *= "Difference: $(diff)\n"
        #     end
        #     errStr *= "post_unsatisfied = $(post_unsatisfied)\n"

        #     if performing_consistency_check
        #         errStr *= "world_refs = $(world_refs)\n"
        #         errStr *= "new world_refs = $([Ss[node.i_modality][idxs[i_instance + r_start]] for i_instance in 1:_ninstances])\n"
        #     end

        #     # for i in 1:_ninstances
        #         # errStr *= "$(DimensionalDatasets.get_channel(Xs, idxs[i + r_start], feature(node.decision)))\t$(Sfs[node.i_modality][i])\t$(!(post_unsatisfied[i]==1))\t$(Ss[node.i_modality][idxs[i + r_start]])\n";
        #     # end

        #     println("ERROR! " * errStr)
        # end

        # if length(unique(post_unsatisfied)) == 1
        #     # Note: this should always be satisfied, since min_samples_leaf is always > 0 and nl,nr>min_samples_leaf
        #     errStr = "An uninformative split was reached."
        #     errStr *= "Something's off with this algorithm\n"
        #     errStr *= "Purity: $(node.purity)\n"
        #     errStr *= "Split: $(decision_str)\n"
        #     errStr *= "Unsatisfied flags: $(post_unsatisfied)"

        #     println("ERROR! " * errStr)
        #     # error(errStr)
        #     makeleaf!(node)
        #     return false
        # end

        ########################################################################################
        ########################################################################################
        ########################################################################################

        # @show post_unsatisfied

        # @logmsg LogDetail "pre-partition" region idxs[region] post_unsatisfied
        node.split_at = partition!(idxs, post_unsatisfied, 0, region)
        node.purity = corrected_this_purity_times_nt/nt
        # @logmsg LogDetail "post-partition" idxs[region] node.split_at

        ind = node.split_at
        oura = node.onlyallowglobal
        mdepth = node.modaldepth

        leftmodaldepth, rightmodaldepth = begin
            if is_propositional_decision(node.decision)
                mdepth, mdepth
            else
                # The left decision nests in the last right ancestor's formula
                # The right decision
                (lastrightancestor(node).modaldepth+1), (lastrightancestor(node).modaldepth+1)
            end
        end

        # onlyallowglobal changes:
        # on the left node, the modality where the decision was taken
        l_oura = copy(oura)
        l_oura[node.i_modality] = false
        r_oura = oura

        # no need to copy because we will copy at the end
        node.l = typeof(node)(region[    1:ind], node.depth+1, leftmodaldepth, l_oura)
        node.r = typeof(node)(region[ind+1:end], node.depth+1, rightmodaldepth, r_oura)

        return true
    end

    ########################################################################################
    ########################################################################################
    ########################################################################################

    ########################################################################################
    #################################### Find best split ###################################
    ########################################################################################

    if performing_consistency_check
        unsatisfied = Vector{Bool}(undef, _ninstances)
    end

    # Optimization-tracking variables
    node.purity_times_nt = typemin(P)
    # node.i_modality = -1
    # node.decision = SimpleDecision(ScalarExistentialFormula{Float64}())
    # node.consistency = nothing

    perform_domain_optimization = is_lookahead_basecase && !performing_consistency_check

    ## Test all decisions or each modality
    for (i_modality, relation, metacondition, test_op, aggr_thresholds) in generate_relevant_decisions(
        Xs,
        Sfs,
        n_subrelations,
        n_subfeatures,
        allow_global_splits,
        node,
        rng,
        max_modal_depth,
        idxs,
        region,
        grouped_featsaggrsnopss,
        grouped_featsnaggrss,
    )
        if isa(_is_classification, Val{true})
            thresh_domain, additional_info = limit_threshold_domain(aggr_thresholds, Yf, Wf, loss_function, test_op, min_samples_leaf, perform_domain_optimization; n_classes = n_classes, nc = nc, nt = nt)
        else
            thresh_domain, additional_info = limit_threshold_domain(aggr_thresholds, Yf, Wf, loss_function, test_op, min_samples_leaf, perform_domain_optimization)
        end
        # Look for the best threshold 'a', as in atoms like "feature >= a"
        for (_threshold, threshold_info) in zip(thresh_domain, additional_info)
            decision = SimpleDecision(ScalarExistentialFormula(relation, ScalarCondition(metacondition, _threshold)))

            # @show decision
            # @show aggr_thresholds
            # @logmsg LogDetail " Testing decision: $(displaydecision(decision))"

            # println(displaydecision(i_modality, decision))

            # TODO avoid ugly unpacking and figure out a different way of achieving this
            # (test_op, _threshold) = (test_operator(decision), threshold(decision))
            ########################################################################
            # Apply decision to all instances
            ########################################################################
            # Note: unsatisfied is also changed
            if isa(_is_classification, Val{true})
                (ncr, nr, ncl, nl) = begin
                    if !isnothing(threshold_info) && !performing_consistency_check
                        threshold_info
                    else
                        # Re-initialize right counts
                        nr = zero(U)
                        ncr = fill(zero(U), n_classes)
                        if performing_consistency_check
                            unsatisfied .= 1
                        end
                        for i_instance in 1:_ninstances
                            gamma = aggr_thresholds[i_instance]
                            issat = SoleData.apply_test_operator(test_op, gamma, _threshold)
                            # @logmsg LogDetail " instance $i_instance/$_ninstances: (f=$(gamma)) -> issat = $(issat)"

                            # Note: in a fuzzy generalization, `issat` becomes a [0-1] value
                            if !issat
                                nr += Wf[i_instance]
                                ncr[Yf[i_instance]] += Wf[i_instance]
                            else
                                if performing_consistency_check
                                    unsatisfied[i_instance] = 0
                                end
                            end
                        end
                        # ncl = Vector{U}(undef, n_classes)
                        # ncl .= nc .- ncr
                        ncl = nc .- ncr
                        nl = nt - nr
                        threshold_info_new = (ncr, nr, ncl, nl)
                        # if !isnothing(threshold_info) && !performing_consistency_check
                        #     if threshold_info != threshold_info_new
                        #         @show nc
                        #         @show nt
                        #         @show Yf
                        #         @show Wf
                        #         @show test_op
                        #         @show _threshold
                        #         @show threshold_info
                        #         @show threshold_info_new
                        #         readline()
                        #     end
                        # end
                        threshold_info_new
                    end
                end
            else
                (rsums, nr, lsums, nl, rsum, lsum) = begin
                    # Initialize right counts
                    # rssq = zero(U)
                    rsum = zero(U)
                    nr   = zero(U)
                    # TODO experiment with running mean instead, because this may cause a lot of memory inefficiency
                    # https://it.wikipedia.org/wiki/Algoritmi_per_il_calcolo_della_varianza
                    rsums = Float64[] # Vector{U}(undef, _ninstances)
                    lsums = Float64[] # Vector{U}(undef, _ninstances)

                    if performing_consistency_check
                        unsatisfied .= 1
                    end
                    for i_instance in 1:_ninstances
                        gamma = aggr_thresholds[i_instance]
                        issat = SoleData.apply_test_operator(test_op, gamma, _threshold)
                        # @logmsg LogDetail " instance $i_instance/$_ninstances: (f=$(gamma)) -> issat = $(issat)"

                        # TODO make this satisfied a fuzzy value
                        if !issat
                            push!(rsums, sums[i_instance])
                            # rsums[i_instance] = sums[i_instance]
                            nr   += Wf[i_instance]
                            rsum += sums[i_instance]
                            # rssq += ssqs[i_instance]
                        else
                            push!(lsums, sums[i_instance])
                            # lsums[i_instance] = sums[i_instance]
                            if performing_consistency_check
                                unsatisfied[i_instance] = 0
                            end
                        end
                    end

                    # Calculate left counts
                    lsum = tsum - rsum
                    # lssq = tssq - rssq
                    nl   = nt - nr

                    (rsums, nr, lsums, nl, rsum, lsum)
                end
            end

            ####################################################################################
            ####################################################################################
            ####################################################################################

            # @logmsg LogDebug "  (n_left,n_right) = ($nl,$nr)"

            # Honor min_samples_leaf
            if !(nl >= min_samples_leaf && (_ninstances - nl) >= min_samples_leaf)
                continue
            end

            purity_times_nt = begin
                if isa(_is_classification, Val{true})
                    loss_function((ncl, nl), (ncr, nr))
                else
                    purity = begin
                        if W isa Ones{Int}
                            loss_function(lsums, lsum, nl, rsums, rsum, nr)
                        else
                            error("TODO expand regression code to weigthed version!")
                            loss_function(lsums, ws_l, nl, rsums, ws_r, nr)
                        end
                    end

                    # TODO use loss_function instead
                    # ORIGINAL: TODO understand how it works
                    # purity_times_nt = (rsum * rsum / nr) + (lsum * lsum / nl)
                    # Variance with ssqs
                    # purity_times_nt = (rmean, lmean = rsum/nr, lsum/nl; - (nr * (rssq - 2*rmean*rsum + (rmean^2*nr)) / (nr-1) + (nl * (lssq - 2*lmean*lsum + (lmean^2*nl)) / (nl-1))))
                    # Variance
                    # var = (x)->sum((x.-StatsBase.mean(x)).^2) / (length(x)-1)
                    # purity_times_nt = - (nr * var(rsums)) + nl * var(lsums))
                    # Simil-variance that is easier to compute but it does not work with few samples on the leaves
                    # var = (x)->sum((x.-StatsBase.mean(x)).^2)
                    # purity_times_nt = - (var(rsums) + var(lsums))
                    # println("purity_times_nt: $(purity_times_nt)")
                end
            end::P

            # If don't need to use lookahead, then I adopt the split only if it's better than the current one
            # Otherwise, I adopt it.
            if (
                !(isa(_using_lookahead, Val{false}) || is_lookahead_basecase)
                ||
                (purity_times_nt > node.purity_times_nt) # && !isapprox(purity_times_nt, node.purity_times_nt))
            )
                # DEBUGprintln((ncl,nl,ncr,nr), purity_times_nt)
                node.i_modality          = i_modality
                node.purity_times_nt     = purity_times_nt
                node.decision            = decision
                # print(decision)
                # println(" NEW BEST $node.i_modality, $node.purity_times_nt/nt")
                # @logmsg LogDetail "  Found new optimum in modality $(node.i_modality): " (node.purity_times_nt/nt) node.decision
                #################################
                node.consistency = begin
                    if performing_consistency_check
                        unsatisfied[1:_ninstances]
                    else
                        nr
                    end
                end

                # Short-circuit if you don't lookahead, and this is a perfect split
                if (isa(_using_lookahead, Val{false}) || is_lookahead_basecase) && istoploss(loss_function, purity_times_nt)
                    # @show "Threshold shortcircuit!"
                    break
                end
            end

            # In case of lookahead, temporarily accept the split,
            #  recurse on my children, and evaluate the purity of the whole subtree
            if (isa(_using_lookahead, Val{true}) && lookahead_depth < lookahead)
                Ss_copy = deepcopy(Ss)
                idxs_copy = deepcopy(idxs) # TODO maybe this reset is not needed?
                is_leaf = splitnode!(node, Ss_copy, idxs_copy)
                if is_leaf
                    # TODO: evaluate the goodneess of the leaf?
                else
                    # node.purity_times_nt
                    # purity_times_nt = loss_function((ncl, nl), (ncr, nr)) ...
                    for childnode in [node.l, node.r]
                        rng_copy =
                        optimize_node!(
                            childnode,
                            Xs,
                            Ss_copy,
                            Y,
                            initconditions,
                            W,
                            grouped_featsaggrsnopss,
                            grouped_featsnaggrss,
                            lookahead_depth+1,
                            ##########################################################################
                            _is_classification,
                            _using_lookahead,
                            _perform_consistency_check
                            ##########################################################################
                            ;
                            loss_function                  = loss_function,
                            lookahead                      = lookahead,
                            max_depth                      = max_depth,
                            min_samples_leaf               = min_samples_leaf,
                            min_purity_increase            = min_purity_increase,
                            max_purity_at_leaf             = max_purity_at_leaf,
                            ##########################################################################
                            max_modal_depth                = max_modal_depth,
                            n_subrelations                 = n_subrelations,
                            n_subfeatures                  = n_subfeatures,
                            allow_global_splits            = allow_global_splits,
                            ##########################################################################
                            idxs                           = deepcopy(idxs_copy),
                            n_classes                      = n_classes,
                            rng                            = copy(rng),
                        )
                    end
                    # TODO: evaluate the goodneess of the subtree?
                end
            end
        end
    end

    # Finally accept the split.
    if (isa(_using_lookahead, Val{false}) || is_lookahead_basecase)
        splitnode!(node, Ss, idxs)
    end

    # println("END split!")
    # readline()
    # node
end


############################################################################################
############################################################################################
############################################################################################

@inline function _fit_tree(
    Xs                        :: MultiLogiset,                         # modal dataset
    Y                         :: AbstractVector{L},                    # label vector
    initconditions            :: AbstractVector{<:InitialCondition},   # world starting conditions
    W                         :: AbstractVector{U}                     # weight vector
    ;
    ##########################################################################
    _is_classification        :: Union{Val{true},Val{false}},
    _using_lookahead          :: Union{Val{true},Val{false}},
    _perform_consistency_check:: Union{Val{true},Val{false}},
    ##########################################################################
    rng = Random.GLOBAL_RNG   :: Random.AbstractRNG,
    print_progress            :: Bool = true,
    kwargs...,
) where{L<:_Label,U}

    _ninstances = ninstances(Xs)

    # Initialize world sets for each instance
    Ss = ModalDecisionTrees.initialworldsets(Xs, initconditions)

    # Distribution of the instances indices throughout the tree.
    #  It will be recursively permuted, and regions of it assigned to the tree nodes (idxs[node.region])
    idxs = collect(1:_ninstances)

    # Create root node
    NodeMetaT = NodeMeta{(isa(_is_classification, Val{true}) ? Int64 : Float64),Float64}
    onlyallowglobal = [(initcond == ModalDecisionTrees.start_without_world) for initcond in initconditions]
    root = NodeMetaT(1:_ninstances, 0, 0, onlyallowglobal)
    
    if print_progress
        # p = ProgressThresh(Inf, 1, "Computing DTree...")
        p = ProgressUnknown("Computing DTree... nodes: ", spinner=true)
    end

    permodality_groups = [
        begin
            _features = features(X)
            _metaconditions = metaconditions(X)

            _grouped_metaconditions = SoleData.grouped_metaconditions(_metaconditions, _features)

            # _grouped_metaconditions::AbstractVector{<:AbstractVector{Tuple{<:ScalarMetaCondition}}}
            # [[(i_metacond, aggregator, metacondition)...]...]

            groups = [begin
                aggrsnops = Dict{Aggregator,AbstractVector{<:ScalarMetaCondition}}()
                aggregators_with_ids = Tuple{<:Integer,<:Aggregator}[]
                for (i_metacond, aggregator, metacondition) in these_metaconditions
                    if !haskey(aggrsnops, aggregator)
                        aggrsnops[aggregator] = Vector{ScalarMetaCondition}()
                    end
                    push!(aggrsnops[aggregator], metacondition)
                    push!(aggregators_with_ids, (i_metacond,aggregator))
                end
                (aggrsnops, aggregators_with_ids)
            end for (i_feature, (_feature, these_metaconditions)) in enumerate(_grouped_metaconditions)]
            grouped_featsaggrsnops = first.(groups)
            grouped_featsnaggrs = last.(groups)

            # grouped_featsaggrsnops::AbstractVector{<:AbstractDict{<:Aggregator,<:AbstractVector{<:ScalarMetaCondition}}}
            # [Dict([aggregator => [metacondition...]]...)...]

            # grouped_featsnaggrs::AbstractVector{<:AbstractVector{Tuple{<:Integer,<:Aggregator}}}
            # [[(i_metacond,aggregator)...]...]

            (grouped_featsaggrsnops, grouped_featsnaggrs)
        end for X in eachmodality(Xs)]

    grouped_featsaggrsnopss = first.(permodality_groups)
    grouped_featsnaggrss = last.(permodality_groups)

    # Process nodes recursively, using multi-threading
    function process_node!(node, rng)
        # Note: better to spawn rng's beforehand, to preserve reproducibility independently from optimize_node!
        rng_l = spawn(rng)
        rng_r = spawn(rng)
        @inbounds optimize_node!(
            node,
            Xs,
            Ss,
            Y,
            initconditions,
            W,
            grouped_featsaggrsnopss,
            grouped_featsnaggrss,
            0,
            ################################################################################
            _is_classification,
            _using_lookahead,
            _perform_consistency_check
            ################################################################################
            ;
            idxs                       = idxs,
            rng                        = rng,
            kwargs...,
        )
        # !print_progress || ProgressMeter.update!(p, node.purity)
        !print_progress || ProgressMeter.next!(p, spinner="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
        if !node.is_leaf
            l = Threads.@spawn process_node!(node.l, rng_l)
            r = Threads.@spawn process_node!(node.r, rng_r)
            wait(l), wait(r)
        end
    end
    @sync Threads.@spawn process_node!(root, rng)

    !print_progress || ProgressMeter.finish!(p)

    return (root, idxs)
end

##############################################################################
##############################################################################
##############################################################################
##############################################################################

@inline function check_input(
    Xs                      :: MultiLogiset,
    Y                       :: AbstractVector{S},
    initconditions          :: Vector{<:InitialCondition},
    W                       :: AbstractVector{U}
    ;
    ##########################################################################
    loss_function           :: Loss,
    lookahead               :: Integer,
    max_depth               :: Union{Nothing,Int},
    min_samples_leaf        :: Int,
    min_purity_increase     :: AbstractFloat,
    max_purity_at_leaf      :: AbstractFloat,
    ##########################################################################
    max_modal_depth         :: Union{Nothing,Int},
    n_subrelations          :: Vector{<:Function},
    n_subfeatures           :: Vector{<:Integer},
    allow_global_splits     :: Vector{Bool},
    ##########################################################################
    kwargs...,
) where {S,U}
    _ninstances = ninstances(Xs)

    if length(Y) != _ninstances
        error("Dimension mismatch between dataset and label vector Y: ($(_ninstances)) vs $(size(Y))")
    elseif length(W) != _ninstances
        error("Dimension mismatch between dataset and weights vector W: ($(_ninstances)) vs $(size(W))")
    ############################################################################
    elseif length(n_subrelations) != nmodalities(Xs)
        error("Mismatching number of n_subrelations with number of modalities: $(length(n_subrelations)) vs $(nmodalities(Xs))")
    elseif length(n_subfeatures)  != nmodalities(Xs)
        error("Mismatching number of n_subfeatures with number of modalities: $(length(n_subfeatures)) vs $(nmodalities(Xs))")
    elseif length(initconditions) != nmodalities(Xs)
        error("Mismatching number of initconditions with number of modalities: $(length(initconditions)) vs $(nmodalities(Xs))")
    elseif length(allow_global_splits) != nmodalities(Xs)
        error("Mismatching number of allow_global_splits with number of modalities: $(length(allow_global_splits)) vs $(nmodalities(Xs))")
    ############################################################################
    # elseif any(nrelations.(eachmodality(Xs)) .< n_subrelations)
    #   error("In at least one modality the total number of relations is less than the number "
    #       * "of relations required at each split\n"
    #       * "# relations:    " * string(nrelations.(eachmodality(Xs))) * "\n\tvs\n"
    #       * "# subrelations: " * string(n_subrelations |> collect))
    # elseif length(findall(n_subrelations .< 0)) > 0
    #   error("Total number of relations $(n_subrelations) must be >= zero ")
    elseif any(nfeatures.(eachmodality(Xs)) .< n_subfeatures)
        error("In at least one modality the total number of features is less than the number "
            * "of features required at each split\n"
            * "# features:    " * string(nfeatures.(eachmodality(Xs))) * "\n\tvs\n"
            * "# subfeatures: " * string(n_subfeatures |> collect))
    elseif length(findall(n_subfeatures .< 0)) > 0
        error("Total number of features $(n_subfeatures) must be >= zero ")
    elseif min_samples_leaf < 1
        error("Min_samples_leaf must be a positive integer "
            * "(given $(min_samples_leaf))")
    # if loss_function in [entropy]
    #   max_purity_at_leaf_thresh = 0.75 # min_purity_increase 0.01
    #   min_purity_increase_thresh = 0.5
    #   if (max_purity_at_leaf >= max_purity_at_leaf_thresh)
    #       println("Warning! It is advised to use max_purity_at_leaf<$(max_purity_at_leaf_thresh) with loss $(loss_function)"
    #           * "(given $(max_purity_at_leaf))")
    #   elseif (min_purity_increase >= min_purity_increase_thresh)
    #       println("Warning! It is advised to use max_purity_at_leaf<$(min_purity_increase_thresh) with loss $(loss_function)"
    #           * "(given $(min_purity_increase))")
    # end
    # elseif loss_function in [gini, zero_one] && (max_purity_at_leaf > 1.0 || max_purity_at_leaf <= 0.0)
    #     error("Max_purity_at_leaf for loss $(loss_function) must be in (0,1]"
    #         * "(given $(max_purity_at_leaf))")
    elseif !isnothing(max_depth) && max_depth < 0
        error("Unexpected value for max_depth: $(max_depth) (expected:"
            * " max_depth >= 0, or max_depth = nothing for unbounded depth)")
    elseif !isnothing(max_modal_depth) && max_modal_depth < 0
        error("Unexpected value for max_modal_depth: $(max_modal_depth) (expected:"
            * " max_modal_depth >= 0, or max_modal_depth = nothing for unbounded depth)")
    end

    if !(lookahead >= 0)
        error("Unexpected value for lookahead: $(lookahead) (expected:"
            * " lookahead >= 0)")
    end

    if SoleData.hasnans(Xs)
        error("This algorithm does not allow NaN values")
    end

    if nothing in Y
        error("This algorithm does not allow nothing values in Y")
    elseif eltype(Y) <: Number && any(isnan.(Y))
        error("This algorithm does not allow NaN values in Y")
    elseif nothing in W
        error("This algorithm does not allow nothing values in W")
    elseif any(isnan.(W))
        error("This algorithm does not allow NaN values in W")
    end

end


############################################################################################
############################################################################################
############################################################################################
################################################################################

function fit_tree(
    # modal dataset
    Xs                        :: MultiLogiset,
    # label vector
    Y                         :: AbstractVector{L},
    # world starting conditions
    initconditions            :: Vector{<:InitialCondition},
    # Weights (unary weigths are used if no weight is supplied)
    W                         :: AbstractVector{U} = default_weights(Y)
    # W                       :: AbstractVector{U} = Ones{Int}(ninstances(Xs)), # TODO check whether this is faster
    ;
    # Lookahead parameter (i.e., depth of the trees to locally optimize for)
    lookahead                 :: Integer = 0,
    # Perform minification: transform dataset so that learning happens faster
    use_minification          :: Bool,
    # Debug-only: checks the consistency of the dataset during training
    perform_consistency_check :: Bool,
    kwargs...,
) where {L<:Union{CLabel,RLabel}, U}
    # Check validity of the input
    check_input(Xs, Y, initconditions, W; lookahead = lookahead, kwargs...)

    # Classification-only: transform labels to categorical form (indexed by integers)
    n_classes = begin
        if L<:CLabel
            class_names, Y = get_categorical_form(Y)
            length(class_names)
        else
            0 # dummy value for the case of regression
        end
    end

    Xs, threshold_backmaps = begin
        if use_minification
            minify(Xs)
        else
            Xs, fill(identity, nmodalities(Xs))
        end
    end

    # println(threshold_backmaps)
    # Call core learning function
    root, idxs = _fit_tree(Xs, Y, initconditions, W,
        ;
        _is_classification = Val(L<:CLabel),
        _using_lookahead = Val((lookahead > 0)),
        _perform_consistency_check = Val(perform_consistency_check),
        lookahead = lookahead,
        n_classes = n_classes,
        kwargs...
    )
    
    # Finally create Tree
    root = begin
        if L<:CLabel
            _convert(root, map((y)->class_names[y], Y[idxs]), class_names, threshold_backmaps)
        else
            _convert(root, Y[idxs], threshold_backmaps)
        end
    end
    DTree{L}(root, worldtype.(eachmodality(Xs)), initconditions)
end
