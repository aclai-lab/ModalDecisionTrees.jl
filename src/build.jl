
include("fit_tree.jl")

################################################################################
############################# Unimodal datasets ################################
################################################################################

function build_stump(X::AbstractLogiset, args...; kwargs...)
    build_stump(MultiLogiset(X), args...; kwargs...)
end

function build_tree(X::AbstractLogiset, args...; kwargs...)
    build_tree(MultiLogiset(X), args...; kwargs...)
end

function build_forest(X::AbstractLogiset, args...; kwargs...)
    build_forest(MultiLogiset(X), args...; kwargs...)
end

################################################################################
############################ Multimodal datasets ###############################
################################################################################

doc_build = """
    build_stump(X, Y, W = nothing; kwargs...)
    build_tree(X, Y, W = nothing; kwargs...)
    build_forest(X, Y, W = nothing; kwargs...)

Train a decision stump (i.e., decision tree with depth 1), a decision tree, or
a random forest model on logiset `X` with labels `Y` and weights `W`.

"""

"""$(doc_build)"""
function build_stump(
    X                 :: MultiLogiset,
    Y                 :: AbstractVector{L},
    W                 :: Union{Nothing,AbstractVector{U},Symbol} = nothing;
    kwargs...,
) where {L<:Label,U}
    params = NamedTuple(kwargs)
    @assert !haskey(params, :max_depth) || params.max_depth == 1 "build_stump " *
        "doesn't allow max_depth != 1."
    build_tree(X, Y, W; max_depth = 1, kwargs...)
end

"""$(doc_build)"""
function build_tree(
    X                   :: MultiLogiset,
    Y                   :: AbstractVector{L},
    W                   :: Union{Nothing,AbstractVector{U},Symbol}   = default_weights(ninstances(X));
    ##############################################################################
    loss_function       :: Union{Nothing,Function}            = nothing,
    max_depth           :: Union{Nothing,Int64}               = nothing,
    min_samples_leaf    :: Int64                              = BOTTOM_MIN_SAMPLES_LEAF,
    min_purity_increase :: AbstractFloat                      = BOTTOM_MIN_PURITY_INCREASE,
    max_purity_at_leaf  :: AbstractFloat                      = BOTTOM_MAX_PURITY_AT_LEAF,
    ##############################################################################
    max_modal_depth     :: Union{Nothing,Int64}               = nothing,
    n_subrelations      :: Union{Function,AbstractVector{<:Function}}                   = identity,
    n_subfeatures       :: Union{Function,AbstractVector{<:Function}}                   = identity,
    initconditions      :: Union{InitialCondition,AbstractVector{<:InitialCondition}}   = start_without_world,
    allow_global_splits :: Union{Bool,AbstractVector{Bool}}                             = true,
    ##############################################################################
    use_minification          :: Bool = false,
    perform_consistency_check :: Bool = DEFAULT_PERFORM_CONSISTENCY_CHECK,
    ##############################################################################
    rng                 :: Random.AbstractRNG = Random.GLOBAL_RNG,
    print_progress      :: Bool = true,
) where {L<:Label,U}
    
    @assert W isa AbstractVector || W in [nothing, :rebalance, :default]

    W = if isnothing(W) || W == :default
        default_weights(Y)
    elseif W == :rebalance
        balanced_weights(Y)
    else
        W
    end

    @assert all(W .>= 0) "Sample weights must be non-negative."

    @assert ninstances(X) == length(Y) == length(W) "Mismatching number of samples in X, Y & W: $(ninstances(X)), $(length(Y)), $(length(W))"
    
    if isnothing(loss_function)
        loss_function = default_loss_function(L)
    end
    
    if allow_global_splits isa Bool
        allow_global_splits = fill(allow_global_splits, nmodalities(X))
    end
    if n_subrelations isa Function
        n_subrelations = fill(n_subrelations, nmodalities(X))
    end
    if n_subfeatures isa Function
        n_subfeatures  = fill(n_subfeatures, nmodalities(X))
    end
    if initconditions isa InitialCondition
        initconditions = fill(initconditions, nmodalities(X))
    end

    @assert isnothing(max_depth) || (max_depth >= 0)
    @assert isnothing(max_modal_depth) || (max_modal_depth >= 0)

    fit_tree(X, Y, initconditions, W
        ;###########################################################################
        loss_function               = loss_function,
        max_depth                   = max_depth,
        min_samples_leaf            = min_samples_leaf,
        min_purity_increase         = min_purity_increase,
        max_purity_at_leaf          = max_purity_at_leaf,
        ############################################################################
        max_modal_depth             = max_modal_depth,
        n_subrelations              = n_subrelations,
        n_subfeatures               = [ n_subfeatures[i](nfeatures(modality)) for (i,modality) in enumerate(eachmodality(X)) ],
        allow_global_splits         = allow_global_splits,
        ############################################################################
        use_minification            = use_minification,
        perform_consistency_check   = perform_consistency_check,
        ############################################################################
        rng                         = rng,
        print_progress              = print_progress,
    )
end

"""$(doc_build)"""
function build_forest(
    X                   :: MultiLogiset,
    Y                   :: AbstractVector{L},
    # Use unary weights if no weight is supplied
    W                   :: Union{Nothing,AbstractVector{U},Symbol} = default_weights(Y);
    ##############################################################################
    # Forest logic-agnostic parameters
    ntrees              = 100,
    partial_sampling    = 0.7,      # portion of sub-sampled samples (without replacement) by each tree
    ##############################################################################
    # Tree logic-agnostic parameters
    loss_function       :: Union{Nothing,Function}          = nothing,
    max_depth           :: Union{Nothing,Int64}             = nothing,
    min_samples_leaf    :: Int64                            = BOTTOM_MIN_SAMPLES_LEAF,
    min_purity_increase :: AbstractFloat                    = BOTTOM_MIN_PURITY_INCREASE,
    max_purity_at_leaf  :: AbstractFloat                    = BOTTOM_MAX_PURITY_AT_LEAF,
    ##############################################################################
    # Modal parameters
    max_modal_depth     :: Union{Nothing,Int64}             = nothing,
    n_subrelations      :: Union{Function,AbstractVector{<:Function}}                   = identity,
    n_subfeatures       :: Union{Function,AbstractVector{<:Function}}                   = x -> ceil(Int64, sqrt(x)),
    initconditions      :: Union{InitialCondition,AbstractVector{<:InitialCondition}}   = start_without_world,
    allow_global_splits :: Union{Bool,AbstractVector{Bool}}                             = true,
    ##############################################################################
    use_minification    :: Bool = false,
    perform_consistency_check :: Bool = DEFAULT_PERFORM_CONSISTENCY_CHECK,
    ##############################################################################
    rng                 :: Random.AbstractRNG = Random.GLOBAL_RNG,
    print_progress      :: Bool = true,
    suppress_parity_warning :: Bool = false,
) where {L<:Label,U}

    @assert W isa AbstractVector || W in [nothing, :rebalance, :default]

    W = if isnothing(W) || W == :default
        default_weights(Y)
    elseif W == :rebalance
        balanced_weights(Y)
    else
        W
    end

    @assert all(W .>= 0) "Sample weights must be non-negative."

    @assert ninstances(X) == length(Y) == length(W) "Mismatching number of samples in X, Y & W: $(ninstances(X)), $(length(Y)), $(length(W))"

    if n_subrelations isa Function
        n_subrelations = fill(n_subrelations, nmodalities(X))
    end
    if n_subfeatures isa Function
        n_subfeatures  = fill(n_subfeatures, nmodalities(X))
    end
    if initconditions isa InitialCondition
        initconditions = fill(initconditions, nmodalities(X))
    end
    if allow_global_splits isa Bool
        allow_global_splits = fill(allow_global_splits, nmodalities(X))
    end

    if ntrees < 1
        error("the number of trees must be >= 1")
    end
    
    if !(0.0 < partial_sampling <= 1.0)
        error("partial_sampling must be in the range (0,1]")
    end
    
    if any(map(f->!(f isa SupportedLogiset), eachmodality(X)))
        @warn "Warning! Consider using structures optimized for model checking " *
            "such as SupportedLogiset."
    end

    tot_samples = ninstances(X)
    num_samples = floor(Int64, partial_sampling * tot_samples)

    trees = Vector{DTree{L}}(undef, ntrees)
    oob_instances = Vector{Vector{Integer}}(undef, ntrees)
    oob_metrics = Vector{NamedTuple}(undef, ntrees)

    rngs = [spawn(rng) for i_tree in 1:ntrees]

    if print_progress
        p = Progress(ntrees, 1, "Computing Forest...")
    end
    Threads.@threads for i_tree in 1:ntrees
        train_idxs = rand(rngs[i_tree], 1:tot_samples, num_samples)

        X_slice = SoleData.instances(X, train_idxs, Val(true))
        Y_slice = @view Y[train_idxs]
        W_slice = SoleModels.slice_weights(W, train_idxs)

        trees[i_tree] = build_tree(
            X_slice
            , Y_slice
            , W_slice
            ;
            ################################################################################
            loss_function        = loss_function,
            max_depth            = max_depth,
            min_samples_leaf     = min_samples_leaf,
            min_purity_increase  = min_purity_increase,
            max_purity_at_leaf   = max_purity_at_leaf,
            ################################################################################
            max_modal_depth      = max_modal_depth,
            n_subrelations       = n_subrelations,
            n_subfeatures        = n_subfeatures,
            initconditions       = initconditions,
            allow_global_splits  = allow_global_splits,
            ################################################################################
            use_minification     = use_minification,
            perform_consistency_check = perform_consistency_check,
            ################################################################################
            rng                  = rngs[i_tree],
            print_progress       = false,
        )

        # grab out-of-bag indices
        oob_instances[i_tree] = setdiff(1:tot_samples, train_idxs)

        tree_preds = apply(trees[i_tree], SoleData.instances(X, oob_instances[i_tree], Val(true)))

        oob_metrics[i_tree] = (;
            actual = Y[oob_instances[i_tree]],
            predicted = tree_preds,
            weights = collect(SoleModels.slice_weights(W, oob_instances[i_tree]))
        )

        !print_progress || next!(p)
    end

    metrics = (;
        oob_metrics = oob_metrics,
    )

    if L<:CLabel
        # For each sample, construct its random forest predictor
        #  by averaging (or majority voting) only those
        #  trees corresponding to boot-strap samples in which the sample did not appear
        oob_classified = Vector{Bool}()
        Threads.@threads for i in 1:tot_samples
            selected_trees = fill(false, ntrees)
            
            # pick every tree trained without i-th sample
            for i_tree in 1:ntrees
                if i in oob_instances[i_tree] # if i is present in the i_tree-th tree, selecte thi tree
                    selected_trees[i_tree] = true
                end
            end
            
            index_of_trees_to_test_with = findall(selected_trees)
            
            if length(index_of_trees_to_test_with) == 0
                continue
            end
            
            X_slice = SoleData.instances(X, [i], Val(true))
            Y_slice = [Y[i]]
            
            preds = apply(trees[index_of_trees_to_test_with], X_slice; suppress_parity_warning = suppress_parity_warning)
            
            push!(oob_classified, Y_slice[1] == preds[1])
        end
        oob_error = 1.0 - (sum(W[findall(oob_classified)]) / sum(W))
        metrics = merge(metrics, (
            oob_error = oob_error,
        ))
    end

    DForest{L}(trees, metrics)
end
