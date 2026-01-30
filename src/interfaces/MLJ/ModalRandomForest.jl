"""
    ModalRandomForest <: MMI.Probabilistic

A modal random forest classifier that combines multiple modal decision trees for
robust probabilistic predictions using ensemble learning and modal logic.

This model extends the concept of random forests by incorporating modal logic operators
and relations, enabling powerful ensemble reasoning about data with complex relational,
temporal, or spatial structures. Each tree in the forest is trained on a bootstrap sample
of the data and uses modal logic for its decision-making process.

# Fields

## Ensemble Parameters
- `sampling_fraction::Float64`: Fraction of data to sample for each tree (default: 0.7).
  Each tree sees 70% of the training data, promoting diversity in the ensemble.
- `ntrees::Int`: Number of trees in the forest (default: 10). More trees generally
  improve accuracy but increase computational cost.

## Pruning Conditions
- `max_depth::Union{Nothing,Int}`: Maximum depth of each decision tree. `nothing` means no limit.
- `min_samples_leaf::Union{Nothing,Int}`: Minimum number of samples required at a leaf node.
- `min_purity_increase::Union{Nothing,Float64}`: Minimum purity increase required for a split.
- `max_purity_at_leaf::Union{Nothing,Float64}`: Maximum purity allowed at a leaf node before stopping.
- `max_modal_depth::Union{Nothing,Int}`: Maximum depth for modal operators in logical formulas.

## Logic Parameters

### Relation Set
- `relations::Union{...}`: Defines the relational structure for modal reasoning. Can be:
  - `nothing`: Uses default relations based on data characteristics
  - `Symbol`: One of the predefined relation sets from `AVAILABLE_RELATIONS`
  - `Vector{<:AbstractRelation}`: Explicitly specified relation set
  - `Function`: A function mapping world type to appropriate relation set

### Feature/Condition Set
- `features::Union{...}`: Defines feature extraction for tree splits. Can be:
  - `nothing`: Defaults to scalar conditions (≥, <) on standard features (min, max) for all variables
  - `Vector{<:Union{SoleData.VarFeature,Base.Callable}}`: Scalar conditions on explicit features
  - `Vector{<:Tuple{Base.Callable,Integer}}`: Features as callables applied to specific variables
  - `Vector{<:Tuple{TestOperator,<:Union{SoleData.VarFeature,Base.Callable}}}`: Explicit (operator, feature) pairs
  - `Vector{<:SoleData.ScalarMetaCondition}`: Explicit scalar condition set

- `conditions::Union{...}`: Alternative specification for conditions (same options as `features`)

- `featvaltype::Type`: Data type for extracted feature values (default: `Float64`)

### Initial Conditions
- `initconditions::Union{...}`: Starting conditions for the learning algorithm. Can be:
  - `nothing`: Uses standard initial conditions (e.g., `start_without_world`)
  - `Symbol`: One of the predefined initial conditions from `AVAILABLE_INITIALCONDITIONS`
  - `InitialCondition`: Explicitly specified initial condition object

## Miscellaneous Parameters
- `downsize::Union{Bool,NTuple{N,Integer},Function}`: Whether/how to reduce data dimensionality
- `force_i_variables::Bool`: Force the use of interval variables in modal logic
- `fixcallablenans::Bool`: Whether to handle NaN values in callable features
- `print_progress::Bool`: Whether to print training progress (automatically enabled for forests with >50 trees)
- `rng::Union{Random.AbstractRNG,Integer}`: Random number generator or seed for reproducibility

## DecisionTree.jl Compatibility Parameters
- `display_depth::Union{Nothing,Int}`: Maximum depth to display when printing trees
- `min_samples_split::Union{Nothing,Int}`: Minimum samples required to split an internal node
- `n_subfeatures::Union{Nothing,Int,Float64,Function}`: Number of features to consider for splits in each tree
- `post_prune::Bool`: Whether to perform post-pruning after tree construction
- `merge_purity_threshold::Union{Nothing,Float64}`: Purity threshold for merging nodes during pruning
- `feature_importance::Symbol`: Method for computing feature importance (default: `:split`)

# How It Works

A random forest works by:
1. Creating `ntrees` bootstrap samples (random samples with replacement) from the training data
2. Training a separate modal decision tree on each sample
3. For predictions, each tree votes and the final prediction is aggregated (usually by averaging probabilities)
4. The diversity from sampling and random feature selection reduces overfitting

# Examples

```julia
# Basic usage with default parameters
model = ModalRandomForest()

# Larger forest with custom sampling
model = ModalRandomForest(
    ntrees = 100,
    sampling_fraction = 0.8,
    max_depth = 15
)

# Custom configuration for temporal data
model = ModalRandomForest(
    ntrees = 50,
    sampling_fraction = 0.7,
    relations = :IA,  # Allen's Interval Algebra for temporal reasoning
    max_modal_depth = 3,
    features = [minimum, maximum, mean],
    min_samples_leaf = 10,
    rng = 42
)

# Full customization with reproducibility
model = ModalRandomForest(
    ntrees = 200,
    sampling_fraction = 0.6,
    max_depth = 20,
    min_samples_leaf = 5,
    min_purity_increase = 0.01,
    relations = custom_relations,
    features = custom_features,
    n_subfeatures = 0.3,  # Consider 30% of features at each split
    print_progress = true,
    rng = 123
)
```

# Performance Considerations

- **More trees** → Better accuracy, slower training/prediction
- **Higher sampling_fraction** → Less diversity, potentially more overfitting
- **Lower sampling_fraction** → More diversity, each tree sees less data
- **Optimal ntrees**: Usually 50-200 trees provide a good balance
- **Parallelization**: Training multiple trees can be parallelized for speed

# See Also
- `ModalDecisionTree`: The base model for individual trees
- `MMI.fit!`: For training the forest
- `MMI.predict`: For making ensemble predictions
- Random forest documentation for general ensemble concepts
"""
mutable struct ModalRandomForest <: MMI.Probabilistic

    ## Ensemble parameters
    # Fraction of training data to sample for each tree (bootstrap sampling)
    # Lower values increase diversity, higher values give each tree more data
    sampling_fraction      :: Float64

    # Number of trees in the forest
    # More trees improve stability and accuracy but increase computation time
    ntrees                 :: Int

    ## Pruning conditions
    # These parameters control when to stop growing each tree
    max_depth              :: Union{Nothing,Int}
    min_samples_leaf       :: Union{Nothing,Int}
    min_purity_increase    :: Union{Nothing,Float64}
    max_purity_at_leaf     :: Union{Nothing,Float64}

    # Maximum depth for modal operators in logical formulas
    max_modal_depth        :: Union{Nothing,Int}

    ## Logic parameters

    # Relation set: defines the modal/relational structure
    # Shared across all trees in the forest
    relations              :: Union{
        Nothing,                                              # defaults to a well-known relation set, depending on the data;
        Symbol,                                               # one of the relation sets specified in AVAILABLE_RELATIONS;
        Vector{<:AbstractRelation},                           # explicitly specify the relation set;
        # Vector{<:Union{Symbol,Vector{<:AbstractRelation}}}, # MULTIMODAL CASE: specify a relation set for each modality;
        Function                                              # A function worldtype -> relation set.
    }

    # Feature set: defines how to extract features from the data
    # Each tree can randomly select from this set at each split
    features             :: Union{
        Nothing,                                                                   # defaults to scalar conditions (with ≥ and <) on well-known feature functions (e.g., minimum, maximum), applied to all variables;
        Vector{<:Union{SoleData.VarFeature,Base.Callable}},                        # scalar conditions with ≥ and <, on an explicitly specified feature set (callables to be applied to each variable, or VarFeature objects);
        Vector{<:Tuple{Base.Callable,Integer}},                                    # scalar conditions with ≥ and <, on a set of features specified as a set of callables to be applied to a set of variables each;
        Vector{<:Tuple{TestOperator,<:Union{SoleData.VarFeature,Base.Callable}}},  # explicitly specify the pairs (test operator, feature);
        Vector{<:SoleData.ScalarMetaCondition},                                    # explicitly specify the scalar condition set.
    }

    # Condition set: alternative/complementary way to specify split conditions
    conditions             :: Union{
        Nothing,                                                                     # defaults to scalar conditions (with ≥ and <) on well-known feature functions (e.g., minimum, maximum), applied to all variables;
        Vector{<:Union{SoleData.VarFeature,Base.Callable}},                          # scalar conditions with ≥ and <, on an explicitly specified feature set (callables to be applied to each variable, or VarFeature objects);
        Vector{<:Tuple{Base.Callable,Integer}},                                      # scalar conditions with ≥ and <, on a set of features specified as a set of callables to be applied to a set of variables each;
        Vector{<:Tuple{TestOperator,<:Union{SoleData.VarFeature,Base.Callable}}},    # explicitly specify the pairs (test operator, feature);
        Vector{<:SoleData.ScalarMetaCondition},                                      # explicitly specify the scalar condition set.
    }

    # Type for the extracted feature values (typically Float64 for numerical stability)
    featvaltype            :: Type

    # Initial conditions for the modal logic learning algorithm
    # Applied consistently across all trees
    initconditions         :: Union{
        Nothing,                                                                     # defaults to standard conditions (e.g., start_without_world)
        Symbol,                                                                      # one of the initial conditions specified in AVAILABLE_INITIALCONDITIONS;
        InitialCondition,                                                            # explicitly specify an initial condition for the learning algorithm.
    }

    ## Miscellaneous
    # Whether to reduce data dimensionality (can be boolean, tuple of sizes, or custom function)
    downsize               :: Union{Bool,NTuple{N,Integer} where N,Function}

    # Force the use of interval variables in modal logic formulas
    force_i_variables      :: Bool

    # Whether to fix/handle NaN values that appear in callable features
    fixcallablenans        :: Bool

    # Display training progress information during fitting
    # Automatically enabled for forests with more than 50 trees
    print_progress         :: Bool

    # Random number generator for reproducible results
    # Controls both sampling and random feature selection
    rng                    :: Union{Random.AbstractRNG,Integer}

    ## DecisionTree.jl compatibility parameters
    # These parameters maintain compatibility with the standard DecisionTree.jl package

    # Maximum depth to display when printing/visualizing trees
    display_depth          :: Union{Nothing,Int}

    # Minimum number of samples required to attempt splitting an internal node
    min_samples_split      :: Union{Nothing,Int}

    # Number of features to randomly consider at each split (random forest feature bagging)
    # Can be integer (exact count), float (proportion), or function
    # This promotes diversity among trees
    n_subfeatures          :: Union{Nothing,Int,Float64,Function}

    # Whether to perform post-pruning to reduce overfitting
    post_prune             :: Bool

    # Purity threshold for merging similar nodes during pruning
    merge_purity_threshold :: Union{Nothing,Float64}

    # Method for computing feature importance scores across the forest
    # :split counts how many times each feature is used for splitting across all trees
    feature_importance     :: Symbol
end

"""
    ModalRandomForest(; kwargs...)

Keyword constructor for `ModalRandomForest` with sensible defaults for ensemble learning.

# Keyword Arguments

## Ensemble Parameters
- `sampling_fraction = 0.7`: Fraction of data for each tree (70% is a good default)
- `ntrees = 10`: Number of trees in the forest

## Pruning Parameters
- `max_depth = nothing`: Maximum tree depth
- `min_samples_leaf = nothing`: Minimum samples per leaf
- `min_purity_increase = nothing`: Minimum purity gain for splits
- `max_purity_at_leaf = nothing`: Maximum leaf purity threshold
- `max_modal_depth = nothing`: Maximum modal operator depth

## Logic Parameters
- `relations = nothing`: Modal relation set
- `features = nothing`: Feature extraction specification
- `conditions = nothing`: Split condition specification
- `featvaltype = Float64`: Type for feature values
- `initconditions = nothing`: Initial conditions for learning

## Miscellaneous
- `downsize = true`: Enable dimensionality reduction
- `force_i_variables = true`: Force interval variables
- `fixcallablenans = false`: Handle NaN in callables
- `print_progress = (ntrees > 50)`: Show progress for large forests (auto-enabled if >50 trees)
- `rng = Random.GLOBAL_RNG`: Random number generator

## DecisionTree.jl Parameters
- `display_depth = nothing`: Display depth for printing
- `min_samples_split = nothing`: Minimum samples to split
- `n_subfeatures = nothing`: Number of features per split (enables feature bagging)
- `post_prune = false`: Enable post-pruning
- `merge_purity_threshold = nothing`: Purity threshold for merging
- `feature_importance = :split`: Importance calculation method

# Returns
- `ModalRandomForest`: A new ensemble model with validated parameters

# Notes
- The constructor automatically validates parameters using `MMI.clean!`
- Progress printing is automatically enabled for forests with more than 50 trees
- Random sampling and feature selection are controlled by the `rng` parameter
- Each tree in the forest is trained independently on a bootstrap sample

# Examples
```julia
# Quick start with defaults (10 trees, 70% sampling)
model = ModalRandomForest()

# Small forest for testing
model = ModalRandomForest(ntrees = 5, sampling_fraction = 0.8)

# Production-ready forest with temporal relations
model = ModalRandomForest(
    ntrees = 100,
    sampling_fraction = 0.7,
    max_depth = 15,
    min_samples_leaf = 5,
    relations = :temporal,
    features = [minimum, maximum, mean, std],
    n_subfeatures = 0.3,  # Consider 30% of features at each split
    rng = 42
)

# Large forest with progress monitoring
model = ModalRandomForest(
    ntrees = 200,  # Progress will be shown automatically
    sampling_fraction = 0.6,
    max_depth = 20,
    relations = custom_relations,
    print_progress = true,  # Explicitly enable
    rng = 123
)
```

# Performance Tips
- Start with `ntrees = 50-100` for most applications
- Use `sampling_fraction = 0.6-0.8` for good diversity/accuracy balance
- Set `n_subfeatures` to promote diversity (e.g., 0.3 means 30% of features)
- Use a fixed `rng` seed for reproducible results
- For large forests (>100 trees), consider parallelization
"""
function ModalRandomForest(;
    # Ensemble parameters with reasonable defaults
    sampling_fraction = 0.7,  # Bootstrap 70% of data for each tree
    ntrees = 10,              # Default to 10 trees (can be increased for better accuracy)

    # Pruning conditions
    max_depth = nothing,
    min_samples_leaf = nothing,
    min_purity_increase = nothing,
    max_purity_at_leaf = nothing,
    max_modal_depth = nothing,

    # Logic parameters
    relations = nothing,
    features = nothing,
    conditions = nothing,
    featvaltype = Float64,
    initconditions = nothing,

    # Miscellaneous parameters
    downsize = true,
    force_i_variables = true,
    fixcallablenans = false,
    # Automatically show progress for large forests (>50 trees)
    print_progress = (ntrees > 50),
    rng = Random.GLOBAL_RNG,

    # DecisionTree.jl compatibility parameters
    display_depth = nothing,
    min_samples_split = nothing,
    n_subfeatures = nothing,  # When set, enables feature bagging (random feature selection)
    post_prune = false,
    merge_purity_threshold = nothing,
    feature_importance = :split,
)
    # Construct the forest model with all parameters
    model = ModalRandomForest(
        sampling_fraction,
        ntrees,
        #
        max_depth,
        min_samples_leaf,
        min_purity_increase,
        max_purity_at_leaf,
        max_modal_depth,
        #
        relations,
        features,
        conditions,
        featvaltype,
        initconditions,
        #
        downsize,
        force_i_variables,
        fixcallablenans,
        print_progress,
        rng,
        #
        display_depth,
        min_samples_split,
        n_subfeatures,
        post_prune,
        merge_purity_threshold,
        feature_importance,
    )

    # Validate the model parameters
    # MMI.clean! checks for invalid parameter combinations and returns warning messages
    message = MMI.clean!(model)

    # Issue a warning if there are any validation issues
    isempty(message) || @warn message

    return model
end
