"""
    ModalDecisionTree <: MMI.Probabilistic

A modal decision tree classifier for probabilistic machine learning tasks using modal logic.

This model extends traditional decision trees by incorporating modal logic operators and
relations, enabling reasoning about data with complex relational structures and temporal
or spatial modalities.

# Fields

## Pruning Conditions
- `max_depth::Union{Nothing,Int}`: Maximum depth of the decision tree. `nothing` means no limit.
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
- `print_progress::Bool`: Whether to print training progress information
- `rng::Union{Random.AbstractRNG,Integer}`: Random number generator or seed for reproducibility

## DecisionTree.jl Compatibility Parameters
- `display_depth::Union{Nothing,Int}`: Maximum depth to display when printing the tree
- `min_samples_split::Union{Nothing,Int}`: Minimum samples required to split an internal node
- `n_subfeatures::Union{Nothing,Int,Float64,Function}`: Number of features to consider for splits
- `post_prune::Bool`: Whether to perform post-pruning after tree construction
- `merge_purity_threshold::Union{Nothing,Float64}`: Purity threshold for merging nodes during pruning
- `feature_importance::Symbol`: Method for computing feature importance (default: `:split`)

# Examples

```julia
# Basic usage with default parameters
model = ModalDecisionTree()

# Custom pruning parameters
model = ModalDecisionTree(
    max_depth = 10,
    min_samples_leaf = 5,
    min_purity_increase = 0.01
)

# Specify custom relations and features
model = ModalDecisionTree(
    relations = :IA,  # Allen's Interval Algebra relations
    max_modal_depth = 3,
    features = [minimum, maximum, mean]
)

# Full customization
model = ModalDecisionTree(
    max_depth = 15,
    min_samples_leaf = 10,
    relations = custom_relations,
    features = custom_features,
    rng = 42,
    print_progress = true
)
```

# See Also
- `MMI.fit!`: For training the model
- `MMI.predict`: For making predictions
- Modal logic documentation for understanding relational structures
"""
mutable struct ModalDecisionTree <: MMI.Probabilistic

    ## Pruning conditions
    # These parameters control when to stop growing the tree
    max_depth              :: Union{Nothing,Int}
    min_samples_leaf       :: Union{Nothing,Int}
    min_purity_increase    :: Union{Nothing,Float64}
    max_purity_at_leaf     :: Union{Nothing,Float64}

    # Maximum depth for modal operators in logical formulas
    max_modal_depth        :: Union{Nothing,Int}

    ## Logic parameters

    # Relation set: defines the modal/relational structure
    # Used for reasoning about relationships between data points or worlds
    relations              :: Union{
        Nothing,                                            # defaults to a well-known relation set, depending on the data;
        Symbol,                                             # one of the relation sets specified in AVAILABLE_RELATIONS;
        Vector{<:AbstractRelation},                         # explicitly specify the relation set;
        # Vector{<:Union{Symbol,Vector{<:AbstractRelation}}}, # MULTIMODAL CASE: specify a relation set for each modality;
        Function                                            # A function worldtype -> relation set.
    }

    # Feature set: defines how to extract features from the data
    # These features are used to construct split conditions in the tree
    features             :: Union{
        Nothing,                                                                     # defaults to scalar conditions (with ≥ and <) on well-known feature functions (e.g., minimum, maximum), applied to all variables;
        Vector{<:Union{SoleData.VarFeature,Base.Callable}},                          # scalar conditions with ≥ and <, on an explicitly specified feature set (callables to be applied to each variable, or VarFeature objects);
        Vector{<:Tuple{Base.Callable,Integer}},                                      # scalar conditions with ≥ and <, on a set of features specified as a set of callables to be applied to a set of variables each;
        Vector{<:Tuple{TestOperator,<:Union{SoleData.VarFeature,Base.Callable}}},    # explicitly specify the pairs (test operator, feature);
        Vector{<:SoleData.ScalarMetaCondition},                                      # explicitly specify the scalar condition set.
    }

    # Condition set: alternative/complementary way to specify split conditions
    # Provides flexibility in defining how nodes make decisions
    conditions             :: Union{
        Nothing,                                                                    # defaults to scalar conditions (with ≥ and <) on well-known feature functions (e.g., minimum, maximum), applied to all variables;
        Vector{<:Union{SoleData.VarFeature,Base.Callable}},                         # scalar conditions with ≥ and <, on an explicitly specified feature set (callables to be applied to each variable, or VarFeature objects);
        Vector{<:Tuple{Base.Callable,Integer}},                                     # scalar conditions with ≥ and <, on a set of features specified as a set of callables to be applied to a set of variables each;
        Vector{<:Tuple{TestOperator,<:Union{SoleData.VarFeature,Base.Callable}}},   # explicitly specify the pairs (test operator, feature);
        Vector{<:SoleData.ScalarMetaCondition},                                     # explicitly specify the scalar condition set.
    }

    # Type for the extracted feature values (typically Float64 for numerical stability)
    featvaltype            :: Type

    # Initial conditions for the modal logic learning algorithm
    # Defines the starting state or assumptions before tree construction
    initconditions         :: Union{
        Nothing,            # defaults to standard conditions (e.g., start_without_world)
        Symbol,             # one of the initial conditions specified in AVAILABLE_INITIALCONDITIONS;
        InitialCondition,   # explicitly specify an initial condition for the learning algorithm.
    }

    ## Miscellaneous
    # Whether to reduce data dimensionality (can be boolean, tuple of sizes, or custom function)
    downsize               :: Union{Bool,NTuple{N,Integer} where N,Function}

    # Force the use of interval variables in modal logic formulas
    force_i_variables      :: Bool

    # Whether to fix/handle NaN values that appear in callable features
    fixcallablenans        :: Bool

    # Display training progress information during fitting
    print_progress         :: Bool

    # Random number generator for reproducible results
    rng                    :: Union{Random.AbstractRNG,Integer}

    ## DecisionTree.jl compatibility parameters
    # These parameters maintain compatibility with the standard DecisionTree.jl package

    # Maximum depth to display when printing/visualizing the tree
    display_depth          :: Union{Nothing,Int}

    # Minimum number of samples required to attempt splitting an internal node
    min_samples_split      :: Union{Nothing,Int}

    # Number of features to randomly consider at each split
    # Can be integer (exact count), float (proportion), or function
    n_subfeatures          :: Union{Nothing,Int,Float64,Function}

    # Whether to perform post-pruning to reduce overfitting
    post_prune             :: Bool

    # Purity threshold for merging similar nodes during pruning
    merge_purity_threshold :: Union{Nothing,Float64}

    # Method for computing feature importance scores
    # :split counts how many times each feature is used for splitting
    feature_importance     :: Symbol
end

"""
    ModalDecisionTree(; kwargs...)

Keyword constructor for `ModalDecisionTree` with sensible defaults.

# Keyword Arguments

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
- `print_progress = false`: Show training progress
- `rng = Random.GLOBAL_RNG`: Random number generator

## DecisionTree.jl Parameters
- `display_depth = nothing`: Display depth for printing
- `min_samples_split = nothing`: Minimum samples to split
- `n_subfeatures = nothing`: Number of features per split
- `post_prune = false`: Enable post-pruning
- `merge_purity_threshold = nothing`: Purity threshold for merging
- `feature_importance = :split`: Importance calculation method

# Returns
- `ModalDecisionTree`: A new model instance with validated parameters

# Notes
The constructor automatically validates parameters using `MMI.clean!` and
issues warnings if any parameter combinations are problematic.

# Examples
```julia
# Minimal usage
model = ModalDecisionTree()

# With custom parameters
model = ModalDecisionTree(
    max_depth = 8,
    min_samples_leaf = 5,
    relations = :temporal,
    print_progress = true,
    rng = 42
)
```
"""
function ModalDecisionTree(;
    # Pruning conditions with sensible defaults
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
    print_progress = false,
    rng = Random.GLOBAL_RNG,

    # DecisionTree.jl compatibility parameters
    display_depth = nothing,
    min_samples_split = nothing,
    n_subfeatures = nothing,
    post_prune = false,
    merge_purity_threshold = nothing,
    feature_importance = :split,
)
    # Construct the model with all parameters
    model = ModalDecisionTree(
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
