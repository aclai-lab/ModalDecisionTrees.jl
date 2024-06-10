__precompile__()

module ModalDecisionTrees

############################################################################################

import Base: show, length

using FunctionWrappers: FunctionWrapper
using Logging: LogLevel, @logmsg
using Printf
using ProgressMeter
using Random
using Reexport
using StatsBase

using SoleBase
using SoleBase: LogOverview, LogDebug, LogDetail
using SoleBase: spawn, nat_sort
using SoleBase: CLabel, RLabel, Label, _CLabel, _Label, get_categorical_form
using SoleBase: bestguess, default_weights, slice_weights

using SoleData
using SoleData: nvariables,
                get_instance,
                slicedataset

using FillArrays

using SoleData: AbstractModalLogiset
import SoleData: feature, test_operator, threshold



import AbstractTrees: print_tree

# Data structures
@reexport using SoleData.DimensionalDatasets
using SoleData: MultiLogiset
using SoleData: Worlds

using SoleData: nfeatures, nrelations,
                            nmodalities, eachmodality, modality,
                            displaystructure,
                            #
                            relations,
                            #
                            MultiLogiset,
                            SupportedLogiset

using SoleData: AbstractWorld, AbstractRelation
using SoleData: AbstractWorlds, Worlds

using SoleData: worldtype

using SoleData: OneWorld

using SoleData: Interval, Interval2D

using SoleData: IARelations, IA2DRelations

using SoleLogics: FullDimensionalFrame
using SoleLogics: normalize

using SoleData: existential_aggregator, universal_aggregator, aggregator_bottom

using SoleModels
import SoleModels: nnodes
import SoleModels: nleaves
import SoleModels: height

############################################################################################

export slicedataset,
       nmodalities, ninstances, nvariables

export DTree,                         # Decision tree
        DForest,                      # Decision forest
        RootLevelNeuroSymbolicHybrid, # Root-level neurosymbolic hybrid model
        #
        nnodes, height, modalheight

############################################################################################

ModalityId = Int

# Utility functions
include("utils.jl")

# Loss functions
include("loss-functions.jl")

# Purity helpers
include("purity.jl")


export RestrictedDecision,
       ScalarExistentialFormula,
       displaydecision

# Definitions for Decision Leaf, Internal, Node, Tree & Forest
include("base.jl")

include("print.jl")

# # Default parameter values
include("default-parameters.jl")

# Metrics for assessing the goodness of a decision leaf/rule
include("leaf-metrics.jl")

# One-step decisions
include("interpret-onestep-decisions.jl")

# Build a decision tree/forest from a dataset
include("build.jl")

# Perform post-hoc manipulation/analysis on a decision tree/forest (e.g., pruning)
include("posthoc.jl")

# Apply decision tree/forest to a dataset
include("apply.jl")

export ModalDecisionTree, ModalRandomForest
export depth, wrapdataset

# Interfaces
include("interfaces/Sole/main.jl")
include("interfaces/MLJ.jl")
include("interfaces/AbstractTrees.jl")

# Experimental features
include("experimentals/main.jl")

# Example datasets
include("other/example-datasets.jl")

end # module
