################################################################################
# Experimental features
################################################################################
module experimentals

using ModalDecisionTrees
using ModalDecisionTrees:
       relation, feature, test_operator, threshold,
       is_propositional_decision,
       is_global_decision

using SoleData
using SoleData.DimensionalDatasets
using SoleLogics


using SoleData: nfeatures, nrelations,
                nmodalities, eachmodality, modality,
                displaystructure,
                #
                relations,
                #
                MultiLogiset,
                SupportedLogiset

using SoleData: scalarlogiset

using SoleData: AbstractWorld, AbstractRelation
using SoleData: AbstractWorlds, Worlds

using SoleLogics: FullDimensionalFrame
using SoleData.DimensionalDatasets
using SoleData: MultiLogiset
using SoleData: Worlds


using SoleData: worldtype

using SoleData: OneWorld

using SoleData: Interval, Interval2D

using SoleData: IARelations

MDT = ModalDecisionTrees
SL  = SoleLogics

include("parse.jl")
include("decisionpath.jl")

end
