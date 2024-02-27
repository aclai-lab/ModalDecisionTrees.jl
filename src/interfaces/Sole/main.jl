using Revise

using SoleLogics
using SoleModels
using SoleModels: info

using SoleData
using SoleData: ScalarCondition, ScalarMetaCondition
using SoleData: AbstractFeature
using SoleData: relation, feature, test_operator, threshold, inverse_test_operator

using ModalDecisionTrees: DTInternal, DTNode, DTLeaf, NSDTLeaf
using ModalDecisionTrees: isleftchild, isrightchild, isinleftsubtree, isinrightsubtree

using ModalDecisionTrees: left, right

using FunctionWrappers: FunctionWrapper

using Memoization

############################################################################################
# MDTv1 translation
############################################################################################

function build_antecedent(a::MultiFormula, initconditions)
    MultiFormula(Dict([i_mod => anchor(f, initconditions[i_mod]) for (i_mod, f) in modforms(a)]))
end

function translate(model::SoleModels.AbstractModel)
    return model
end

function translate(
    forest::DForest,
    info = (;),
)
    pure_trees = [translate(tree) for tree in trees(forest)]

    info = merge(info, (;
        metrics = metrics(forest),
    ))

    return SoleModels.DecisionForest(pure_trees, info)
end

function translate(
    tree::DTree,
    info = (;),
)
    pure_root = translate(ModalDecisionTrees.root(tree), ModalDecisionTrees.initconditions(tree))

    info = merge(info, SoleModels.info(pure_root))
    info = merge(info, (;))

    return SoleModels.DecisionTree(pure_root, info)
end

function translate(
    tree::DTLeaf,
    initconditions,
    all_ancestors::Vector{<:DTInternal} = DTInternal[],
    all_ancestor_formulas::Vector = [],
    pos_ancestors::Vector{<:DTInternal} = DTInternal[],
    info = (;),
    shortform = nothing,
)
    info = merge(info, (;
        supporting_labels      = ModalDecisionTrees.supp_labels(tree),
        supporting_predictions = ModalDecisionTrees.predictions(tree),
    ))
    if !isnothing(shortform)
        info = merge(info, (;
            shortform = build_antecedent(shortform, initconditions),
        ))
    end
    return SoleModels.ConstantModel(ModalDecisionTrees.prediction(tree), info)
end

function translate(
    tree::NSDTLeaf,
    initconditions,
    all_pos_ancestors::Vector{<:DTInternal} = DTInternal[],
    all_ancestor_formulas::Vector = [],
    ancestors::Vector{<:DTInternal} = DTInternal[],
    info = (;),
    shortform = nothing,
)
    info = merge(info, (;
        supporting_labels      = ModalDecisionTrees.supp_labels(tree),
        supporting_predictions = ModalDecisionTrees.predictions(tree),
    ))
    if !isnothing(shortform)
        info = merge(info, (;
            shortform = build_antecedent(shortform, initconditions),
        ))
    end
    return SoleModels.FunctionModel(ModalDecisionTrees.predicting_function(tree), info)
end

############################################################################################
############################################################################################
############################################################################################

function _condition(feature::AbstractFeature, test_op, threshold::T) where {T}
    # t = FunctionWrapper{Bool,Tuple{U,T}}(test_op)
    metacond = ScalarMetaCondition(feature, test_op)
    cond = ScalarCondition(metacond, threshold)
    return cond
end

function _atom(φ::ScalarCondition)
    test_op = test_operator(φ)
    return Atom(_condition(feature(φ), test_op, threshold(φ)))
end

function _atom_inv(φ::ScalarCondition)
    test_op = inverse_test_operator(test_operator(φ))
    return Atom(_condition(feature(φ), test_op, threshold(φ)))
end

function _atom(p::String)
    return Atom(p)
end

function _atom_inv(p::String)
    # return Atom(startswith(p, "¬") ? p[nextind(p,1):end] : "¬$p")
    # return startswith(p, "¬") ? Atom(p[nextind(p,1):end]) : ¬(Atom(p))
    return ¬(Atom(p))
end

get_atom(φ::Atom) = φ
get_atom_inv(φ::Atom) = ¬(φ)

get_atom(φ::ExistentialTopFormula) = ⊤
get_atom_inv(φ::ExistentialTopFormula) = ⊥
get_diamond_op(φ::ExistentialTopFormula) = DiamondRelationalConnective(relation(φ))
get_box_op(φ::ExistentialTopFormula) = BoxRelationalConnective(relation(φ))

get_atom(φ::ScalarExistentialFormula) = _atom(φ.p)
get_atom_inv(φ::ScalarExistentialFormula) = _atom_inv(φ.p)
get_diamond_op(φ::ScalarExistentialFormula) = DiamondRelationalConnective(relation(φ))
get_box_op(φ::ScalarExistentialFormula) = BoxRelationalConnective(relation(φ))

function is_propositional(node::DTNode)
    f = formula(ModalDecisionTrees.decision(node))
    isprop = (relation(f) == identityrel)
    return isprop
end

function get_lambda(parent::DTNode, child::DTNode)
    f = formula(ModalDecisionTrees.decision(parent))
    isprop = (relation(f) == identityrel)
    if isinleftsubtree(child, parent)
        p = get_atom(f)
        diamond_op = get_diamond_op(f)
        return isprop ? SyntaxTree(p) : diamond_op(p)
    elseif isinrightsubtree(child, parent)
        p_inv = get_atom_inv(f)
        box_op = get_box_op(f)
        return isprop ? SyntaxTree(p_inv) : box_op(p_inv)
    else
        error("Cannot compute pathformula on malformed path: $((child, parent)).")
    end
end

include("complete.jl")
include("restricted.jl")
