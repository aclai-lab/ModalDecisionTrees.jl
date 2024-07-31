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

function translate(model::SoleModels.AbstractModel; kwargs...)
    return model
end

# TODO remove
function translate(
    model::Union{DTree,DForest};
    info = (;),
    kwargs...
)
    return translate(model, info; kwargs...)
end

function translate(
    forest::DForest,
    info = (;);
    kwargs...
)
    pure_trees = [translate(tree; kwargs...) for tree in trees(forest)]

    info = merge(info, (;
        metrics = metrics(forest),
    ))

    return SoleModels.DecisionForest(pure_trees, info)
end

function translate(
    tree::DTree,
    info = (;);
    kwargs...
)
    pure_root = translate(ModalDecisionTrees.root(tree), ModalDecisionTrees.initconditions(tree); kwargs...)

    info = merge(info, SoleModels.info(pure_root))
    info = merge(info, (;))

    return SoleModels.DecisionTree(pure_root, info)
end


function translate(
    tree::DTLeaf,
    initconditions,
    args...;
    info = (;),
    shortform = nothing,
    optimize_shortforms = nothing,
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
    args...;
    info = (;),
    shortform = nothing,
    optimize_shortforms = nothing,
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

function translate(
    node::DTInternal{L,D},
    initconditions,
    path::Vector{<:DTInternal} = DTInternal[],
    pos_path::Vector{<:DTInternal} = DTInternal[],
    ancestors::Vector{<:DTInternal} = DTInternal[],
    ancestor_formulas::Vector = [];
    info = (;),
    shortform::Union{Nothing,MultiFormula} = nothing,
    optimize_shortforms::Bool = true
) where {L,D<:AbstractDecision}
    if D<:RestrictedDecision
        forthnode = node
        new_ancestors = DTInternal{L,<:RestrictedDecision}[ancestors..., forthnode]
        new_path = DTInternal{L,<:RestrictedDecision}[path..., forthnode]
        new_pos_path = DTInternal{L,<:RestrictedDecision}[pos_path..., forthnode]
        φl = pathformula(new_pos_path, left(forthnode), false)
    elseif D<:DoubleEdgedDecision
        forthnode = forth(node)
        subtree_nodes = []
        cur_node = node
        while cur_node != forthnode
            push!(subtree_nodes, cur_node)
            @assert isinleftsubtree(forthnode, cur_node) || isinrightsubtree(forthnode, cur_node) "Translation error! Illegal case detected."
            cur_node = isinleftsubtree(forthnode, cur_node) ? left(cur_node) : right(cur_node)
        end
        # @show length(subtree_nodes)
        # @show displaydecision.(decision.(subtree_nodes))
        # println(displaydecision.(decision.(subtree_nodes)))
        push!(subtree_nodes, forthnode)
        new_ancestors = DTInternal{L,<:DoubleEdgedDecision}[ancestors..., forthnode]
        new_path = DTInternal{L,<:DoubleEdgedDecision}[path..., subtree_nodes...]
        new_pos_path = DTInternal{L,<:DoubleEdgedDecision}[pos_path..., subtree_nodes...]
        # DEBUG
        # for (i, (νi, νj)) in enumerate(zip(new_path[2:end], new_path[1:end-1]))
        #     if !(isleftchild(νi, νj) || isrightchild(νi, νj))
        #         error("ERROR")
        #         @show νi
        #         @show νj
        #     end
        # end
        φl = pathformula(new_path, left(forthnode), false)
    else
        error("Unexpected decision type: $(D)")
    end
    φr = SoleLogics.normalize(¬(φl); allow_atom_flipping=true, prefer_implications = true)
    new_ancestor_formulas = [ancestor_formulas..., φl]

    # φr = pathformula(new_pos_path, right(forthnode), true)

    # @show syntaxstring(φl)
    pos_shortform, neg_shortform = begin
        if length(path) == 0
            (
                φl,
                φr,
            )
        else
            # my_conjuncts = [begin
            #     # anc_prefix = new_path[1:nprefix]
            #     # cur_node = new_path[nprefix+1]
            #     anc_prefix = new_path[1:(nprefix+1)]
            #     new_pos_path = similar(anc_prefix, 0)
            #     for i in 1:(length(anc_prefix)-1)
            #         if isinleftsubtree(anc_prefix[i+1], anc_prefix[i])
            #             push!(new_pos_path, anc_prefix[i])
            #         end
            #     end
            #     φ = pathformula(new_pos_path, anc_prefix[end], false)
            #     (isinleftsubtree(node, anc_prefix[end]) ? φ : ¬φ)
            # end for nprefix in 1:(length(new_path)-1)]
            # @assert length(ancestor_formulas) == length(ancestors)
            my_conjuncts = [begin
                (isinleftsubtree(node, anc) ? φ : SoleLogics.normalize(¬(φ); allow_atom_flipping=true, prefer_implications = true))
            end for (φ, anc) in zip(ancestor_formulas, ancestors)]

            my_left_conjuncts = [my_conjuncts..., φl]
            my_right_conjuncts = [my_conjuncts..., φr]

            # println()
            # println()
            # println()
            # @show syntaxstring.(my_left_conjuncts)
            # @show syntaxstring.(my_right_conjuncts)
            # @show syntaxstring(∧(my_left_conjuncts...))
            # @show syntaxstring(∧(my_right_conjuncts...))
            if optimize_shortforms
                # if D <: DoubleEdgedDecision
                #     error("optimize_shortforms is untested with DoubleEdgedDecision's.")
                # end
                # Remove nonmaximal positives (for each modality)
                modalities = unique(i_modality.(new_ancestors))
                my_filtered_left_conjuncts = similar(my_left_conjuncts, 0)
                my_filtered_right_conjuncts = similar(my_right_conjuncts, 0)
                for i_mod in modalities
                    this_mod_mask = map((anc)->i_modality(anc) == i_mod, new_ancestors)
                    # @show this_mod_mask
                    this_mod_ancestors = new_ancestors[this_mod_mask]
                    # @show syntaxstring.(formula.(decision.(this_mod_ancestors)))
                    ispos_ancestors = [isinleftsubtree(ν2, ν1) for (ν2, ν1) in zip(this_mod_ancestors[2:end], this_mod_ancestors[1:end-1])]
                    # @show ispos_ancestors

                    begin
                        this_mod_conjuncts = my_left_conjuncts[this_mod_mask]
                        # ispos = map(anc->isinleftsubtree(left(forthnode), anc), this_mod_ancestors)
                        ispos = [ispos_ancestors..., true]
                        lastpos = findlast(x->x == true, ispos)
                        # @show i_mod, ispos
                        if !isnothing(lastpos)
                            this_mod_conjuncts = [this_mod_conjuncts[lastpos], this_mod_conjuncts[(!).(ispos)]...]
                        end
                        # @show this_mod_conjuncts
                        append!(my_filtered_left_conjuncts, this_mod_conjuncts)
                    end
                    begin
                        this_mod_conjuncts = my_right_conjuncts[this_mod_mask]
                        # ispos = map(anc->isinleftsubtree(right(forthnode), anc), this_mod_ancestors)
                        ispos = [ispos_ancestors..., false]
                        lastpos = findlast(x->x == true, ispos)
                        # @show i_mod, ispos
                        if !isnothing(lastpos)
                            this_mod_conjuncts = [this_mod_conjuncts[lastpos], this_mod_conjuncts[(!).(ispos)]...]
                        end
                        # @show this_mod_conjuncts
                        append!(my_filtered_right_conjuncts, this_mod_conjuncts)
                    end
                end

                # @show syntaxstring(∧(my_filtered_left_conjuncts...))
                # @show syntaxstring(∧(my_filtered_right_conjuncts...))
                ∧(my_filtered_left_conjuncts...), ∧(my_filtered_right_conjuncts...)
            else
                ∧(my_left_conjuncts...), ∧(my_right_conjuncts...)
            end
        end
    end

    # pos_conj = pathformula(new_pos_path[1:end-1], new_pos_path[end], false)
    # @show pos_conj
    # @show syntaxstring(pos_shortform)
    # @show syntaxstring(neg_shortform)

    # # shortforms for my children
    # pos_shortform, neg_shortform = begin
    #     if isnothing(shortform)
    #         φl, φr
    #     else
    #         dl, dr = Dict{Int64,SoleLogics.SyntaxTree}(deepcopy(modforms(shortform))), Dict{Int64,SoleLogics.SyntaxTree}(deepcopy(modforms(shortform)))

    #         dl[i_modality(node)] = modforms(φl)[i_modality(node)]
    #         dr[i_modality(node)] = modforms(φr)[i_modality(node)]
    #         MultiFormula(dl), MultiFormula(dr)
    #     end
    # end

    forthnode_as_a_leaf = ModalDecisionTrees.this(forthnode)
    this_as_a_leaf = translate(forthnode_as_a_leaf, initconditions, new_path, new_pos_path, ancestors, ancestor_formulas; shortform = shortform, optimize_shortforms = optimize_shortforms)

    info = merge(info, (;
        this = this_as_a_leaf,
        # supporting_labels = SoleModels.info(this_as_a_leaf, :supporting_labels),
        supporting_labels = ModalDecisionTrees.supp_labels(forthnode_as_a_leaf),
        # supporting_predictions = SoleModels.info(this_as_a_leaf, :supporting_predictions),
        supporting_predictions = ModalDecisionTrees.predictions(forthnode_as_a_leaf),
    ))

    if !isnothing(shortform)
        # @show syntaxstring(shortform)
        info = merge(info, (;
            shortform = build_antecedent(shortform, initconditions),
        ))
    end

    SoleModels.Branch(
        build_antecedent(φl, initconditions),
        translate(left(forthnode), initconditions, new_path, new_pos_path, new_ancestors, new_ancestor_formulas; shortform = pos_shortform, optimize_shortforms = optimize_shortforms),
        translate(right(forthnode), initconditions, new_path, pos_path, new_ancestors, new_ancestor_formulas; shortform = neg_shortform, optimize_shortforms = optimize_shortforms),
        info
    )
end


# function translate(
#     node::DTInternal{L,D},
#     initconditions,
#     all_ancestors::Vector{<:DTInternal} = DTInternal[],
#     all_ancestor_formulas::Vector = [],
#     pos_ancestors::Vector{<:DTInternal} = DTInternal[];
#     info = (;),
#     shortform::Union{Nothing,MultiFormula} = nothing,
#     optimize_shortforms = false,
# ) where {L,D<:RestrictedDecision}
#     new_all_ancestors = DTInternal{L,<:RestrictedDecision}[all_ancestors..., node]
#     new_pos_ancestors = DTInternal{L,<:RestrictedDecision}[pos_ancestors..., node]
#     φl = pathformula(new_pos_ancestors, left(node), false)
#     φr = SoleLogics.normalize(¬(φl); allow_atom_flipping=true, prefer_implications = true)
#     new_all_ancestor_formulas = [all_ancestor_formulas..., φl]

#     # @show φl, φr

#     # φr = pathformula(new_pos_ancestors, right(node), true)

#     # @show syntaxstring(φl)
#     pos_shortform, neg_shortform = begin
#         if length(all_ancestors) == 0
#             (
#                 φl,
#                 φr,
#             )
#         else
#             # my_conjuncts = [begin
#             #     # anc_prefix = new_all_ancestors[1:nprefix]
#             #     # cur_node = new_all_ancestors[nprefix+1]
#             #     anc_prefix = new_all_ancestors[1:(nprefix+1)]
#             #     new_pos_all_ancestors = similar(anc_prefix, 0)
#             #     for i in 1:(length(anc_prefix)-1)
#             #         if isinleftsubtree(anc_prefix[i+1], anc_prefix[i])
#             #             push!(new_pos_all_ancestors, anc_prefix[i])
#             #         end
#             #     end
#             #     φ = pathformula(new_pos_all_ancestors, anc_prefix[end], false)
#             #     (isinleftsubtree(node, anc_prefix[end]) ? φ : ¬φ)
#             # end for nprefix in 1:(length(new_all_ancestors)-1)]

#             my_conjuncts = [begin
#                 (isinleftsubtree(node, anc) ? φ : SoleLogics.normalize(¬(φ); allow_atom_flipping=true, prefer_implications = true))
#             end for (φ, anc) in zip(all_ancestor_formulas, all_ancestors)]

#             my_left_conjuncts = [my_conjuncts..., φl]
#             my_right_conjuncts = [my_conjuncts..., φr]

#             # Remove nonmaximal positives (for each modality)
#             modalities = unique(i_modality.(new_all_ancestors))
#             my_filtered_left_conjuncts = similar(my_left_conjuncts, 0)
#             my_filtered_right_conjuncts = similar(my_right_conjuncts, 0)
#             for i_mod in modalities
#                 this_mod_mask = map((anc)->i_modality(anc) == i_mod, new_all_ancestors)
#                 this_mod_ancestors = new_all_ancestors[this_mod_mask]

#                 begin
#                     this_mod_conjuncts = my_left_conjuncts[this_mod_mask]
#                     ispos = map(anc->isinleftsubtree(left(node), anc), this_mod_ancestors)
#                     lastpos = findlast(x->x, ispos)
#                     # @show i_mod, ispos
#                     if !isnothing(lastpos)
#                         this_mod_conjuncts = [this_mod_conjuncts[lastpos], this_mod_conjuncts[(!).(ispos)]...]
#                     end
#                     append!(my_filtered_left_conjuncts, this_mod_conjuncts)
#                 end
#                 begin
#                     this_mod_conjuncts = my_right_conjuncts[this_mod_mask]
#                     ispos = map(anc->isinleftsubtree(right(node), anc), this_mod_ancestors)
#                     lastpos = findlast(x->x, ispos)
#                     # @show i_mod, ispos
#                     if !isnothing(lastpos)
#                         this_mod_conjuncts = [this_mod_conjuncts[lastpos], this_mod_conjuncts[(!).(ispos)]...]
#                     end
#                     append!(my_filtered_right_conjuncts, this_mod_conjuncts)
#                 end
#             end

#             ∧(my_filtered_left_conjuncts...), ∧(my_filtered_right_conjuncts...)
#         end
#     end

#     # pos_conj = pathformula(new_pos_ancestors[1:end-1], new_pos_ancestors[end], false)
#     # @show pos_conj
#     # @show syntaxstring(pos_shortform)
#     # @show syntaxstring(neg_shortform)

#     # # shortforms for my children
#     # pos_shortform, neg_shortform = begin
#     #     if isnothing(shortform)
#     #         φl, φr
#     #     else
#     #         dl, dr = Dict{Int64,SoleLogics.SyntaxTree}(deepcopy(modforms(shortform))), Dict{Int64,SoleLogics.SyntaxTree}(deepcopy(modforms(shortform)))

#     #         dl[i_modality(node)] = modforms(φl)[i_modality(node)]
#     #         dr[i_modality(node)] = modforms(φr)[i_modality(node)]
#     #         MultiFormula(dl), MultiFormula(dr)
#     #     end
#     # end

#     info = merge(info, (;
#         this = translate(ModalDecisionTrees.this(node), initconditions, new_all_ancestors, all_ancestor_formulas, new_pos_ancestors; shortform = shortform, optimize_shortforms = optimize_shortforms),
#         supporting_labels = ModalDecisionTrees.supp_labels(node),
#     ))
#     if !isnothing(shortform)
#         # @show syntaxstring(shortform)
#         info = merge(info, (;
#             shortform = build_antecedent(shortform, initconditions),
#         ))
#     end

#     SoleModels.Branch(
#         build_antecedent(φl, initconditions),
#         translate(left(node), initconditions, new_all_ancestors, new_all_ancestor_formulas, new_pos_ancestors; shortform = pos_shortform, optimize_shortforms = optimize_shortforms),
#         translate(right(node), initconditions, new_all_ancestors, new_all_ancestor_formulas, pos_ancestors; shortform = neg_shortform, optimize_shortforms = optimize_shortforms),
#         info
#     )
# end

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

# function is_propositional(node::DTNode)
#     f = formula(ModalDecisionTrees.decision(node))
#     isprop = (relation(f) == identityrel)
#     return isprop
# end

function get_lambda(parent::DTNode, child::DTNode)
    d = ModalDecisionTrees.decision(parent)
    f = formula(d)
    # isprop = (relation(f) == identityrel)
    isprop = is_propositional_decision(d)
    if isinleftsubtree(child, parent)
        p = get_atom(f)
        if isprop
            return SyntaxTree(p)
        else
            diamond_op = get_diamond_op(f)
            return diamond_op(p)
        end
    elseif isinrightsubtree(child, parent)
        p_inv = get_atom_inv(f)
        if isprop
            return SyntaxTree(p_inv)
        else
            box_op = get_box_op(f)
            return box_op(p_inv)
        end
    else
        error("Cannot compute pathformula on malformed path: $((child, parent)).")
    end
end

include("complete.jl")
include("restricted.jl")
