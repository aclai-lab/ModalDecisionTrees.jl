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

function build_antecedent(a::MultiFormula{F}, initconditions) where {F<:AbstractSyntaxStructure}
    MultiFormula(Dict([i_mod => anchor(f, initconditions[i_mod]) for (i_mod, f) in modforms(a)]))
end

function translate(
    forest::DForest,
    info = (;),
)
    pure_trees = [translate(tree) for tree in trees(forest)]

    info = merge(info, (;
        metrics = metrics(forest),
    ))

    return DecisionForest(pure_trees, info)
end

function translate(
    tree::DTree,
    info = (;),
)
    pure_root = translate(ModalDecisionTrees.root(tree), ModalDecisionTrees.initconditions(tree))

    info = merge(info, SoleModels.info(pure_root))
    info = merge(info, (;))

    return DecisionTree(pure_root, info)
end

function translate(
    node::DTInternal{L,D},
    initconditions,
    all_ancestors::Vector{<:DTInternal} = DTInternal[],
    all_ancestor_formulas::Vector = [],
    pos_ancestors::Vector{<:DTInternal} = DTInternal[],
    info = (;),
    shortform::Union{Nothing,MultiFormula} = nothing,
) where {L,D}
    new_all_ancestors = [all_ancestors..., node]
    new_pos_ancestors = [pos_ancestors..., node]
    φl = pathformula(new_pos_ancestors, left(node), false)
    φr = SoleLogics.normalize(¬(φl); allow_atom_flipping=true, prefer_implications = true)
    new_all_ancestor_formulas = [all_ancestor_formulas..., φl]

    # @show φl, φr

    # φr = pathformula(new_pos_ancestors, right(node), true)

    # @show syntaxstring(φl)
    pos_shortform, neg_shortform = begin
        if length(all_ancestors) == 0
            (
                φl,
                φr,
            )
        else
            # my_conjuncts = [begin
            #     # anc_prefix = new_all_ancestors[1:nprefix]
            #     # cur_node = new_all_ancestors[nprefix+1]
            #     anc_prefix = new_all_ancestors[1:(nprefix+1)]
            #     new_pos_all_ancestors = similar(anc_prefix, 0)
            #     for i in 1:(length(anc_prefix)-1)
            #         if isinleftsubtree(anc_prefix[i+1], anc_prefix[i])
            #             push!(new_pos_all_ancestors, anc_prefix[i])
            #         end
            #     end
            #     φ = pathformula(new_pos_all_ancestors, anc_prefix[end], false)
            #     (isinleftsubtree(node, anc_prefix[end]) ? φ : ¬φ)
            # end for nprefix in 1:(length(new_all_ancestors)-1)]

            my_conjuncts = [begin
                (isinleftsubtree(node, anc) ? φ : SoleLogics.normalize(¬(φ); allow_atom_flipping=true, prefer_implications = true))
            end for (φ, anc) in zip(all_ancestor_formulas, all_ancestors)]

            my_left_conjuncts = [my_conjuncts..., φl]
            my_right_conjuncts = [my_conjuncts..., φr]

            # Remove nonmaximal positives (for each modality)
            modalities = unique(i_modality.(new_all_ancestors))
            my_filtered_left_conjuncts = similar(my_left_conjuncts, 0)
            my_filtered_right_conjuncts = similar(my_right_conjuncts, 0)
            for i_mod in modalities
                this_mod_mask = map((anc)->i_modality(anc) == i_mod, new_all_ancestors)
                this_mod_ancestors = new_all_ancestors[this_mod_mask]

                begin
                    this_mod_conjuncts = my_left_conjuncts[this_mod_mask]
                    ispos = map(anc->isinleftsubtree(left(node), anc), this_mod_ancestors)
                    lastpos = findlast(x->x, ispos)
                    # @show i_mod, ispos
                    if !isnothing(lastpos)
                        this_mod_conjuncts = [this_mod_conjuncts[lastpos], this_mod_conjuncts[(!).(ispos)]...]
                    end
                    append!(my_filtered_left_conjuncts, this_mod_conjuncts)
                end
                begin
                    this_mod_conjuncts = my_right_conjuncts[this_mod_mask]
                    ispos = map(anc->isinleftsubtree(right(node), anc), this_mod_ancestors)
                    lastpos = findlast(x->x, ispos)
                    # @show i_mod, ispos
                    if !isnothing(lastpos)
                        this_mod_conjuncts = [this_mod_conjuncts[lastpos], this_mod_conjuncts[(!).(ispos)]...]
                    end
                    append!(my_filtered_right_conjuncts, this_mod_conjuncts)
                end
            end

            ∧(my_filtered_left_conjuncts...), ∧(my_filtered_right_conjuncts...)
        end
    end

    # pos_conj = pathformula(new_pos_ancestors[1:end-1], new_pos_ancestors[end], false)
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

    info = merge(info, (;
        this = translate(ModalDecisionTrees.this(node), initconditions, new_all_ancestors, all_ancestor_formulas, new_pos_ancestors, (;), shortform),
        supporting_labels = ModalDecisionTrees.supp_labels(node),
    ))
    if !isnothing(shortform)
        # @show syntaxstring(shortform)
        info = merge(info, (;
            shortform = build_antecedent(shortform, initconditions),
        ))
    end

    SoleModels.Branch(
        build_antecedent(φl, initconditions),
        translate(left(node), initconditions, new_all_ancestors, new_all_ancestor_formulas, new_pos_ancestors, (;), pos_shortform),
        translate(right(node), initconditions, new_all_ancestors, new_all_ancestor_formulas, pos_ancestors, (;), neg_shortform),
        info
    )
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

function get_atom(φ::ScalarExistentialFormula)
    test_op = test_operator(φ)
    return Atom(_condition(feature(φ), test_op, threshold(φ)))
end

function get_atom_inv(φ::ScalarExistentialFormula)
    test_op = inverse_test_operator(test_operator(φ))
    return Atom(_condition(feature(φ), test_op, threshold(φ)))
end

function get_diamond_op(φ::ScalarExistentialFormula)
    return DiamondRelationalConnective{typeof(relation(φ))}()
end

function get_box_op(φ::ScalarExistentialFormula)
    return BoxRelationalConnective{typeof(relation(φ))}()
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

############################################################################################
############################################################################################
############################################################################################

# # Compute path formula using semantics from TODO cite
# # @memoize function pathformula(
# function pathformula(
#     nodes::Vector{<:DTNode{L,<:RestrictedDecision{<:ScalarExistentialFormula}}},
#     multi::Bool,
#     dontincrease::Bool = true,
# ) where {L}
#     return pathformula(nodes[1:(end-1)], nodes[end], nodes, multi, dontincrease)
# end

function pathformula(
    pos_ancestors::Vector{<:DTInternal{L,<:RestrictedDecision{<:ScalarExistentialFormula}}},
    node::DTNode{LL},
    multimodal::Bool,
    dontincrease::Bool = true,
    addlast = true,
) where {L,LL}

    if length(pos_ancestors) == 0
        # return error("pathformula cannot accept 0 pos_ancestors. node = $(node).")
        # @show get_lambda(node, left(node))
        return MultiFormula(i_modality(node), get_lambda(node, left(node)))
    else
        # Compute single-modality formula to check.
        if !multimodal
            pos_ancestors = filter(a->i_modality(a) == i_modality(last(pos_ancestors)), pos_ancestors)
            # @assert length(pos_ancestors) > 0
        end

        if length(pos_ancestors) == 1
            # @show prediction(this(node))
            anc = first(pos_ancestors)
            # @show get_lambda(anc, node)
            return MultiFormula(i_modality(anc), get_lambda(anc, node))
        else
            nodes = begin
                if addlast
                    [pos_ancestors..., node]
                else
                    pos_ancestors
                end
            end
            f = formula(ModalDecisionTrees.decision(nodes[1]))
            p = MultiFormula(i_modality(nodes[1]), SyntaxTree(get_atom(f)))
            isprop = (relation(f) == identityrel)

            _dontincrease = isprop
            φ = pathformula(Vector{DTInternal{Union{L,LL},<:RestrictedDecision{<:ScalarExistentialFormula}}}(nodes[2:(end-1)]), nodes[end], multimodal, _dontincrease, addlast)

            # @assert length(unique(anc_mods)) == 1 "At the moment, translate does not work " *
            #     "for MultiFormula formulas $(unique(anc_mods))."
            # @show addlast
            if (addlast && isinleftsubtree(nodes[end], nodes[end-1])) || (!addlast && isinleftsubtree(node, nodes[end])) # Remember: don't use isleftchild, because it fails in the multimodal case.
                # @show "DIAMOND"
                if isprop
                    return dontincrease ? φ : (p ∧ φ)
                else
                    ◊ = get_diamond_op(f)
                    return ◊(p ∧ φ)
                end
            elseif (addlast && isinrightsubtree(nodes[end], nodes[end-1])) || (!addlast && isinrightsubtree(node, nodes[end])) # Remember: don't use isrightchild, because it fails in the multimodal case.
                # @show "BOX"
                if isprop
                    return dontincrease ? φ : (p → φ)
                else
                    □ = get_box_op(f)
                    return □(p → φ)
                end
            else
                error("Cannot compute pathformula on malformed path: $((nodes[end], nodes[end-1])).")
            end
        end
    end
end

# lambda(node::DTInternal) = decision2formula(decision(node))
# lambda_inv(node::DTInternal) = ¬decision2formula(decision(node))

# isback(backnode::DTInternal, back::DTInternal) = (backnode == back(node))
# isleft(leftnode::DTInternal, node::DTInternal) = (leftnode == left(node))
# isright(rightnode::DTInternal, node::DTInternal) = (rightnode == right(node))

# function lambda(node::DTInternal, parent::DTInternal)
#     if isleft(node, parent)
#         lambda(parent)
#     elseif isright(node, parent)
#         lambda_inv(parent)
#     else
#         error("Cannot compute lambda of two nodes that are not parent-child: $(node) and $(parent).")
#     end
# end


# function isimplicative(f::Formula)
#     t = tree(f)
#     return token(t) == → ||
#         (any(isa.(token(t), [BoxRelationalConnective, □])) && first(children(t)) == →)
# end

# function pathformula(ancestors::Vector{<:DTInternal{L,<:DoubleEdgedDecision}}, leaf::DTNode{L}) where {L}
#     nodes = [ancestors..., leaf]
#     depth = length(ancestors)

#     if depth == 0
#         SoleLogics.⊤
#     elseif depth == 1
#         lambda(node, first(ancestor))
#     else
#         _lambda = lambda(first(ancestors), second(ancestors))
#         pi1, pi2, ctr, ctr_child = begin
#         TODO
#         for a in ancestors...
#         isback
#         ctr
#         end
#         agreement = !xor(isleft(second(ancestors), first(ancestors)), isleft(ctr_child, ctr))

#         f1 = pureformula(pi1)
#         f2 = pureformula(pi2)

#         if !(_lambda isa... ExistsTrueDecision)
#             if !xor(agreement, !isimplicative(f2))
#                 _lambda ∧ (f1 ∧ f2)
#             else
#                 _lambda → (f1 → f2)
#             end
#         else
#             relation = relation(_lambda)
#             if !xor(agreement, !isimplicative(f2))
#                 DiamondRelationalConnective(relation)()(f1 ∧ f2)
#             else
#                 BoxRelationalConnective(relation)()(f1 → f2)
#             end
#         end
#     end
# end

############################################################################################
############################################################################################
############################################################################################
