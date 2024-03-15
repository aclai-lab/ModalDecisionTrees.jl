


# Compute path formula using semantics from TODO cite
@memoize function pathformula(
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
            isprop = is_propositional_decision(decision(nodes[1]))

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
