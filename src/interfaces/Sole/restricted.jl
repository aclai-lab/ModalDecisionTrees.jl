

function translate(
    node::DTInternal{L,D},
    initconditions,
    all_ancestors::Vector{<:DTInternal} = DTInternal[],
    all_ancestor_formulas::Vector = [],
    pos_ancestors::Vector{<:DTInternal} = DTInternal[],
    info = (;),
    shortform::Union{Nothing,MultiFormula} = nothing,
) where {L,D<:RestrictedDecision}
    new_all_ancestors = DTInternal{L,<:RestrictedDecision}[all_ancestors..., node]
    new_pos_ancestors = DTInternal{L,<:RestrictedDecision}[pos_ancestors..., node]
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
