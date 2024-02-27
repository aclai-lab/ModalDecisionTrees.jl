using SoleData: ExistentialTopFormula

function translate(
    node::DTInternal{L,D},
    initconditions,
    all_ancestors::Vector{<:DTInternal} = DTInternal[],
    all_ancestor_formulas::Vector = [],
    pos_ancestors::Vector{<:DTInternal} = DTInternal[],
    info = (;),
    shortform::Union{Nothing,MultiFormula} = nothing,
) where {L,D<:DoubleEdgedDecision}
    new_all_ancestors = DTInternal{L,<:DoubleEdgedDecision}[all_ancestors..., node]
    new_pos_ancestors = DTInternal{L,<:DoubleEdgedDecision}[pos_ancestors..., node]
    φl = pathformula(new_all_ancestors, left(node), false)
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

            my_conjuncts = [begin
                (isinleftsubtree(node, anc) ? φ : SoleLogics.normalize(¬(φ); allow_atom_flipping=true, prefer_implications = true))
            end for (φ, anc) in zip(all_ancestor_formulas, all_ancestors)]

            my_left_conjuncts = [my_conjuncts..., φl]
            my_right_conjuncts = [my_conjuncts..., φr]

            ∧(my_left_conjuncts...), ∧(my_right_conjuncts...)
        end
    end

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

# isback(backnode::DTInternal, back::DTInternal) = (backnode == back(node))

function isimplicative(f::Formula)
    t = tree(f)
    isimpl = (token(t) == (→)) ||
            (SoleLogics.isbox(token(t)) && SoleLogics.isunary(token(t)) && first(children(t)) == (→))
    println(syntaxstring(f), "\t", isimpl)
    return isimpl
end

function pathformula(
    ancestors::Vector{<:DTInternal{L,<:DoubleEdgedDecision}},
    node::DTNode{LL},
    multimodal::Bool,
    args...;
    kwargs...
) where {L,LL}
    φ, isimpl = _pathformula_complete(DTNode{Union{L,LL},<:DoubleEdgedDecision}[ancestors..., node], multimodal, args...; kwargs...)
    println(syntaxstring(φ))
    println()
    println()
    println()
    return φ
end

# TODO @memoize
function _pathformula_complete(
    path::Vector{<:DTNode{L,<:DoubleEdgedDecision}},
    multimodal::Bool,
    # dontincrease::Bool = true,
    # addlast = true,
    # perform_checks = true,
) where {L}

    if !multimodal
        println([ν isa DTLeaf ? ModalDecisionTrees.prediction(ν) : displaydecision(decision(ν)) for ν in path])
        path = filter(ν->((ν isa DTLeaf) || i_modality(ν) == i_modality(last(path[1:end-1]))), path)
        # @assert length(path) > 0
    end

    h = length(path)-1
    # println([displaydecision.(decision.(path[1:end-1]))..., ModalDecisionTrees.prediction(path[end])])
    println([ν isa DTLeaf ? ModalDecisionTrees.prediction(ν) : displaydecision(decision(ν)) for ν in path])
    @show h

    if h == 0
        return (SoleLogics.⊤, false)
        # return error("pathformula cannot accept path of height 0.")
    elseif h == 1
        ν0, ν1 = path
        return (MultiFormula(i_modality(ν0), get_lambda(ν0, ν1)), false)
    else
        ν0, ν1 = path[1], path[2]
        _lambda = get_lambda(ν0, ν1)
        @show syntaxstring(_lambda)
        path1, path2, ctr, ctr_child = begin
            # # if perform_checks
            # contributors = filter(ν->back(ν) == ν1, path)
            # @assert length(contributors) <= 1
            # return length(contributors) == 1 ? first(contributors) : ν1
            i_ctr, ctr = begin
                i_ctr, ctr = nothing, nothing
                i_ctr, ctr = 2, path[2]
                for (i_node, ν) in enumerate(path[1:end-1])
                    if back(ν) == ν1
                        i_ctr, ctr = i_node, ν
                        break
                    end
                end
                i_ctr, ctr
            end
            path1, path2 = path[2:i_ctr], path[i_ctr:end]
            @show i_ctr
            ctr_child = path[i_ctr+1]
            path1, path2, ctr, ctr_child
        end
        agreement = !xor(isleftchild(ν1, ν0), isleftchild(ctr_child, ctr))

        f1, _ = _pathformula_complete(path1, true)
        f2, f2_isimpl = _pathformula_complete(path2, true)

        # DEBUG:
        # λ = MultiFormula(i_modality(ν0), _lambda)
        # f1 = f1 == ⊤ ? MultiFormula(1, f1) : f1
        # f2 = f2 == ⊤ ? MultiFormula(1, f2) : f2
        # @show syntaxstring(λ)
        # @show syntaxstring(f1)
        # @show syntaxstring(f2)
        # END DEBUG

        # f2_isimpl = isimplicative(f2)
        ded = decision(ν0)
        isprop = is_propositional_decision(ded)

        if isprop
            λ = MultiFormula(i_modality(ν0), _lambda)
            if !xor(agreement, !f2_isimpl)
                # return (λ ∧ (f1 ∧ f2), false)
                if f1 == ⊤ && f2 != ⊤           return (λ ∧ f2,        false)
                elseif f1 != ⊤ && f2 == ⊤       return (λ ∧ f1,        false)
                elseif f1 != ⊤ && f2 != ⊤       return (λ ∧ (f1 ∧ f2), false)
                else                            return (λ,             false)
                end
            else
                # return (λ → (f1 → f2), true)
                if f1 == ⊤ && f2 != ⊤           return (λ → f2,        true)
                elseif f1 != ⊤ && f2 == ⊤       return (λ → f1,        true)
                elseif f1 != ⊤ && f2 != ⊤       return (λ → (f1 → f2), true)
                else                            return (λ,             true)
                end
            end
        else
            rel = relation(formula(ded))
            if !xor(agreement, !f2_isimpl)
                ◊ = SoleLogics.diamond(rel)
                # return (◊(f1 ∧ f2), false)
                if f1 == ⊤ && f2 != ⊤           return (◊(f2), false)
                elseif f1 != ⊤ && f2 == ⊤       return (◊(f1), false)
                elseif f1 != ⊤ && f2 != ⊤       return (◊(f1 ∧ f2), false)
                else                            return (◊(⊤), false)
                end
            else
                □ = SoleLogics.box(rel)
                # return (□(f1 → f2), true)
                if f1 == ⊤ && f2 != ⊤           return (□(f2), true)
                elseif f1 != ⊤ && f2 == ⊤       return (□(f1), true)
                elseif f1 != ⊤ && f2 != ⊤       return (□(f1 → f2), true)
                else                            return (⊤, true)
                end
            end
        end
    end
end
