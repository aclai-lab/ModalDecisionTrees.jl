
############################################################################################
# Decisions
############################################################################################
using SoleLogics
using SoleLogics: identityrel, globalrel
using SoleData.DimensionalDatasets: alpha
using SoleData: ScalarOneStepFormula,
                ScalarExistentialFormula,
                ExistentialTopFormula,
                ScalarUniversalFormula


struct NoNode end


# looking for the docstring? see displaydecision(node::DTInternal, args...; kwargs...)
function displaydecision(
    i_modality::ModalityId,
    decision::AbstractDecision;
    variable_names_map::Union{
        Nothing,
        AbstractVector{<:AbstractVector},
        AbstractVector{<:AbstractDict}
    } = nothing,
    kwargs...,
)
    _variable_names_map = isnothing(variable_names_map) ? nothing : variable_names_map[i_modality]
    "{$i_modality} $(displaydecision(decision; variable_names_map = _variable_names_map, kwargs...))"
end

# function displaydecision_inverse(decision::AbstractDecision, args...; node = nothing, kwargs...)
#     syntaxstring(dual(decision), args...; node = node, kwargs...)
# end

# function displaydecision_inverse(i_modality::ModalityId, decision::AbstractDecision, args...; node = nothing, kwargs...)
#     displaydecision(i_modality, dual(decision), args...; node = node, kwargs...)
# end


is_propositional_decision(d::Atom) = true
is_global_decision(d::Atom) = false
is_propositional_decision(d::ScalarOneStepFormula) = (SoleData.relation(d) == identityrel)
is_global_decision(d::ScalarOneStepFormula) = (SoleData.relation(d) == globalrel)
is_propositional_decision(d::ExistentialTopFormula) = (SoleData.relation(d) == identityrel)
is_global_decision(d::ExistentialTopFormula) = (SoleData.relation(d) == globalrel)

import SoleData: relation, atom, metacond, feature, test_operator, threshold

struct RestrictedDecision{F<:ScalarExistentialFormula} <: AbstractDecision
    formula  :: F
end

formula(d::RestrictedDecision) = d.formula

relation(d::RestrictedDecision) = relation(formula(d))
atom(d::RestrictedDecision) = atom(formula(d))
metacond(d::RestrictedDecision) = metacond(formula(d))
feature(d::RestrictedDecision) = feature(formula(d))
test_operator(d::RestrictedDecision) = test_operator(formula(d))
threshold(d::RestrictedDecision) = threshold(formula(d))

is_propositional_decision(d::RestrictedDecision) = is_propositional_decision(formula(d))
is_global_decision(d::RestrictedDecision) = is_global_decision(formula(d))

function displaydecision(d::RestrictedDecision; node = NoNode(), displayedges = true, kwargs...)
    outstr = ""
    outstr *= "RestrictedDecision("
    outstr *= syntaxstring(formula(d); kwargs...)
    outstr *= ")"
    outstr
end

function RestrictedDecision(
    d::RestrictedDecision{<:ScalarExistentialFormula},
    threshold_backmap::Function
)
    f = formula(d)
    cond = SoleLogics.value(atom(f))
    newcond = ScalarCondition(metacond(cond), threshold_backmap(threshold(cond)))
    RestrictedDecision(ScalarExistentialFormula(relation(f), newcond))
end

mutable struct DoubleEdgedDecision{F<:Formula} <: AbstractDecision
    formula   :: F
    _back     :: Base.RefValue{N} where N<:AbstractNode # {L,DoubleEdgedDecision}
    _forth    :: Base.RefValue{N} where N<:AbstractNode # {L,DoubleEdgedDecision}

    function DoubleEdgedDecision{F}(formula::F) where {F<:Formula}
        @assert F <: Union{Atom,ExistentialTopFormula} "Cannot instantiate " *
            "DoubleEdgedDecision with formula of type $(F)."
        ded = new{F}()
        ded.formula = formula
        ded
    end

    function DoubleEdgedDecision(formula::F) where {F<:Formula}
        DoubleEdgedDecision{F}(formula)
    end
end

formula(ded::DoubleEdgedDecision) = ded.formula
back(ded::DoubleEdgedDecision) = isdefined(ded, :_back) ? ded._back[] : nothing
forth(ded::DoubleEdgedDecision) = isdefined(ded, :_forth) ? ded._forth[] : nothing
_back(ded::DoubleEdgedDecision) = isdefined(ded, :_back) ? ded._back : nothing
_forth(ded::DoubleEdgedDecision) = isdefined(ded, :_forth) ? ded._forth : nothing

formula!(ded::DoubleEdgedDecision, formula) = (ded.formula = formula)
_back!(ded::DoubleEdgedDecision, _back) = (ded._back = _back)
_forth!(ded::DoubleEdgedDecision, _forth) = (ded._forth = _forth)

# TODO remove?
is_propositional_decision(ded::DoubleEdgedDecision) = is_propositional_decision(formula(ded))
is_global_decision(ded::DoubleEdgedDecision) = is_global_decision(formula(ded))

function displaydecision(ded::DoubleEdgedDecision; node = NoNode(), displayedges = true, kwargs...)
    outstr = ""
    outstr *= "DoubleEdgedDecision("
    outstr *= syntaxstring(formula(ded); kwargs...)
    if displayedges
        # outstr *= ", " * (isnothing(_back(ded)) ? "-" : "$(typeof(_back(ded))){decision = $(displaydecision(_back(ded)[])), height = $(height(_back(ded)[]))}")
        outstr *= ", " * (isnothing(_back(ded)) ? "-" : begin
            νb = _back(ded)[]
            # @show νb
            # @show node
            if νb == node
                "back{loop}"
            else
                if node isa NoNode "?" else "" end *
                    "back{decision = $(displaydecision(decision(νb); node = νb, displayedges = false)), height = $(height(νb))}"
            end
        end)
        # outstr *= ", " * (isnothing(_forth(ded)) ? "-" : "$(typeof(_forth(ded))){decision = $(displaydecision(_forth(ded)[])), height = $(height(_forth(ded)[]))}")
        outstr *= ", " * (isnothing(_forth(ded)) ? "-" : begin
            νf = _forth(ded)[]
            if νf == node
                "forth{loop}"
            else
                if node isa NoNode "?" else "" end *
                    "forth{decision = $(displaydecision(decision(νf); node = νf, displayedges = false)), height = $(height(νf))}"
            end
        end)
    end
    outstr *= ")"
    # outstr *= "DoubleEdgedDecision(\n\t"
    # outstr *= syntaxstring(formula(ded))
    # # outstr *= "\n\tback: " * (isnothing(back(ded)) ? "-" : displaymodel(back(ded), args...; kwargs...))
    # # outstr *= "\n\tforth: " * (isnothing(forth(ded)) ? "-" : displaymodel(forth(ded), args...; kwargs...))
    # outstr *= "\n\tback: " * (isnothing(_back(ded)) ? "-" : "$(typeof(_back(ded)))")
    # outstr *= "\n\tforth: " * (isnothing(_forth(ded)) ? "-" : "$(typeof(_forth(ded)))")
    # outstr *= "\n)"
    outstr
end

function DoubleEdgedDecision(
    d::DoubleEdgedDecision,
    threshold_backmap::Function
)
    return error("TODO implement")
end
