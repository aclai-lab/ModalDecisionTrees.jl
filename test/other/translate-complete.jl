
using SoleLogics
using SoleModels
using AbstractTrees
using ModalDecisionTrees
using ModalDecisionTrees: DTLeaf, DTInternal
using SoleData: ExistentialTopFormula
using ModalDecisionTrees: AbstractNode, DoubleEdgedDecision, DTNode, Label
using ModalDecisionTrees: back!, forth!


diamondtop = ExistentialTopFormula(IA_A)
diamondtop2 = ExistentialTopFormula(IA_L)

function MDT(prediction)
    return DTLeaf(prediction)
end
function MDT(p::Atom, left::AbstractNode, right::AbstractNode)
    decision = DoubleEdgedDecision(ScalarExistentialFormula(identityrel, value(p)))
    return DTInternal(1, decision, left, right)
end

function MDT(φ::ExistentialTopFormula, left::AbstractNode, right::AbstractNode)
    decision = DoubleEdgedDecision(φ)
    return DTInternal(1, decision, left, right)
end

ν3 = MDT(
    Atom("p₂"),
    MDT("L₂"),
    MDT("L₁"),
)

ν1 = MDT(
    diamondtop,
    MDT(
        Atom(" ̅p₁"),
        MDT("L₁"),
        ν3,
    ),
    MDT("L₁"),
)

# back!((ν3), ModalDecisionTrees.right(ν1))
back!(ν3, ν1)

t = MDT(
    diamondtop2,
    ν1,
    MDT("L₂"),
)

printmodel(t)
# treemap(x->(printmodel(x), children(x)), t);

# paths = treemap(x->((x isa DTInternal ? ModalDecisionTrees.decision(x) : ModalDecisionTrees.prediction(x)), children(x)), t)

# collect(Leaves(t))
# collect(PostOrderDFS(t))


# treemap(_t->(_t,children(_t)), t)

using ModalDecisionTrees: translate

t2 = ModalDecisionTrees.left(t)

using Test
initconditions = [ModalDecisionTrees.StartWithoutWorld()]
@test_nowarn translate.(Leaves(t), initconditions)
@test_nowarn translate(ν3, initconditions)
@test_nowarn translate(ν1, initconditions)
@test_nowarn translate(t, initconditions)
@test_nowarn translate(t2, initconditions)
printmodel(translate(t2, initconditions))
printmodel(t)

translate(t2, initconditions)
translate(t, initconditions)
printmodel(t2)
