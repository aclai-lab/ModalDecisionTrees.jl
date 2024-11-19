using Test
using SoleData
using SoleLogics
using SoleData: ScalarExistentialFormula
using ModalDecisionTrees
using ModalDecisionTrees: DTLeaf, DTInternal
using ModalDecisionTrees: isbackloop, isforthloop
using ModalDecisionTrees: DoubleEdgedDecision, _back!, _forth!
using ModalDecisionTrees: back, forth

using AbstractTrees

decision1 = ScalarExistentialFormula(globalrel, VariableMin(1), >=, 10)
decision2 = ScalarExistentialFormula(IA_A, VariableMin(2), <, 0)
decision3 = ScalarExistentialFormula(IA_L, VariableMin(3), <=, 0)

ded1 = DoubleEdgedDecision(decision1)
ded2 = DoubleEdgedDecision(decision2)
ded3 = DoubleEdgedDecision(decision3)

reg_leaf, cls_leaf = DTLeaf([1.0,2.0]), DTLeaf([1,2])

branch = DTInternal(2, ded1, cls_leaf, cls_leaf)

@test !isbackloop(branch)
@test !isforthloop(branch)

_back!(ded1, Ref(branch))
_forth!(ded1, Ref(branch))

@test isbackloop(branch)
@test isforthloop(branch)

branch = DTInternal(1, ded2, cls_leaf, branch)
_back!(ded2, Ref(branch))
_forth!(ded2, Ref(branch))
branch = DTInternal(2, ded3, branch, cls_leaf)
_back!(ded3, Ref(branch))
_forth!(ded3, Ref(branch))

@test isbackloop(branch)
@test isforthloop(branch)

AbstractTrees.print_tree(branch)

t = branch
initconditions = [ModalDecisionTrees.StartWithoutWorld(), ModalDecisionTrees.StartWithoutWorld()]
pure_tree = ModalDecisionTrees.translate(t, initconditions)
