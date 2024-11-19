using Test

using AbstractTrees
using SoleLogics
using SoleData
using SoleModels
using ModalDecisionTrees
using ModalDecisionTrees: DTLeaf, DTInternal

reg_leaf, cls_leaf = DTLeaf([1.0,2.0]), DTLeaf([1,2])

decision1 = ScalarExistentialFormula(globalrel, VariableMin(1), >=, 10)
decision2 = ScalarExistentialFormula(IA_A, VariableMin(2), <, 0)
decision3 = ScalarExistentialFormula(IA_L, VariableMin(3), <=, 0)

branch = DTInternal(2, decision1, cls_leaf, cls_leaf)
branch = DTInternal(2, decision2, cls_leaf, branch)
branch = DTInternal(2, decision3, branch, cls_leaf)

@test_nowarn AbstractTrees.print_tree(branch)

branchpure = ModalDecisionTrees.translate(branch, [ModalDecisionTrees.StartWithoutWorld(), ModalDecisionTrees.StartWithoutWorld()])
@test_nowarn AbstractTrees.print_tree(branchpure)

complete = ModalDecisionTrees.restricted2complete(branch)
completepure = ModalDecisionTrees.translate(complete, [ModalDecisionTrees.StartWithoutWorld(), ModalDecisionTrees.StartWithoutWorld()])
branchpure |> printmodel
completepure |> printmodel

printmodel(branchpure; show_shortforms = true)
printmodel(completepure; show_shortforms = true)

branchpure    |> listrules
completepure  |> listrules

branchpure = ModalDecisionTrees.translate(branch, [ModalDecisionTrees.StartWithoutWorld(), ModalDecisionTrees.StartWithoutWorld()]; optimize_shortforms = true)
completepure = ModalDecisionTrees.translate(complete, [ModalDecisionTrees.StartWithoutWorld(), ModalDecisionTrees.StartWithoutWorld()]; optimize_shortforms = true)

printmodel(branchpure; show_shortforms = true)
printmodel(completepure; show_shortforms = true)

printmodel(branchpure)
printmodel(completepure)

branchpure    |> listrules
completepure  |> listrules

using D3Trees

text = ["one\n(second line)", "2", "III", "four"]
style = ["", "fill:red", "r:14px", "opacity:0.7"]
link_style = ["", "stroke:blue", "", "stroke-width:10px"]
tooltip = ["pops", "up", "on", "hover"]
@test_nowarn t = D3Tree(children,
   text=text,
   style=style,
   tooltip=tooltip,
   link_style=link_style,
   title="My Tree",
   init_expand=10,
)
@test_nowarn t = D3Tree(tree)
# inchrome(t)
# inbrowser(t, "firefox")
# inchrome(t)
# inbrowser(t, "firefox")
