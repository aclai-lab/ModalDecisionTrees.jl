using ModalDecisionTrees
using MLJ
using DataFrames
using SoleModels
using SoleData
using Logging
using SoleLogics
using Random
using Test

N = 5
y = [i <= div(N,2)+1 for i in 1:N]

# Split dataset
p = randperm(Random.MersenneTwister(1), N)
train_idxs, test_idxs = p[1:round(Int, N*.8)], p[round(Int, N*.8)+1:end]


_size = ((x)->(hasmethod(size, (typeof(x),)) ? size(x) : missing))

X_static = DataFrame(
    ID = 1:N,
    a = randn(N),
    b = [-2.0, 1.0, 2.0, missing, 3.0],
    c = [1, 2, 3, 4, 5],
    d = [0, 1, 0, 1, 0],
    e = ['M', 'F', missing, 'M', 'F'],
)
_size.(X_static)

@test_throws AssertionError MLJ.fit!(machine(ModalDecisionTree(;), X_static, y), rows=train_idxs)

X_static = DataFrame(
    ID = 1:N,
    # a = randn(N),
    b = [-2.0, -1.0, 2.0, 2.0, 3.0],
    c = [1, 2, 3, 4, 5],
    d = [0, 1, 0, 1, 0],
)
_size.(X_static)

@test_throws AssertionError MLJ.fit!(machine(ModalDecisionTree(;), X_static, y), rows=train_idxs)

mach = MLJ.fit!(machine(ModalDecisionTree(; min_samples_leaf = 2), Float64.(X_static[:,Not(:ID)]), y), rows=train_idxs)
@test depth(fitted_params(mach).tree) == 1

mach = MLJ.fit!(machine(ModalDecisionTree(; min_purity_increase=-Inf, min_samples_leaf = 1), Float64.(X_static[:,Not(:ID)]), y), rows=train_idxs)
@test depth(fitted_params(mach).tree) > 0

X_multi1 = DataFrame(
    ID = 1:N,
    t1 = [randn(2), randn(2), randn(2), randn(2), randn(2)], # good
    t2 = [randn(2), randn(2), randn(2), randn(2), randn(2)], # good
)
_size.(X_multi1)

MLJ.fit!(machine(ModalDecisionTree(;), X_multi1, y), rows=train_idxs)

X_multi2 = DataFrame(
    ID = 1:N,
    t3 = [randn(2), randn(2), randn(2), randn(2), randn(2)], # good
    twrong1 = [randn(2), randn(2), randn(5), randn(2), randn(4)], # good but actually TODO
)
_size.(X_multi2)

MLJ.fit!(machine(ModalDecisionTree(;), X_multi2, y), rows=train_idxs)

X_images1 = DataFrame(
    ID = 1:N,
    R1 = [randn(2,2), randn(2,2), randn(2,2), randn(2,2), randn(2,2)], # good
)
_size.(X_images1)

MLJ.fit!(machine(ModalDecisionTree(;), X_images1, y), rows=train_idxs)

X_images1 = DataFrame(
    ID = 1:N,
    R1 = [randn(2,3), randn(2,3), randn(2,3), randn(2,3), randn(2,3)], # good
)
_size.(X_images1)

logiset = scalarlogiset(X_images1[:,Not(:ID)]; use_onestep_memoization=true, conditions = [
    ScalarMetaCondition(VariableMax(1), ≥),
    ScalarMetaCondition(VariableMax(1), <),
    ScalarMetaCondition(VariableMin(1), ≥),
    ScalarMetaCondition(VariableMin(1), <),
], relations = [globalrel])
ModalDecisionTrees.build_tree(logiset, y)
ModalDecisionTrees.build_tree(logiset, y;
    max_depth           = nothing,
    min_samples_leaf    = ModalDecisionTrees.BOTTOM_MIN_SAMPLES_LEAF,
    min_purity_increase = ModalDecisionTrees.BOTTOM_MIN_PURITY_INCREASE,
    max_purity_at_leaf  = ModalDecisionTrees.BOTTOM_MAX_PURITY_AT_LEAF,
)
ModalDecisionTrees.build_tree(MultiLogiset(logiset), y)

multilogiset, _ = ModalDecisionTrees.wrapdataset(X_images1[:,Not(:ID)], ModalDecisionTree(; min_samples_leaf = 1))

kwargs = (loss_function = nothing, max_depth = nothing, min_samples_leaf = 1, min_purity_increase = 0.002, max_purity_at_leaf = Inf, max_modal_depth = nothing, n_subrelations = identity, n_subfeatures = identity, initconditions = ModalDecisionTrees.StartAtCenter(), allow_global_splits = true, use_minification = false, perform_consistency_check = false, rng = Random.GLOBAL_RNG, print_progress = false)


ModalDecisionTrees.build_tree(multilogiset, y;
    kwargs...
)
MLJ.fit!(machine(ModalDecisionTree(; min_samples_leaf = 1, relations = (d)->[globalrel]), X_images1[:,Not(:ID)], y), rows=train_idxs, verbosity=2)
MLJ.fit!(machine(ModalDecisionTree(; min_samples_leaf = 1, relations = (d)->[globalrel]), X_images1[:,Not(:ID)], y), verbosity=2)
@test_throws CompositeException MLJ.fit!(machine(ModalDecisionTree(; min_samples_leaf = 1, initconditions = :start_at_center, relations = (d)->SoleLogics.AbstractRelation[]), X_images1[:,Not(:ID)], y), verbosity=2)

X_images1 = DataFrame(
    ID = 1:N,
    R1 = [randn(2,3), randn(2,3), randn(2,3), randn(2,3), randn(2,3)], # good
    G1 = [randn(3,3), randn(3,3), randn(3,3), randn(3,3), randn(3,3)], # good
    B1 = [randn(3,3), randn(3,3), randn(3,3), randn(3,3), randn(3,3)], # good
)
_size.(X_images1)

MLJ.fit!(machine(ModalDecisionTree(; min_samples_leaf = 2), X_images1[:,Not(:ID)], y), rows=train_idxs)

X_images2 = DataFrame(
    ID = 1:N,
    R2 = [ones(5,5),  ones(5,5),  ones(5,5),  zeros(5,5), zeros(5,5)], # good
    G2 = [randn(5,5), randn(5,5), randn(5,5), randn(5,5), randn(5,5)], # good
    B2 = [randn(5,5), randn(5,5), randn(5,5), randn(5,5), randn(5,5)], # good
)
_size.(X_images2)

X_all = innerjoin([Float64.(X_static), X_multi1, X_multi2, X_images1, X_images2]... , on = :ID)[:, Not(:ID)]
_size.(X_all)

MLJ.fit!(machine(ModalDecisionTree(;), X_all, y), rows=train_idxs)

X_all = innerjoin([X_multi1, X_images2]... , on = :ID)[:, Not(:ID)]
mach = MLJ.fit!(machine(ModalDecisionTree(; min_samples_leaf = 1), X_all, y), rows=train_idxs)

multilogiset, var_grouping = ModalDecisionTrees.wrapdataset(X_all, ModalDecisionTree(; min_samples_leaf = 1))


ModalDecisionTrees.build_tree(multilogiset, y;
    kwargs...
)

############################################################################################
############################################################################################
############################################################################################

# Multimodal tree:
X_all = DataFrame(
    mode0 = [1.0, 0.0, 0.0, 0.0, 0.0],
    mode1 = [zeros(5), ones(5), zeros(5), zeros(5), zeros(5)],
    mode2 = [zeros(5,5), zeros(5,5), ones(5,5), zeros(5,5), zeros(5,5)],
)
mach = MLJ.fit!(machine(ModalDecisionTree(; min_samples_leaf = 1), X_all, y), rows=train_idxs)

report(mach).printmodel(1000; threshold_digits = 2);

printmodel.(listrules(report(mach).model; use_shortforms=true, use_leftmostlinearform = true));

printmodel.(joinrules(listrules(report(mach).model; use_shortforms=true, use_leftmostlinearform = true)));

model = ModalDecisionTree(min_purity_increase = 0.001)

@test_logs min_level=Logging.Error machine(model, X_multi1, y) |> fit!
@test_logs min_level=Logging.Error machine(model, X_multi2, y) |> fit!
@test_logs min_level=Logging.Error machine(model, X_images1, y) |> fit!
@test_logs min_level=Logging.Error machine(model, X_images2, y) |> fit!
machine(model, X_all, y) |> fit!
# @test_throws AssertionError machine(model, X_all, y) |> fit!

############################################################################################
############################################################################################
############################################################################################

using MultiData
using SoleData
using ModalDecisionTrees
using MLJ
using DataFrames
using SoleModels
using Random
using Test
using Logging

N = 5

# Multimodal tree:
_X_all = DataFrame(
    mode0 = [1.0, 0.0, 0.0, 0.0, 0.0],
    mode1 = [zeros(5), ones(5), zeros(5), zeros(5), zeros(5)],
    mode2 = [zeros(5,5), zeros(5,5), ones(5,5), zeros(5,5), zeros(5,5)],
)

X_all = MultiDataset(_X_all)
y = [i <= div(N,2)+1 for i in 1:N]

# Split dataset
p = randperm(Random.MersenneTwister(1), N)
train_idxs, test_idxs = p[1:round(Int, N*.8)], p[round(Int, N*.8)+1:end]

@test_logs min_level=Logging.Error wrapdataset(X_all, ModalDecisionTree(;))
@test_logs min_level=Logging.Error mach = MLJ.fit!(machine(ModalDecisionTree(; min_samples_leaf = 1), X_all, y), rows=train_idxs)

# Very multimodal tree:
N = 100

# Multimodal tree:
# X_all = DataFrame(
#     mode0 = [min(rand(), 1/i) for i in 1:N],
#     mode1 = [max.(rand(5), 1/i) for i in 1:N],
#     mode2 = [begin
#         a = zeros(5,5)
#         idx = rand(1:ceil(Int, 4*(i/N)))
#         a[idx:1+idx,2:3] = max.(rand(2, 2), 1/i)
#     end for i in 1:N],
# )
X_all = DataFrame(
    mode0 = [rand() for i in 1:N],
    mode1 = [rand(5) for i in 1:N],
    mode2 = [rand(5,5) for i in 1:N],
)

X_all = MultiDataset(X_all)
y = [i <= div(N,2)+1 for i in 1:N]

# Split dataset
p = randperm(Random.MersenneTwister(1), N)
train_idxs, test_idxs = p[1:round(Int, N*.8)], p[round(Int, N*.8)+1:end]

multilogiset, var_grouping = ModalDecisionTrees.wrapdataset(X_all, ModalDecisionTree(; min_samples_leaf = 1))


mach = MLJ.fit!(machine(ModalDecisionTree(; max_purity_at_leaf = Inf, min_samples_leaf = 1, min_purity_increase = -Inf), X_all, y), rows=train_idxs)

preds = string.(predict_mode(mach, X_all))

report(mach).model
report(mach).solemodel
printmodel(report(mach).solemodel; show_shortforms = true)

longform_ruleset = (listrules(report(mach).model; use_shortforms=false));
shortform_ruleset = (listrules(report(mach).model; use_shortforms=true));

longform_ruleset .|> antecedent .|> x->syntaxstring(x; threshold_digits = 2) .|> println;
shortform_ruleset .|> antecedent .|> x->syntaxstring(x; threshold_digits = 2) .|> println;


as = (longform_ruleset .|> antecedent);
as = as .|> (x->normalize(x; allow_atom_flipping=true, prefer_implications = true))
bs = (shortform_ruleset .|> antecedent);
bs = bs .|> (x->normalize(x; allow_atom_flipping=true, prefer_implications = true))

# (as[2], bs[2]) .|> x->syntaxstring(x; threshold_digits = 2) .|> println
# (as[13], bs[13]) .|> x->syntaxstring(x; threshold_digits = 2) .|> println
# as .|> x->syntaxstring(x; threshold_digits = 2)
# bs .|> x->syntaxstring(x; threshold_digits = 2)

# @test isequal(as, bs)
# @test all(((x,y),)->isequal(x,y), collect(zip((longform_ruleset .|> antecedent .|> x->normalize(x; allow_atom_flipping=true)), (shortform_ruleset .|> antecedent .|> x->normalize(x; allow_atom_flipping=true)))))




# Longform set is mutually exclusive & collectively exhaustive
longform_y_per_rule = [SoleModels.apply(r, multilogiset) for r in longform_ruleset]
m1 = hcat(longform_y_per_rule...)
@test all(r->count(!isnothing, r) >= 1, eachrow(m1));
@test all(r->count(!isnothing, r) < 2, eachrow(m1));
@test all(r->count(!isnothing, r) == 1, eachrow(m1));

# Path formula CORRECTNESS! Very very important!!
map(s->filter(!isnothing, s), eachrow(m1))
longform_y = map(s->filter(!isnothing, s)[1], eachrow(m1))
@test preds == longform_y

# Shortform set is mutually exclusive & collectively exhaustive
shortform_y_per_rule = [SoleModels.apply(r, multilogiset) for r in shortform_ruleset]
m2 = hcat(shortform_y_per_rule...)
@test all(r->count(!isnothing, r) >= 1, eachrow(m2));
@test all(r->count(!isnothing, r) < 2, eachrow(m2));
@test all(r->count(!isnothing, r) == 1, eachrow(m2));

# Path formula CORRECTNESS! Very very important!!
map(s->filter(!isnothing, s), eachrow(m2))
shortform_y = map(s->filter(!isnothing, s)[1], eachrow(m2))
@test shortform_y == preds

# More consistency
_shortform_y_per_rule = [map(r->SoleModels.apply(r, multilogiset, i_instance), shortform_ruleset) for i_instance in 1:ninstances(multilogiset)]
for j in 1:size(m1, 1)
for i in 1:size(m1, 2)
@test m2[j,i] == hcat(_shortform_y_per_rule...)[i,j]
end
end
@test eachcol(hcat(_shortform_y_per_rule...)) == eachrow(hcat(shortform_y_per_rule...))

# More consistency
_longform_y_per_rule = [map(r->SoleModels.apply(r, multilogiset, i_instance), longform_ruleset) for i_instance in 1:ninstances(multilogiset)]
for j in 1:size(m1, 1)
for i in 1:size(m1, 2)
@test m1[j,i] == hcat(_longform_y_per_rule...)[i,j]
end
end
@test eachcol(hcat(_longform_y_per_rule...)) == eachrow(hcat(longform_y_per_rule...))


@test longform_y_per_rule == shortform_y_per_rule
@test _longform_y_per_rule == _shortform_y_per_rule

# filter.(!isnothing, eachrow(hcat(longform_y_per_rule...)))
# # filter.(!isnothing, eachcol(hcat(longform_y_per_rule...)))
# filter.(!isnothing, eachrow(hcat(shortform_y_per_rule...)))
# # filter.(!isnothing, eachcol(hcat(shortform_y_per_rule...)))

printmodel.(listrules(report(mach).model; use_shortforms=true));
printmodel.(listrules(report(mach).model; use_shortforms=true, use_leftmostlinearform = true));
printmodel.(joinrules(longform_ruleset); show_metrics = true);
printmodel.(joinrules(shortform_ruleset); show_metrics = true);

printmodel.(joinrules(listrules(report(mach).model)));


@test_nowarn printmodel.(listrules(report(mach).model; use_shortforms=true, use_leftmostlinearform = true))
@test_nowarn printmodel.(joinrules(listrules(report(mach).model; use_shortforms=true, use_leftmostlinearform = true)))
