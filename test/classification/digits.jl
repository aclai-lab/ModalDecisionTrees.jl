using Random
using ModalDecisionTrees
using MLJBase

include("$(dirname(dirname(pathof(ModalDecisionTrees))))/test/data/load.jl")

_X, _y = load_digits()
Xcube = cat(map(r->reshape(r, (8,8,1)), eachrow(_X))...; dims=4)

Xcube_small = Xcube
Xcube_small = mapslices(x->[
    sum(x[1:2, 1:2]) sum(x[1:2, 3:4]) sum(x[1:2, 5:6]) sum(x[1:2, 7:8]);
    sum(x[3:4, 1:2]) sum(x[3:4, 3:4]) sum(x[3:4, 5:6]) sum(x[3:4, 7:8]);
    sum(x[5:6, 1:2]) sum(x[5:6, 3:4]) sum(x[5:6, 5:6]) sum(x[5:6, 7:8]);
    sum(x[7:8, 1:2]) sum(x[7:8, 3:4]) sum(x[7:8, 5:6]) sum(x[7:8, 7:8]);
], Xcube; dims = [1,2])

Xnt = NamedTuple(zip(Symbol.(1:length(eachslice(Xcube_small; dims=3))), eachslice.(eachslice(Xcube_small; dims=3); dims=3)))
X = Xnt
y = string.(_y.-1)

N = length(y)
p = randperm(Random.MersenneTwister(1), N)
train_idxs, test_idxs = p[1:round(Int, N*.1)], p[round(Int, N*.1)+1:end]

############################################################################################

# # Full training TODO too costly
# mach = @time machine(ModalDecisionTree(;), X, y) |> fit!
# @show nnodes(fitted_params(mach).rawmodel) # @test nnodes(fitted_params(mach).rawmodel) == 191
# @show sum(predict_mode(mach, X) .== y) / length(y) # @test sum(predict_mode(mach, X) .== y) / length(y) > 0.92

############################################################################################

mach = @time machine(ModalDecisionTree(;), X, y) |> m->fit!(m, rows = train_idxs)
@test nnodes(fitted_params(mach).rawmodel) == 57
@test sum(predict_mode(mach, rows = test_idxs) .== y[test_idxs]) / length(y[test_idxs]) > 0.43


mach = @time machine(ModalDecisionTree(;
    n_subfeatures       = 0,
    max_depth           = 6,
    min_samples_leaf    = 5,
), X, y) |> m->fit!(m, rows = train_idxs)
@test nnodes(fitted_params(mach).rawmodel) == 45
@test sum(predict_mode(mach, rows = test_idxs) .== y[test_idxs]) / length(y[test_idxs]) > 0.41


mach = machine(ModalRandomForest(;
    n_subfeatures       = 0.7,
    ntrees              = 10,
    sampling_fraction   = 0.7,
    max_depth           = -1,
    min_samples_leaf    = 1,
    min_samples_split   = 2,
    min_purity_increase = 0.0,
    rng = Random.MersenneTwister(1)
), X, y) |> m->fit!(m, rows = train_idxs)
@test nnodes(fitted_params(mach).rawmodel) == 736
@test_nowarn predict_mode(mach, rows = test_idxs)
@test_nowarn MLJ.predict(mach, rows = test_idxs)
@test sum(predict_mode(mach, rows = test_idxs) .== y[test_idxs]) / length(y[test_idxs]) > 0.54


############################################################################################

# NamedTuple dataset
mach = @time machine(ModalDecisionTree(;), Xnt, y) |> m->fit!(m, rows = train_idxs)
@test nnodes(fitted_params(mach).rawmodel) == 57
@test sum(predict_mode(mach, rows = test_idxs) .== y[test_idxs]) / length(y[test_idxs]) > 0.43

mach = @time machine(ModalDecisionTree(;
    relations = :IA7,
    features = [minimum],
    initconditions = :start_at_center,
), Xnt, y) |> m->fit!(m, rows = train_idxs)
@test nnodes(fitted_params(mach).rawmodel) == 43
@test sum(predict_mode(mach, rows = test_idxs) .== y[test_idxs]) / length(y[test_idxs]) > 0.58

mach = @time machine(ModalDecisionTree(;
    relations = :IA7,
    features = [minimum, maximum],
    # initconditions = :start_at_center,
    featvaltype = Float32,
), selectrows(Xnt, train_idxs), selectrows(y, train_idxs)) |> m->fit!(m)
@test nnodes(fitted_params(mach).rawmodel) == 57
@test sum(predict_mode(mach, selectrows(Xnt, test_idxs)) .== y[test_idxs]) / length(y[test_idxs]) > 0.43


############################################################################################
############################################################################################
############################################################################################

mach = @time machine(ModalDecisionTree(;), Xnt, y) |> m->fit!(m, rows = train_idxs)
@test nnodes(fitted_params(mach).rawmodel) == 57
@test sum(predict_mode(mach, rows = test_idxs) .== y[test_idxs]) / length(y[test_idxs]) > 0.43

mach = @time machine(ModalDecisionTree(;
    n_subfeatures       = 0,
    max_depth           = 6,
    min_samples_leaf    = 5,
), Xnt, y) |> m->fit!(m, rows = train_idxs)
@test nnodes(fitted_params(mach).rawmodel) == 45
@test sum(predict_mode(mach, rows = test_idxs) .== y[test_idxs]) / length(y[test_idxs]) > 0.41


@test_throws CompositeException mach = machine(ModalRandomForest(;
    n_subfeatures       = 3,
    ntrees              = 10,
    sampling_fraction   = 0.7,
    max_depth           = -1,
    min_samples_leaf    = 1,
    min_samples_split   = 2,
    min_purity_increase = 0.0,
), Xnt, y) |> m->fit!(m, rows = train_idxs)

mach = machine(ModalRandomForest(;
    n_subfeatures       = 0.2,
    ntrees              = 10,
    sampling_fraction   = 0.7,
    max_depth           = -1,
    min_samples_leaf    = 1,
    min_samples_split   = 2,
    min_purity_increase = 0.0,
), Xnt, y) |> m->fit!(m, rows = train_idxs)
@test nnodes(fitted_params(mach).rawmodel) == 730
@test sum(predict_mode(mach, rows = test_idxs) .== y[test_idxs]) / length(y[test_idxs]) > 0.55

############################################################################################
############################################################################################
############################################################################################

using ImageFiltering
using StatsBase

kernel = [1 0 -1;
          2 0 -2;
          1 0 -1]
im = imfilter(rand(10,10), kernel)
im = imfilter(rand(2,2), kernel)

recvedge(x) = (imfilter(x, [1;; -1]))
rechedge(x) = (imfilter(x, [1;; -1]'))
recvsobel(x) = (imfilter(x, [1 0 -1; 2 0 -2; 1 0 -1]))
rechsobel(x) = (imfilter(x, [1 0 -1; 2 0 -2; 1 0 -1]'))

vedge(x) = StatsBase.mean(recvedge(x)) # prod(size(x)) == 1 ? Inf : StatsBase.mean(recvedge(x))
hedge(x) = StatsBase.mean(rechedge(x)) # prod(size(x)) == 1 ? Inf : StatsBase.mean(rechedge(x))
vsobel(x) = StatsBase.mean(recvsobel(x)) # prod(size(x)) == 1 ? Inf : StatsBase.mean(recvsobel(x))
hsobel(x) = StatsBase.mean(rechsobel(x)) # prod(size(x)) == 1 ? Inf : StatsBase.mean(rechsobel(x))

svedge(x) = StatsBase.sum(recvedge(x)) # prod(size(x)) == 1 ? Inf : StatsBase.sum(recvedge(x))
shedge(x) = StatsBase.sum(rechedge(x)) # prod(size(x)) == 1 ? Inf : StatsBase.sum(rechedge(x))
svsobel(x) = StatsBase.sum(recvsobel(x)) # prod(size(x)) == 1 ? Inf : StatsBase.sum(recvsobel(x))
shsobel(x) = StatsBase.sum(rechsobel(x)) # prod(size(x)) == 1 ? Inf : StatsBase.sum(rechsobel(x))

train_idxs, test_idxs = p[1:round(Int, N*.2)], p[round(Int, N*.2)+1:end]

# train_idxs = train_idxs[1:10]
mach = @time machine(ModalDecisionTree(;
    relations = :IA7,
    features = [hedge, vedge],
    # initconditions = :start_at_center,
    featvaltype = Float32,
), selectrows(Xnt, train_idxs), selectrows(y, train_idxs)) |> m->fit!(m)
@show nnodes(fitted_params(mach).rawmodel) # @test nnodes(fitted_params(mach).rawmodel) == 71
@show sum(predict_mode(mach, selectrows(Xnt, test_idxs)) .== y[test_idxs]) / length(y[test_idxs]) # @test sum(predict_mode(mach, selectrows(Xnt, test_idxs)) .== y[test_idxs]) / length(y[test_idxs]) > 0.73

preds, tree2 = report(mach).sprinkle(selectrows(Xnt, test_idxs), selectrows(y, test_idxs));

@show MLJBase.accuracy(preds, selectrows(y, test_idxs)) # @test MLJBase.accuracy(preds, selectrows(y, test_idxs)) > 0.75

# printmodel.(joinrules(listrules(report(mach).model)); show_metrics = true, threshold_digits = 2);
printmodel.(joinrules(listrules(ModalDecisionTrees.translate(tree2))); show_metrics = true, threshold_digits = 2);
readmetrics.(joinrules(listrules(ModalDecisionTrees.translate(tree2))))

# train_idxs = train_idxs[1:10]
mach = @time machine(ModalDecisionTree(;
    relations = :IA7,
    features = [shedge, svedge],
    # initconditions = :start_at_center,
    featvaltype = Float32,
), selectrows(Xnt, train_idxs), selectrows(y, train_idxs)) |> m->fit!(m)
@show nnodes(fitted_params(mach).rawmodel) # @test nnodes(fitted_params(mach).rawmodel) == 79
@show sum(predict_mode(mach, selectrows(Xnt, test_idxs)) .== y[test_idxs]) / length(y[test_idxs]) # @test sum(predict_mode(mach, selectrows(Xnt, test_idxs)) .== y[test_idxs]) / length(y[test_idxs]) > 0.73

preds, tree2 = report(mach).sprinkle(selectrows(Xnt, test_idxs), selectrows(y, test_idxs));

@show MLJBase.accuracy(preds, selectrows(y, test_idxs)) # @test MLJBase.accuracy(preds, selectrows(y, test_idxs)) > 0.75

# printmodel.(joinrules(listrules(report(mach).model)); show_metrics = true, threshold_digits = 2);
printmodel.(joinrules(listrules(ModalDecisionTrees.translate(tree2))); show_metrics = true, threshold_digits = 2);
readmetrics.(joinrules(listrules(ModalDecisionTrees.translate(tree2))))
