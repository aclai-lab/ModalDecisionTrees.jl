using ModalDecisionTrees
using MLJ
using Test

################################################################################

X, y = @load_iris

# X = DataFrame(X)
# X = scalarlogiset(X)

w = abs.(randn(length(y)))
# w = fill(1, length(y))
# w = rand([1,2], length(y))




model = ModalRandomForest(; ntrees = 50, loss_function=ModalDecisionTrees.RandomLoss(), max_depth = 4, min_purity_increase=nothing, min_samples_leaf=1, rng=10)

@test_broken mach = @time machine(model, X, y, w) |> fit!

myforest = mach.report[:fit].model
@test myforest isa DForest

for t in myforest.trees
    scores = ModalDecisionTrees.apply_proba(t, X, y; anomaly_detection=true, path_length_hlim = 5)
end

model = ModalDecisionTree(; loss_function=ModalDecisionTrees.RandomLoss(), min_samples_leaf=1, min_purity_increase=ModalDecisionTrees.BOTTOM_MIN_PURITY_INCREASE, rng=1)

mach = @time machine(model, X, y, w) |> fit!

model = ModalDecisionTree(; loss_function=ModalDecisionTrees.RandomLoss(), max_depth = 4, min_samples_leaf=1, rng=10)

mach = @time machine(model, X, y, w) |> fit!
mach.report[:fit].model

###
model = ModalDecisionTree(; loss_function=ModalDecisionTrees.RandomLoss(), max_depth = 6, min_samples_leaf=1, rng=1)
mach = @time machine(model, X, y, w) |> fit!
m1 = mach.report[:fit].rawmodel

# model = ModalDecisionTree(; loss_function=ModalDecisionTrees.RandomLoss(), max_depth = 4, min_samples_leaf=1, rng=9)
# mach = @time machine(model, X, y, w) |> fit!
# m2 = mach.report[:fit].rawmodel

# model = ModalDecisionTree(; loss_function=ModalDecisionTrees.RandomLoss(), max_depth = 4, min_samples_leaf=1, rng=8)
# mach = @time machine(model, X, y, w) |> fit!
# m3 = mach.report[:fit].rawmodel

models = [m1]
for i in 2:50
    model = model = ModalDecisionTree(; loss_function=ModalDecisionTrees.RandomLoss(), max_depth = 6, min_samples_leaf=1, rng=i)
    mach = @time machine(model, X, y, w) |> fit!
    m = mach.report[:fit].rawmodel
    push!(models, m)
end

scores = ModalDecisionTrees.apply_proba(models, 
                SoleData.MultiLogiset(SoleData.scalarlogiset(DataFrame(X))), 
                y; 
                anomaly_detection=true, 
                path_length_hlim = 2)

###

sampling_fraction = 0.7

ψ = ceil(sampling_fraction * ninstances(X))

Xnew = (sepal_length = [6.4, 7.2, 7.4],
        sepal_width = [2.8, 3.0, 2.8],
        petal_length = [5.6, 5.8, 6.1],
        petal_width = [2.1, 1.6, 1.9],)

yhat = MLJ.predict(mach, Xnew)
yhat = MLJ.predict_mode(mach, Xnew)

yhat = MLJ.predict_mode(mach, X)
@test MLJ.accuracy(y, yhat) < 0.5
