using Test
using ModalDecisionTrees
using SoleModels
using MLJ

################################################################################

X, y = @load_iris

w = abs.(randn(length(y)))
# w = fill(1, length(y))
# w = rand([1,2], length(y))
model = ModalDecisionTree()

mach = @time machine(model, X, y, w) |> fit!

Xnew = (sepal_length = [6.4, 7.2, 7.4],
        sepal_width = [2.8, 3.0, 2.8],
        petal_length = [5.6, 5.8, 6.1],
        petal_width = [2.1, 1.6, 1.9],)

yhat = MLJ.predict(mach, Xnew)
yhat = MLJ.predict_mode(mach, Xnew)

yhat = MLJ.predict_mode(mach, X)

@test MLJ.accuracy(y, yhat) > 0.8

@test_nowarn fitted_params(mach).rawmodel
@test_nowarn report(mach).model

@test_nowarn printmodel(prune(fitted_params(mach).rawmodel, simplify=true, min_samples_leaf = 20), max_depth = 3)
@test_nowarn printmodel(prune(fitted_params(mach).rawmodel, simplify=true, min_samples_leaf = 20))

@test_nowarn printmodel(report(mach).model, header = false)
@test_nowarn printmodel(report(mach).model, header = :brief)
@test_nowarn printmodel(report(mach).model, header = true)

io = IOBuffer()
@test_nowarn printmodel(io, report(mach).model, show_subtree_info = true)
# String(take!(io))

@test_nowarn printmodel.((SoleModels.listrules(report(mach).model,)));

################################################################################
