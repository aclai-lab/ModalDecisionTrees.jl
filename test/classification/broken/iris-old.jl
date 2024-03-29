# Classification Test - Iris Data Set
# https://archive.ics.uci.edu/ml/datasets/iris

@testset "iris.jl" begin

features, labels = load_data("iris")
labels = String.(labels)
classes = sort(unique(labels))
n = length(labels)

# train a decision stump (depth=1)
model = build_stump(labels, features)
preds = apply_tree(model, features)
@test MLJ.accuracy(labels, preds) > 0.6
@test depth(model) == 1
probs = apply_tree_proba(model, features, classes)
@test reshape(sum(probs, dims=2), n) ≈ ones(n)

# train full-tree classifier (over-fit)
model = build_tree(labels, features)
preds = apply_tree(model, features)
@test MLJ.accuracy(labels, preds) == 1.0
@test length(model) == 9
@test depth(model) == 5
@test preds isa Vector{String}
print_model(model)
probs = apply_tree_proba(model, features, classes)
@test reshape(sum(probs, dims=2), n) ≈ ones(n)

# prune tree to 8 leaves
pruning_purity = 0.9
pt = prune(model, pruning_purity)
@test length(pt) == 8
preds = apply_tree(pt, features)
@test 0.99 < MLJ.accuracy(labels, preds) < 1.0

# prune tree to 3 leaves
pruning_purity = 0.6
pt = prune(model, pruning_purity)
@test length(pt) == 3
preds = apply_tree(pt, features)
@test 0.95 < MLJ.accuracy(labels, preds) < 1.0
probs = apply_tree_proba(model, features, classes)
@test reshape(sum(probs, dims=2), n) ≈ ones(n)

# prune tree to a stump, 2 leaves
pruning_purity = 0.5
pt = prune(model, pruning_purity)
@test length(pt) == 2
preds = apply_tree(pt, features)
@test 0.66 < MLJ.accuracy(labels, preds) < 1.0


# run n-fold cross validation for pruned tree
println("\n##### nfoldCV Classification Tree #####")
nfolds = 3
accuracy = nfoldCV_tree(labels, features, nfolds)
@test mean(accuracy) > 0.8

# train random forest classifier
ntrees = 10
n_subfeatures = 2
sampling_fraction = 0.5
model = build_forest(labels, features, n_subfeatures, ntrees, sampling_fraction)
preds = apply_forest(model, features)
@test MLJ.accuracy(labels, preds) > 0.95
@test preds isa Vector{String}
probs = apply_forest_proba(model, features, classes)
@test reshape(sum(probs, dims=2), n) ≈ ones(n)

# run n-fold cross validation for forests
println("\n##### nfoldCV Classification Forest #####")
n_subfeatures = 2
ntrees = 10
n_folds = 3
sampling_fraction = 0.5
accuracy = nfoldCV_forest(labels, features, nfolds, n_subfeatures, ntrees, sampling_fraction)
@test mean(accuracy) > 0.9

# train adaptive-boosted decision stumps
n_iterations = 15
model, coeffs = build_adaboost_stumps(labels, features, n_iterations)
preds = apply_adaboost_stumps(model, coeffs, features)
@test MLJ.accuracy(labels, preds) > 0.9
@test preds isa Vector{String}
probs = apply_adaboost_stumps_proba(model, coeffs, features, classes)
@test reshape(sum(probs, dims=2), n) ≈ ones(n)

# run n-fold cross validation for boosted stumps, using 7 iterations and 3 folds
println("\n##### nfoldCV Classification Adaboosted Stumps #####")
n_iterations = 15
nfolds = 3
accuracy = nfoldCV_stumps(labels, features, nfolds, n_iterations)
@test mean(accuracy) > 0.9

end # @testset
