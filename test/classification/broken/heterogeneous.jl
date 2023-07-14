### Classification - Heterogeneously typed features (ints, floats, bools, strings)

@testset "heterogeneous.jl" begin

m, n = 10^2, 5

tf = [trues(Int(m/2)) falses(Int(m/2))]
inds = Random.randperm(m)
labels = string.(tf[inds])

features = Array{Any}(undef, m, n)
features[:,:] = randn(m, n)
features[:,2] = string.(tf[Random.randperm(m)])
features[:,3] = map(t -> round.(Int, t), features[:,3])
features[:,4] = tf[inds]

model = build_tree(labels, features)
preds = apply_tree(model, features)
@test MLJBase.accuracy(labels, preds) > 0.9

n_subfeatures = 2
ntrees = 3
model = build_forest(labels, features, n_subfeatures, ntrees)
preds = apply_forest(model, features)
@test MLJBase.accuracy(labels, preds) > 0.9

n_subfeatures = 7
model, coeffs = build_adaboost_stumps(labels, features, n_subfeatures)
preds = apply_adaboost_stumps(model, coeffs, features)
@test MLJBase.accuracy(labels, preds) > 0.9

end # @testset
