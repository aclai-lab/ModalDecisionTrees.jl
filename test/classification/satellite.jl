using MLJ
using ModalDecisionTrees
using MLDatasets # TODO remove?
using DataFrames
using Random

# Regression, spatial problem
using RDatasets
channing = RDatasets.dataset("robustbase", "radarImage")

N = maximum(channing[:,:XCoord])
M = maximum(channing[:,:YCoord])

Xcube = fill(Inf, N, M, 3)

for r in eachrow(channing)
    Xcube[r[:XCoord], r[:YCoord], 1] = r[:Band1]
    Xcube[r[:XCoord], r[:YCoord], 2] = r[:Band2]
    Xcube[r[:XCoord], r[:YCoord], 3] = r[:Band3]
end

samplemap = (x->all(!isinf,x)).(eachslice(Xcube; dims=(1,2)))
_s = size(samplemap)
samplemap[[1,end],:] .= 0
samplemap[:,[1,end]] .= 0

samplemap = cat(moving_average(eachslice(samplemap; dims=1); window_size=2, window_step=1)...; dims=2)'
samplemap = cat(moving_average(eachslice(samplemap; dims=2); window_size=2, window_step=1)...; dims=2)

samplemap = hcat(eachslice(samplemap; dims=2)..., zeros(size(samplemap, 1)))
samplemap = hcat(eachslice(samplemap; dims=1)..., zeros(size(samplemap, 2)))'

samplemap = (samplemap .== 1.0)

@assert _s == size(samplemap)

samples = [begin
    X = Xcube[(idx[1]-1):(idx[1]+1), (idx[2]-1):(idx[2]+1), [1,2]]
    y = Xcube[idx[1], idx[2], 3]
    (X, y)
end for idx in findall(isone, samplemap)]

samples = filter(s->all(!isinf, first(s)), samples)

shuffle!(samples)

X = DataFrame([((x)->x[:,:,1]).(first.(samples)), ((x)->x[:,:,2]).(first.(samples))], :auto)
y = last.(samples)

N = length(y)

mach = machine(ModalDecisionTree(min_samples_leaf=4), X, y)

# Split dataset
p = randperm(Random.MersenneTwister(1), N)
train_idxs, test_idxs = p[1:round(Int, N*.8)], p[round(Int, N*.8)+1:end]

# Fit
MLJ.fit!(mach, rows=train_idxs)

yhat = MLJ.predict(mach, rows=test_idxs)
mae = MLJ.mae(mean.(yhat), y[test_idxs])
mae = MLJ.mae(MLJ.predict_mean(mach, rows=test_idxs), y[test_idxs])
mae = MLJ.mae(MLJ.predict_mean(mach, rows=train_idxs), y[train_idxs])

t = ModalDecisionTree(relations = :RCC5, min_samples_leaf=2)
mach = machine(t, X, y)

MLJ.fit!(mach, rows=train_idxs)

report(mach).printmodel(1000; threshold_digits = 2);

listrules(report(mach).model; use_shortforms=true, use_leftmostlinearform = true)

fs = SoleData.antecedent.(listrules(report(mach).model; use_shortforms=true, use_leftmostlinearform = true))
fsnorm = map(f->normalize(modforms(f)[1]; allow_atom_flipping = true), fs)

# TODO: expand to implicationstate
function knowntoimply(t1::SyntaxTree, t2::SyntaxTree)
    # @show t1
    # @show t2
    _diamg = SoleLogics.DiamondRelationalConnective(globalrel)
    _boxg = SoleLogics.BoxRelationalConnective(globalrel)
    @assert arity(_diamg) == 1
    @assert arity(_boxg) == 1
    if token(t1) == _boxg && token(t2) == _diamg
        knowntoimply(children(t1)[1], children(t2)[1])
    elseif token(t1) == _boxg && token(t2) == _boxg
        knowntoimply(children(t1)[1], children(t2)[1])
    elseif token(t1) == _diamg && token(t2) == _diamg
        knowntoimply(children(t1)[1], children(t2)[1])
    elseif token(t1) isa Atom{<:ScalarCondition} && token(t2) isa Atom{<:ScalarCondition}
        c1 = SoleLogics.value(token(t1))
        c2 = SoleLogics.value(token(t2))
        # if SoleData.metacond(c1) == SoleData.metacond(c2)
        #     @show c1, c2
        #     @show SoleData.test_operator(c1)(SoleData.threshold(c1), SoleData.threshold(c2))
        # end
        (SoleData.metacond(c1) == SoleData.metacond(c2) && SoleData.test_operator(c1)(SoleData.threshold(c1), SoleData.threshold(c2)))
    else
        false
    end
end
function _simplify(φ::SyntaxTree)
    if token(φ) in [CONJUNCTION, DISJUNCTION]
        φ = LeftmostLinearForm(φ)
        chs = SoleLogics.grandchildren(φ)
        for i in length(chs):-1:1
            ch1 = chs[i]
            for ch2 in chs
                if (token(φ) == CONJUNCTION && knowntoimply(ch2, ch1)) ||
                   (token(φ) == DISJUNCTION && knowntoimply(ch1, ch2))
                    deleteat!(chs, i)
                    break
                end
            end
        end
        tree(LeftmostLinearForm(SoleLogics.connective(φ), chs))
    else
        φ
    end
end

_simplify.(fsnorm)

syntaxstring.(_simplify.(fsnorm)) .|> println;

printmodel.(listrules(report(mach).model); show_metrics = true, threshold_digits = 2);
mae = MLJ.mae(MLJ.predict_mean(mach, rows=test_idxs), y[test_idxs])
mae = MLJ.mae(MLJ.predict_mean(mach, rows=train_idxs), y[train_idxs])
