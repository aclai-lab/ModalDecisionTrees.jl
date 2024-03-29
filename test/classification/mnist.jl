using Test
using Logging
using MLJ
using SoleData
using SoleModels
using ModalDecisionTrees
using MLDatasets

DOWNSIZE_WINDOW = (3,3)

Xcube, y = begin
    if MNIST isa Base.Callable # v0.7
        trainset = MNIST(:train)
        trainset[:]
    else # v0.5
        MNIST.traindata()
    end
end

y = string.(y)

N = length(y)

p = 1:100
p_test = 101:1000 # N

begin
    X = SoleData.cube2dataframe(Xcube)

    X_train, y_train = X[p,:], y[p]
    X_test, y_test = X[p_test,:], y[p_test]

    model = ModalDecisionTree()

    mach = @time machine(model, X_train, y_train) |> fit!

    report(mach).printmodel(1000; threshold_digits = 2);


    yhat_test = MLJ.predict_mode(mach, X_test)

    @test MLJ.accuracy(y_test, yhat_test) > 0.2
end


_s = collect(size(Xcube))
insert!(_s, length(_s), 1)
Xcube = reshape(Xcube, _s...)
X = SoleData.cube2dataframe(Xcube, ["black"])

X_train, y_train = X[p,:], y[p]
X_test, y_test = X[p_test,:], y[p_test]

# begin
#     model = ModalDecisionTree()

#     mach = @time machine(model, X_train, y_train) |> fit!

#     report(mach).printmodel(1000; threshold_digits = 2);


#     yhat_test = MLJ.predict_mode(mach, X_test)
#     MLJ.accuracy(y_test, yhat_test)
#     @test MLJ.accuracy(y_test, yhat_test) > 0.2
# end


# begin
#     model = ModalDecisionTree(; relations = :IA7,)

#     mach = @time machine(model, X_train, y_train) |> fit!

#     report(mach).printmodel(1000; threshold_digits = 2);


#     yhat_test = MLJ.predict_mode(mach, X_test)
#     MLJ.accuracy(y_test, yhat_test)
#     @test MLJ.accuracy(y_test, yhat_test) > 0.2
# end

begin
    model = ModalDecisionTree(; downsize = DOWNSIZE_WINDOW)

    mach = @time machine(model, X_train, y_train) |> fit!

    report(mach).printmodel(1000; threshold_digits = 2);

    yhat_test = MLJ.predict_mode(mach, X_test)
    MLJ.accuracy(y_test, yhat_test)
    @test MLJ.accuracy(y_test, yhat_test) > 0.12
end

begin
    model = ModalDecisionTree(; relations = :IA7, downsize = DOWNSIZE_WINDOW)

    mach = @time machine(model, X_train, y_train) |> fit!

    report(mach).printmodel(1000; threshold_digits = 2);


    yhat_test = MLJ.predict_mode(mach, X_test)
    MLJ.accuracy(y_test, yhat_test)
    @test MLJ.accuracy(y_test, yhat_test) > 0.26
end

begin
    recheight(x) = Float32(size(x, 1))
    recwidth(x) = Float32(size(x, 2))
    model = ModalDecisionTree(;
        relations = :IA7,
        features = [minimum, maximum, recheight, recwidth],
        featvaltype = Float32,
        downsize = DOWNSIZE_WINDOW,
    )

    mach1 = @time machine(model, X_train, y_train) |> fit!

    model = ModalDecisionTree(;
        relations = :IA7,
        features = [recheight, recwidth, minimum, maximum],
        featvaltype = Float32,
        downsize = DOWNSIZE_WINDOW,
    )

    mach2 = @time machine(model, X_train, y_train) |> fit!

    report(mach1).printmodel(1000; threshold_digits = 2);
    report(mach2).printmodel(1000; threshold_digits = 2);
    @test_broken displaymodel(fitted_params(mach1).solemodel) == displaymodel(fitted_params(mach2).solemodel)

    yhat_test = MLJ.predict_mode(mach1, X_test)
    MLJ.accuracy(y_test, yhat_test)
    @test MLJ.accuracy(y_test, yhat_test) > 0.25
end

begin
    recheight(x) = Float32(size(x, 1))
    recwidth(x) = Float32(size(x, 2))
    model = ModalDecisionTree(;
        relations = :IA7,
        features = [minimum, maximum, recheight, recwidth],
        initconditions = :start_at_center,
        featvaltype = Float32,
        downsize = DOWNSIZE_WINDOW,
        # features = [minimum, maximum, UnivariateFeature{Float64}(recheight), UnivariateFeature{Float64}(recwidth)],
        # features = [minimum, maximum, UnivariateFeature{Float32}(1, recheight), UnivariateFeature{Float32}(1, recwidth)],
    )

    mach = @time machine(model, X_train, y_train) |> fit!

    report(mach).printmodel(1000; threshold_digits = 2);

    yhat_test = MLJ.predict_mode(mach, X_test)
    MLJ.accuracy(y_test, yhat_test)
    @test MLJ.accuracy(y_test, yhat_test) > 0.25
end

begin
    model = ModalDecisionTree(;
        relations = :IA3,
        features = [minimum],
        initconditions = :start_at_center,
        featvaltype = Float32,
        downsize = DOWNSIZE_WINDOW,
        # features = [minimum, maximum, UnivariateFeature{Float64}(recheight), UnivariateFeature{Float64}(recwidth)],
        # features = [minimum, maximum, UnivariateFeature{Float32}(1, recheight), UnivariateFeature{Float32}(1, recwidth)],
    )

    mach = @time machine(model, X_train, y_train) |> fit!

    report(mach).printmodel(1000; threshold_digits = 2);

    yhat_test = MLJ.predict_mode(mach, X_test)
    MLJ.accuracy(y_test, yhat_test)
    @test MLJ.accuracy(y_test, yhat_test) > 0.25
end


begin
    model = ModalDecisionTree(;
        relations = :IA7,
        features = [minimum],
        initconditions = :start_at_center,
        # downsize = (x)->ModalDecisionTrees.MLJInterface.moving_average(x, (10,10)),
        downsize = (x)->ModalDecisionTrees.MLJInterface.moving_average(x, DOWNSIZE_WINDOW),
    )

    mach = @test_logs min_level=Logging.Error @time machine(model, X_train, y_train) |> fit!

    model = ModalDecisionTree(;
        relations = :IA7,
        features = [minimum],
        initconditions = :start_at_center,
        downsize = DOWNSIZE_WINDOW,
        # downsize = (10,10),
        # features = [minimum, maximum, UnivariateFeature{Float64}(recheight), UnivariateFeature{Float64}(recwidth)],
        # features = [minimum, maximum, UnivariateFeature{Float32}(1, recheight), UnivariateFeature{Float32}(1, recwidth)],
    )

    mach = @test_logs min_level=Logging.Error @time machine(model, X_train, y_train) |> fit!

    report(mach).printmodel(1000; threshold_digits = 2);

    yhat_test = MLJ.predict_mode(mach, X_test)

    @test MLJ.accuracy(y_test, yhat_test) > 0.25

    yhat_test2, tree2 = report(mach).sprinkle(X_test, y_test);

    @test yhat_test2 == yhat_test

    soletree2 = ModalDecisionTrees.translate(tree2)
    @test_nowarn printmodel(soletree2; show_metrics = true);
    @test_nowarn printmodel.(listrules(soletree2); show_metrics = true, threshold_digits = 2);
    @test_nowarn printmodel.(joinrules(listrules(soletree2)); show_metrics = true, threshold_digits = 2);


    SoleModels.info.(listrules(soletree2), :supporting_labels);
    _leaves = consequent.(listrules(soletree2))
    SoleModels.readmetrics.(_leaves)
    zip(SoleModels.readmetrics.(_leaves),_leaves) |> collect |> sort


    # @test MLJ.accuracy(y_test, yhat_test) > 0.4 with DOWNSIZE_WINDOW = (10,10)
    @test MLJ.accuracy(y_test, yhat_test) > 0.27
end
