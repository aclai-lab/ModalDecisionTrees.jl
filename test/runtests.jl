using Test

# For MLDatasets
ENV["DATADEPS_ALWAYS_ACCEPT"] = true
# Pkg.update()

println("Julia version: ", VERSION)

function run_tests(list)
    println("\n" * ("#"^50))
    for test in list
        println("TEST: $test")
        include(test)
        println("=" ^ 50)
    end
end

test_suites = [
    ("Base", ["base.jl"]),
    ("Classification, modal", [
        "classification/japanesevowels.jl",
        "classification/digits.jl",
        "classification/mnist.jl",
        # "classification/demo-juliacon2022.jl",
    ]),
    ("Classification", [
        "classification/iris.jl",
        "classification/iris-params.jl",
    ]),
    ("Regression", [
        "regression/simple.jl",
        # "regression/ames.jl",
        "regression/digits-regression.jl",
        # "regression/random.jl",
    ]),
    ("Miscellaneous", [
        "multimodal-datasets-multiformulas-construction.jl",
    ]),
    ("Other", [
        "other/parse-and-translate-restricted.jl",
        "other/restricted2complete.jl",
        # "other/translate-complete.jl",
    ]),

    ("Pluto Demo", ["$(dirname(dirname(pathof(ModalDecisionTrees))))/pluto-demo.jl", ]),
]

@testset "ModalDecisionTrees.jl" begin
    for ts in 1:length(test_suites)
        name = test_suites[ts][1]
        list = test_suites[ts][2]
        let
            @testset "$name" begin
                run_tests(list)
            end
        end
    end
end
