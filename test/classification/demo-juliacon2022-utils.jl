

using Tables
using DataFrames
using StatsBase
import MLJModelInterface: fit

variable_names_latex = [
"\\text{hand tip}_X^L",
"\\text{hand tip}_Y^L",
"\\text{hand tip}_Z^L",
"\\text{hand tip}_X^R",
"\\text{hand tip}_Y^R",
"\\text{hand tip}_Z^R",
"\\text{elbow}_X^L",
"\\text{elbow}_Y^L",
"\\text{elbow}_Z^L",
"\\text{elbow}_X^R",
"\\text{elbow}_Y^R",
"\\text{elbow}_Z^R",
"\\text{wrist}_X^L",
"\\text{wrist}_Y^L",
"\\text{wrist}_Z^L",
"\\text{wrist}_X^R",
"\\text{wrist}_Y^R",
"\\text{wrist}_Z^R",
"\\text{thumb}_X^L",
"\\text{thumb}_Y^L",
"\\text{thumb}_Z^L",
"\\text{thumb}_X^R",
"\\text{thumb}_Y^R",
"\\text{thumb}_Z^R",
]

function show_latex(tree; file_suffix = "", variable_names = nothing, silent = true)
    include("../results/utils/print-tree-to-latex.jl")

    savedir = "latex"


    additional_dict = Dict{String, String}(
        "predictionI have command" => "\\fcolorbox{black}{pastel1}{\\ \\ I have command\\ \\ }",
        "predictionAll clear"      => "\\fcolorbox{black}{pastel2}{\\ \\ All clear\\ \\ }",
        "predictionNot clear"      => "\\fcolorbox{black}{pastel3}{\\ \\ Not clear\\ \\ }",
        "predictionSpread wings"   => "\\fcolorbox{black}{pastel4}{\\ \\ Spread wings\\ \\ }",
        "predictionFold wings"     => "\\fcolorbox{black}{pastel5}{\\ \\ Fold wings\\ \\ }",
        "predictionLock wings"     => "\\fcolorbox{black}{pastel6}{\\ \\ Lock wings\\ \\ }",
    )
    common_kwargs = (
        conversion_dict = additional_dict,
        # threshold_scale_factor = 3,
        threshold_show_decimals = 2,    
        hide_modality_ids = true,
        variable_names_map = variable_names,
        # replace_dict = Dict([
        # #     "\\{1\\}" => "",
        # #     "{1}" => "",
        #     "NN" => "N",
        # ]),
        scale = 1.,
        height = ["25em", "25em", "20em", "22em", "22em", "22em"],
        decisions_at_nodes = false,
        edges_textsize = [:Large, :small, :Large, :small, :small, :small],
        tree_names = ["t_minmax_static", "t_minmax_temporal", "t_neuro_static", "t_minmax_neuro_temporal", "t_minmax_neuro_static", "t_neuro_temporal"],
        latex_preamble = """
    \\definecolor{pastel1}{RGB}{161, 201, 244}
    \\definecolor{pastel2}{RGB}{255, 180, 130}
    \\definecolor{pastel3}{RGB}{141, 229, 161}
    \\definecolor{pastel4}{RGB}{255, 159, 155}
    \\definecolor{pastel5}{RGB}{208, 187, 255}
    \\definecolor{pastel6}{RGB}{222, 187, 155}
    \\definecolor{pastel7}{RGB}{250, 176, 228}
    \\definecolor{pastel8}{RGB}{207, 207, 207}
    \\definecolor{pastel9}{RGB}{255, 254, 163}
    \\definecolor{pastel10}{RGB}{185, 242, 240}
    """,
        # space_unit = (2.2, 3.9)./.2,
        space_unit = (2.2, 3.9)./.75,
        # min_n_inst = 
    )

    main_tex_file = "tree$(file_suffix == "" ? "" : "-$(file_suffix)").tex"
    save_tree_latex(
        [tree],
        savedir;
        main_tex_file = main_tex_file,
        common_kwargs...,
    )
    
    cd(savedir)
    if !silent
        run(`pdflatex $(main_tex_file)`);
    else
        # run(`bash -c "echo 2"`);
        # run(`bash -c "echo 2 2\\\>\\&1 \\\> /dev/null"`);
        run(`bash -c "pdflatex $(main_tex_file)  2\\\>\\&1 \\\> /dev/null"`);
    end
    pdf_name = replace(main_tex_file, ".tex" => ".pdf")
    run(`evince $pdf_name`);
    cd("..")
end

############################################################################################
############################################################################################
############################################################################################

using LinearAlgebra
using StatsBase
using SoleModels: Label, RLabel, CLabel

struct ConfusionMatrix{T<:Number}
    ########################################################################################
    class_names::Vector
    matrix::Matrix{T}
    ########################################################################################
    overall_accuracy::Float64
    kappa::Float64
    mean_accuracy::Float64
    accuracies::Vector{Float64}
    F1s::Vector{Float64}
    sensitivities::Vector{Float64}
    specificities::Vector{Float64}
    PPVs::Vector{Float64}
    NPVs::Vector{Float64}
    ########################################################################################

    function ConfusionMatrix(matrix::AbstractMatrix)
        ConfusionMatrix(Symbol.(1:size(matrix, 1)), matrix)
    end
    function ConfusionMatrix(
        class_names::Vector,
        matrix::AbstractMatrix{T},
    ) where {T<:Number}

        @assert size(matrix,1) == size(matrix,2) "Cannot instantiate ConfusionMatrix with matrix of size ($(size(matrix))"
        n_classes = size(matrix,1)
        @assert length(class_names) == n_classes "Cannot instantiate ConfusionMatrix with mismatching n_classes ($(n_classes)) and class_names $(class_names)"

        ALL = sum(matrix)
        TR = LinearAlgebra.tr(matrix)
        F = ALL-TR

        overall_accuracy = TR / ALL
        prob_chance = (sum(matrix,dims=1) * sum(matrix,dims=2))[1] / ALL^2
        kappa = (overall_accuracy - prob_chance) / (1.0 - prob_chance)

        ####################################################################################
        TPs = Vector{Float64}(undef, n_classes)
        TNs = Vector{Float64}(undef, n_classes)
        FPs = Vector{Float64}(undef, n_classes)
        FNs = Vector{Float64}(undef, n_classes)

        for i in 1:n_classes
            class = i
            other_classes = [(1:i-1)..., (i+1:n_classes)...]
            TPs[i] = sum(matrix[class,class])
            TNs[i] = sum(matrix[other_classes,other_classes])
            FNs[i] = sum(matrix[class,other_classes])
            FPs[i] = sum(matrix[other_classes,class])
        end
        ####################################################################################

        # https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification
        accuracies = (TPs .+ TNs)./ALL
        mean_accuracy = StatsBase.mean(accuracies)

        # https://en.wikipedia.org/wiki/F-score
        F1s           = TPs./(TPs.+.5*(FPs.+FNs))

        # https://en.wikipedia.org/wiki/Sensitivity_and_specificity
        sensitivities = TPs./(TPs.+FNs)
        specificities = TNs./(TNs.+FPs)
        PPVs          = TPs./(TPs.+FPs)
        NPVs          = TNs./(TNs.+FNs)

        new{T}(class_names,
            matrix,
            overall_accuracy,
            kappa,
            mean_accuracy,
            accuracies,
            F1s,
            sensitivities,
            specificities,
            PPVs,
            NPVs,
        )
    end

    function ConfusionMatrix(
        actual::AbstractVector{L},
        predicted::AbstractVector{L},
        weights::Union{Nothing,AbstractVector{Z}} = nothing;
        force_class_order = nothing,
    ) where {L<:CLabel,Z}
        @assert length(actual) == length(predicted) "Cannot compute ConfusionMatrix with mismatching number of actual $(length(actual)) and predicted $(length(predicted)) labels."

        if isnothing(weights)
            weights = default_weights(actual)
        end
        @assert length(actual) == length(weights)   "Cannot compute ConfusionMatrix with mismatching number of actual $(length(actual)) and weights $(length(weights)) labels."

        class_labels = begin
            class_labels = unique([actual; predicted])
            if isnothing(force_class_order)
                class_labels = sort(class_labels, lt=SoleBase.nat_sort)
            else
                @assert length(setdiff(force_class_order, class_labels)) == 0
                class_labels = force_class_order
            end
            # Binary case: retain order of classes YES/NO
            if length(class_labels) == 2 &&
                    startswith(class_labels[1], "YES") &&
                    startswith(class_labels[2], "NO")
                class_labels = reverse(class_labels)
            end
            class_labels
        end

        _ninstances = length(actual)
        _actual    = zeros(Int, _ninstances)
        _predicted = zeros(Int, _ninstances)

        n_classes = length(class_labels)
        for i in 1:n_classes
            _actual[actual .== class_labels[i]] .= i
            _predicted[predicted .== class_labels[i]] .= i
        end

        matrix = zeros(eltype(weights),n_classes,n_classes)
        for (act,pred,w) in zip(_actual, _predicted, weights)
            matrix[act,pred] += w
        end
        ConfusionMatrix(class_labels, matrix)
    end
end

overall_accuracy(cm::ConfusionMatrix) = cm.overall_accuracy
kappa(cm::ConfusionMatrix)            = cm.kappa

class_counts(cm::ConfusionMatrix) = sum(cm.matrix,dims=2)

function Base.show(io::IO, cm::ConfusionMatrix)

    max_num_digits = maximum(length(string(val)) for val in cm.matrix)

    println(io, "Confusion Matrix ($(length(cm.class_names)) classes):")
    for (i,(row,class_name,sensitivity)) in enumerate(zip(eachrow(cm.matrix),cm.class_names,cm.sensitivities))
        for val in row
            print(io, lpad(val,max_num_digits+1," "))
        end
        println(io, "\t\t\t$(round(100*sensitivity, digits=2))%\t\t$(class_name)")
    end

    ############################################################################
    println(io, "accuracy =\t\t$(round(overall_accuracy(cm), digits=4))")
    println(io, "κ =\t\t\t$(round(cm.kappa, digits=4))")
    ############################################################################
    println(io, "sensitivities:\t\t$(round.(cm.sensitivities, digits=4))")
    println(io, "specificities:\t\t$(round.(cm.specificities, digits=4))")
    println(io, "PPVs:\t\t\t$(round.(cm.PPVs, digits=4))")
    println(io, "NPVs:\t\t\t$(round.(cm.NPVs, digits=4))")
    print(io,   "F1s:\t\t\t$(round.(cm.F1s, digits=4))")
    println(io, "\tmean_F1:\t$(round(cm.mean_accuracy, digits=4))")
    print(io,   "accuracies:\t\t$(round.(cm.accuracies, digits=4))")
    println(io, "\tmean_accuracy:\t$(round(cm.mean_accuracy, digits=4))")
end


############################################################################################
############################################################################################
############################################################################################
