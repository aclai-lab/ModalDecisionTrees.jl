# Modal Decision Trees & Forests

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://aclai-lab.github.io/ModalDecisionTrees.jl)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://aclai-lab.github.io/ModalDecisionTrees.jl/dev)
[![Build Status](https://api.cirrus-ci.com/github/aclai-lab/ModalDecisionTrees.jl.svg?branch=main)](https://cirrus-ci.com/github/aclai-lab/ModalDecisionTrees.jl)
[![Coverage](https://codecov.io/gh/aclai-lab/ModalDecisionTrees.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/aclai-lab/ModalDecisionTrees.jl)
<!-- [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle) -->

### Interpretable models for native time-series & image classification!

This package provides algorithms for learning *decision trees* and *decision forests* with enhanced abilities.
Leveraging the express power of Modal Logic, these models can extract *temporal/spatial patterns*, and can natively handle *time series* and *images* (without any data preprocessing). Currently available via [MLJ.jl](https://github.com/alan-turing-institute/MLJ.jl) and [*Sole.jl*](https://github.com/aclai-lab/Sole.jl).
#### Features & differences with [DecisionTree.jl](https://github.com/JuliaAI/DecisionTree.jl):
The MLJ models provided (`ModalDecisionTree` and `ModalRandomForest`) can act as drop in replacements for DecisionTree.jl's tree and forest models. The main difference is that the two models provided are [probabilistic](https://alan-turing-institute.github.io/MLJ.jl/dev/adding_models_for_general_use/#Overview) and can perform both classification (with y labels of type `String` or `CategoricalValue`), and regression (with numeric y labels).
<!-- Also feature_importance = :impurity is not supported -->
Additionally, these models:
- Are able to handle variables that are `AbstractVector{<:Real}` or `AbstractMatrix{<:Real}`;
- Support [multimodal](https://en.wikipedia.org/wiki/Multimodal_learning) learning (e.g., learning from *combinations* of scalars, time series and images);
- A unique algorithm that extends CART and C4.5;
<!-- - Fully optimized implementation (fancy data structures, multithreading, memoization, minification, Pareto-based pruning optimizations, etc); -->
<!-- - TODO -->
<!-- - Four pruning conditions: max_depth, min_samples_leaf, min_purity_increase, max_purity_at_leaf -->
<!-- TODO - Top-down pre-pruning & post-pruning -->
<!-- - Bagging (Random Forests) TODO dillo meglio -->

#### Current limitations (also see [TODOs](#todos)):
- Only supports numeric features;
- Does not support `missing` or `NaN` values.

#### JuliaCon 2022 lightning talk (8 minutes)

<div align="center">
<a href="https://youtu.be/8F1vZsl8Zvg">
  <img src="https://img.youtube.com/vi/8F1vZsl8Zvg/0.jpg">
</a>
</div>


<!-- 
## Installation

Simply type the following commands in Julia's REPL:

```julia
using Pkg; Pkg.add("ModalDecisionTrees");
```
-->

## Installation & Usage

Simply type the following commands in Julia's REPL:

```julia
# Install package
using Pkg; Pkg.add("MLJ");
using Pkg; Pkg.add("ModalDecisionTrees");

# Import packages
using MLJ
using ModalDecisionTrees
using Random

# Load an example dataset (a temporal one)
X, y = ModalDecisionTrees.load_japanesevowels()
N = length(y)

# Instantiate an MLJ machine based on a Modal Decision Tree with ≥ 4 samples at leaf
mach = machine(ModalDecisionTree(min_samples_leaf=4), X, y)

# Split dataset
p = randperm(N)
train_idxs, test_idxs = p[1:round(Int, N*.8)], p[round(Int, N*.8)+1:end]

# Fit
fit!(mach, rows=train_idxs)

# Perform predictions, compute accuracy
yhat = predict_mode(mach, X[test_idxs,:])
accuracy = MLJ.accuracy(yhat, y[test_idxs])

# Print model
report(mach).printmodel(3)

# Access raw model
model = fitted_params(mach).model
```


<!--
# TODO
# Render raw model
Pkg.add("GraphRecipes"); Pkg.add("Plots")

using GraphRecipes
using Plots

#wrapped_model = ModalDecisionTrees.wrap(model.root, (variable_names_map = report(mach).var_grouping,))
# for _method in [:spectral, :sfdp, :circular, :shell, :stress, :spring, :tree, :buchheim, :arcdiagram, :chorddiagram]
wrapped_model = ModalDecisionTrees.wrap(model.root, (; threshold_display_method = x->round(x, digits=2)), use_feature_abbreviations = true)
for _method in [:tree, :buchheim]
	for _nodeshape in [:rect] # , [:rect, :ellipse]
		display(plot(
 		TreePlot(wrapped_model), 
 		method = _method,
 		nodeshape = _nodeshape,
 		# nodesize = (3,10),
 		# root = :left,
 		curves = false,
		fontsize = 10,
		size=(860, 640),
		title = "$(_method)"
		))
	end
end
-->

<!-- TODO (`Y isa Vector{<:{Integer,String}}`) -->

<!--
Detailed usage instructions are available for each model using the doc method. For example:

```julia
using MLJ
doc("DecisionTreeClassifier", pkg="ModalDecisionTrees")
```

Available models are: AdaBoostStumpClassifier, DecisionTreeClassifier, DecisionTreeRegressor, RandomForestClassifier, RandomForestRegressor.


-->
<!-- 
## Visualization

A DecisionTree model can be visualized using the print_tree-function of its native interface (for an example see above in section 'Classification Example'). -->

<!-- ## TODOs

- [x]  Enable loss functions different from Shannon's entropy (*untested*)
- [x]  Enable regression (*untested*)
- [x]  Proper test suite
- [ ]  Visualizations of modal rules/patterns
<!-- - [x]  AbstractTrees interface -->

## Theoretical foundations

Most of the works in *symbolic learning* are based either on Propositional Logics (PLs) or First-order Logics (FOLs); PLs are the simplest kind of logic and can only handle *tabular data*, while FOLs can express complex entity-relation concepts. Machine Learning with FOLs enables handling data with complex topologies, such as time series, images, or videos; however, these logics are computationally challenging. Instead, Modal Logics (e.g. [Interval Logic](https://en.wikipedia.org/wiki/Interval_temporal_logic)) represent a perfect trade-off in terms of computational tractability and expressive power, and naturally lend themselves for expressing some forms of *temporal/spatial reasoning*.

Recently, symbolic learning techniques such as Decision Trees, Random Forests and Rule-Based models have been extended to the use of Modal Logics of time and space. *Modal Decision Trees* and *Modal Random Forests* have been applied to classification tasks, showing statistical performances that are often comparable to those of functional methods (e.g., neural networks), while providing, at the same time, highly-interpretable classification models. Examples of these tasks are COVID-19 diagnosis from cough/breath audio [[1]](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4102488), [[2]](https://drops.dagstuhl.de/opus/volltexte/2021/14783/pdf/LIPIcs-TIME-2021-7.pdf), land cover classification from aereal images [[3]](https://arxiv.org/abs/2109.08325), EEG-related tasks [[4]](https://link.springer.com/chapter/10.1007/978-3-031-06242-1_53), and gas turbine trip prediction.
This technology also offers a natural extension for *multimodal* learning [[5]](http://ceur-ws.org/Vol-2987/paper7.pdf).

## Credits

*ModalDecisionTrees.jl* lives within the [*Sole.jl*](https://github.com/aclai-lab/Sole.jl) framework for *symbolic machine learning*.

The package is developed by the [ACLAI Lab](https://aclai.unife.it/en/) @ University of Ferrara.

Thanks to Ben Sadeghi ([@bensadeghi](https://github.com/bensadeghi/)), author of [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl),
which inspired the construction of this package.
