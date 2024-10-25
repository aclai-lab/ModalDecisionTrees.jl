using ModalDecisionTrees
using Documenter

DocMeta.setdocmeta!(ModalDecisionTrees, :DocTestSetup, :(using ModalDecisionTrees); recursive=true)

makedocs(;
    modules=[ModalDecisionTrees, ModalDecisionTrees.MLJInterface, ModalDecisionTrees.experimentals],
    authors="Federico Manzella, Giovanni Pagliarini, Eduard I. Stan",
    repo=Documenter.Remotes.GitHub("aclai-lab", "ModalDecisionTrees.jl"),
    sitename="ModalDecisionTrees.jl",
    format=Documenter.HTML(;
        size_threshold = 4000000,
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://aclai-lab.github.io/ModalDecisionTrees.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
    warnonly = true, # TODO remove?
)

deploydocs(;
    repo = "github.com/aclai-lab/ModalDecisionTrees.jl",
    devbranch = "dev",
    target = "build",
    branch = "gh-pages",
    versions = ["main" => "main", "stable" => "v^", "v#.#", "dev" => "dev"],
)
