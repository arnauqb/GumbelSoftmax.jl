using GumbelSoftmax
using Documenter

DocMeta.setdocmeta!(GumbelSoftmax, :DocTestSetup, :(using GumbelSoftmax); recursive=true)

makedocs(;
    modules=[GumbelSoftmax],
    authors="Arnau Quera-Bofarull",
    repo="https://github.com/arnauqb/GumbelSoftmax.jl/blob/{commit}{path}#{line}",
    sitename="GumbelSoftmax.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://arnauqb.github.io/GumbelSoftmax.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/arnauqb/GumbelSoftmax.jl",
    devbranch="main",
)
