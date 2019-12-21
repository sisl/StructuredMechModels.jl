using Documenter, StructuredMechModels
using StructuredMechModels.Torch

makedocs(;
    modules=[StructuredMechModels],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
        "API" => "api.md"
    ],
    repo="https://github.com/sisl/StructuredMechModels.jl/blob/{commit}{path}#L{line}",
    sitename="StructuredMechModels.jl",
    authors="rejuvyesh <mail@rejuvyesh.com>",
    assets=String[],
)

deploydocs(;
    repo="github.com/sisl/StructuredMechModels.jl",
)
