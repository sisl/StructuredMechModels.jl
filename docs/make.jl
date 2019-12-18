using Documenter, StructuredMechModels

makedocs(;
    modules=[StructuredMechModels],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/rejuvyesh/StructuredMechModels.jl/blob/{commit}{path}#L{line}",
    sitename="StructuredMechModels.jl",
    authors="rejuvyesh <mail@rejuvyesh.com>",
    assets=String[],
)

deploydocs(;
    repo="github.com/rejuvyesh/StructuredMechModels.jl",
)
