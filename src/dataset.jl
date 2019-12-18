using Random
using MLDataUtils

"""
$(TYPEDEF)
"""
struct Dataset{F,G,H}
    xs::F
    us::G
    xsp::H
end

function rmNaN(data::Dataset)
    goodidx1 = Set([a[2] for a in findall(sum((!isnan).(data.xs), dims=1) .== size(data.xs, 1))])
    goodidx2 = Set([a[2] for a in findall(sum((!isnan).(data.xsp), dims=1) .== size(data.xsp, 1))])
    allgood = collect(intersect(goodidx1, goodidx2))
    newxs = data.xs[:, allgood]
    newus = data.us[:, allgood]
    newxsp = data.xsp[:, allgood]
    @assert !any(isnan, newxs)
    @assert !any(isnan, newus)
    @assert !any(isnan, newxsp)
    newd = Dataset(newxs, newus, newxsp)
    oldlen = length(data)
    newlen = length(newd)
    @info "Previously, $oldlen. Now, $newlen"
end

"""
Adds noise to position and velocity measuresments in a dataset

$(SIGNATURES)
"""
function addnoise(data::Dataset, std::Float64=0.001)
    Dataset(data.xs .+ std .* randn(size(data.xs)), data.us, data.xsp .+ std .* randn(size(data.xsp)))
end

function MLDataUtils.splitobs(data::Dataset, at::AbstractFloat=0.7)
    train, test = MLDataUtils.splitobs((data.xs, data.us, data.xsp))
    return Dataset(train...), Dataset(test...)
end

function Random.shuffle(r::AbstractRNG, d::Dataset)
    newbid = randperm(r, length(d))
    return Dataset(d.xs[:, newbid], d.us[:, newbid], d.xsp[:, newbid])
end

Base.length(d::Dataset) = size(d.xs)[end]

function Base.hcat(ds::Dataset...)
    newxs = reduce(hcat, [d.xs for d in ds])
    newus = reduce(hcat, [d.us for d in ds])
    newxsp = reduce(hcat, [d.xsp for d in ds])
    return Dataset(newxs, newus, newxsp)
end

function MLDataUtils.eachbatch(data::Dataset, size::Int)
    return MLDataUtils.eachbatch((data.xs, data.us, data.xsp), size=size)
end

function MLDataUtils.shuffleobs(data::Dataset)
    return MLDataUtils.shuffleobs((data.xs, data.us, data.xsp))
end
