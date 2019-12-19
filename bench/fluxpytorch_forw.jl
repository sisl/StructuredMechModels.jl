#!/usr/bin/env julia

using Random
using PyCall
using BenchmarkTools
using Zygote
using Flux
using StructuredMechModels
const SMM = StructuredMechModels


function forward_pytorch_naive(B)
    dt = 0.05
    nn = SMM.Torch.Naive(2, 1, [1, 1, 0, 0])
    x = SMM.Torch.torch.from_numpy(randn(B, 4))
    u = SMM.Torch.torch.from_numpy(randn(B, 1))
    xp = SMM.Torch.torch.from_numpy(randn(B, 4))
    loss, info = SMM.Torch.structmechmod.trainer.compute_loss(nn.nn, x, u, xp, dt)
end


function forward_flux_naive(B)
    dt = 0.05
    nn = SMM.Naive(2, 1, [1, 1, 0, 0])
    function loss(x, u, xp)
        xp_hat = SMM.odestep(SMM.RK4(), nn, x, u, dt)
        SMM.xloss(xp, xp_hat)
    end
    x = randn(4, B)
    u = randn(1, B)
    xp = randn(4, B)
    loss(x, u, xp)
end

function forward_pytorch_ca(B)
    dt = 0.05
    nn = SMM.Torch.ControlAffine(2, 1, [1, 1, 0, 0])
    x = SMM.Torch.torch.from_numpy(randn(B, 4))
    u = SMM.Torch.torch.from_numpy(randn(B, 1))
    xp = SMM.Torch.torch.from_numpy(randn(B, 4))
    loss, info = SMM.Torch.structmechmod.trainer.compute_loss(nn.nn, x, u, xp, dt)
end


function forward_flux_ca(B)
    dt = 0.05
    nn = SMM.ControlAffine(2, 1, [1, 1, 0, 0])
    function loss(x, u, xp)
        xp_hat = SMM.odestep(SMM.RK4(), nn, x, u, dt)
        SMM.xloss(xp, xp_hat)
    end
    x = randn(4, B)
    u = randn(1, B)
    xp = randn(4, B)
    loss(x, u, xp)
end


for b in [64, 128, 256, 512]
    @info "PyTorch Naive $b"
    @btime forward_pytorch_naive($b)

    @info "Flux Naive $b"
    @btime forward_flux_naive($b)

    @info "PyTorch CA $b"
    @btime forward_pytorch_ca($b)

    @info "Flux CA $b"
    @btime forward_flux_ca($b)
end

#=
julia> include("bench/fluxpytorch_forw.jl")
[ Info: PyTorch Naive 64
3.459 ms (151 allocations: 10.91 KiB)
[ Info: Flux Naive 64
1.989 ms (337 allocations: 2.36 MiB)
[ Info: PyTorch CA 64
12.578 ms (247 allocations: 15.14 KiB)
[ Info: Flux CA 64
5.916 ms (29229 allocations: 12.97 MiB)
[ Info: PyTorch Naive 128
3.603 ms (151 allocations: 15.44 KiB)
[ Info: Flux Naive 128
3.756 ms (337 allocations: 3.93 MiB)
[ Info: PyTorch CA 128
14.566 ms (247 allocations: 19.67 KiB)
[ Info: Flux CA 128
11.491 ms (56513 allocations: 25.67 MiB)
[ Info: PyTorch Naive 256
4.760 ms (151 allocations: 24.42 KiB)
[ Info: Flux Naive 256
8.014 ms (337 allocations: 7.06 MiB)
[ Info: PyTorch CA 256
20.620 ms (247 allocations: 28.66 KiB)
[ Info: Flux CA 256
23.608 ms (110785 allocations: 51.09 MiB)
[ Info: PyTorch Naive 512
5.093 ms (151 allocations: 42.42 KiB)
[ Info: Flux Naive 512
24.017 ms (341 allocations: 13.32 MiB)
[ Info: PyTorch CA 512
26.165 ms (247 allocations: 46.66 KiB)
[ Info: Flux CA 512
67.250 ms (219521 allocations: 101.97 MiB)
=#
