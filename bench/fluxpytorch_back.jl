#!/usr/bin/env julia

using Random
using PyCall
using BenchmarkTools
using Zygote
using Flux
using StructuredMechModels
const SMM = StructuredMechModels


function backward_flux(model, loss, x, u, xp)
    ps = Zygote.Params(Flux.params(model))
    gs = Zygote.gradient(ps) do
        loss(x, u, xp)
    end
end


function backward_pytorch_naive(B)
    dt = 0.05
    nn = SMM.Torch.Naive(2, 1, [1, 1, 0, 0])
    x = SMM.Torch.torch.from_numpy(randn(B, 4))
    u = SMM.Torch.torch.from_numpy(randn(B, 1))
    xp = SMM.Torch.torch.from_numpy(randn(B, 4))
    loss, info = SMM.Torch.structmechmod.trainer.compute_loss(nn.nn, x, u, xp, dt)
    loss.backward()
end


function backward_flux_naive(B)
    dt = 0.05
    nn = SMM.Naive(2, 1, [1, 1, 0, 0])
    function loss(x, u, xp)
        xp_hat = SMM.odestep(SMM.RK4(), nn, x, u, dt)
        SMM.xloss(xp, xp_hat)
    end
    x = randn(4, B)
    u = randn(1, B)
    xp = randn(4, B)
    backward_flux(nn.nn, loss, x, u, xp)
end

function backward_pytorch_ca(B)
    dt = 0.05
    nn = SMM.Torch.ControlAffine(2, 1, [1, 1, 0, 0])
    x = SMM.Torch.torch.from_numpy(randn(B, 4))
    u = SMM.Torch.torch.from_numpy(randn(B, 1))
    xp = SMM.Torch.torch.from_numpy(randn(B, 4))
    loss, info = SMM.Torch.structmechmod.trainer.compute_loss(nn.nn, x, u, xp, dt)
    loss.backward()
end


function backward_flux_ca(B)
    dt = 0.05
    nn = SMM.ControlAffine(2, 1, [1, 1, 0, 0])
    function loss(x, u, xp)
        xp_hat = SMM.odestep(SMM.RK4(), nn, x, u, dt)
        SMM.xloss(xp, xp_hat)
    end
    x = randn(4, B)
    u = randn(1, B)
    xp = randn(4, B)
    backward_flux(nn, loss, x, u, xp)
end


for b in [64, 128, 256, 512]
    @info "PyTorch Naive $b"
    @btime backward_pytorch_naive($b)

    @info "Flux Naive $b"
    @btime backward_flux_naive($b)

    @info "PyTorch CA $b"
    @btime backward_pytorch_ca($b)

    @info "Flux CA $b"
    @btime backward_flux_ca($b)
end

#=
julia> include("bench/fluxpytorch_back.jl")
[ Info: PyTorch Naive 64
5.048 ms (158 allocations: 11.22 KiB)
[ Info: Flux Naive 64
 3.352 ms (3192 allocations: 7.42 MiB)
[ Info: PyTorch CA 64
41.798 ms (254 allocations: 15.45 KiB)
[ Info: Flux CA 64
585.014 ms (2465222 allocations: 139.65 MiB)
[ Info: PyTorch Naive 128
5.939 ms (158 allocations: 15.75 KiB)
[ Info: Flux Naive 128
5.703 ms (4472 allocations: 11.37 MiB)
[ Info: PyTorch CA 128
48.838 ms (254 allocations: 19.98 KiB)
[ Info: Flux CA 128
864.625 ms (4906430 allocations: 290.24 MiB)
[ Info: PyTorch Naive 256
11.171 ms (158 allocations: 24.73 KiB)
[ Info: Flux Naive 256
18.896 ms (7033 allocations: 19.29 MiB)
[ Info: PyTorch CA 256
67.413 ms (254 allocations: 28.97 KiB)
[ Info: Flux CA 256
1.963 s (9787103 allocations: 630.31 MiB)
[ Info: PyTorch Naive 512
13.937 ms (158 allocations: 42.73 KiB)
[ Info: Flux Naive 512
40.104 ms (12171 allocations: 35.13 MiB)
[ Info: PyTorch CA 512
117.407 ms (254 allocations: 46.97 KiB)
[ Info: Flux CA 512
5.268 s (19548297 allocations: 1.43 GiB)
=#
