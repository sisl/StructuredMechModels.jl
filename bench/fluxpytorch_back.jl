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
