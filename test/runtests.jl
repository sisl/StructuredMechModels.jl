using StructuredMechModels
using Test

using BatchedRoutines
using Zygote
using TrajectoryOptimization
using Flux

function ngradient(f, xs::AbstractArray...)
    grads = zero.(xs)
    for (x, Δ) in zip(xs, grads), i in 1:length(x)
        δ = sqrt(eps())
        tmp = x[i]
        x[i] = tmp - δ/2
        y1 = f(xs...)
        x[i] = tmp + δ/2
        y2 = f(xs...)
        x[i] = tmp
        Δ[i] = (y2-y1)/δ
    end
    return grads
end

gradcheck(f, xs...) =
    all(isapprox.(ngradient(f, xs...),
                  gradient(f, xs...), rtol = 1e-5, atol = 1e-5))

gradtest(f, xs::AbstractArray...) = gradcheck((xs...) -> sum(sin.(f(xs...))), xs...)


@testset "StructuredMechModels.jl" begin

    @testset "batchmul" begin
        A = rand(4, 2, 3)
        B = rand(2, 4, 3)
        C = StructuredMechModels.batchmul(A, B)
        C_f = batched_gemm('N', 'N', A, B)
        C_s = StructuredMechModels.batchmul_slow(A, B)
        @test all(isapprox(C, C_s, rtol=1e-5,atol=1e-5))
        @test all(isapprox(C_f, C_s, rtol=1e-5,atol=1e-5))
    end

    @testset "grads" begin
        x = rand(4, 5)
        u = rand(1, 5)
        target = rand(4, 5)
        nn = StructuredMechModels.DeLan(2, 1, [1, 1, 0, 0])
        g = Zygote.gradient(() -> Flux.mse(nn(x, u), target), params(nn))
        @test gradtest(nn, x, u)
    end

    @testset "pytorch" begin
        @testset "$mtype" for mtype in (:Naive, :DeLan, :ControlAffine)
            @testset "$act" for act in ("ReLU", "SoftPlus", "Tanh")
                nn_torch = getfield(StructuredMechModels.Torch, mtype)(2, 1, [1, 1, 0, 0], activation=act)
                nn = getfield(StructuredMechModels, mtype)(2, 1, [1, 1, 0, 0], activation=getfield(StructuredMechModels, Symbol(act))())
                Flux.loadparams!(nn, StructuredMechModels.Torch.params(nn_torch))
                x = rand(4, 8);
                u = rand(1, 8);
                @test all(isapprox.(nn(x, u), nn_torch(x, u)))
            end
        end
    end

    @testset "model" begin
        @testset "$mtype" for mtype in (:Naive, :DeLan, :ControlAffine)
            @testset "$act" for act in ("ReLU", "SoftPlus", "Tanh")
                nn = getfield(StructuredMechModels, mtype)(2, 1, [1, 1, 0, 0], activation=getfield(StructuredMechModels, Symbol(act))())
                model = Model(nn)
                x = rand(4, 8);
                u = rand(1, 8);
                @test all(isapprox.(nn(x, u), model.params.f(x, u)))

                @testset "jac" begin
                    model_d = TrajectoryOptimization.discretize_model(model, :rk4)
                    x = rand(4)
                    u = rand(1)
                    Z = TrajectoryOptimization.PartedMatrix(model_d)
                    @test sum(TrajectoryOptimization.jacobian!(Z, model_d, x, u, 0.1)) != 0.0
                end
            end
        end
    end
end
