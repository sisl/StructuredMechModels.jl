using TrajectoryOptimization
using MLDataUtils
using LinearAlgebra
using Statistics
using JLD2
using TensorBoardLogger
using Juno
using Logging
using LoggingExtras
using Random
using Zygote


mutable struct StepupwithMaximaScheduler
    baseval
    maxval
    stepsize
    gamma
    last_epoch
end
StepupwithMaximaScheduler() = StepupwithMaximaScheduler(10, 100, 1, 2, -1)

function step!(s::StepupwithMaximaScheduler; epoch::Int=-1)
    if epoch < 0
        epoch = s.last_epoch + 1
    end
    s.last_epoch = s
    return value(s)
end
function value(s::StepupwithMaximaScheduler)
    return min(s.maxval, s.baseval * s.gamma^round(Int, s.last_epoch / s.stepsize))
end

ignore(f) = f()
Zygote.@adjoint ignore(f) = f(), _ -> nothing

flat(xs) = reduce(vcat, vec.(xs))

function _train!(model, opt, data, loss)
    train_t = @elapsed begin
        ps = Zygote.Params(Flux.params(model))
        ignore() do
            epoch_train_losses = 0.0
            epoch_grad_norms = 0.0
        end
        @progress for d in data
            try
                gs = Zygote.gradient(ps) do
                    tl = loss(d...)
                    ignore() do
                        epoch_train_losses += tl
                    end
                    return tl
                end
                
                ignore() do
                    epoch_grad_norms += norm(flat(values(gs.grads)))
                    count += 1
                end
                Flux.Optimise.update!(opt, ps, gs)
            catch ex
                if ex isa Flux.Optimise.StopException
                    break
                else
                    rethrow(ex)
                end
            end
        end
    end
    ignore() do
        epoch_train_losses = epoch_train_losses / count
        epoch_grad_norms = epoch_grad_norms / count
    end
    return (t=train_t, loss=epoch_train_losses, gnorm=epoch_grad_norms)
end

function xloss(x, x̂)
    l2 = sum((x̂ - x).^2, dims=1)
    l2_mean = sum(l2) * 1 // length(l2)
    return l2_mean
end

struct RK4
end

struct MidPoint
end

function odestep(a::RK4, f, x, u, dt)
    k1 = f(x, u);         k1 *= dt;
    k2 = f(x .+ k1/2, u);  k2 *= dt;
    k3 = f(x .+ k2/2, u);  k3 *= dt;
    k4 = f(x .+ k3, u);    k4 *= dt;
    xp = x .+ (k1 .+ 2*k2 .+ 2*k3 .+ k4) ./ 6
    return xp
end

function odestep(a::MidPoint, f, x, u, dt)
    k1 = f(x, u);    k1 *= dt/2;
    xp = f(x + k1, u)
    return xp
end

function train!(nn::AbstractRigidBody, opt, data, testdata, nepochs, savedir, dt)
    function loss(x, u, xp)
        xp_hat = odestep(RK4(), nn, x, u, dt)
        xloss(xp, xp_hat)
    end

    logdir = joinpath(savedir, "logs")
    mkpath(logdir)
    logger = TeeLogger(current_logger(), TBLogger(logdir))

    with_logger(logger) do
        for epoch_idx in 1:nepochs
            metrics = _train!(nn, opt, data, loss)
            @info "train" train_loss=metrics.loss grad_norm=metrics.gnorm train_time=metrics.t

            test_t = @elapsed begin
                @info "test" val_loss=mean(loss(td...) for td in testdata) log_step_increment=0
            end
            @info "test" test_time=test_t log_step_increment=0

            save_t = @elapsed begin
                if epoch_idx % 20 == 0
                    @info "Saving nn..."
                    weights = Flux.params(nn)
                    JLD2.@save joinpath(savedir, "ckpt_epoch_$epoch_idx.jld2") weights
                end
            end
            @info "save" save_time=save_t log_step_increment=0

            @info "train" epoch=epoch_idx log_step_increment=0
        end
    end
    @info "Done!"

end
