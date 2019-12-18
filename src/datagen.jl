"""
$(SIGNATURES)

Mostly useful for generating data for training offline.
"""
function batch_rollout(env::Env, init_xs, uss, T, B, dt)
    @assert size(init_xs)[end] == B
    xs = zeros(Float64, env.sys.n, T, B)
    @views xs[:, 1, :] .= init_xs
    for b in 1:B
        for k = 1:T-1
            @views TrajectoryOptimization.evaluate!(xs[:, k+1, b], env.sys, xs[:, k, b], uss[:, k, b], dt)
            @views xs[:, k+1, b] .= wraptopi(xs[:, k+1, b], env.thetamask)
        end
    end
    return xs, uss
end

"""
Rollout a trajectory using a policy in an environment for `T` timesteps.

$(SIGNATURES)
"""
function rollout(env::Env, policy, T, dt)
    xs = zeros(Float64, env.sys.n, T)
    us = zeros(Float64, env.sys.m, T-1)
    @views xs[:, 1] = env.x0
    for t in 1:T-1
        @views us[:, t] = policy(convert(Float64, t)*dt, xs[:, t], env.thetamask)
        @views TrajectoryOptimization.evaluate!(xs[:, t+1], env.sys, xs[:, t], us[:, t], dt)
        @views xs[:, t+1] .=  xs[:, t+1] # wraptopi(xs[:, t+1], env.thetamask)
    end
    return xs, us
end

"""
Converts a batch of trajectories of position, velocities and input forces into a Dataset

$(SIGNATURES)
"""
function trajbatch2dataset(xs, uss)
    A = slidingwindow(i->i+1, xs, 1, LearnBase.ObsDim.Constant(2))
    newxs = []
    newxsp = []
    newus = []
    for (i, (x, xp)) in enumerate(A)
        u = uss[:, i, :]
        push!(newus, u)
        push!(newxs, dropdims(x, dims=2))
        push!(newxsp, xp)
    end
    return Dataset(reduce(hcat, newxs), reduce(hcat, newus), reduce(hcat, newxsp))
end
