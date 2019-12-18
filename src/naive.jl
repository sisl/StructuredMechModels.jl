struct NaiveRigidBody <: AbstractRigidBody
    nn
    qdim
    udim
    thetamask
end
function NaiveRigidBody(qdim::Int, udim::Int, hidden_sizes, thetamask)

    return NaiveRigidBody(Chain(
        Dense(qdim*2+udim, hidden_sizes[1], tanh),
        (Dense(hidden_sizes[i], hidden_sizes[i+1], tanh) for i in 1:(length(hidden_sizes)-1))...,
        Dense(hidden_sizes[end], qdim, identity)
    ) |> f64, qdim, udim, thetamask)
end

function (f::NaiveRigidBody)(x, u)
    q = x[1:f.qdim, :]
    v = x[f.qdim+1:2*f.qdim, :]
    return vcat(v, f.nn(vcat(x, u)))
end
Flux.@functor NaiveRigidBody
