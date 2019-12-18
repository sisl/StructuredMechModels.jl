using Flux

struct ReLU
end

(a::ReLU)(x) = relu(x)

der(a::ReLU, x) = ceil(clamp(x, 0, 1))

struct SoftPlus
    beta
end

SoftPlus() = SoftPlus(1.0)

(a::SoftPlus)(x) = softplus(x)

function der(a::SoftPlus, x)
    cx = clamp(x, -20, 20)
    exp_x = exp(convert(eltype(x), a.beta) * cx)
    out = exp_x / (exp_x + convert(eltype(x), 1.0))
    return out
end

struct Tanh
end

(a::Tanh)(x) = tanh(x)

der(a::Tanh, x) = convert(eltype(x), 1.0) - tanh(x)^2


struct Identity
end

(a::Identity)(x) = x

der(a::Identity, x) = clamp(x, 1, 1)

struct Differential{F, G, S, T}
    W::S
    b::T
    g::F
    g_prime::G
end

function Differential(in::Integer, out::Integer, g=Identity();
               initW = Flux.glorot_uniform, initb = Flux.zeros)
    return Differential(initW(out, in), (initb(out)), g, x->der(g, x))
end

@Flux.treelike Differential

function (a::Differential)((x, prev))
    W, b, g, g_prime = a.W, a.b, a.g, a.g_prime
    o = W*x .+ b
    op = g.(o)
    #newd = reshape(g_prime.(o), size(o, 1), 1, size(o, 2)) # hidden x B
    #derW = newd .* W
    derW = reshape(g_prime.(o), size(o, 1), 1, size(o, 2)) .* W
    der = batchmul(derW, prev)
    return op, der
end

struct DifferentialNet
    layers::Chain
    DifferentialNet(xs...) = new(Chain(xs...))
end

Flux.@functor DifferentialNet


function (a::DifferentialNet)(q::AbstractArray)
    xdim, B = size(q)
    qd_dq = repeat(eye(eltype(q), xdim), 1, 1, B)
    qd, qd_dq = a.layers((q, qd_dq))
    return qd, qd_dq
end

struct GeneralizedForces
    forces
end

Flux.@functor GeneralizedForces

function (a::GeneralizedForces)(q::AbstractArray, v::AbstractArray, u::AbstractArray)
    F = sum([f(q, v, u) for f in a.forces])
    return F
end

struct GeneralizedForce
    layers::Chain
end
Flux.@functor GeneralizedForce

function (a::GeneralizedForce)(q, v, u)
    x = vcat(q, v, u)
    return a.layers(x)
end

function GeneralizedForce(qdim, udim, hidden_sizes)
    return GeneralizedForce(Chain(Dense(2*qdim+udim, hidden_sizes[1], tanh),
                                     (Dense(hidden_sizes[i], hidden_sizes[i+1], tanh)
                                      for i in 1:(length(hidden_sizes)-1))...,
                                     Dense(hidden_sizes[end], qdim)))
end

struct ControlAffineForce
    layers::Chain
end

Flux.@functor ControlAffineForce

function (a::ControlAffineForce)(q, v, u)
    bsize = size(q)[end]
    qdim = size(q, 1)
    udim = size(u, 1)
    B = a.layers(q)
    B = reshape(B, (qdim, udim, bsize))
    F = batchmul(B, reshape(u, size(u, 1), 1, size(u, 2)))
    F = dropdims(F, dims=2)
    @assert size(F) == (qdim, bsize)
    return F
end

function ControlAffineForce(qdim, udim, hidden_sizes)
    return ControlAffineForce(Chain(Dense(qdim, hidden_sizes[1], tanh),
                                       (Dense(hidden_sizes[i], hidden_sizes[i+1], tanh)
                                        for i in 1:(length(hidden_sizes)-1))...,
                                       Dense(hidden_sizes[end], qdim * udim)))
end

struct ControlAffineLinearForce
    B
end

Flux.@functor ControlAffineLinearForce

function (a::ControlAffineLinearForce)(q, v, u)
    bsize = size(q)[end]
    qdim = size(q, 1)
    B = reshape(a.B, size(a.B)..., 1)
    F = batchmul(repeat(B, 1, 1, bsize), reshape(u, size(u, 1), 1, size(u, 2)))
    F = dropdims(F, dims=2)
    @assert size(F) == (qdim, bsize)
    return F
end

function ControlAffineLinearForce(qdim, udim)
    return ControlAffineLinearForce(randn(qdim, udim))
end

struct QVForce
    layers::Chain
end

Flux.@functor QVForce

function (a::QVForce)(q, v, u)
    bsize = size(q)[end]
    qdim = size(q, 1)
    F = a.layers(vcat(q, v))
    F = reshape(F, (qdim, bsize))
    @assert size(F) == (qdim, bsize)
    return F
end

function QVForce(qdim, udim, hidden_sizes)
    return QVForce(Chain(Dense(qdim*2, hidden_sizes[1], tanh),
                         (Dense(hidden_sizes[i], hidden_sizes[i+1], tanh)
                          for i in 1:(length(hidden_sizes)-1))...,
                         Dense(hidden_sizes[end], qdim)))
end

function DelanMassMatrix(qdim, hidden_sizes, activation=ReLU())
    return DifferentialNet(Differential(qdim, hidden_sizes[1], activation),
                           (Differential(hidden_sizes[i], hidden_sizes[i+1], activation)
                            for i in 1:(length(hidden_sizes)-1))...,
                           Differential(hidden_sizes[end], round(Int, qdim*(qdim+1)/2)))
end

function DelanPotential(qdim, hidden_sizes, activation=ReLU())
    return DifferentialNet(Differential(qdim, hidden_sizes[1], activation),
                           (Differential(hidden_sizes[i], hidden_sizes[i+1], activation)
                            for i in 1:(length(hidden_sizes)-1))...,
                           Differential(hidden_sizes[end], 1))
end

struct DelanZeroPotential
end

(a::DelanZeroPotential)(q) = (zeros(1, size(q)[end]), zeros(1, size(q, 1), size(q)[end]))


struct DelanRigidBody{M, V, F, I<:Integer, G, H<:AbstractVector{<:Real}} <: AbstractRigidBody
    massmatrix::M
    potential::V
    generalized_forces::F
    qdim::I
    udim::I
    bias::G
    thetamask::H
end

Flux.@functor DelanRigidBody

function DelanRigidBody(qdim::Int, udim::Int, hidden_sizes, thetamask; activation=ReLU(), bias=10.)
    mm = DelanMassMatrix(qdim, hidden_sizes, activation)
    pot = DelanPotential(qdim, hidden_sizes, activation)
    ff = GeneralizedForce(qdim, udim, hidden_sizes)
    return DelanRigidBody(mm, pot, ff, qdim, udim, bias, thetamask)
end

"""
Converts a batch of vectors to a batch of LowerTriangular matrices.
"""
function embedtoLwithbias(x, bias, qdim)
    L_diag = fill_diagonal(x[1:qdim, :] .+ convert(eltype(x), bias))
    L_ltr = fill_lower_triangular(x[qdim+1:end, :])
    L = L_diag .+ L_ltr
    return L
end

function (f::DelanRigidBody)(x, u)
    q = x[1:f.qdim, :]
    v = x[f.qdim+1:2*f.qdim, :]

    embedtoL(x) = embedtoLwithbias(x, 0., f.qdim)

    L_params, dLparamsdq = f.massmatrix(q)

    L = embedtoLwithbias(L_params, f.bias, f.qdim)

    M = batchmul(L, permutedims(L, (2, 1, 3)))

    pot, gradpot = f.potential(q)
    G = dropdims(gradpot, dims=1)

    batched_v = reshape(v, size(v, 1), 1, size(v, 2))

    dLparamsdt = batchmul(dLparamsdq, batched_v)

    dLparamsdt = dropdims(dLparamsdt, dims=2)
    dLdt = embedtoL(dLparamsdt)

    dMdt = batchmul(L, permutedims(dLdt, (2, 1, 3))) + batchmul(dLdt, permutedims(L, (2, 1, 3)))

    dMdtv = batchmul(dMdt, batched_v)
    dMdtv = dropdims(dMdtv, dims=2)

    function dKEdq_i(dLdq_i)
        _mx = batchmul(L, permutedims(dLdq_i, (2, 1, 3))) + batchmul(dLdq_i, permutedims(L, (2, 1, 3)))
        t = batchmul(_mx, batched_v)
        dKEdqi = sum((dropdims(t, dims=2) .* v), dims=1)
        return dKEdqi
    end

    # XXX: Zygote doesn't work with this anymore :/
    #dKEdq = slicemap(x->dKEdq_i(embedtoL(x)), dLparamsdq; dims=(1, 3))
    #dKEdq = dropdims(dKEdq, dims=1)
    dKEdq = map(i->dKEdq_i(embedtoL(dLparamsdq[:, i, :])), 1:f.qdim)
    dKEdq = vcat(dKEdq...)

    Cv = dMdtv - convert(eltype(x), 0.5) * dKEdq

    F = f.generalized_forces(q, v, u)
    totalF = F - Cv - G
    qddot_ls = [(view(M, :, :,i) \ view(totalF, :, i)) for i in 1:size(L, 3)]
    qddot = hcat(qddot_ls...)
    return vcat(v, qddot)
end

"""
Create (a batch of) diagonal matrix from a vector of inputs

$(SIGNATURES)
"""
function fill_diagonal(x)
    m = size(x, 1)
    y = Zygote.Buffer(x, (m, m, size(x)[2:end]...)...)
    for R in CartesianIndices((m, m, size(x)[end]))
        y[R]= convert(eltype(x), 0.)
    end
    for i in 1:size(x)[end]
        vptr = 1
        @inbounds for j in 1:m
            y[j, j, i] = x[vptr, i]
            vptr += 1
        end
    end
    return copy(y)
end


"""
Create (a batch of) triangular matrix from a vector of inputs

$(SIGNATURES)
"""
function fill_lower_triangular(x)
    m = size(x)[1]
    n = (sqrt(8m+1)+1)/2
    n = round(Int, n)
    y = Zygote.Buffer(x, (n, n, size(x)[2:end]...)...)

    for R in CartesianIndices((n, n, size(x)[end]))
        y[R]= convert(eltype(x), 0.)
    end
    for i in 1:size(x)[end]
        vptr = 1
        @inbounds for j in 1:n, k in 1:j-1
            y[j, k, i] = x[vptr, i]
            vptr += 1
        end
    end
    return copy(y)
end
