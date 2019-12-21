module StructuredMechModels

using Flux
using LinearAlgebra
using TrajectoryOptimization
using DocStringExtensions

import TrajectoryOptimization: Model

include("utils.jl")

abstract type AbstractRigidBody end

"""
$(TYPEDEF)
"""
struct Env
    sys
    x0
    thetamask
end

Env(sys, x0) = Env(sys, x0, zeros(size(x0)))

"""
Helper function to interface parameterized rigidbody dynamics functions with TrajectoryOptimization
"""
function rigid_body_dynamics!(ẋ, x, u, p)
    res = p.f(x, u)
    ẋ[:] = res
end

function rigid_body_jacobian!(Z, x, u, p)
    error("Shouldn't be called! TrajOpt uses ForwardDiff to generate the jacobian")
end

"""
Converts parameterized rigidbody dynamics functions to `Model` useful for `TrajectoryOptimization`

$(SIGNATURES)
"""
function Model(nn::AbstractRigidBody)
    p = (f=nn,)
    return AnalyticalModel{Nominal,TrajectoryOptimization.Continuous}(
        (xdot, x, u)->rigid_body_dynamics!(xdot, x, u, p),
        (Z,x,u)->rigid_body_jacobian!(Z, x, u, p), 2*nn.qdim, nn.udim, 0, p, Dict{Symbol,Any}())
end

include("naive.jl")

"""
Models the rigidbody as a Black-Box Neural Network

$(SIGNATURES)

**Arguments**:

- `qdim:: Int`: Dimensions of position configuration
- `udim:: Int`: Dimensions of active control
- `thetamask`: Binary Vector with ones corresponding to angles in the configuration space
"""
function Naive(qdim::Int, udim::Int, thetamask; activation=Tanh(), hidden_sizes=(128, 128, 128))
    nn = f64(NaiveRigidBody(qdim, udim, hidden_sizes, thetamask))
    return nn
end

include("delan.jl")

"""
Models the rigidbody as a Structured Mechanical Model with no constraints on the external forces.

$(SIGNATURES)

**Arguments**:

- `qdim`: Dimensions of position configuration
- `udim`: Dimensions of active control
- `thetamask`: Binary Vector with ones corresponding to angles in the configuration space

Useful when no prior knowledge about the active forces on a mechanical system.
"""
function DeLan(qdim, udim, thetamask; activation=Tanh(), hidden_sizes=(32, 32, 32), bias=10.0)
    nn = f64(DelanRigidBody(qdim, udim, hidden_sizes, thetamask, activation=activation, bias=bias))
    return nn
end

"""
Models the rigidbody as a Structured Mechanical Model with the constraints that active forces on the system are control affine.

$(SIGNATURES)

**Arguments**:

- `qdim`: Dimensions of position configuration
- `udim`: Dimensions of active control
- `thetamask`: Binary Vector with ones corresponding to angles in the configuration space

Almost all mechanical systems are control affine in the input forces for control.
"""
function ControlAffine(qdim, udim, thetamask; activation=Tanh(), hidden_sizes=(32, 32, 32), bias=10.0)
    mm = DelanMassMatrix(qdim, hidden_sizes, activation)
    pot = DelanPotential(qdim, hidden_sizes, activation)
    ff = GeneralizedForces([ControlAffineForce(qdim, udim, hidden_sizes), QVForce(qdim, udim, hidden_sizes)])
    nn = f64(DelanRigidBody(mm, pot, ff, qdim, udim, bias, thetamask))
    return nn
end

include("dataset.jl")

include("datagen.jl")

#include("supervised.jl")

include("policy.jl")

include("pytorch.jl")
using .Torch


module Dynamics
    include(joinpath("dynamics", "furuta.jl"))
    include(joinpath("dynamics", "doublecartpole.jl"))
    include(joinpath("dynamics", "triplecartpole.jl"))
end
using .Dynamics

export Model

end # module
