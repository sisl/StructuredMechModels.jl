module Torch

using DocStringExtensions
using StructuredMechModels: AbstractRigidBody
using PyCall
using Zygote: Params


const torch = PyNULL()
const nn = PyNULL()
const optim = PyNULL()
const F = PyNULL()
const py_utils = PyNULL()
const structmechmod = PyNULL()

abstract type PyTorchRigidBody <: AbstractRigidBody end

"""
$(TYPEDEF)
"""
struct Naive{I<:Integer} <: PyTorchRigidBody
    nn::PyObject
    qdim::I
    udim::I
    thetamask
end

"""
Models the rigidbody as a Black-Box Neural Network

$(SIGNATURES)

Arguments:
    - `qdim`: Dimensions of position configuration
    - `udim`: Dimensions of active control
    - `thetamask`: Binary Vector with ones corresponding to angles in the configuration space
"""
function Naive(qdim::Int, udim::Int, thetamask; activation="Tanh", hidden_sizes=(128, 128, 128))
    nn = structmechmod.rigidbody.NaiveRigidBody(qdim, udim, torch.from_numpy(thetamask), hidden_sizes)
    return Naive(nn, qdim, udim, thetamask)
end


struct DeLan{I<:Integer} <: PyTorchRigidBody
    nn::PyObject
    qdim::I
    udim::I
    thetamask
end

"""
Models the rigidbody as a Structured Mechanical Model with no constraints on the external forces.

$(SIGNATURES)

Arguments:
    - `qdim`: Dimensions of position configuration
    - `udim`: Dimensions of active control
    - `thetamask`: Binary Vector with ones corresponding to angles in the configuration space

Useful when no prior knowledge about the active forces on a mechanical system.
"""
function DeLan(qdim, udim, thetamask; activation="Tanh", hidden_sizes=(32, 32, 32))
    @assert all(hidden_sizes .== hidden_sizes[1])
    nn = structmechmod.rigidbody.DeLan(qdim, hidden_sizes[1], length(hidden_sizes), torch.from_numpy(thetamask), udim=udim, activation=activation)
    return DeLan(nn, qdim, udim, thetamask)
end

struct ControlAffine{I<:Integer} <: PyTorchRigidBody
    nn::PyObject
    qdim::I
    udim::I
    thetamask
end

"""
Models the rigidbody as a Structured Mechanical Model with the constraints that active forces on the system are control affine.

$(SIGNATURES)

Arguments:
    - `qdim`: Dimensions of position configuration
    - `udim`: Dimensions of active control
    - `thetamask`: Binary Vector with ones corresponding to angles in the configuration space

Almost all mechanical systems are control affine in the input forces for control.
"""
function ControlAffine(qdim, udim, thetamask; activation="Tanh", hidden_sizes=(32, 32, 32))
    @assert all(hidden_sizes .== hidden_sizes[1])
    forces = structmechmod.models.GeneralizedForces([structmechmod.models.ControlAffineForceNet(qdim, udim, hidden_sizes),
                                               structmechmod.models.QVForceNet(qdim, hidden_sizes)])
    nn = structmechmod.rigidbody.DeLan(qdim, hidden_sizes[1], length(hidden_sizes),
                                 torch.from_numpy(thetamask), udim=udim, activation=activation, forces=forces)
    return ControlAffine(nn, qdim, udim, thetamask)
end


"""


$(SIGNATURES)

Arguments:
    - `x`: Input state (includes state and actions)
    - `u`: Input control
"""
function (f::PyTorchRigidBody)(x, u)
    q = torch.from_numpy(permutedims(x[1:f.qdim, :], (2, 1)))
    v = torch.from_numpy(permutedims(x[f.qdim+1:2*f.qdim, :], (2, 1)))
    u = torch.from_numpy(permutedims(u, (2, 1)))

    qdot, vdot = f.nn(0, (q, v), u)
    qdot, vdot = map(x->permutedims(x, (2, 1)), (qdot.detach().numpy(), vdot.detach().numpy()))
    return vcat(qdot, vdot)
end

"""
Returns the parameters of `PyTorchRigidBody` as required by `Flux`.

$(SIGNATURES)
"""
function params(f::PyTorchRigidBody)
    ps = f.nn.get_params()
    return Params(ps)
end

"""

$(SIGNATURES)
"""
function train!(model::PyTorchRigidBody, train_data, valid_data, hparams)
    train_data = map(x->(permutedims(x, (2, 1))), (train_data.xs, train_data.us, train_data.xsp))
    valid_data = map(x->(permutedims(x, (2, 1))), (valid_data.xs, valid_data.us, valid_data.xsp))
    structmechmod.trainer.train(model.nn, train_data, valid_data, hparams)
end

"""
$(SIGNATURES)
"""
function load_state!(model::PyTorchRigidBody, path)
    model.nn.load_state_dict(torch.load(path))
    return model
end


function __init__()
    copy!(torch, pyimport("torch"))
    copy!(nn, pyimport("torch.nn"))
    copy!(optim, pyimport("torch.optim"))
    copy!(optim, pyimport("torch.nn.functional"))
    copy!(structmechmod, pyimport("structmechmod"))
end
end
