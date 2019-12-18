using MeshCat, MeshCatMechanisms, RigidBodyDynamics, GeometryTypes, CoordinateTransformations
using Zygote
using LinearAlgebra: I
using BatchedRoutines


"""
Visualize a trajectory in the browser
"""
function viz_traj(qs, ts, mechanism, urdf)
    vis = Visualizer()
    open(vis)
    mvis = MechanismVisualizer(mechanism, URDFVisuals(urdf), vis)
    settransform!(vis["/Cameras/default"], compose(Translation(0.0, 1.0, 1.0), LinearMap(RotZ(pi/2))));
    setanimation!(mvis, ts, qs)
end

"""
Matrix-matrix multiplication _batched_ and adjoint defined
"""
function batchmul(A::AbstractArray{T,3},B::AbstractArray{V,3}) where {T, V}
    # XXX: To deal with Dual numbers from ForwardDiff
    if T ∉ (Float64, Float32) || V ∉ (Float64, Float32)
        return batchmul_slow(A, B)
    end
    batched_gemm('N','N', A, B)
end

function batchmul_slow(A, B)
    batchmm = [A[:,:,i] * B[:, :, i] for i in 1:size(A, 3)]
    C = reduce((xs,x)->cat(xs, x; dims=3),
               [reshape(batchmm[i], size(batchmm[i])..., 1) for i in 1:size(A)[end]])
    return C
end

Zygote.@adjoint function batchmul(A, B)
    batchmul(A, B), Δ -> (batched_gemm('N' ,'T', Δ , B), batched_gemm( 'T' ,'N', A, Δ))
end

eye(T, p) = Matrix{T}(I, p, p)
Zygote.@nograd eye
