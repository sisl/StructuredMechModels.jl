using ControlSystems

abstract type AbstractController end

struct TVLQRController <: AbstractController
    nom_x
    nom_u
    K
    xf
    uf
    L
    N
    Δt
    LQR_eps
end

function TVDARE(model, nom_x, nom_u, Q, R, Qf, dt)
    P = Qf
    n = model.n
    m = model.m
    N = length(nom_x)
    K = [zeros(m,n) for i = 1:N-1]

    model_d = discretize_model(model, :rk4)

    J = TrajectoryOptimization.PartedMatrix(model_d)

    for i = 1:length(nom_u)
        t = N - i
        xt = nom_x[t]
        ut = nom_u[t]

        jacobian!(J, model_d, xt, ut, dt)

        A = J.xx
        B = J.xu

        APA = A' * P * A
        APB = A' * P * B
        BPA = B' * P * A
        BPB = B' * P * B

        Kt = -(R + BPB)\BPA
        K[t] .= Kt

        P = Q + Kt' * R * Kt + (A + B * Kt)' * P * (A + B * Kt)
    end

    return K
end

struct ALTRO
    x
    u
    K
end

function ALTRO(model, x0, xf, Q, R, Qf, N, dt;
               ilqr_opts=(iterations=200,
                          iterations_linesearch=10,
                          verbose=true, cost_tolerance=1.0e-6),
               al_opts=(verbose=false,
                        iterations=30,
                        penalty_scaling=10.0,
                        cost_tolerance=1.0e-6,
                        cost_tolerance_intermediate=1.0e-5,
                        constraint_tolerance=1.0e-4),
               )
    model_d = discretize_model(model, :rk4)
    T = eltype(x0)
    n = model_d.n;
    m = model_d.m;

    goal = goal_constraint(xf);

    #N = 251;

    U = [ones(m) for k = 1:N-1];

    #dt = 0.01;

    obj = LQRObjective(Q,R,Qf,xf,N);
    con = Constraints(N);
    con[N] += goal;

    prob = Problem(model_d,obj,constraints=con,x0=x0,N=N,dt=dt,xf=xf);
    initial_controls!(prob, U);

    opts_ilqr = iLQRSolverOptions(;ilqr_opts...);

    opts_al = AugmentedLagrangianSolverOptions{T}(;opts_uncon=opts_ilqr, al_opts...);

    opts_altro = ALTROSolverOptions{T}(verbose=false,
                                       opts_al=opts_al);

    prob, solver = TrajectoryOptimization.solve(prob, opts_altro); # solve with ALTRO
    prob.X[end] = xf

    return ALTRO(prob.X, prob.U, solver.solver_al.solver_uncon.K)
end

function TVLQRController(model, x0, xf, Q, R, Qf, N, dt;
                         TVLQR_Q = nothing,
                         TVLQR_R = nothing,
                         TVLQR_Qf = nothing,
                         LQR_eps = 1e-1,
                         ilqr_opts=(iterations=200,
                                    iterations_linesearch=10,
                                    verbose=true, cost_tolerance=1.0e-6),
                         al_opts=(verbose=false,
                                  iterations=30,
                                  penalty_scaling=10.0,
                                  cost_tolerance=1.0e-6,
                                  cost_tolerance_intermediate=1.0e-5,
                                  constraint_tolerance=1.0e-4),
                         )

    if isnothing(TVLQR_Q)
        TVLQR_Q = Q
    end
    if isnothing(TVLQR_R)
        TVLQR_R = R
    end
    if isnothing(TVLQR_Qf)
        TVLQR_Qf = Qf
    end

    nom_traj = ALTRO(model, x0, xf, Q, R, Qf, N, dt, ilqr_opts=ilqr_opts, al_opts=al_opts)
    return TVLQRController(model, x0, xf, nom_traj, TVLQR_Q, TVLQR_R, TVLQR_Qf, N, dt, LQR_eps)
end

"""
Q, R, Qf refer to the TVLQR costs
"""
function TVLQRController(model, x0, xf, nom_traj, Q, R, Qf, N, dt, LQR_eps)
    model_d = discretize_model(model, :rk4)

    X = nom_traj.x
    U = nom_traj.u
    K = TVDARE(model, nom_traj.x, nom_traj.u, Q, R, Qf, dt)
    # K = nom_traj.K

    J = TrajectoryOptimization.PartedMatrix(model_d)
    uf = zeros(model.m)

    jacobian!(J, model_d, xf, uf, dt)

    A = J.xx
    B = J.xu
    L = dlqr(A, B, Q, R)
    return TVLQRController(X, U, K, xf, uf, -L, N, dt, LQR_eps)
end

function (ctrl::TVLQRController)(t, x, thetamask)
    i = convert(Int, ceil(t/ctrl.Δt))
    i_ = min(i, ctrl.N-1)
    # i = min(t, ctrl.N-1)

    dx = x - ctrl.xf
    @views dx .= wraptopi(dx, thetamask)
    #@show norm(dx)

    if (i_ < i || norm(dx) < ctrl.LQR_eps)
        x_ = ctrl.xf
        u_ = ctrl.uf
        K = ctrl.L
        #@show dx
    else
        x_ = ctrl.nom_x[i_]
        u_ = ctrl.nom_u[i_]
        K  = ctrl.K[i_]
    end

    dx = x - x_
    @views dx .= wraptopi(dx, thetamask)


    du = K * dx
    u = u_ + du

    return u
end

struct RandomController <: AbstractController
end

function RandomController(model, std)
    return (t, x, thetamask) -> std * randn(model.m)
end


function addnoise(ctrl::AbstractController, noisestd)
    function noisy(t, x, thetamask)
        u = ctrl(t, x, thetamask)
        return u .+ randn(size(u)...) .* noisestd
    end
    return noisy
end

"""
Clip the output of a controller to a maximum force value `maxu`.

$(SIGNATURES)
"""
function clip(ctrl::AbstractController, maxu)
    function cu(t, x, thetamask)
        u = ctrl(t, x, thetamask)
        return clamp.(u, -maxu, maxu)
    end
    return cu
end

function wraptopi(radians)
     return radians .- 2*pi*floor.((radians .+ pi)/(2*pi))
end

function wraptopi(radians, mask)
    oldrad = copy(radians)
    return mask .* wraptopi(radians) .+ (1.0 .- mask) .* oldrad
end
