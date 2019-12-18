import TrajectoryOptimization
urdf_folder = joinpath(@__DIR__, "urdfs")
urdf_triplecartpole = joinpath(urdf_folder, "triplecartpole.urdf")

triplecartpole_model = TrajectoryOptimization.Model(urdf_triplecartpole, [1., 0., 0., 0.])
triplecartpole_model_full = TrajectoryOptimization.Model(urdf_triplecartpole, [1., 1., 1., 1.])
