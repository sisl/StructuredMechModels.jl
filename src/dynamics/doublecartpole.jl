import TrajectoryOptimization

urdf_folder = joinpath(@__DIR__, "urdfs")
urdf_doublecartpole = joinpath(urdf_folder, "doublecartpole.urdf")

doublecartpole_model = TrajectoryOptimization.Model(urdf_doublecartpole, [1., 0., 0.])
doublecartpole_model_full = TrajectoryOptimization.Model(urdf_doublecartpole, [1., 1., 1.])
