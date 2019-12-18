import TrajectoryOptimization
urdf_folder = joinpath(@__DIR__, "urdfs")
urdf_furuta = joinpath(urdf_folder, "FurutaPendulum.urdf")

furuta_model = TrajectoryOptimization.Model(urdf_furuta) 
