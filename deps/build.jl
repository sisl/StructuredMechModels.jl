using PyCall

const PKG = "git+https://github.com/sisl/structmechmod"

try
    pyimport("structmechmod")
catch e
    try
        run(`$(PyCall.pyprogramname) -m pip install $PKG`)
    catch ee
        if !(typeof(ee) <: PyCall.PyError)
            rethrow(ee)
        end
        @warn("""
    Python dependencies not installed.
    Either
    - Rebuild `PyCall` to use Conda by running the following in Julia REPL
        - `ENV[PYTHON]=""; using Pkg; Pkg.build("PyCall"); Pkg.build("StructuredMechModels)
    - Or install the dependencies by running `pip`
        - `pip install $PKG`
        """
              )
    end
end
