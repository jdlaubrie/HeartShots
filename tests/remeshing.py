# python3
# import pathlib library to define directories for the project
from pathlib import Path
path = Path()
test_path = path.cwd()
project_path = test_path.parent.resolve()
data_path = project_path.joinpath("data")
scratch_path = project_path.joinpath("scratch")

# import the project itself
import sys
sys.path.append(str(project_path))
from HeartShots import *

# define names and paths for data and scratch
mesh_name = 'ventricles'
mesh_data = str(data_path) + "/" + mesh_name + ".stl"
mesh_scratch = str(scratch_path) + "/" + mesh_name

# call pygalmesh function to convert and re-mesh
pygalmesh_converter(mesh_data, mesh_scratch)