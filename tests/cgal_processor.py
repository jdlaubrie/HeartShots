import sys
from pathlib import Path

from meshio import Mesh

from HeartShots.mesh.cgal_processor import CgalProcessor

#  python3 tests/cgal_processor.py /tmp/ventricles.stl
if __name__ == '__main__':
    path = Path(sys.argv[1])
    mesh: Mesh = CgalProcessor().get(path)

    print(mesh.points)
