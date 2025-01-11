import os
import tempfile
from pathlib import Path

import meshio
from CGAL import CGAL_Mesh_3
from CGAL.CGAL_Mesh_3 import Polyhedral_mesh_domain_3, Mesh_3_parameters, Default_mesh_criteria
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from meshio import Mesh


class CgalProcessor:

    def get(self, path: Path) -> Mesh:
        """
        Get mesh from path
        Is based on writing temporary files to allow meshio and CGAL to process
        :param path: the path of the mesh file
        :return: a cgal processed mesh wrapped in a meshio class (Mesh)
        """
        datafile: str
        if path.suffix != '.off':
            datafile = str(self._write_tmp_off_file(mesh=self._get_mesh(path)))
        else:
            datafile = str(path)

        # Create input polyhedron
        polyhedron = Polyhedron_3(datafile)

        # Create domain
        domain = Polyhedral_mesh_domain_3(polyhedron)

        # Create param
        params = Mesh_3_parameters()
        params.set_lloyd(60, 10000, 0.01, 0.001)
        params.no_exude()
        params.no_perturb()

        # Mesh criteria (no cell_size set)
        criteria = Default_mesh_criteria()
        criteria.facet_angle(20.0)
        criteria.facet_size(3.0)
        criteria.facet_distance(0.9)
        criteria.cell_radius_edge_ratio(3.0)
        criteria.cell_size(4.0)

        # Mesh generation
        c3t3 = CGAL_Mesh_3.make_mesh_3(domain, criteria, params)

        mesh = meshio.read(self._write_tmp_mesh_file(c3t3))
        return mesh

    def _get_mesh(self, path: Path) -> Mesh:
        return meshio.read(str(path))

    def _write_tmp_off_file(self, mesh: Mesh) -> Path:
        fh, off_file = tempfile.mkstemp(suffix=".off")
        os.close(fh)
        meshio.write(off_file, mesh)
        return Path(off_file)

    def _write_tmp_mesh_file(self, mesh: CGAL_Mesh_3):
        fh, mesh_file = tempfile.mkstemp(suffix=".mesh")
        os.close(fh)
        mesh.output_to_medit(mesh_file)
        return Path(mesh_file)
