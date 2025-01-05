import pygalmesh

__all__ = ['pygalmesh_converter']

def pygalmesh_converter(mesh_data, mesh_scratch):
    mesh = pygalmesh.generate_volume_mesh_from_surface_mesh(
        mesh_data,
        min_facet_angle=20.0,
        max_radius_surface_delaunay_ball=3.0,
        max_facet_distance=0.9,
        max_circumradius_edge_ratio=3.0,
        max_cell_circumradius=4.0,  # element size
        verbose=True)

    mesh.write(mesh_scratch + ".msh", file_format="gmsh22")  # binary=True
    mesh.write(mesh_scratch + "_test.inp")
