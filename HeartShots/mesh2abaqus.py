# python3
import pygalmesh
import os, sys
import pandas as pd
import numpy as np

#=============================================================================#
class Mesh(object):

    def __init__(self):

        self.nodes = None
        self.elements = None
        self.faces = None

        self.left_ventricle_nodes = None
        self.right_ventricle_nodes = None
        self.epicardium_nodes = None

        self.left_ventricle_surface = None
        self.right_ventricle_surface = None
        self.epicardium_surface = None

        self.aorta_nodes = None
        self.mitral_nodes = None

    #=========================================================================#
    def MeshConvertion(self, mesh_name, elem_size=4.0):

        mesh0 = pygalmesh.generate_volume_mesh_from_surface_mesh(
            mesh_name + ".stl",
            min_facet_angle=20.0,
            max_radius_surface_delaunay_ball=3.0,
            max_facet_distance=0.9,
            max_circumradius_edge_ratio=3.0,
            max_cell_circumradius=elem_size,             #element size
            verbose=True)

        mesh0.write(mesh_name + ".msh",file_format="gmsh22")  #binary=True
        mesh0.write(mesh_name + "_pre.inp")

    #=========================================================================#
    def ReadAbaqusTestMesh(self, mesh_name):

        mesh_file = open(mesh_name + "_pre.inp", "r")
        mesh_lines = mesh_file.readlines()
        mesh_file.close()

        n_lines = len(mesh_lines)

        mesh_dict = {}
        i = 0
        for line in mesh_lines:
            if line[:1] == '*':
                mesh_dict[line] = i
            i += 1

        for key in mesh_dict.keys():
            # count nodes
            if key[:5] == '*NODE':
                node_key = key
                nnodes = 0
                for i in range(mesh_dict[key]+1,n_lines):
                    if mesh_lines[i][:1] == '*':
                        break
                    nnodes += 1
            # count elements. 3d elements
            if key[:8] == '*ELEMENT' and key[-5:-1] == 'C3D4':
                elem_key = key
                nelem = 0
                for i in range(mesh_dict[key]+1,n_lines):
                    if mesh_lines[i][:1] == '*':
                        break
                    nelem += 1
            # count faces. 2d elements
            if key[:8] == '*ELEMENT' and key[-5:-1] == 'R3D3':
                face_key = key
                nfaces = 0
                for i in range(mesh_dict[key]+1,n_lines):
                    if mesh_lines[i][:1] == '*':
                        break
                    nfaces += 1

    #----------------------------------------------------------------#
        # get the nodes of abaqus mesh
        df_node = pd.Series(
                mesh_lines[mesh_dict[node_key]+1:mesh_dict[node_key]+nnodes+1])
        df_node = df_node.replace(r'\n', '', regex=True)               #remove end-line
        df_node = df_node.str.split(',', expand=True)                  #split columns
        nodes = np.array(df_node.values[:,1:], dtype=np.float64)

        # get the element connectivity
        df_elem = pd.Series(
                mesh_lines[mesh_dict[elem_key]+1:mesh_dict[elem_key]+nelem+1])
        df_elem = df_elem.replace(r'\n', '', regex=True)
        df_elem = df_elem.str.split(',', expand=True)
        elements = np.array(df_elem.values[:,1:], dtype=np.int64) - 1

    #----------------------------------------------------------------#
        # search the nodes in the surfaces
        # meshIO generates a face and its inverse
        faces_bool = np.zeros((nfaces),dtype=bool)
        for i in range(nfaces):
            if np.remainder(i,2) == 0: faces_bool[i] = True

        nfaces2 = int(nfaces/2)

        # read faces connectivity
        df_face = pd.Series(
                mesh_lines[mesh_dict[face_key]+1:mesh_dict[face_key]+nfaces+1])
        df_face = df_face.replace(r'\n', '', regex=True)
        df_face = df_face.str.split(',', expand=True)
        faces = np.array(df_face.values[faces_bool,1:], dtype=np.int64) - 1

        # connect a face with its element and define the surface
        face_of_elem = np.zeros((nfaces2,2),dtype=np.int64)
        face_normal = np.zeros((nfaces2,3),dtype=np.float64)
        face_center = np.zeros((nfaces2,3),dtype=np.float64)
        for face in range(nfaces2):
            face_nodes = faces[face,:]

            # find the element of the face and surface
            elem1, pos1 = np.where(elements==face_nodes[0])
            elem2, pos2 = np.where(elements[elem1]==face_nodes[1])
            elem3, pos3 = np.where(elements[elem1[elem2]]==face_nodes[2])
            face_of_elem[face,0] = elem1[elem2[elem3]]
            local_node = np.zeros((3),dtype=np.int64)
            local_node[0] = pos1[elem2[elem3]]
            local_node[1] = pos2[elem3]
            local_node[2] = pos3
            local_node.sort()
            if np.all(np.equal(local_node,[0,1,2])):
                face_of_elem[face,1] = 0
            elif np.all(np.equal(local_node,[0,1,3])):
                face_of_elem[face,1] = 1
            elif np.all(np.equal(local_node,[1,2,3])):
                face_of_elem[face,1] = 2
            elif np.all(np.equal(local_node,[0,2,3])):
                face_of_elem[face,1] = 3

            # compute the normal of the face
            tan1 = nodes[face_nodes[0]] - nodes[face_nodes[1]]
            tan2 = nodes[face_nodes[2]] - nodes[face_nodes[1]]
            face_normal[face,:] = np.cross(tan1,tan2)/\
                            np.linalg.norm(np.cross(tan1,tan2))
            face_center[face,:] = np.mean(nodes[face_nodes],axis=0)

        mesh_keys = {'node': node_key, 'elem': elem_key}

        # stock mesh features in mesh object
        self.nodes = nodes
        self.elements = elements
        self.faces = faces
        self.face_of_elem = face_of_elem
        self.face_center = face_center
        self.face_normal = face_normal
        self.node_key = node_key
        self.elem_key = elem_key

        self.nnode = self.nodes.shape[0]
        self.nelem = self.elements.shape[0]
        self.nfaces = self.faces.shape[0]

    #=========================================================================#
    def AngleFilter(self, faces_in_surface, idface, tolerance):

        # initialize list of faces and nodes in the surface
        face_list = [idface]
        node_list = []
        for node in self.faces[idface]:
            node_list.append(node)

        # Define direction of the inital face
        tan1 = self.nodes[self.faces[idface,0]] - \
                self.nodes[self.faces[idface,1]]
        tan2 = self.nodes[self.faces[idface,2]] - \
                self.nodes[self.faces[idface,1]]
        normal0 = np.cross(tan1,tan2)/np.linalg.norm(np.cross(tan1,tan2))
        angle0 = np.arccos(normal0)*180.0/np.pi

        # look for new faces while the list of nodes is not empty
        while node_list:
            # get the faces containing the first node in the list
            node_in_faces = np.where(self.faces==node_list[0])[0]

            # verify if the new faces are in the list of faces
            for iface in node_in_faces:
                # if the face is in, then replace the reference direction
                if iface in face_list:
                    vec1 = self.nodes[self.faces[iface,0]] - \
                            self.nodes[self.faces[iface,1]]
                    vec2 = self.nodes[self.faces[iface,2]] - \
                            self.nodes[self.faces[iface,1]]
                    normal = np.cross(vec1,vec2)/np.linalg.norm(np.cross(vec1,vec2))
                    angle0 = np.arccos(normal)*180.0/np.pi
                # if the face is out, verify is in the same surface
                else:
                    vec1 = self.nodes[self.faces[iface,0]] - self.nodes[self.faces[iface,1]]
                    vec2 = self.nodes[self.faces[iface,2]] - self.nodes[self.faces[iface,1]]
                    normal = np.cross(vec1,vec2)/np.linalg.norm(np.cross(vec1,vec2))
                    angle = np.arccos(normal)*180.0/np.pi
                    if np.allclose(angle,angle0,atol=tolerance) and iface in faces_in_surface:
                        face_list.append(iface)
                        for inode in self.faces[iface]:
                            if not inode in node_list: node_list.append(inode)

            # remove the first node after its analysis
            node_list.pop(0)

        # copy the list of face into a numpy array
        faces_in_surface = np.array(face_list, dtype=np.int64, copy=True)
        faces_in_surface.sort()

        return faces_in_surface

    #=========================================================================#
    def PlaneFilter(self, idface, plane_nodes):

        id_center = np.mean(self.nodes[self.faces[idface]], axis=0)
        if len(plane_nodes.shape) > 1:
            nplanes = plane_nodes.shape[0]
        else:
            nplanes = 1
            plane_nodes = plane_nodes.reshape(1,3)

        plane_ref = np.zeros((nplanes,3),dtype=np.float64)
        plane_normal = np.zeros((nplanes,3),dtype=np.float64)
        # define constraint plane
        for i in range(nplanes):
            ref = self.nodes[plane_nodes[i,0]]
            tan1 = self.nodes[plane_nodes[i,1]] - ref
            tan2 = self.nodes[plane_nodes[i,2]] - ref
            normal = np.cross(tan1,tan2)/\
                np.linalg.norm(np.cross(tan1,tan2))
            if np.dot(normal,id_center-ref) < 0.0:
                normal *= -1.0
            plane_ref[i,:] = ref
            plane_normal[i,:] = normal

        surface_bool = np.zeros((self.faces.shape[0]),dtype=bool)
        # evaluate the face is into the region
        for iface in range(self.faces.shape[0]):
            center = np.mean(self.nodes[self.faces[iface]], axis=0)
            plane_bool = np.zeros((nplanes),dtype=bool)
            for i in range(nplanes):
                projection = np.dot(plane_normal[i,:], center-plane_ref[i,:])
                if projection>0.0: plane_bool[i] = True
            if np.all(plane_bool): surface_bool[iface] = True

        faces_in_surface = np.where(surface_bool)[0]
        return faces_in_surface

    #=========================================================================#
    def ContourSelection(self, points_in_surface, plane_nodes, TOL):

        # define constraint plane
        ref = self.nodes[plane_nodes[0]]
        tan1 = self.nodes[plane_nodes[1]] - ref
        tan2 = self.nodes[plane_nodes[2]] - ref
        normal = np.cross(tan1,tan2)/\
                np.linalg.norm(np.cross(tan1,tan2))

        # evaluate the point is into near the plane
        plane_nset = []
        for ipoin in points_in_surface:
            projection = np.abs(np.dot(normal, self.nodes[ipoin]-ref))
            if projection<TOL: plane_nset.append(ipoin)

        contour_nset = np.array(plane_nset,dtype=np.int64,copy=True)

        return contour_nset

    #=========================================================================#
    def DefineSurfacesAndSets(self, left_planes, right_planes, aorta_nodes=None, mitral_nodes=None):

    #-------------------------------------------------------------------------#
        # left ventricle geometry
        endo_left_faces = self.PlaneFilter(1544, left_planes)
        endo_left_faces = self.AngleFilter(endo_left_faces, 1544, 40.9)
        endo_left_nodes = np.unique(self.faces[endo_left_faces].flatten())
        endo_left_elem = self.face_of_elem[endo_left_faces,:]

        endo_left = {}
        endo_left['s1'] = np.sort(endo_left_elem[np.where(endo_left_elem[:,1]==0)[0],0]) + 1
        endo_left['s2'] = np.sort(endo_left_elem[np.where(endo_left_elem[:,1]==1)[0],0]) + 1
        endo_left['s3'] = np.sort(endo_left_elem[np.where(endo_left_elem[:,1]==2)[0],0]) + 1
        endo_left['s4'] = np.sort(endo_left_elem[np.where(endo_left_elem[:,1]==3)[0],0]) + 1

        aorta_nodes = self.ContourSelection(endo_left_nodes, left_planes[0,:], 1.0)
        mitral_nodes = self.ContourSelection(endo_left_nodes, left_planes[1,:], 2.0)

    #-------------------------------------------------------------------------#
        # right ventricle geometry
        endo_right_faces = self.PlaneFilter(566, right_planes)
        endo_right_faces = self.AngleFilter(endo_right_faces, 566, 30.9)
        endo_right_nodes = np.unique(self.faces[endo_right_faces].flatten())
        endo_right_elem = self.face_of_elem[endo_right_faces]

        endo_right = {}
        endo_right['s1'] = np.sort(endo_right_elem[np.where(endo_right_elem[:,1]==0)[0],0]) + 1
        endo_right['s2'] = np.sort(endo_right_elem[np.where(endo_right_elem[:,1]==1)[0],0]) + 1
        endo_right['s3'] = np.sort(endo_right_elem[np.where(endo_right_elem[:,1]==2)[0],0]) + 1
        endo_right['s4'] = np.sort(endo_right_elem[np.where(endo_right_elem[:,1]==3)[0],0]) + 1

        pulmonary_nodes = self.ContourSelection(endo_right_nodes, right_planes[0,:], 1.0)
        tricuspid_nodes = self.ContourSelection(endo_right_nodes, right_planes[1,:], 2.0)

    #-------------------------------------------------------------------------#
        epi_bool = np.ones((self.faces.shape[0]),dtype=bool)
        for i in range(self.faces.shape[0]):
            if i in endo_left_faces or i in endo_right_faces:
                epi_bool[i] = False
        epi_faces = np.where(epi_bool)[0]
        epi_nodes = np.unique(self.faces[epi_faces].flatten())
        epi_elem = self.face_of_elem[epi_faces]

        epi = {}
        epi['s1'] = np.sort(epi_elem[np.where(epi_elem[:,1]==0)[0],0]) + 1
        epi['s2'] = np.sort(epi_elem[np.where(epi_elem[:,1]==1)[0],0]) + 1
        epi['s3'] = np.sort(epi_elem[np.where(epi_elem[:,1]==2)[0],0]) + 1
        epi['s4'] = np.sort(epi_elem[np.where(epi_elem[:,1]==3)[0],0]) + 1

    #-----------------------------------------------------------------------------#
        self.left_ventricle_faces = endo_left_faces
        self.left_ventricle_nodes = endo_left_nodes
        self.left_ventricle_elem = endo_left_elem
        self.left_ventricle_surface = endo_left

        self.right_ventricle_faces = endo_right_faces
        self.right_ventricle_nodes = endo_right_nodes
        self.right_ventricle_elem = endo_right_elem
        self.right_ventricle_surface = endo_right

        self.epicardium_faces = epi_faces
        self.epicardium_nodes = epi_nodes
        self.epicardium_elem = epi_elem
        self.epicardium_surface = epi

        self.aorta_nodes = aorta_nodes
        self.mitral_nodes = mitral_nodes
        self.pulmonary_nodes = pulmonary_nodes
        self.tricuspid_nodes = tricuspid_nodes

    #=========================================================================#
    def WriteAbaqusInput(self):

        nnodes = mesh.nnode
        nelem = mesh.nelem

        node_key = mesh.node_key
        elem_key = mesh.elem_key

        file_out = open("./heart_healthy.inp", "w")

        # write the heading of abaqus input file
        file_out.write('*Heading\n')
        file_out.write('** Job name: heart, Model name: heart_model\n')
        file_out.write('** Generated by: a python code (JD Laubrie)\n')
        file_out.write('*Preprint, echo=NO, model=NO, history=NO, contact=NO\n')

        #-----------------------------------------------------------------------------#
        # define the heart part of abaqus input file
        file_out.write('**\n')
        file_out.write('** PARTS\n')
        file_out.write('**\n')
        file_out.write('*Part, name=HEART\n')
        file_out.write(node_key)
        for i in range(nnodes):
            file_out.write('{}, {}, {}, {}\n'.format(i+1, mesh.nodes[i,0], 
                                            mesh.nodes[i,1], mesh.nodes[i,2]))
        file_out.write(elem_key)
        mesh.elements += 1
        for i in range(nelem):
            file_out.write('{},{},{},{},{}\n'.format(i+1, mesh.elements[i,0], 
                    mesh.elements[i,1], mesh.elements[i,2], mesh.elements[i,3]))

        #-----------------------------------------------------------------------------#
        # define sets in the heart part
        file_out.write('*Elset, elset=body, generate\n')
        file_out.write(' {}, {}, {}\n'.format(1,nelem,1))

        mesh.epicardium_nodes += 1
        file_out.write('*Nset, nset=nepicardium\n')
        idx = 1
        for i in range(mesh.epicardium_nodes.shape[0]):
            if i == (16*idx-1):
                file_out.write(' {0:4d}\n'.format(mesh.epicardium_nodes[i]))
                idx += 1
            elif i == (mesh.epicardium_nodes.shape[0]-1):
                if np.remainder(i,16) == 0:
                    file_out.write(' {0:4d},\n'.format(mesh.epicardium_nodes[i]))
                else:
                    file_out.write(' {0:4d}\n'.format(mesh.epicardium_nodes[i]))
            else:
                file_out.write(' {0:4d},'.format(mesh.epicardium_nodes[i]))

        mesh.left_ventricle_nodes += 1
        file_out.write('*Nset, nset=nleft_ventricle\n')
        idx = 1
        for i in range(mesh.left_ventricle_nodes.shape[0]):
            if i == (16*idx-1):
                file_out.write(' {0:4d}\n'.format(mesh.left_ventricle_nodes[i]))
                idx += 1
            elif i == (mesh.left_ventricle_nodes.shape[0]-1):
                if np.remainder(i,16) == 0:
                    file_out.write(' {0:4d},\n'.format(mesh.left_ventricle_nodes[i]))
                else:
                    file_out.write(' {0:4d}\n'.format(mesh.left_ventricle_nodes[i]))
            else:
                file_out.write(' {0:4d},'.format(mesh.left_ventricle_nodes[i]))

        mesh.right_ventricle_nodes += 1
        file_out.write('*Nset, nset=nright_ventricle\n')
        idx = 1
        for i in range(mesh.right_ventricle_nodes.shape[0]):
            if i == (16*idx-1):
                file_out.write(' {0:4d}\n'.format(mesh.right_ventricle_nodes[i]))
                idx += 1
            elif i == (mesh.right_ventricle_nodes.shape[0]-1):
                if np.remainder(i,16) == 0:
                    file_out.write(' {0:4d},\n'.format(mesh.right_ventricle_nodes[i]))
                else:
                    file_out.write(' {0:4d}\n'.format(mesh.right_ventricle_nodes[i]))
            else:
                file_out.write(' {0:4d},'.format(mesh.right_ventricle_nodes[i]))

        mesh.aorta_nodes += 1
        file_out.write('*Nset, nset=naorta\n')
        idx = 1
        for i in range(mesh.aorta_nodes.shape[0]):
            if i == (16*idx-1):
                file_out.write(' {0:4d}\n'.format(mesh.aorta_nodes[i]))
                idx += 1
            elif i == (mesh.aorta_nodes.shape[0]-1):
                if np.remainder(i,16) == 0:
                    file_out.write(' {0:4d},\n'.format(mesh.aorta_nodes[i]))
                else:
                    file_out.write(' {0:4d}\n'.format(mesh.aorta_nodes[i]))
            else:
                file_out.write(' {0:4d},'.format(mesh.aorta_nodes[i]))

        mesh.mitral_nodes += 1
        file_out.write('*Nset, nset=nmitral\n')
        idx = 1
        for i in range(mesh.mitral_nodes.shape[0]):
            if i == (16*idx-1):
                file_out.write(' {0:4d}\n'.format(mesh.mitral_nodes[i]))
                idx += 1
            elif i == (mesh.mitral_nodes.shape[0]-1):
                if np.remainder(i,16) == 0:
                    file_out.write(' {0:4d},\n'.format(mesh.mitral_nodes[i]))
                else:
                    file_out.write(' {0:4d}\n'.format(mesh.mitral_nodes[i]))
            else:
                file_out.write(' {0:4d},'.format(mesh.mitral_nodes[i]))

        mesh.pulmonary_nodes += 1
        file_out.write('*Nset, nset=npulmonary\n')
        idx = 1
        for i in range(mesh.pulmonary_nodes.shape[0]):
            if i == (16*idx-1):
                file_out.write(' {0:4d}\n'.format(mesh.pulmonary_nodes[i]))
                idx += 1
            elif i == (mesh.pulmonary_nodes.shape[0]-1):
                if np.remainder(i,16) == 0:
                    file_out.write(' {0:4d},\n'.format(mesh.pulmonary_nodes[i]))
                else:
                    file_out.write(' {0:4d}\n'.format(mesh.pulmonary_nodes[i]))
            else:
                file_out.write(' {0:4d},'.format(mesh.pulmonary_nodes[i]))

        mesh.tricuspid_nodes += 1
        file_out.write('*Nset, nset=ntricuspid\n')
        idx = 1
        for i in range(mesh.tricuspid_nodes.shape[0]):
            if i == (16*idx-1):
                file_out.write(' {0:4d}\n'.format(mesh.tricuspid_nodes[i]))
                idx += 1
            elif i == (mesh.tricuspid_nodes.shape[0]-1):
                if np.remainder(i,16) == 0:
                    file_out.write(' {0:4d},\n'.format(mesh.tricuspid_nodes[i]))
                else:
                    file_out.write(' {0:4d}\n'.format(mesh.tricuspid_nodes[i]))
            else:
                file_out.write(' {0:4d},'.format(mesh.tricuspid_nodes[i]))

        #-----------------------------------------------------------------------------#
        # define element sets for surfaces in the heart part
        for key in mesh.left_ventricle_surface.keys():
            file_out.write('*Elset, elset=left_ventricle_'+key+', internal\n')
            idx = 1
            for i in range(mesh.left_ventricle_surface[key].shape[0]):
                if i == (16*idx-1):
                    file_out.write(' {0:4d}\n'.format(mesh.left_ventricle_surface[key][i]))
                    idx += 1
                elif i == (mesh.left_ventricle_surface[key].shape[0]-1):
                    if np.remainder(i,16) == 0:
                        file_out.write(' {0:4d},\n'.format(mesh.left_ventricle_surface[key][i]))
                    else:
                        file_out.write(' {0:4d}\n'.format(mesh.left_ventricle_surface[key][i]))
                else:
                    file_out.write(' {0:4d},'.format(mesh.left_ventricle_surface[key][i]))

        for key in mesh.right_ventricle_surface.keys():
            file_out.write('*Elset, elset=right_ventricle_'+key+', internal\n')
            idx = 1
            for i in range(mesh.right_ventricle_surface[key].shape[0]):
                if i == (16*idx-1):
                    file_out.write(' {0:4d}\n'.format(mesh.right_ventricle_surface[key][i]))
                    idx += 1
                elif i == (mesh.right_ventricle_surface[key].shape[0]-1):
                    if np.remainder(i,16) == 0:
                        file_out.write(' {0:4d},\n'.format(mesh.right_ventricle_surface[key][i]))
                    else:
                        file_out.write(' {0:4d}\n'.format(mesh.right_ventricle_surface[key][i]))
                else:
                    file_out.write(' {0:4d},'.format(mesh.right_ventricle_surface[key][i]))

        for key in mesh.epicardium_surface.keys():
            file_out.write('*Elset, elset=epicardium_'+key+', internal\n')
            idx = 1
            for i in range(mesh.epicardium_surface[key].shape[0]):
                if i == (16*idx-1):
                    file_out.write(' {0:4d}\n'.format(mesh.epicardium_surface[key][i]))
                    idx += 1
                elif i == (mesh.epicardium_surface[key].shape[0]-1):
                    if np.remainder(i,16) == 0:
                        file_out.write(' {0:4d},\n'.format(mesh.epicardium_surface[key][i]))
                    else:
                        file_out.write(' {0:4d}\n'.format(mesh.epicardium_surface[key][i]))
                else:
                    file_out.write(' {0:4d},'.format(mesh.epicardium_surface[key][i]))

        #-----------------------------------------------------------------------------#
        # define the surfaces in the heart part
        file_out.write('*Surface, type=ELEMENT, name=left_ventricle\n')
        file_out.write('left_ventricle_s1, S1\n')
        file_out.write('left_ventricle_s2, S2\n')
        file_out.write('left_ventricle_s3, S3\n')
        file_out.write('left_ventricle_s4, S4\n')

        file_out.write('*Surface, type=ELEMENT, name=right_ventricle\n')
        file_out.write('right_ventricle_s1, S1\n')
        file_out.write('right_ventricle_s2, S2\n')
        file_out.write('right_ventricle_s3, S3\n')
        file_out.write('right_ventricle_s4, S4\n')

        file_out.write('*Surface, type=ELEMENT, name=epicardium\n')
        file_out.write('epicardium_s1, S1\n')
        file_out.write('epicardium_s2, S2\n')
        file_out.write('epicardium_s3, S3\n')
        file_out.write('epicardium_s4, S4\n')

        #-----------------------------------------------------------------------------#
        # define the section of the myocardium (heart tissue)
        file_out.write('** Section: section_myocardium\n')
        file_out.write('*Solid Section, elset=body, material=myocardio\n')
        file_out.write(',\n')
        file_out.write('*End Part\n')

        #-----------------------------------------------------------------------------#
        # define the assembly of abaqus input file
        file_out.write('**\n')
        file_out.write('** ASSEMBLY\n')
        file_out.write('**\n')
        file_out.write('*Assembly, name=Assembly\n')
        file_out.write('**\n')
        file_out.write('*Instance, name=HEART-1, part=HEART\n')
        file_out.write('*End Instance\n')
        file_out.write('**\n')
        file_out.write('*End Assembly\n')
        file_out.close()

        return

#=============================================================================#
# general setting parameters
stl_mesh = "heart_model1"
from_stl = False

# initialize mesh object
mesh = Mesh()

if from_stl:
    # convert the STL mesh into a GMSH or ABAQUS mesh
    mesh.MeshConvertion(stl_mesh, elem_size=3.0)
else:
    # this routine reads the abaqus mesh generated by pygalmesh
    mesh.ReadAbaqusTestMesh(stl_mesh)

    # I use these lines to seek the reference faces
    print(np.where(mesh.faces==3589))
    print(np.where(mesh.faces==1635))

    # selected nodes
    """
    aorta_nodes = np.array([176,378,656,722,1404,2091,2275,2561,3201,3395,3408,
            3719,3869,4260,4305,4897,7341,8809,9767,10089,11894], dtype=np.int64)
    mitral_nodes = np.array([1186,3733,9889,2256,968,2849,9850,3090,8953,607,1953,
            3798,11884,1755,77,4584,1242,3790,4885,1958,11840,1993,4721,2390,6191,
            3474,11081,4579,2527,9769,4696,4293,11605,2052,9563,803,2976,158], dtype=np.int64)
    aorta_nodes -= 1
    mitral_nodes -= 1
    """

    left_planes = np.array([[819, 1882, 2274],                     #aorta
                            [9888, 3732, 1275]], dtype=np.int64)   #mitral
    right_planes = np.array([[1325, 1675, 9977],                   #pulmonary
                             [8829, 332, 3298]], dtype=np.int64)   #tricuspid
    # this routine define sets and surfaces based in some constraints
    mesh.DefineSurfacesAndSets(left_planes, right_planes)

    # write the abaqus INP file
    mesh.WriteAbaqusInput()

