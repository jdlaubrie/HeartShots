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
    def MeshConvertion(self, mesh_name):

        mesh0 = pygalmesh.generate_volume_mesh_from_surface_mesh(
            mesh_name + ".stl",
            min_facet_angle=20.0,
            max_radius_surface_delaunay_ball=3.0,
            max_facet_distance=0.9,
            max_circumradius_edge_ratio=3.0,
            max_cell_circumradius=4.0,             #element size
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
            # count elements
            if key[:8] == '*ELEMENT' and key[-5:-1] == 'C3D4':
                elem_key = key
                nelem = 0
                for i in range(mesh_dict[key]+1,n_lines):
                    if mesh_lines[i][:1] == '*':
                        break
                    nelem += 1
            # count faces
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
    def DefineSurfacesAndSets(self):

        # left ventricle geometry
        left_planes = np.array([[581, 1013, 3407],
                            [3790, 714, 1968]], dtype=np.int64)
        endo_left_faces = self.PlaneFilter(2544, left_planes)
        endo_left_faces = self.AngleFilter(endo_left_faces, 687, 40.9)
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
        right_planes = np.array([[3224, 5143, 6494],
                            [3300, 538, 536]], dtype=np.int64)
        endo_right_faces = self.PlaneFilter(711, right_planes)
        endo_right_faces = self.AngleFilter(endo_right_faces, 711, 44.9)
        endo_right_nodes = np.unique(self.faces[endo_right_faces].flatten())
        endo_right_elem = self.face_of_elem[endo_right_faces]

        endo_right = {}
        endo_right['s1'] = np.sort(endo_right_elem[np.where(endo_right_elem[:,1]==0)[0],0]) + 1
        endo_right['s2'] = np.sort(endo_right_elem[np.where(endo_right_elem[:,1]==1)[0],0]) + 1
        endo_right['s3'] = np.sort(endo_right_elem[np.where(endo_right_elem[:,1]==2)[0],0]) + 1
        endo_right['s4'] = np.sort(endo_right_elem[np.where(endo_right_elem[:,1]==3)[0],0]) + 1

        pulmonary_nodes = self.ContourSelection(endo_right_nodes, right_planes[0,:], 2.0)
        tricuspid_nodes = self.ContourSelection(endo_right_nodes, right_planes[1,:], 3.0)

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

    #base_faces = AngleFilter(nodes, elements, faces, 666, 20.0)
    #base_nodes = np.unique(faces[base_faces].flatten())

    #base_center = np.mean(nodes[base_nodes],axis=0)
    #tan1 = nodes[base_nodes[0]] - base_center
    #tan2 = nodes[base_nodes[1]] - base_center
    #normal = np.cross(tan1,tan2)/np.linalg.norm(np.cross(tan1,tan2))
    #normal_nodes = []
    #for i in range(nodes.shape[0]):
    #    inner = np.dot(nodes[i]-base_center,normal)
    #    angle = np.arccos(inner/np.linalg.norm(nodes[i]-base_center))
    #    if np.abs(angle) < (5.0*np.pi/180.0):
    #        normal_nodes.append(i)
    #normal_nodes = np.array(normal_nodes,dtype=np.int64,copy=True)
    #far_node0 = np.argmax(np.linalg.norm(nodes[normal_nodes] - base_center, axis=1))
    #far_node = normal_nodes[far_node0]


    #base_nodes += 1
    #normal_nodes += 1

    #=========================================================================#
    def OrientationField(self):

        nelem = self.nelem
        nfaces2 = self.nfaces

        endo_left_faces = self.left_ventricle_faces
        endo_right_faces = self.right_ventricle_faces
        epi_faces = self.epicardium_faces

        mitral_center = np.mean(self.nodes[self.mitral_nodes], axis=0)
        vector1 = self.nodes[self.mitral_nodes[0]] - mitral_center
        vector2 = self.nodes[self.mitral_nodes[1]] - mitral_center
        axis_vector = np.cross(vector1,vector2)/\
                np.linalg.norm(np.cross(vector1,vector2))

        face_tan1 = np.zeros((nfaces2,3),dtype=np.float64)
        face_tan2 = np.zeros((nfaces2,3),dtype=np.float64)
        for face in range(nfaces2):
            # compute direction 1 of face
            face_tan1[face,:] = np.cross(axis_vector,self.face_normal[face])/\
                    np.linalg.norm(np.cross(axis_vector,self.face_normal[face]))
            face_tan2[face,:] = np.cross(self.face_normal[face],face_tan1[face])/\
                    np.linalg.norm(np.cross(self.face_normal[face],face_tan1[face]))

        direction1 = np.zeros((nelem,3),dtype=np.float64)
        direction2 = np.zeros((nelem,3),dtype=np.float64)
        elem_angle = np.zeros((nelem),dtype=np.float64)
        for elem in range(nelem):
            # compute the center of the elements
            elem_center = np.mean(self.nodes[self.elements[elem,:]],axis=0)

            # find the closest face in left endocardio
            iface_left = np.argmin(np.linalg.norm(elem_center - \
                    self.face_center[endo_left_faces], axis=1))
            # find the closest face in right endocardio
            iface_right = np.argmin(np.linalg.norm(elem_center - \
                    self.face_center[endo_right_faces], axis=1))
            # find the closest face in epicardio
            iface_epi = np.argmin(np.linalg.norm(elem_center - \
                    self.face_center[epi_faces], axis=1))

            # measure the distance to the face in the respective instance
            elem_to_left = np.linalg.norm(elem_center - \
                    self.face_center[endo_left_faces[iface_left]])
            elem_to_right = np.linalg.norm(elem_center - \
                    self.face_center[endo_right_faces[iface_right]])
            elem_to_epi = np.linalg.norm(elem_center - \
                    self.face_center[epi_faces[iface_epi]])

            # rotate the system depending on element location
            if elem_to_left < elem_to_right:
                # get the tangential directions respect to the left ventricle
                direction1[elem,:] = face_tan1[endo_left_faces[iface_left]]
                direction2[elem,:] = face_tan2[endo_left_faces[iface_left]]

                total_length = elem_to_left + elem_to_epi
                elem_angle[elem] = -120.0*elem_to_left/total_length
            else:
                # get the tangential directions respect to the right ventricle
                direction1[elem,:] = face_tan1[endo_right_faces[iface_right]]
                direction2[elem,:] = face_tan2[endo_right_faces[iface_right]]

                total_length = elem_to_right + elem_to_epi
                elem_angle[elem] = -120.0*elem_to_right/total_length

        self.rotation_angle = elem_angle
        self.orientation1 = direction1
        self.orientation2 = direction2

#=============================================================================#
def DefineInjectionPoints(mesh):

    nnodes = mesh.nnode

    endo_left_nodes = mesh.left_ventricle_nodes
    endo_left_faces = mesh.left_ventricle_faces
    face_center = mesh.face_center
    face_normal = mesh.face_normal

    mitral_center = np.mean(mesh.nodes[mesh.mitral_nodes], axis=0)
    vector1 = mesh.nodes[mesh.mitral_nodes[0]] - mitral_center
    vector2 = mesh.nodes[mesh.mitral_nodes[1]] - mitral_center
    axis_vector = np.cross(vector1,vector2)/\
                np.linalg.norm(np.cross(vector1,vector2))

    injection_center = []
    left_center = np.mean(mesh.nodes[endo_left_nodes],axis=0)
    for face in endo_left_faces:
        add_point = False

        projection = np.dot(face_center[face,:] - left_center,axis_vector)
        if projection>-45.0 and projection<15.0:
            if len(injection_center)==0:
                injection_center.append(face)
            else:
                for i in injection_center:
                    distance = np.linalg.norm(face_center[face,:] - \
                                face_center[i,:])
                    if distance>11.0:
                        add_point = True
                    else:
                        add_point = False
                        break
                if add_point:
                    injection_center.append(face)

        if len(injection_center)==10: break

    injection_center = np.array(injection_center, dtype=np.int64, copy=True)

    injection = []
    c_index = []
    for node in range(nnodes):
        ref_length = 100.0
        for i in injection_center:
            central_point = face_center[i] - 5.0*face_normal[i]
            radius = np.linalg.norm(mesh.nodes[node] - central_point)
            if radius < ref_length: ref_length = radius
        if ref_length < 5.0:
            injection.append(node)
            c_index.append(1.0)
        elif ref_length < 10.0 and ref_length > 5.0:
            injection.append(node)
            c_index.append(-0.2*ref_length + 2.0)

    injection = np.array(injection, dtype=np.int64, copy=True)
    c_index = np.array(c_index, dtype=np.float64, copy=True)

    return injection, c_index

#=============================================================================#
def WriteAbaqusInput(mesh, injection_zone=None, c_index=None):

    nnodes = mesh.nnode
    nelem = mesh.nelem

    node_key = mesh.node_key
    elem_key = mesh.elem_key

    if np.all(injection_zone) and np.all(c_index):
        injection = True
    else:
        injection = False

    # writing the abaqus input file
    if not injection:
        file_out = open("heart.inp", "w")
    else:
        file_out = open("heart_inj.inp", "w")

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

    #file_out.write('*Nset, nset=base\n')
    #idx = 1
    #for i in range(base_nodes.shape[0]):
    #    if i == (16*idx-1):
    #        file_out.write(' {0:4d}\n'.format(base_nodes[i]))
    #        idx += 1
    #    elif i == (base_nodes.shape[0]-1):
    #        if np.remainder(i,16) == 0:
    #            file_out.write(' {0:4d},\n'.format(base_nodes[i]))
    #        else:
    #            file_out.write(' {0:4d}\n'.format(base_nodes[i]))
    #    else:
    #        file_out.write(' {0:4d},'.format(base_nodes[i]))

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
    # define the distribution of local orientations by element
    file_out.write('*Distribution, name=elemdir, location=ELEMENT, Table=elemdir_table\n')
    file_out.write('** Description: Generated by python (JD Laubrie)\n')
    file_out.write(' , 1.0, 0.0, 0.0, 0.0, 1.0, 0.0\n')
    for elem in range(nelem):
        file_out.write('{}, '.format(elem+1))
        file_out.write('{}, '.format(mesh.orientation1[elem,0]))
        file_out.write('{}, '.format(mesh.orientation1[elem,1]))
        file_out.write('{}, '.format(mesh.orientation1[elem,2]))
        file_out.write('{}, '.format(mesh.orientation2[elem,0]))
        file_out.write('{}, '.format(mesh.orientation2[elem,1]))
        file_out.write('{}\n'.format(mesh.orientation2[elem,2]))

    # define the distribution of local system rotation by element
    file_out.write('*Distribution, name=elemangle, location=ELEMENT, Table=elemangle_table\n')
    file_out.write('** Description: Generated by python (JD Laubrie)\n')
    file_out.write(' , 0.0\n')
    for elem in range(nelem):
        file_out.write('{}, '.format(elem+1))
        file_out.write('{}\n'.format(mesh.rotation_angle[elem]))
    file_out.write('*Orientation, name=myofibre, system=RECTANGULAR\n')
    file_out.write(' elemdir\n')
    #file_out.write(' , 1.0, 0.0, 0.0, 0.0, 1.0, 0.0\n')
    file_out.write(' 3, elemangle\n')
    #file_out.write(' 3, 0.0\n')

    #-----------------------------------------------------------------------------#
    # define the section of the myocardium (heart tissue)
    file_out.write('** Section: section_myocardium\n')
    file_out.write('*Solid Section, elset=body, orientation=myofibre, material=myocardio\n')
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

    #-----------------------------------------------------------------------------#
    # lines needed for the definition of the orientation distributions
    file_out.write('*Distribution Table, name=elemdir_table\n')
    file_out.write(' coord3D, coord3D\n')
    file_out.write('*Distribution Table, name=elemangle_table\n')
    file_out.write(' angle\n')

    #-----------------------------------------------------------------------------#
    # define signal function for the load
    file_out.write('*Amplitude, name=load_amp\n')
    file_out.write(' 0.0, 0.0, 1.0, 1.0\n')

    #-----------------------------------------------------------------------------#
    # define material parameters
    file_out.write('**\n')
    file_out.write('** MATERIALS\n')
    file_out.write('**\n')
    file_out.write('*Include, input=parameter0.inp\n')
    file_out.write('*Material, name=myocardio\n')
    file_out.write('*Density\n')
    file_out.write(' <density>,\n')
    file_out.write('*User material, constants=11\n')
    file_out.write(' <D>, <a_iso>, <b_iso>, <a_f>, <b_f>, <phi_f>, <a_s>, <b_s>\n')
    file_out.write(' <phi_s>, <a_fs>, <b_fs>\n')
    file_out.write('*Depvar\n')
    file_out.write(' 3,\n')
    #file_out.write('*Hyperelastic, neo hooke\n')
    #file_out.write(' 0.05, 0.50\n')

    #-----------------------------------------------------------------------------#
    # define predefined fields, this case the injections
    if injection:
        file_out.write('**\n')
        file_out.write('** PREDEFINED FIELDS\n')
        file_out.write('**\n')
        file_out.write('** Name: C_index Type= Field\n')
        file_out.write('*Initial Conditions, type=FIELD, variable=1\n')
        for i in range(injection_zone.shape[0]):
            file_out.write('HEART-1.{}, {}\n'.format(injection_zone[i],c_index[i]))

    #-----------------------------------------------------------------------------#
    # define the load step
    file_out.write('** ----------------------------------------------------------------\n')
    file_out.write('** \n')
    file_out.write('** STEP: Step-1\n')
    file_out.write('** \n')
    file_out.write('*Step, name=Step-1, nlgeom=YES\n')
    file_out.write('*Static\n')
    file_out.write('0.05, 1.0, 1e-05, 1.0\n')
    file_out.write('** \n')
    file_out.write('** BOUNDARY CONDITIONS\n')
    file_out.write('** \n')
    file_out.write('** Name: fix_bound Type: Symmetry/Antisymmetry/Encastre\n')
    file_out.write('*Boundary\n')
    file_out.write('HEART-1.npulmonary, ENCASTRE\n')
    file_out.write('*Boundary\n')
    file_out.write('HEART-1.ntricuspid, ENCASTRE\n')
    file_out.write('*Boundary\n')
    file_out.write('HEART-1.naorta, ENCASTRE\n')
    file_out.write('** \n')
    file_out.write('** LOADS\n')
    file_out.write('** \n')
    file_out.write('** Name: pressure   Type: Pressure\n')
    file_out.write('*Dsload, amplitude=load_amp\n')
    file_out.write('HEART-1.left_ventricle, P, 0.004\n')
    file_out.write('** \n')
    file_out.write('** OUTPUT REQUESTS\n')
    file_out.write('** \n')
    file_out.write('*Restart, write, frequency=0\n')
    file_out.write('** \n')
    file_out.write('** FIELD OUTPUT: F-Output-1\n')
    file_out.write('** \n')
    file_out.write('*Output, field, number interval=20\n')
    file_out.write('*Node Output\n')
    file_out.write(' CF, RF, U\n')
    file_out.write('*Element Output, direction=YES\n')
    file_out.write(' FV, LE, S\n')
    file_out.write('** \n')
    file_out.write('** HISTORY OUTPUT: H-Output-1\n')
    file_out.write('** \n')
    file_out.write('*Output, history, variable=PRESELECT, number interval=20\n')
    file_out.write('*End Step\n')

    #-----------------------------------------------------------------------------#
    file_out.close()

#=============================================================================#
stl_mesh = "heart_model1"
injection = False

mesh = Mesh()
#mesh.MeshConvertion(stl_mesh)

# this routine reads the abaqus mesh generated by pygalmesh
mesh.ReadAbaqusTestMesh(stl_mesh)

# I use these lines to seek the reference faces
#print(np.where(faces==1519))
#print(np.where(faces==2675))
#print(np.where(faces==127))

# this routine define sets and surfaces based in some constraints
mesh.DefineSurfacesAndSets()

mesh.OrientationField()

injection_zone = None
c_index = None
if injection:
    injection_zone, c_index = DefineInjectionPoints(mesh)
    injection_zone += 1

    if not np.all(injection_zone) or not np.all(c_index):
        raise ValueError("Injection parameters are still void despite the computations.")

WriteAbaqusInput(mesh, injection_zone=injection_zone, c_index=c_index)



#os.system("gmsh out.msh")
