import pandas as pd
import numpy as np

class ABQMesh(object):

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

    def read_abq_test_mesh(self, mesh_data):

        mesh_file = open(mesh_data, "r")
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
                for i in range(mesh_dict[key] + 1, n_lines):
                    if mesh_lines[i][:1] == '*':
                        break
                    nnodes += 1
            # count elements. 3d elements
            if key[:8] == '*ELEMENT' and key[-5:-1] == 'C3D4':
                elem_key = key
                nelem = 0
                for i in range(mesh_dict[key] + 1, n_lines):
                    if mesh_lines[i][:1] == '*':
                        break
                    nelem += 1
            # count faces. 2d elements
            if key[:8] == '*ELEMENT' and key[-5:-1] == 'R3D3':
                face_key = key
                nfaces = 0
                for i in range(mesh_dict[key] + 1, n_lines):
                    if mesh_lines[i][:1] == '*':
                        break
                    nfaces += 1

        # ----------------------------------------------------------------#
        # get the nodes of abaqus mesh
        df_node = pd.Series(
            mesh_lines[mesh_dict[node_key] + 1:mesh_dict[node_key] + nnodes + 1])
        df_node = df_node.replace(r'\n', '', regex=True)  # remove end-line
        df_node = df_node.str.split(',', expand=True)  # split columns
        nodes = np.array(df_node.values[:, 1:], dtype=np.float64)

        # get the element connectivity
        df_elem = pd.Series(
            mesh_lines[mesh_dict[elem_key] + 1:mesh_dict[elem_key] + nelem + 1])
        df_elem = df_elem.replace(r'\n', '', regex=True)
        df_elem = df_elem.str.split(',', expand=True)
        elements = np.array(df_elem.values[:, 1:], dtype=np.int64) - 1

        # ----------------------------------------------------------------#
        # search the nodes in the surfaces
        # meshIO generates a face and its inverse
        faces_bool = np.zeros((nfaces), dtype=bool)
        for i in range(nfaces):
            if np.remainder(i, 2) == 0: faces_bool[i] = True

        nfaces2 = int(nfaces / 2)

        # read faces connectivity
        df_face = pd.Series(
            mesh_lines[mesh_dict[face_key] + 1:mesh_dict[face_key] + nfaces + 1])
        df_face = df_face.replace(r'\n', '', regex=True)
        df_face = df_face.str.split(',', expand=True)
        faces = np.array(df_face.values[faces_bool, 1:], dtype=np.int64) - 1

        # connect a face with its element and define the surface
        face_of_elem = np.zeros((nfaces2, 2), dtype=np.int64)
        face_normal = np.zeros((nfaces2, 3), dtype=np.float64)
        face_center = np.zeros((nfaces2, 3), dtype=np.float64)
        for face in range(nfaces2):
            face_nodes = faces[face, :]

            # find the element of the face and surface
            elem1, pos1 = np.where(elements == face_nodes[0])
            elem2, pos2 = np.where(elements[elem1] == face_nodes[1])
            elem3, pos3 = np.where(elements[elem1[elem2]] == face_nodes[2])
            face_of_elem[face, 0] = elem1[elem2[elem3]]
            local_node = np.zeros((3), dtype=np.int64)
            local_node[0] = pos1[elem2[elem3]]
            local_node[1] = pos2[elem3]
            local_node[2] = pos3
            local_node.sort()
            if np.all(np.equal(local_node, [0, 1, 2])):
                face_of_elem[face, 1] = 0
            elif np.all(np.equal(local_node, [0, 1, 3])):
                face_of_elem[face, 1] = 1
            elif np.all(np.equal(local_node, [1, 2, 3])):
                face_of_elem[face, 1] = 2
            elif np.all(np.equal(local_node, [0, 2, 3])):
                face_of_elem[face, 1] = 3

            # compute the normal of the face
            tan1 = nodes[face_nodes[0]] - nodes[face_nodes[1]]
            tan2 = nodes[face_nodes[2]] - nodes[face_nodes[1]]
            face_normal[face, :] = np.cross(tan1, tan2) / \
                                   np.linalg.norm(np.cross(tan1, tan2))
            face_center[face, :] = np.mean(nodes[face_nodes], axis=0)

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
