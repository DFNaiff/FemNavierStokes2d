import itertools
import collections

import numpy as np

import scipy.sparse


class Mini():
    def __init__(self):
        """Initialize Mini elements, composed of integrals on the unit triangle {0, e_1, e_2}"""
        pass

    def unit_forcing_vector(self):
        """
        Generates the forcing vector integral int_{T0} \phi_0^j dV.
        
        Returns
        -------
        np.ndarray
            The unit forcing vector.
        """
        #The int_{T0} \phi_0^j dV
        return np.array([[0.0916666666666668], [0.0916666666666668], [0.0916666666666668], [9/40]]).flatten()

    def unit_mass_matrix(self):
        """
        Generate the unit mass matrix int_{T0} \phi_0^i \phi_0^j dV.

        Returns
        -------
        np.ndarray
            The unit mass matrix.
        """
        #The matrix int_{T0} \phi_0^i \phi_0^j dV
        return np.array([[0.0494047619047588, 0.00773809523809499, 0.00773809523809632, 0.0267857142857135], [0.00773809523809499, 0.049404761904762, 0.00773809523809499, 0.0267857142857135], [0.00773809523809632, 0.00773809523809499, 0.0494047619047588, 0.0267857142857135], [0.0267857142857135, 0.0267857142857135, 0.0267857142857135, 81/560]])

    def unit_stiffness_matrix(self, K):
        """
        Generate the unit stiffness matrix int_{T0} \nabla \phi_0^i \cdot K \nabla phi_0^j dV.

        Parameters
        ----------
        K : np.ndarray
            The stiffness tensor.

        Returns
        -------
        np.ndarray
            The unit stiffness matrix.
        """
        k11, k12, k21, k22 = K[0, 0], K[0, 1], K[1, 0], K[1, 1]
        return np.array([[0.950000000000002*k11 + 0.725*k12 + 0.725*k21 + 0.949999999999999*k22, -0.049999999999998*k11 + 0.225*k12 - 0.275*k21 + 0.449999999999999*k22, 0.450000000000002*k11 - 0.275*k12 + 0.225*k21 - 0.0500000000000007*k22, -1.35000000000001*k11 - 0.675000000000001*k12 - 0.675000000000001*k21 - 1.35*k22], [-0.049999999999998*k11 - 0.275*k12 + 0.225*k21 + 0.449999999999999*k22, 0.950000000000002*k11 + 0.225*k12 + 0.225*k21 + 0.449999999999999*k22, 0.450000000000002*k11 + 0.725*k12 + 0.225*k21 + 0.449999999999999*k22, -1.35*k11 - 0.675000000000001*k12 - 0.675000000000001*k21 - 1.35*k22], [0.450000000000002*k11 + 0.225*k12 - 0.275*k21 - 0.0500000000000007*k22, 0.450000000000002*k11 + 0.225*k12 + 0.725*k21 + 0.449999999999999*k22, 0.450000000000002*k11 + 0.225*k12 + 0.225*k21 + 0.949999999999999*k22, -1.35000000000001*k11 - 0.675000000000001*k12 - 0.675000000000001*k21 - 1.35*k22], [-1.35000000000001*k11 - 0.675000000000001*k12 - 0.675000000000001*k21 - 1.35*k22, -1.35*k11 - 0.675000000000001*k12 - 0.675000000000001*k21 - 1.35*k22, -1.35000000000001*k11 - 0.675000000000001*k12 - 0.675000000000001*k21 - 1.35*k22, (81/20)*k11 + (81/40)*k12 + (81/40)*k21 + (81/20)*k22]])

    def unit_convection_matrix(self, v):
        #The matrix int_{T0} phi_0^i (\nabla phi_0^j \cdot v) dV
        v1, v2 = v[0], v[1]
        #For dumb reasons, I computed first the matrix -int_{T0} phi_0^j (\nabla phi_0^i \cdot v) dV
        A = np.array([[0.166666666666666*v1 + 0.166666666666667*v2, 0.0166666666666675*v1 + 0.0916666666666666*v2, 0.0916666666666668*v1 + 0.0166666666666666*v2, 0.224999999999994*v1 + 0.225*v2], [-0.0166666666666675*v1 + 0.075*v2, -0.166666666666668*v1, -0.0916666666666668*v1 - 0.075*v2, -0.225000000000001*v1], [0.0750000000000011*v1 - 0.0166666666666666*v2, -0.0749999999999993*v1 - 0.0916666666666666*v2, -0.166666666666667*v2, -0.225*v2], [-0.225000000000001*v1 - 0.225*v2, 0.225000000000001*v1, 0.225*v2, 0]])
        #So I'm transposing and negating it to make it correct 
        return -A.T

    def unit_velocity_pressure_convection_matrix(self, v):
        v1, v2 = v[0], v[1]
        return np.array([[0.241666666666666*v1 + 0.241666666666667*v2, 0.0916666666666668*v1 + 0.166666666666667*v2, 0.166666666666666*v1 + 0.0916666666666666*v2], [-0.091666666666667*v1 + 0.075*v2, -0.241666666666667*v1, -0.166666666666667*v1 - 0.075*v2], [0.075*v1 - 0.0916666666666666*v2, -0.0750000000000002*v1 - 0.166666666666667*v2, -0.241666666666667*v2], [-9/40*v1 - 9/40*v2, (9/40)*v1, (9/40)*v2]])
    
    def unit_pressure_velocity_convection_matrix(self, v):
        v1, v2 = v[0], v[1]
        return np.array([[0.0916666666666666*v1 + 0.0916666666666666*v2, 0.0916666666666666*v1 + 0.0916666666666666*v2, 0.0916666666666666*v1 + 0.0916666666666666*v2, (9/40)*v1 + (9/40)*v2], [-0.0916666666666666*v1, -0.0916666666666666*v1, -0.0916666666666666*v1, -9/40*v1], [-0.0916666666666666*v2, -0.0916666666666666*v2, -0.0916666666666666*v2, -9/40*v2]])

class Linear():
    def __init__(self):
        pass

    def unit_forcing_vector(self):
        #The int_{T0} \phi_0^j dV
        return np.array([[1/6], [1/6], [1/6]]).flatten()

    def unit_line_forcing_vector(self):
        #The int_{T0} \phi_0^j dV
        return np.array([[1/2], [1/2]]).flatten()

    def unit_mass_matrix(self):
        #The matrix int_{T0} \phi_0^i \phi_0^j dV
        return np.array([[1/12, 1/24, 1/24], [1/24, 1/12, 1/24], [1/24, 1/24, 1/12]])

    def unit_stiffness_matrix(self, K):
        #The matrix int_{T0} \nabla \phi_0^i \cdot K \nabla phi_0^j dV
        k11, k12, k21, k22 = K[0, 0], K[0, 1], K[1, 0], K[1, 1]
        return np.array([[(1/2)*k11 + (1/2)*k12 + (1/2)*k21 + (1/2)*k22, -1/2*k11 - 1/2*k21, -1/2*k12 - 1/2*k22], [-1/2*k11 - 1/2*k12, (1/2)*k11, (1/2)*k12], [-1/2*k21 - 1/2*k22, (1/2)*k21, (1/2)*k22]])

    def unit_convection_matrix(self, v):
        #The matrix -int_{T0} phi_0^j (\nabla phi_0^i \cdot v) dV
        v1, v2 = v[0], v[1]
        #For dumb reasons, I computed first the matrix -int_{T0} phi_0^j (\nabla phi_0^i \cdot v) dV
        A = np.array([[(1/6)*v1 + (1/6)*v2, (1/6)*v1 + (1/6)*v2, (1/6)*v1 + (1/6)*v2], [-1/6*v1, -1/6*v1, -1/6*v1], [-1/6*v2, -1/6*v2, -1/6*v2]])
        return -A.T

    
class MiniAssembler():
    def __init__(self, mesh):
        self.mesh = mesh
        self.mini = Mini()
        self.linear = Linear()
        self.main_element = "mini"
        self.has_boundary_normals = False

    def set_main_element(self, element):
        assert element in ["mini", "linear"]
        self.main_element = element

    def set_line_elements(self):
        line_elements_counter = collections.defaultdict(lambda : list())
        for j, triangle in enumerate(self.elements):
            for i in range(3):
                # Extract the vertices of each edge and sort them to ensure uniqueness
                edge = tuple(sorted([triangle[i], triangle[(i + 1) % 3]]))
                line_elements_counter[edge].append(j)
        line_elements = np.vstack(list(line_elements_counter.keys()))
        self.line_elements = line_elements
        self.line_elements_to_elements = line_elements_counter
        self.line_lengths = np.sqrt(np.sum((self.points[line_elements][:, 0, :] - 
                                            self.points[line_elements][:, 1, :])**2,
                                           axis=-1))


    def set_boundary_vectors(self):
        raise NotImplementedError
        boundary_centroids = np.mean(self.points[self.boundary_elements], axis=1)
        boundary_inner_point_map = np.empty(shape=self.boundary_elements.shape[0], dtype=int)
        boundary_element_map = np.empty(shape=self.boundary_elements.shape[0], dtype=int)
        for i, b in enumerate(self.boundary_elements):
            for j, e in enumerate(self.elements):
                diff = np.setdiff1d(e, b)
                if len(diff) == 1:
                    boundary_inner_point_map[i] = diff[0]
                    boundary_element_map[i] = j
                    break
        boundary_points = self.points[self.boundary_elements, :-1]
        boundary_tangent_vector = boundary_points[:, 1, :] - boundary_points[:, 0, :]
        boundary_normal_vector = np.einsum('ij, ...j -> ...i', np.array([[0, -1], [1, 0]]), boundary_tangent_vector)
        inner_point = self.points[boundary_inner_point_map][..., :-1]
        revert = ((inner_point - boundary_centroids[:, :-1])*boundary_normal_vector).sum(axis=-1) > 0
        boundary_normal_vector[revert] *= -1
        boundary_lengths = np.linalg.norm(boundary_normal_vector, axis=-1) #Since is a rotation, same length as tangent
        boundary_normal_vector /= (boundary_lengths[..., None] + 1e-6)
        self.boundary_centroids = boundary_centroids
        self.boundary_element_map = boundary_element_map
        self.boundary_normals = boundary_normal_vector
        self.boundary_lengths = boundary_lengths
        self.has_boundary_normals = True


    def element_translation_matrix_and_vector(self, i):
        element_nodes = self.mesh.cells_dict["triangle"][i]
        node_coordinates = self.mesh.points[element_nodes, :]
        node_coordinates = node_coordinates[:, :-1]
        x, y, z = node_coordinates[0], node_coordinates[1], node_coordinates[2]
        A = np.vstack([y-x,z-x]).T
        b = x
        return A, b

    def triangle_element_area(self, i):
        A, b = self.element_translation_matrix_and_vector(i)
        nodedet = np.linalg.det(A)
        return nodedet/2

    def element_forcing_vector(self, i, forcing):
        nodes = self.mesh.cells_dict["triangle"]
        A, b = self.element_translation_matrix_and_vector(i)
        node_det = np.linalg.det(A)
        base = self.belel.unit_forcing_vector()
        element_forcing = forcing*node_det*base
        return element_forcing

    def line_element_forcing_vector(self, i, forcing):
        base = self.belel.unit_line_forcing_vector()
        node_det = self.line_lengths[i]
        element_forcing = forcing*node_det*base
        return element_forcing

    def element_mass_matrix(self, i, mass=1.0, unitfunction=None):
        A, b = self.element_translation_matrix_and_vector(i)
        node_det = np.linalg.det(A)
        base = self.belel.unit_mass_matrix()
        M = node_det*base*mass
        return M

    def element_convection_matrix(self, i, u, unitfunction=None):
        A, b = self.element_translation_matrix_and_vector(i)
        node_det = np.linalg.det(A)
        v = np.linalg.solve(A, u)
        base = self.belel.unit_convection_matrix(v)
        M = node_det*base
        return M

    def element_stiffness_matrix(self, i,
                                stiffness=1.0, unitfunction=None):
        A, b = self.element_translation_matrix_and_vector(i)
        Ainv = np.linalg.inv(A)
        if np.isscalar(stiffness):
            D = stiffness*np.eye(2)
        else:
            D = stiffness
        K = Ainv@(D@(Ainv.T))
        node_det = np.linalg.det(A)
        base = self.belel.unit_stiffness_matrix(K)
        M = node_det*base
        return M

    def element_velocity_pressure_convection_matrix(self, i, u):
        if self.main_element != "mini":
            raise ValueError
        A, b = self.element_translation_matrix_and_vector(i)
        node_det = np.linalg.det(A)
        v = np.linalg.solve(A, u)
        M = node_det*self.mini.unit_velocity_pressure_convection_matrix(v)
        return M

    def element_pressure_velocity_convection_matrix(self, i, u):
        if self.main_element != "mini":
            raise ValueError
        A, b = self.element_translation_matrix_and_vector(i)
        node_det = np.linalg.det(A)
        v = np.linalg.solve(A, u)
        M = node_det*self.mini.unit_pressure_velocity_convection_matrix(v)
        return M

    def mass_matrix(self, mass_fn=1.0, mass_fn_type='function'):
        nodes = self.mesh.points
        elements = self.mesh.cells_dict["triangle"]
        num_nodes = nodes.shape[0]
        num_elements = elements.shape[0]
        # M = np.zeros((num_nodes + num_elements, num_nodes + num_elements))
        
        M = scipy.sparse.dok_matrix((self.nvariables, self.nvariables))
        for i, nodes in enumerate(self.mesh.cells_dict["triangle"]):
            if not callable(mass_fn):
                mass_value = mass_fn
            else:
                if mass_fn_type == 'function':
                    nodepoints = self.mesh.points[nodes, :]
                    centroid = nodepoints.mean(axis=0)
                    mass_value = mass_fn(centroid)
                elif mass_fn_type == 'element':
                    mass_value = mass_fn(i)
            element_mass = self.element_mass_matrix(i, mass_value)
            element_index = num_nodes + i
            if self.main_element == "mini":
                element_index = num_nodes + i
                extended_nodes = np.hstack([nodes, [element_index]])
            else:
                extended_nodes = nodes
            for ab, enodeab in enumerate_product(extended_nodes, 2):
                a, b = ab
                enodea, enodeb = enodeab
                M[enodea, enodeb] += element_mass[a, b]
        return M

    def convection_matrix(self, velocity_fn, velocity_fn_type='function'):
        num_nodes = self.mesh.points.shape[0]
        num_elements = len(self.mesh.cells_dict["triangle"])
        # G = np.zeros((num_nodes + num_elements, num_nodes + num_elements))
        G = scipy.sparse.dok_matrix((self.nvariables, self.nvariables))
        for i, nodes in enumerate(self.mesh.cells_dict["triangle"]):
            if not callable(velocity_fn):
                velocity = velocity_fn
            else:
                if velocity_fn_type == 'function':
                    nodepoints = self.mesh.points[nodes, :]
                    centroid = nodepoints.mean(axis=0)
                    velocity = velocity_fn(centroid)
                elif velocity_fn_type == 'element':
                    velocity = velocity_fn(i)
            element_convection = self.element_convection_matrix(i, velocity)
            if self.main_element == "mini":
                element_index = num_nodes + i
                extended_nodes = np.hstack([nodes, [element_index]])
            else:
                extended_nodes = nodes
            for ab, enodeab in enumerate_product(extended_nodes, 2):
                a, b = ab
                enodea, enodeb = enodeab
                G[enodea, enodeb] += element_convection[a, b]
        return G

    def stiffness_matrix(self, stiffness_fn=1.0, stiffness_fn_type='function'):
        num_nodes = self.mesh.points.shape[0]
        num_elements = len(self.mesh.cells_dict["triangle"])
        # K = np.zeros((num_nodes + num_elements, num_nodes + num_elements))
        K = scipy.sparse.dok_matrix((self.nvariables, self.nvariables))
        for i, nodes in enumerate(self.mesh.cells_dict["triangle"]):
            if not callable(stiffness_fn):
                stiffness = stiffness_fn
            else:
                if stiffness_fn_type == 'function':
                    nodepoints = self.mesh.points[nodes, :]
                    centroid = nodepoints.mean(axis=0)
                    stiffness = stiffness_fn(centroid)
                elif stiffness_fn_type == 'element':
                    stiffness = stiffness_fn(i)
            element_stiffness = self.element_stiffness_matrix(i, stiffness)
            if self.main_element == "mini":
                element_index = num_nodes + i
                extended_nodes = np.hstack([nodes, [element_index]])
            else:
                extended_nodes = nodes
            for ab, enodeab in enumerate_product(extended_nodes, 2):
                a, b = ab
                enodea, enodeb = enodeab
                K[enodea, enodeb] += element_stiffness[a, b]
        return K

    def velocity_pressure_convection_matrix(self, velocity_fn, velocity_fn_type='function'):
        num_nodes = self.mesh.points.shape[0]
        num_elements = len(self.mesh.cells_dict["triangle"])
        # G = np.zeros((num_nodes + num_elements, num_nodes))
        G = scipy.sparse.dok_matrix((num_nodes + num_elements, num_nodes))
        for i, nodes in enumerate(self.mesh.cells_dict["triangle"]):
            if not callable(velocity_fn):
                velocity = velocity_fn
            else:
                if velocity_fn_type == 'function':
                    nodepoints = self.mesh.points[nodes, :]
                    centroid = nodepoints.mean(axis=0)
                    velocity = velocity_fn(centroid)
                elif velocity_fn_type == 'element':
                    velocity = velocity_fn(i)
            element_convection = self.element_velocity_pressure_convection_matrix(i, velocity)
            element_index = num_nodes + i
            extended_nodes = np.hstack([nodes, [element_index]])
            for a, enodea in enumerate(extended_nodes):
                for b, nodeb in enumerate(nodes):
                    G[enodea, nodeb] += element_convection[a, b]
        return G

    def pressure_velocity_convection_matrix(self, velocity_fn, velocity_fn_type='function'):
        num_nodes = self.mesh.points.shape[0]
        num_elements = len(self.mesh.cells_dict["triangle"])
        # G = np.zeros((num_nodes, num_nodes + num_elements))
        G = scipy.sparse.dok_matrix((num_nodes, num_nodes + num_elements))
        for i, nodes in enumerate(self.mesh.cells_dict["triangle"]):
            if not callable(velocity_fn):
                velocity = velocity_fn
            else:
                if velocity_fn_type == 'function':
                    nodepoints = self.mesh.points[nodes, :]
                    centroid = nodepoints.mean(axis=0)
                    velocity = velocity_fn(centroid)
                elif velocity_fn_type == 'element':
                    velocity = velocity_fn(i)
            element_convection = self.element_pressure_velocity_convection_matrix(i, velocity)
            element_index = num_nodes + i
            extended_nodes = np.hstack([nodes, [element_index]])
            for a, nodea in enumerate(nodes):
                for b, enodeb in enumerate(extended_nodes):
                    G[nodea, enodeb] += element_convection[a, b]
        return G
        
    def forcing_vector(self, forcing_fn=None, forcing_fn_type='function'):
        num_nodes = self.mesh.points.shape[0]
        num_elements = len(self.mesh.cells_dict["triangle"])
        fvec = np.zeros(self.nvariables)
        for i, nodes in enumerate(self.mesh.cells_dict["triangle"]):
            if not callable(forcing_fn):
                if forcing_fn is None:
                    forcing = 1.0
                else:
                    forcing = forcing_fn
            else:
                if forcing_fn_type == 'function':
                    nodepoints = self.mesh.points[nodes, :]
                    centroid = nodepoints.mean(axis=0)
                    forcing = forcing_fn(centroid)
                elif forcing_fn_type == 'element':
                    forcing = forcing_fn(i)
            element_forcing = self.element_forcing_vector(i, forcing)
            if self.main_element == "mini":
                element_index = num_nodes + i
                extended_nodes = np.hstack([nodes, [element_index]])
            else:
                extended_nodes = nodes
            for i, enode in enumerate(extended_nodes):
                fvec[enode] += element_forcing[i]
        return fvec
    
    def line_forcing_vector(self, forcing_fn=None, forcing_fn_type='function'):
        num_nodes = self.mesh.points.shape[0]
        num_line_elements = len(self.line_elements)
        fvec = np.zeros(self.nvariables)
        for i, nodes in enumerate(self.line_elements):
            if not callable(forcing_fn):
                if forcing_fn is None:
                    forcing = 1.0
                else:
                    forcing = forcing_fn
            else:
                if forcing_fn_type == 'function':
                    nodepoints = self.mesh.points[nodes, :]
                    centroid = nodepoints.mean(axis=0)
                    forcing = forcing_fn(centroid)
                elif forcing_fn_type == 'element':
                    forcing = forcing_fn(i)
            element_forcing = self.line_element_forcing_vector(i, forcing)
            if self.main_element == "mini":
                raise NotImplementedError
                element_index = num_nodes + i
                extended_nodes = np.hstack([nodes, [element_index]])
            else:
                extended_nodes = nodes
            for i, enode in enumerate(extended_nodes):
                fvec[enode] += element_forcing[i]
        return fvec

    def apply_dirichlet_to_matrix(self, markers, matrix, where='u',
                                  data_dict_name="gmsh:physical",
                                  excluded_markers = []):
        mesh = self.mesh
        npoints = mesh.points.shape[0]
        boundary_elements = mesh.cells_dict["line"]
        boundary_markers = mesh.cell_data_dict[data_dict_name]["line"]
        leap = ['u', 'v', 'p'].index(where)
        if len(excluded_markers) > 0:
            excluded_nodes = []
            for excluded_marker in excluded_markers:
                excluded_nodes_for_marker = boundary_elements[boundary_markers == excluded_marker, :].flatten()
                excluded_nodes.append(excluded_nodes_for_marker)
            excluded_nodes = np.hstack(excluded_nodes)
        else:
            excluded_nodes = np.zeros([0], dtype=int)
        boundary_nodes = []
        for marker in markers:
            boundary_nodes_for_marker = boundary_elements[boundary_markers == marker, :].flatten()
            boundary_nodes.append(boundary_nodes_for_marker)
        boundary_nodes = np.unique(np.hstack(boundary_nodes))
        if len(excluded_nodes) > 0:
            boundary_nodes = np.setdiff1d(boundary_nodes, excluded_nodes)
        boundary_nodes += leap*(self.npoints + self.nelements)
        matrix[boundary_nodes, :] = 0  # Zeroing the row
        matrix[boundary_nodes, boundary_nodes] = 1
        return matrix

    def apply_dirichlet_to_vector(self, values, markers, vector, where='u',
                                  data_dict_name="gmsh:physical",
                                  excluded_markers = []):
        mesh = self.mesh
        npoints = mesh.points.shape[0]
        boundary_elements = mesh.cells_dict["line"]
        boundary_markers = mesh.cell_data_dict[data_dict_name]["line"]
        leap = ['u', 'v', 'p'].index(where)
        if len(excluded_markers) > 0:
            excluded_nodes = []
            for excluded_marker in excluded_markers:
                excluded_nodes_for_marker = boundary_elements[boundary_markers == excluded_marker, :].flatten()
                excluded_nodes.append(excluded_nodes_for_marker)
            excluded_nodes = np.hstack(excluded_nodes)
        else:
            excluded_nodes = np.zeros([0], dtype=int)
        for value, marker in zip(values, markers):
            boundary_nodes = np.unique(boundary_elements[boundary_markers == marker, :].flatten())
            boundary_nodes = np.setdiff1d(boundary_nodes, excluded_nodes)
            if callable(value):
                val = value(self.points[boundary_nodes, :])
            else:
                val = value
            boundary_nodes += leap*(self.npoints + self.nelements)
            vector[boundary_nodes] = val
        return vector

    def get_linear_element_equation(self, values):
        #ax + by + c
        mesh = self.mesh
        connectivity = mesh.cells_dict["triangle"]
        element_points = mesh.points[connectivity, :-1]
        element_values = values[connectivity]
        element_matrix = np.concatenate([element_points,
                                        np.ones(element_points.shape[:-1])[..., None]],
                                        axis=-1)
        element_coefficients = np.linalg.solve(element_matrix, element_values)
        return element_coefficients

    def get_centroid_values_from_linear(self, values):
        element_coefficients = self.get_linear_element_equation(values)
        centroids = self.centroids
        centroids[:, -1] = 1.0
        return (element_coefficients*centroids).sum(axis=-1)

    def split_velocities(self, uxyp):
        n = self.npoints + self.nelements
        ux = uxyp[..., :n]
        uy = uxyp[..., n:2*n]
        p = uxyp[..., 2*n:]
        return ux, uy, p

    def get_element_velocities(self, uxyp):
        n = self.npoints + self.nelements
        ns = self.npoints
        ux = uxyp[..., :n]
        uy = uxyp[..., n:2*n]
        p = uxyp[..., 2*n:]
        uxn, uxe = ux[..., :ns], ux[..., ns:]
        uyn, uye = uy[..., :ns], uy[..., ns:]
        ue = np.stack([uxe, uye], axis=-1)
        return ue

    def is_inside(self, points, strictly=False, tol=0):
        triangles = self.points[self.elements, :-1]
        # Expand dimensions of points to match the triangles' shape
        points_shape = (1,) * (triangles.ndim - 3) + points.shape
        points_expanded = points.reshape(points_shape)
        
        # Compute vectors
        v0 = triangles[..., 2, :] - triangles[..., 0, :]
        v1 = triangles[..., 1, :] - triangles[..., 0, :]
        v2 = points_expanded[..., np.newaxis, :] - triangles[..., 0, :]
        
        # Compute dot products
        d00 = np.sum(v0 * v0, axis=-1, keepdims=True)
        d01 = np.sum(v0 * v1, axis=-1, keepdims=True)
        d11 = np.sum(v1 * v1, axis=-1, keepdims=True)
        d20 = np.sum(v2 * v0, axis=-1, keepdims=True)
        d21 = np.sum(v2 * v1, axis=-1, keepdims=True)
        
        # Compute denominator
        denom = d00 * d11 - d01 * d01
        
        # Compute barycentric coordinates
        lambda1 = (d11 * d20 - d01 * d21) / denom
        lambda2 = (d00 * d21 - d01 * d20) / denom
        lambda3 = 1 - lambda1 - lambda2
        
        # Check if all barycentric coordinates are within the range (0, 1)
        if strictly:
            inside = (lambda1 > -tol) & (lambda2 > -tol) & (lambda3 > -tol)
        else:
            inside = (lambda1 >= -tol) & (lambda2 >= -tol) & (lambda3 >= -tol)
        
        # Reduce to the expected shape
        inside = np.squeeze(inside, -1)

        #If needed to validate, this is a code
        #np.all(is_inside(element_points, x) == np.eye(x.shape[0], dtype=bool))
        #x = assembler.centroids[..., :-1]
        #element_points = mesh.points[mesh.cells_dict['triangle'], :-1]
        #np.all(is_inside(element_points, x) == np.eye(x.shape[0], dtype=bool))        
        return inside

    def is_inside_domain(self, x, tol=0):
        return np.any(self.is_inside(x), axis=-1)

    @property
    def npoints(self):
        return self.mesh.points.shape[0]
    
    @property
    def nelements(self):
        return self.mesh.cells_dict["triangle"].shape[0]

    @property
    def elements(self):
        return self.mesh.cells_dict["triangle"]

    @property
    def boundary_elements(self):
        return self.mesh.cells_dict["line"]
    
    @property
    def npointse(self):
        return self.npoints + self.nelements

    @property
    def nvariables(self):
        if self.main_element == 'mini':
            return self.npointse
        elif self.main_element == "linear":
            return self.npoints

    @property
    def points(self):
        return self.mesh.points

    @property
    def centroids(self):
        elements = self.mesh.cells_dict["triangle"]
        return self.mesh.points[elements, :].mean(axis=1)
    
    @property
    def pointse(self):
        return np.vstack([self.points, self.centroids])

    @property
    def belel(self):
        if self.main_element == 'mini':
            return self.mini
        elif self.main_element == "linear":
            return self.linear
        else:
            raise NotImplementedError

def enumerate_product(L, n):
    indices = list(itertools.product(range(len(L)), repeat=n))
    values = list(itertools.product(L, repeat=n))
    return zip(indices, values)
