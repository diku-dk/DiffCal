import numpy as np
import torch
import pytorch3d.transforms as pt
from typing import Union, Optional, Dict, List


class Object:
    surface_file: str
    """Surface mesh file of the object."""
    volume_file: str
    """Volume mesh file of the object."""
    actuation_file: Optional[str] = None
    """Actuation file of the object"""
    vertices: Union[np.ndarray, torch.Tensor]
    """Vertices of the object."""
    surface_vertices: Union[np.ndarray, torch.Tensor]
    """Vertices on the surface of the object."""
    surface_tris: Union[np.ndarray, torch.Tensor]
    """Indices of the surface triangles."""
    vertex_normals: Union[np.ndarray, torch.Tensor]
    """Vertex normals at the surface."""
    face_normals: Union[np.ndarray, torch.Tensor]
    """Face normals at the surface."""
    tetrahedra: Union[np.ndarray, torch.Tensor]
    """Indices of tetrahedral elements."""
    has_surface_mesh: bool = False
    """Flag indicating whether this object has a surface mesh."""
    has_volume_mesh: bool = False
    """Flag indicating whether this object has a surface mesh."""
    dirichlet_boundary_conditions: np.ndarray = np.empty((0, 3, 2), dtype=np.float)
    """Dirichlet Boundary conditions of the object."""
    dirichlet_boundary_particles: np.ndarray = np.empty(0, dtype=np.int)
    """Particles affected by Dirichlet Boundary conditions."""
    scale: float
    """Scale of the object."""
    displacement: np.ndarray
    """X,Y,Z-displacement of the object."""
    orientation: np.ndarray
    """Initial orientation of the object (as a quaternion)."""
    euler_orientation: np.ndarray
    """Euler angles corresponding to the quaternion."""
    normalize: bool
    """Whether or not to center object around orig for dirichlet boundary condition computation."""
    actuation_type: int = -1
    """Type of actuation for this robot. -1 => no actuation, 0 => Moving dirichlet, 1 => Cables, 2 => Linear."""
    tet_wise_parameter_bounding_boxes: np.ndarray = np.empty((0, 3, 2), dtype=np.float)
    """Parameter distribution bounding boxes."""
    tet_wise_parameter_distribution: np.ndarray = np.empty(0, dtype=np.int)
    """Map from tet index to parameter index."""

    def __init__(self,
                 surface_file: Optional[str] = None,
                 volume_file: Optional[str] = None,
                 scale: float = 1.0,
                 displacement: np.ndarray = np.array([0., 0., 0.], dtype=np.float32),
                 orientation: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                 normalize: bool = False,
                 vertices: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 surface_tris: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 tetrahedra: Optional[Union[np.ndarray, torch.Tensor]] = None):
        """
        Initializes a scene Object. If given surface and volume files, these will be read from disk.
        If given vertices and surface tris/tetrahedra, these will be the vertices, surface indices and elements of
        the object.
        :param surface_file: Path to surface scaled_data.
        :param volume_file: Path to volume scaled_data.
        :param scale: Scale of the object.
        :param displacement: Initial displacement, which should be applied to the object.
        :param orientation: Initial orientation, which should be applied to the object.
        :param vertices: Vertices of the surface _or_ volume mesh.
        :param normalize: Whether or not to center object around orig for dirichlet boundary condition comp.
        :param surface_tris: Indices into 'vertices' for the surface.
        :param tetrahedra: Indices into 'vertices' for volume elements.
        """
        if surface_file is not None:
            self.read_surface_mesh(surface_file)
        elif surface_tris is  not None:
            self.surface_vertices = vertices
            self.surface_tris = surface_tris
            self.has_surface_mesh = not (surface_tris is None)

        if volume_file is not None:
            self.read_volume_mesh(volume_file)
        elif tetrahedra is not None:
            self.vertices = vertices
            self.tetrahedra = tetrahedra.reshape(-1,4)
            self.has_volume_mesh = not (tetrahedra is None)

        self.compute_vertex_normals()

        self.scale = scale
        self.displacement = displacement
        self.orientation = orientation
        self.euler_orientation =\
            self.compute_euler_angles(pt.quaternion_to_matrix(torch.from_numpy(orientation)).numpy())
        self.normalize = normalize
        self.compute_neighbors()
        self.compute_2_ring()

    def read_surface_mesh(self,
                          filename: str):
        """
        Reads an .obj/.stl file using TriMesh.
        :param filename: Path to read scaled_data from.
        """
        self.surface_file = filename
        vertices = np.empty((0, 3), dtype=np.float32)
        indices = np.empty((0, 3), dtype=np.uint32)
        with open(filename, 'r') as fh:
            lines = fh.readlines()
            for line in lines:
                tokens = line.split()
                if len(tokens) == 0:
                    continue
                if tokens[0] == 'v':
                    vertices = np.append(vertices, [list(map(lambda x: float(x.strip()), tokens[1:4]))], axis=0)
                if tokens[0] == 'f':
                    indices = np.append(indices, [list(map(lambda x: int(x.strip().split('/')[0]) - 1, tokens[1:4]))],
                                        axis=0)

        face_normals = np.cross(vertices[indices[:, 1]] - vertices[indices[:, 0]],
                                vertices[indices[:, 2]] - vertices[indices[:, 0]])
        face_normals /= np.vstack([np.linalg.norm(face_normals, axis=1),
                                   np.linalg.norm(face_normals, axis=1),
                                   np.linalg.norm(face_normals, axis=1)]).T
        self.face_normals = face_normals
        self.surface_vertices = vertices
        if not self.has_volume_mesh:
            self.surface_tris = indices
        else:
            self.surface_tris = Object.re_align_surface_triangles(Object.get_index_map(self.vertices, vertices),
                                                                  indices)
        self.has_surface_mesh = True

    def compute_vertex_normals(self):
        index_map = Object.get_index_map(self.vertices, self.surface_vertices)
        index_map = dict((y, x) for x, y in index_map.items())
        vertex_normals = np.zeros_like(self.vertices)
        vertex_normal_count = np.zeros(self.vertices.shape[0])
        for ft, f in enumerate(self.surface_tris):
            for fi in f:
                if fi not in index_map.keys():
                    continue
                vertex_normals[index_map[fi]] += self.face_normals[ft]
                vertex_normal_count[index_map[fi]] += 1
        self.vertex_normals = vertex_normals
        self.vertex_normals[np.where(vertex_normal_count != 0)] \
            / np.vstack([vertex_normal_count[np.where(vertex_normal_count != 0)],
                         vertex_normal_count[np.where(vertex_normal_count != 0)],
                         vertex_normal_count[np.where(vertex_normal_count != 0)]]).T

    def read_volume_mesh(self,
                         filename: str):
        """
        Reads a volume mesh (.tet format).
        :param filename: Path to read scaled_data from.
        """
        self.volume_file = filename
        vertices = np.empty((0, 3), dtype=np.float32)
        tetrahedra = np.empty((0, 4), dtype=np.uint32)
        with open(filename, 'r') as fh:
            lines = fh.readlines()
            for line in lines:
                tokens = line.split()
                if len(tokens) == 0:
                    continue
                elif tokens[0] == 'v':
                    vertices = np.append(vertices, [list(map(lambda x: float(x.strip()), tokens[1:4]))], axis=0)
                elif tokens[0] == 't':
                    tetrahedra = np.append(tetrahedra, list(map(lambda x: int(x.strip()), tokens[1:5])))
        if self.has_surface_mesh:
            self.surface_tris = Object.re_align_surface_triangles(Object.get_index_map(vertices, self.surface_vertices),
                                                                  self.surface_tris)
        self.vertices = vertices.astype(np.float32)
        self.tetrahedra = tetrahedra.reshape(-1,4)
        self.tet_wise_parameter_distribution.resize(self.tetrahedra.shape[0], refcheck=False)
        self.tet_wise_parameter_distribution[:] = -1
        self.has_volume_mesh = True

    def add_parameter_division(self,
                               bbox: Union[np.ndarray, torch.Tensor, List[float]],
                               parameter_id: int):
        if type(bbox) == list:
            bbox = np.array(bbox)

        offset = np.mean(self.vertices, axis=0) if self.normalize else np.zeros(3, dtype=np.float32)
        for ti, t in enumerate(self.tetrahedra):
            if self.tet_wise_parameter_distribution[ti] != -1:
                continue
            center = self.vertices[t].mean(0) - offset
            if bbox[0, 0] <= center[0] <= bbox[0, 1] and \
               bbox[1, 0] <= center[1] <= bbox[1, 1] and \
               bbox[2, 0] <= center[2] <= bbox[2, 1]:
                self.tet_wise_parameter_distribution[ti] = parameter_id

        if type(bbox) == np.ndarray:
            self.tet_wise_parameter_bounding_boxes = np.append(self.tet_wise_parameter_bounding_boxes,
                                                               [bbox], axis=0)
        elif type(bbox) == torch.Tensor:
            self.tet_wise_parameter_bounding_boxes = np.append(self.tet_wise_parameter_bounding_boxes,
                                                               [torch.from_numpy(bbox)], axis=0)

    def add_dirichlet_boundary_conditions(self,
                                          bbox: Union[np.ndarray, torch.Tensor, List[List[float]]]):
        """
        Adds vertex indices to the list of particles which should be affected by dirichlet boundary conditions.
        :param bbox: Bounding box.
        """
        if self.dirichlet_boundary_particles.shape[0] == 0:
            self.dirichlet_boundary_particles = np.zeros(self.vertices.shape[0], dtype=bool)
        if type(bbox) == list:
            bbox = np.array(bbox)
        if self.normalize:
            center = 0.5 * (np.max(self.vertices, axis=0) - np.min(self.vertices, axis=0)) + np.min(self.vertices, axis=0)
            self.dirichlet_boundary_particles = \
                np.logical_or(self.dirichlet_boundary_particles,
                              np.logical_and(
                                  np.logical_and(self.vertices[:, 0] - center[0] >= bbox[0, 0],
                                                 self.vertices[:, 0] - center[0] <= bbox[0, 1]),
                                  np.logical_and(
                                      np.logical_and(self.vertices[:, 1] - center[1] >= bbox[1, 0],
                                                     self.vertices[:, 1] - center[1] <= bbox[1, 1]),
                                      np.logical_and(self.vertices[:, 2] - center[2] >= bbox[2, 0],
                                                     self.vertices[:, 2] - center[2] <= bbox[2, 1]))))
        else:
            self.dirichlet_boundary_particles = \
                np.logical_or(self.dirichlet_boundary_particles,
                              np.logical_and(
                                  np.logical_and(self.vertices[:, 0] >= bbox[0, 0],
                                                 self.vertices[:, 0] <= bbox[0, 1],
                                                 self.vertices[:, 1] >= bbox[1, 0]),
                                  np.logical_and(self.vertices[:, 1] <= bbox[1, 1],
                                                 self.vertices[:, 2] >= bbox[2, 0],
                                                 self.vertices[:, 2] <= bbox[2, 1])))
        if type(bbox) == np.ndarray:
            self.dirichlet_boundary_conditions = np.append(self.dirichlet_boundary_conditions,
                                                           [bbox], axis=0)
        elif type(bbox) == torch.Tensor:
            self.dirichlet_boundary_conditions = np.append(self.dirichlet_boundary_conditions,
                                                           [torch.from_numpy(bbox)], axis=0)

    @staticmethod
    def get_index_map(tet_vertices: np.ndarray,
                      tri_vertices: np.ndarray) -> Dict[int, int]:
        """
        Computes mapping between tetrahedra vertices and triangle vertices.
        :param tet_vertices: Vertices of the volume mesh.
        :param tri_vertices: Vertices of the surface mesh.
        :return: Map between volume mesh vertices and surface mesh vertices.
        """
        correspondence = {}
        for i, vo in enumerate(tri_vertices):
            for j, vt in enumerate(tet_vertices):
                if np.linalg.norm(vo - vt) <= 1e-8:
                    correspondence[i] = j
                    break
        return correspondence

    @staticmethod
    def re_align_surface_triangles(correspondence_map: Dict[int, int],
                                   surface_triangles: np.ndarray) -> np.ndarray:
        """
        Re-aligns a surface mesh to the vertex indices of a volume mesh.
        :param correspondence_map: Map between surface and volume vertex indices.
        :param surface_triangles: The triangles of the surface mesh.
        :return: The re-aligned surface mesh, which can be used with the vertices of the volume mesh.
        """
        re_aligned_surface_triangles = np.empty((0, 3), dtype=np.int)
        for vi in surface_triangles:
            re_aligned_surface_triangles =\
                np.append(re_aligned_surface_triangles, [[correspondence_map[vi[0]],
                                                          correspondence_map[vi[1]],
                                                          correspondence_map[vi[2]]]], axis=0)
        return re_aligned_surface_triangles

    @staticmethod
    def compute_euler_angles(rotation: np.ndarray):
        """
        Computes roll, pitch and yaw (i.e. Euler angles) given a rotation matrix
        :param rotation:
        :return: roll, pitch, yaw (x, y, z angles)
        """
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        pitch = np.arctan2(-rotation[2, 0], np.sqrt(rotation[2, 1]**2. + rotation[2, 2]**2))
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])

        roll = np.clip(np.rad2deg(roll), 0.0, 360.0)
        pitch = np.clip(np.rad2deg(pitch), 0.0, 360.0)
        yaw = np.clip(np.rad2deg(yaw), 0.0, 360.0)
        return roll, pitch, yaw

    @staticmethod
    def euler_angles_to_rotation_matrix(roll: float,
                                        pitch: float,
                                        yaw: float):
        """
        Computes a 3x3 rotation matrix from Euler angles.
        :param roll: X-axis rotation in degrees.
        :param pitch:  Y-axis rotation in degrees.
        :param yaw: Z-axis rotation in degrees.
        :return: 3x3 rotation matrix.
        """
        gamma, beta, alpha = np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(yaw)
        r_x = np.eye(3, dtype=np.float32)
        r_y = np.eye(3, dtype=np.float32)
        r_z = np.eye(3, dtype=np.float32)

        r_x[1, 1] = np.cos(gamma)
        r_x[1, 2] = -np.sin(gamma)
        r_x[2, 1] = np.sin(gamma)
        r_x[2, 2] = np.cos(gamma)

        r_y[0, 0] = np.cos(beta)
        r_y[0, 2] = np.sin(beta)
        r_y[2, 0] = -np.sin(beta)
        r_y[2, 2] = np.cos(beta)

        r_z[0, 0] = np.cos(alpha)
        r_z[0, 1] = -np.sin(alpha)
        r_z[1, 0] = np.sin(alpha)
        r_z[1, 1] = np.cos(alpha)

        rotation = np.dot(r_z, np.dot(r_y, r_x))

        return rotation

    def compute_neighbors(self):
        """
        This function computes adjacent tetrahedral information. The result is returned in a Neihbor lookup array, N.
        The N-array works such that if n = N[e,v] then tetrahedron n is adajacent to tetrahedorn e. The tetrahedron
        n is sharing the face opposite node v of tetrahedron e.
        Concretely if T[e,:] = [i,j,k,m] and say v = 3 then it will be the triangle face i,j,k that are
        shared between tetrahedra e and n.
        If n is -1 then it means that there are no neighboring tetrahedra on the other side
        of face i,j,k. This means that face i,j,k is a surface boundary face of the tetrahedral mesh.
        :param T:    The tetrahedra of the mesh
        :return:     The neighbor tetrahedra lookup array
        """
        T = self.tetrahedra
        N = -np.ones(T.shape, dtype=np.int32)
        face_info = np.zeros((4 * len(T), 5), dtype=np.int32)
        for e in range(len(T)):
            i, j, k, m = T[e]
            face_info[e * 4 + 0, :] = sorted([j, k, m]) + [e, 0]
            face_info[e * 4 + 1, :] = sorted([i, k, m]) + [e, 1]
            face_info[e * 4 + 2, :] = sorted([i, j, m]) + [e, 2]
            face_info[e * 4 + 3, :] = sorted([i, j, k]) + [e, 3]
        face_info = face_info[np.lexsort((face_info[:, 0], face_info[:, 1], face_info[:, 2]))]
        for i in range(4 * len(T) - 1):
            if np.array_equal(face_info[i, 0:3], face_info[i + 1, 0:3]):
                # We have found a shared face in tetrahedral mesh
                elem, elem_opp_node = face_info[i, 3:5]
                twin, twin_opp_node = face_info[i + 1, 3:5]
                N[elem, elem_opp_node] = twin
                N[twin, twin_opp_node] = elem
        self.neighbors = N

    def compute_2_ring(self):
        T = self.tetrahedra
        self.two_ring_neighbourhood = -np.ones((T.shape[0], 20), dtype=np.int32)
        for ni, n in enumerate(self.neighbors):
            self.two_ring_neighbourhood[ni] = np.append(n, np.array([self.neighbors[nj] for nj in self.neighbors[ni]]).flatten()).flatten()