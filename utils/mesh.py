#
#
#      0======================0
#      |    Mesh utilities    |
#      0======================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      functions related to meshes
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 10/02/2017
#


# ----------------------------------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Basic libs
import numpy as np
import time


# ----------------------------------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#


def rasterize_mesh(vertices, faces, dl, verbose=False):
    """
    Creation of point cloud from mesh via rasterization. All models are rescaled to fit in a 1 meter radius sphere
    :param vertices: array of vertices
    :param faces: array of faces
    :param dl: parameter controlling density. Distance between each point
    :param verbose: display parameter
    :return: point cloud
    """

    ######################################
    # Eliminate useless faces and vertices
    ######################################

    # 3D coordinates of faces
    faces3D = vertices[faces, :]
    sides = np.stack([faces3D[:, i, :] - faces3D[:, i - 1, :] for i in [2, 0, 1]], axis=1)

    # Indices of big enough faces
    keep_bool = np.min(np.linalg.norm(sides, axis=-1), axis=-1) > 1e-9
    faces = faces[keep_bool]

    ##################################
    # Place random points on each face
    ##################################

    # 3D coordinates of faces
    faces3D = vertices[faces, :]

    # Area of each face
    opposite_sides = np.stack([faces3D[:, i, :] - faces3D[:, i - 1, :] for i in [2, 0, 1]], axis=1)
    lengths = np.linalg.norm(opposite_sides, axis=-1)

    # Points for each face
    all_points = []
    all_vert_inds = []
    for face_verts, face, l, sides in zip(faces, faces3D, lengths, opposite_sides):

        # All points generated for this face
        face_points = []

        # Safe check for null faces
        if np.min(l) < 1e-9:
            continue

        # Smallest faces, only place one point in the center
        if np.max(l) < dl:
            face_points.append(np.mean(face, axis=0))
            continue

        # Chose indices so that A is the largest angle
        A_idx = np.argmax(l)
        B_idx = (A_idx + 1) % 3
        C_idx = (A_idx + 2) % 3
        i = -sides[B_idx] / l[B_idx]
        j = sides[C_idx] / l[C_idx]

        # Create a mesh grid of points along the two smallest sides
        s1 = (l[B_idx] % dl) / 2
        s2 = (l[C_idx] % dl) / 2
        x, y = np.meshgrid(np.arange(s1, l[B_idx], dl), np.arange(s2, l[C_idx], dl))
        points = face[A_idx, :] + (np.expand_dims(x.ravel(), 1) * i + np.expand_dims(y.ravel(), 1) * j)
        points = points[x.ravel() / l[B_idx] + y.ravel() / l[C_idx] <= 1, :]
        face_points.append(points)

        # Add points on the three edges
        for edge_idx in range(3):
            i = sides[edge_idx] / l[edge_idx]
            A_idx = (edge_idx + 1) % 3
            s1 = (l[edge_idx] % dl) / 2
            x = np.arange(s1, l[edge_idx], dl)
            points = face[A_idx, :] + np.expand_dims(x.ravel(), 1) * i
            face_points.append(points)

        # Add vertices
        face_points.append(face)

        # Compute vertex indices
        dists = np.sum(np.square(np.expand_dims(np.vstack(face_points), 1) - face), axis=2)
        all_vert_inds.append(face_verts[np.argmin(dists, axis=1)])

        # Save points and inds
        all_points += face_points

    return np.vstack(all_points).astype(np.float32), np.hstack(all_vert_inds)


def cylinder_mesh(cylinder, precision=24):

    # Get parameters
    center = cylinder[:3]
    h = cylinder[3]
    r = cylinder[4]

    # Create vertices
    theta = 2.0 * np.pi / precision
    thetas = np.arange(precision) * theta
    circleX = r * np.cos(thetas)
    circleY = r * np.sin(thetas)
    top_vertices = np.vstack((circleX, circleY, circleY * 0 + h / 2)).T
    bottom_vertices = np.vstack((circleX, circleY, circleY * 0 - h / 2)).T
    vertices = np.array([[0, 0, h / 2],
                         [0, 0, -h / 2]])
    vertices = np.vstack((vertices, top_vertices, bottom_vertices))
    vertices += center

    # Create faces
    top_faces = [[0, 2 + i, 2 + ((i + 1) % precision)] for i in range(precision)]
    bottom_faces = [[1, 2 + precision + i, 2 + precision + ((i + 1) % precision)] for i in range(precision)]
    side_faces1 = [[2 + i, 2 + precision + i, 2 + precision + ((i + 1) % precision)] for i in range(precision)]
    side_faces2 = [[2 + precision + ((i + 1) % precision), 2 + i, 2 + ((i + 1) % precision)] for i in range(precision)]
    faces = np.array(top_faces + bottom_faces + side_faces1 + side_faces2, dtype=np.int32)

    return vertices.astype(np.float32), faces
