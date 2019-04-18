#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Smallest bounding sphere algorithm
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Basic libs
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

# C++ extension
import cpp_bounding_circle.bounding_circle as cpp_bounding_circle


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main functions
#       \********************/
#

def smallest_bounding_cylinder(points, verbose=False):
    """
    Computes the smallest bounding cylinder for a set of points
    :param points: [N, 3] array
    :return: (center3D, height, radius) cylinder defined by center point, height and radius
    """

    # Compute min and max along z axis
    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])

    # Compute height and center_z
    center_z = (z_max + z_min) / 2
    height = z_max - z_min

    # Compute smallest bounding circle in 2D plane (x, y)
    center2D, radius = cpp_smallest_bounding_circle(points[:, :2], verbose)

    return np.array([center2D[0], center2D[1], center_z, height, radius], dtype=np.float32)


def py_smallest_bounding_sphere(points, implementation='iterative'):
    """
    Computes the smallest bounding sphere for a set of points in dimension 2 or 3
    :param points: [N, d] array
    :return: (center, radius) sphere defined by center point and radius
    """

    if implementation == 'iterative':
        return welzl_algorithm_iterative(points)
    else:
        return welzl_algorithm(points, np.zeros((0, points.shape[1])))


def cpp_smallest_bounding_circle(points, verbose=False):
    """
    C++ implementation of the recursive Welzl algorithm. If the number of points is too high, reduce it for performance.
    :param points:
    :return:
    """

    if points.shape[0] > 500:
        # Keep the 100 furthest point in 8 directions
        inds = []
        for a in range(-1, 2):
            for b in range(-1, 2):
                if a != 0 or b != 0:
                    inds.append(np.argpartition(points.dot([a, b]), 100)[:100])
        inds = np.unique(np.hstack(inds))

        return cpp_bounding_circle.compute(points[inds, :].astype(np.float32))

    else:
        return cpp_bounding_circle.compute(points.astype(np.float32))


def cpp_smallest_bounding_circle_debug():
    """
    C++ implementation of the recursive Welzl algorithm. If the number of points is too high, reduce it for performance.
    :param points:
    :return:
    """

    if points.shape[0] > 200:

        # Keep the 100 furthest point in 8 directions
        inds = []
        for a in range(-1, 2):
            for b in range(-1, 2):
                if a != 0 or b != 0:
                    inds.append(np.argpartition(points.dot([a, b]), 100)[:100])
        inds = np.unique(np.hstack(inds))

        center1, radius1 = cpp_bounding_circle.compute(points[inds, :].astype(np.float32))

        center2, radius2 = cpp_bounding_circle.compute(points.astype(np.float32))

        center3, radius3 = py_smallest_bounding_sphere(points[inds, :])

        fig = plt.figure()
        plt.plot(points[:, 0], points[:, 1], 'b.')
        plt.plot(points[inds, 0], points[inds, 1], 'r.')
        circle1 = plt.Circle(center1, radius1, color='r', fill=False, linewidth=3)
        circle2 = plt.Circle(center2, radius2, color='g', fill=False, ls='--', linewidth=2)
        circle3 = plt.Circle(center3, radius3, color='b', fill=False, ls='--', linewidth=2)
        fig.axes[0].add_artist(circle1)
        fig.axes[0].add_artist(circle2)
        fig.axes[0].add_artist(circle3)

        fig.axes[0].axis('equal')
        fig.axes[0].set_xlim((center2[0] - radius2 * 1.5, center2[0] + radius2 * 1.5))
        fig.axes[0].set_ylim((center2[1] - radius2 * 1.5, center2[1] + radius2 * 1.5))
        fig.axes[0].set_xlabel('x')
        fig.axes[0].set_ylabel('y')
        plt.show()

        return cpp_bounding_circle.compute(points[inds, :].astype(np.float32))

    else:
        return cpp_bounding_circle.compute(points.astype(np.float32))

# ----------------------------------------------------------------------------------------------------------------------
#
#           Utilities
#       \***************/
#


def welzl_algorithm(P, R):
    """
    Implementation of the welzl recursive algorithm
    :param P: set of points to be enclosed
    :param R: set of points ont the boundary sphere
    :return: (center, radius) sphere defined by center point and radius
    """

    if P.shape[0] < 1 or R.shape[0] >= (P.shape[1] + 1):

        if R.shape[0] <= 1:
            # P and R are empty, return a zeros circle
            return np.zeros(P.shape[1]), -1.0

        if R.shape[0] == 1:
            # P is empty, and smallest sphere containing the one point of R has radius zero
            return R[0], 0.0

        elif R.shape[0] == 2:
            # P is empty, and two points define the smallest sphere
            return np.mean(R, axis=0), np.linalg.norm(R[0] - R[1]) / 2.0

        elif R.shape[0] == 3:
            print("hello there")
            # P is empty, and 3 points define the smallest sphere
            R01 = R[0] - R[1]
            R12 = R[1] - R[2]
            R20 = R[2] - R[0]
            A = np.cross(R01, R12)
            alpha = - R12.dot(R12) * R01.dot(R20)
            beta = - R20.dot(R20) * R01.dot(R12)
            gamma = - R01.dot(R01) * R20.dot(R12)
            radius = 0.5 * np.sqrt(R01.dot(R01) * R12.dot(R12) * R20.dot(R20) / A.dot(A))
            center = (alpha * R[0] + beta * R[1] + gamma * R[2]) / (2.0 * A.dot(A))
            return center, radius

        elif R.shape[0] == 4:
            # P is empty, and 4 points define the smallest sphere
            U = R[1:] - R[:1]
            l01 = U[0].dot(U[0])
            l02 = U[1].dot(U[1])
            l03 = U[2].dot(U[2])
            c12 = np.cross(U[1], U[2])
            c20 = np.cross(U[2], U[0])
            c01 = np.cross(U[0], U[1])
            v = (l01 * c12 + l02 * c20 + l03 * c01) / (2 * U[0].dot(c12))
            return R[0] + v, np.linalg.norm(v)

        else:
            # There are too many points in R, is it possible?
            raise ValueError('Undefined circle in Welzl algorithm')

    center, radius = welzl_algorithm(P[1:, :], R)

    if np.linalg.norm(P[0, :] - center) < radius:
        return center, radius

    return welzl_algorithm(P[1:, :], np.vstack((R, P[0, :])))


def welzl_algorithm_iterative(points):
    """
    Implementation of the welzl recursive algorithm in an iterative way
    :param P: set of points to be enclosed
    :param R: set of points ont the boundary sphere
    :return: (center, radius) sphere defined by center point and radius
    """

    # Initiate state and stack
    stack = []
    state = 0
    fcn_call_idx = 0
    retval = None

    # Initiate P and R
    P = points
    R = np.zeros((0, points.shape[1]))

    # First parameters
    stack.append((fcn_call_idx, P, R))

    while stack:

        # When we enter an iteration of this loop, we can either be:
        #   (state = 0) => starting a new function call
        #   (state = 1) => returning the result of the first function call
        #   (state = 2) => returning the result of the second function call

        if state == 0:

            # Get current values of the parameters
            fcn_call_idx, P, R = stack[-1]

            if P.shape[0] < 1 or R.shape[0] >= (P.shape[1] + 1):

                if R.shape[0] <= 1:
                    #  P and R are empty, return a zeros circle
                    retval = (np.zeros(P.shape[1]), -1.0)
                    state = fcn_call_idx
                    stack.pop()
                    continue

                if R.shape[0] == 1:
                    #  P is empty, and smallest sphere containing the one point of R has radius zero
                    retval = (R[0], 0.0)
                    state = fcn_call_idx
                    stack.pop()
                    continue

                elif R.shape[0] == 2:
                    #  P is empty, and two points define the smallest sphere
                    retval = (np.mean(R, axis=0), np.linalg.norm(R[0] - R[1]) / 2.0)
                    state = fcn_call_idx
                    stack.pop()
                    continue

                elif R.shape[0] == 3:
                    #  P is empty, and 3 points define the smallest sphere
                    R01 = R[0] - R[1]
                    R12 = R[1] - R[2]
                    R20 = R[2] - R[0]
                    A = np.cross(R01, R12)
                    alpha = - R12.dot(R12) * R01.dot(R20)
                    beta = - R20.dot(R20) * R01.dot(R12)
                    gamma = - R01.dot(R01) * R20.dot(R12)
                    radius = 0.5 * np.sqrt(R01.dot(R01) * R12.dot(R12) * R20.dot(R20) / A.dot(A))
                    center = (alpha * R[0] + beta * R[1] + gamma * R[2]) / (2.0 * A.dot(A))
                    retval = (center, radius)
                    state = fcn_call_idx
                    stack.pop()
                    continue

                elif R.shape[0] == 4:
                    #  P is empty, and 4 points define the smallest sphere
                    U = R[1:] - R[:1]
                    l01 = U[0].dot(U[0])
                    l02 = U[1].dot(U[1])
                    l03 = U[2].dot(U[2])
                    c12 = np.cross(U[1], U[2])
                    c20 = np.cross(U[2], U[0])
                    c01 = np.cross(U[0], U[1])
                    v = (l01 * c12 + l02 * c20 + l03 * c01) / (2 * U[0].dot(c12))
                    retval = (R[0] + v, np.linalg.norm(v))
                    state = fcn_call_idx
                    stack.pop()
                    continue

                else:
                    #  There are too many points in R, is it possible?
                    raise ValueError('Undefined circle in Welzl algorithm')


            #  First call of the function
            stack.append((1, P[1:], R))
            state = 0
            continue

        elif state == 1:

            # Get current values of the parameters
            fcn_call_idx, P, R = stack[-1]

            # Values are returned from the first function call
            center, radius = retval

            if np.linalg.norm(P[0, :] - center) < radius:

                # Return the current retval (what is function call idx really?)
                state = fcn_call_idx
                stack.pop()
                continue

            # Second call of the function
            stack.append((2, P[1:], np.vstack((R, P[0]))))
            state = 0
            continue

        elif state == 2:

            # Get current values of the parameters
            fcn_call_idx, P, R = stack[-1]

            # Return the current retval (what is function call idx really?)
            state = fcn_call_idx
            stack.pop()
            continue

    return retval



if __name__ == '__main__':

    # test here
    d = 2
    N = 5000
    points = np.random.randn(N, d).astype(np.float32) * 100.0
    #points = np.random.uniform(-10, 10, N * d).reshape((N, d))

    for i in range(10):
        t1 = time.time()
        center_points, sphere_radius = smallest_bounding_circle(points)
        t2 = time.time()
        print('{:.1f} ms'.format(1000 * (t2 - t1)))

    fig = plt.figure()
    plt.plot(points[:, 0], points[:, 1], '.')
    plt.plot(center_points[0], center_points[1], '+')
    circle = plt.Circle(center_points, sphere_radius, color='r', fill=False)
    fig.axes[0].add_artist(circle)

    fig.axes[0].axis('equal')
    fig.axes[0].set_xlim((center_points[0]-sphere_radius * 1.5, center_points[0]+sphere_radius * 1.5))
    fig.axes[0].set_ylim((center_points[1]-sphere_radius * 1.5, center_points[1]+sphere_radius * 1.5))
    fig.axes[0].set_xlabel('x')
    fig.axes[0].set_ylabel('y')
    plt.show()

