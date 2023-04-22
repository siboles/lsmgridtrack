import numpy as np
import pandas as pds
import vtkmodules.all as vtk


def _define_2d_rotation_matrix(
    e1: np.ndarray, e2: np.ndarray, e1p: np.ndarray, e2p: np.ndarray
) -> np.ndarray:
    Q = np.array([[np.dot(e1, e1p), np.dot(e1, e2p)], [0.0, np.dot(e2, e2p)]])
    Q[1, 0] = Q[0, 1]
    return Q


def _define_3d_rotation_matrix(
    e1: np.ndarray, e2: np.ndarray, e1p: np.ndarray, e2p: np.ndarray
) -> np.ndarray:
    e3 = np.cross(e1, e2)
    e3p = np.cross(e1p, e2p)

    Q = np.array(
        [
            [np.dot(e1, e1p), np.dot(e1, e2p), np.dot(e1, e3p)],
            [0.0, np.dot(e2, e2p), np.dot(e2, e3p)],
            [0.0, 0.0, np.dot(e3, e3p)],
        ]
    )
    Q[1, 0] = Q[0, 1]
    Q[2, 0] = Q[0, 2]
    Q[2, 1] = Q[1, 2]
    return Q


def _get_nearest_point_orientation(points: np.ndarray, surface: vtk.vtkPolyData):
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()
    locator.Update()

    tangents = []
    normals = []
    for i in range(points.shape[1]):
        point = points[i, :].tolist()
        point_id = locator.FindClosestPoint(point)
        tangents.append(surface.GetPointData().GetArray("Tangents").GetTuple(point_id))
        normals.append(surface.GetPointData().GetArray("Normals").GetTuple(point_id))

    return tangents, normals


def transform_to_surface_local_csys(data: pds.DataFrame, surface: vtk.vtkPolyData):
    tangents, normals = _get_nearest_point_orientation(
        data["X", "Y", "Z"].values, surface
    )
    print(tangents, normals)
