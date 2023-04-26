from typing import Union

import numpy as np
import vtkmodules.all as vtk


def _define_2d_rotation_matrix(
    e1: np.ndarray, e2: np.ndarray, e1p: np.ndarray, e2p: np.ndarray
) -> np.ndarray:
    Q = np.array(
        [
            [np.dot(e1, e1p), np.dot(e1, e2p), 0.0],
            [np.dot(e2, e1p), np.dot(e2, e2p), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return Q


def _define_3d_rotation_matrix(
    e1: np.ndarray, e3: np.ndarray, e1p: np.ndarray, e3p: np.ndarray
) -> np.ndarray:
    e3p *= -1.0
    e1p /= np.linalg.norm(e1p)
    e3p /= np.linalg.norm(e3p)
    e2 = np.cross(e3, e1)
    e2p = np.cross(e3p, e1p)
    e1p = np.cross(e2p, e3p)

    Q = np.array(
        [
            [np.dot(e1, e1p), np.dot(e1, e2p), np.dot(e1, e3p)],
            [np.dot(e2, e1p), np.dot(e2, e2p), np.dot(e2, e3p)],
            [np.dot(e3, e1p), np.dot(e3, e2p), np.dot(e3, e3p)],
        ]
    )
    return Q


def _get_nearest_point_orientation(
    data: Union[vtk.vtkRectilinearGrid, vtk.vtkImageData], surface: vtk.vtkPolyData
):
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()
    locator.Update()

    tangents = []
    normals = []
    for i in range(data.GetNumberOfPoints()):
        point = data.GetPoint(i)
        point_id = locator.FindClosestPoint(point)
        tangents.append(surface.GetPointData().GetArray("Tangents").GetTuple(point_id))
        normals.append(surface.GetPointData().GetArray("Normals").GetTuple(point_id))

    return tangents, normals


def _transform_data_arrays(
    data: Union[vtk.vtkRectilinearGrid, vtk.vtkImageData],
    rotation_matrices: list[np.ndarray],
) -> Union[vtk.vtkRectilinearGrid, vtk.vtkImageData]:
    for i in range(data.GetPointData().GetNumberOfArrays()):
        data_array = data.GetPointData().GetArray(i)
        if data_array.GetNumberOfComponents() == 3:
            for j in range(data_array.GetNumberOfTuples()):
                data_array.SetTuple3(
                    j, *np.matmul(rotation_matrices[j], data_array.GetTuple(j))
                )
        if data_array.GetNumberOfComponents() == 9:
            for j in range(data_array.GetNumberOfTuples()):
                value = np.array(data_array.GetTuple(j)).reshape(3, 3)
                data_array.SetTuple9(
                    j,
                    np.matmul(
                        rotation_matrices[j],
                        np.matmul(value, rotation_matrices[j].T),
                    ).ravel(),
                )
    return data


def transform_to_local_csys_3d(
    data: Union[vtk.vtkRectilinearGrid, vtk.vtkImageData], surface: vtk.vtkPolyData
) -> Union[vtk.vtkRectilinearGrid, vtk.vtkImageData]:
    tangents, normals = _get_nearest_point_orientation(data, surface)
    rotation_matrices = []
    for tangent, normal in zip(tangents, normals):
        rotation_matrices.append(
            _define_3d_rotation_matrix(
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
                np.array(tangent),
                np.array(normal),
            )
        )

    rotated_data = _transform_data_arrays(data, rotation_matrices)

    return rotated_data


def transform_to_local_csys_2d(
    data: Union[vtk.vtkRectilinearGrid, vtk.vtkImageData], surface: vtk.vtkPolyData
) -> Union[vtk.vtkRectilinearGrid, vtk.vtkImageData]:
    tangents, normals = _get_nearest_point_orientation(data, surface)
    rotation_matrices = []
    for tangent, normal in zip(tangents, normals):
        rotation_matrices.append(
            _define_2d_rotation_matrix(
                np.array([1.0, 0.0]),
                np.array([0.0, 1.0]),
                np.array(tangent[0:2]),
                np.array(normal[0:2]),
            )
        )

    return data


def globally_transform_3d(
    data: Union[vtk.vtkRectilinearGrid, vtk.vtkImageData], surface: vtk.vtkPolyData
) -> Union[vtk.vtkRectilinearGrid, vtk.vtkImageData]:
    tangents = surface.GetPointData().GetArray("Tangents")
    normals = surface.GetPointData().GetArray("Normals")
    mean_tangent = np.mean(tangents, axis=0)
    mean_normal = np.mean(normals, axis=0)
    rotation_matrix = _define_3d_rotation_matrix(
        np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), mean_tangent, mean_normal
    )

    rotated_data = _transform_data_arrays(data, [rotation_matrix])

    return rotated_data


def write_to_vtk_image_data(data: vtk.vtkImageData, name: str):
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(f"{name}.vti")
    writer.SetInputData(data)
    writer.Write()


def write_to_vtk_grid(data: vtk.vtkRectilinearGrid, name: str):
    writer = vtk.vtkXMLRectilinearGridWriter()
    writer.SetFileName(f"{name}.vtr")
    writer.SetInputData(data)
    writer.Write()


def read_vtk_grid(filename: str) -> vtk.vtkRectilinearGrid:
    if filename.endswith(".vtr"):
        reader = vtk.vtkXMLRectilinearGridReader()
    elif filename.endswith(".vti"):
        reader = vtk.vtkXMLImageDataReader()
    else:
        raise ValueError("File should be either vtkImageData or vtkRectilinearGrid")

    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def read_vtk_surface(filename) -> vtk.vtkPolyData:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()
