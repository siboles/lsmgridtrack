import logging
from typing import Union

import numpy as np
import pandas as pds
import vtkmodules.all as vtk
from vtkmodules.util import numpy_support

log = logging.getLogger(__name__)


CELL_VECTOR_COLUMN_NAMES = (
    (
        "Max Tensile Strain Component 1",
        "Max Tensile Strain Component 2",
        "Max Tensile Strain Component 3",
    ),
    (
        "Max Compressive Strain Component 1",
        "Max Compressive Strain Component 2",
        "Max Compressive Strain Component 3",
    ),
    (
        "Reference Cell Direction Component 1",
        "Reference Cell Direction Component 2",
        "Reference Cell Direction Component 3",
    ),
)

CELL_POSITION_COLUMNS = (
    "Reference Cell Centroid X",
    "Reference Cell Centroid Y",
    "Reference Cell Centroid Z",
)


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


def _mean_vtk_array(arr: vtk.vtkFloatArray) -> np.ndarray:
    narray = numpy_support.vtk_to_numpy(arr)
    return narray.mean(axis=0)


def _get_nearest_point_orientation_3d(
    data: Union[vtk.vtkRectilinearGrid, vtk.vtkImageData], surface: vtk.vtkPolyData
):
    locator = vtk.vtkKdTreePointLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()
    locator.Update()
    mean_normal = _mean_vtk_array(surface.GetPointData().GetArray("Normals"))
    mean_tangent = _mean_vtk_array(surface.GetPointData().GetArray("Tangents"))

    tangents = []
    normals = []
    for i in range(data.GetNumberOfPoints()):
        point = data.GetPoint(i)
        point_id = locator.FindClosestPoint(point)
        if point_id < 0:
            log.warning(
                "Closest point erroneous: setting normal and tangent to surface average"
            )
            tangents.append(mean_tangent)
            normals.append(mean_normal)
        else:
            tangents.append(
                np.array(surface.GetPointData().GetArray("Tangents").GetTuple(point_id))
            )
            normals.append(
                np.array(surface.GetPointData().GetArray("Normals").GetTuple(point_id))
            )

    return tangents, normals


def _get_nearest_point_orientation_2d(
    data: Union[vtk.vtkRectilinearGrid, vtk.vtkImageData], surface: vtk.vtkPolyData
):
    locator = vtk.vtkStaticPointLocator2D()
    locator.SetDataSet(surface)
    locator.AutomaticOn()
    locator.BuildLocator()
    locator.Update()

    mean_normal = _mean_vtk_array(surface.GetPointData().GetArray("Normals"))
    mean_tangent = _mean_vtk_array(surface.GetPointData().GetArray("Tangents"))

    tangents = []
    normals = []
    for i in range(data.GetNumberOfPoints()):
        point = data.GetPoint(i)
        point_id = locator.FindClosestPoint(point)
        if point_id < 0:
            log.warning(
                "Closest point erroneous: setting normal and tangent to surface average"
            )
            tangents.append(mean_tangent)
            normals.append(mean_normal)
        else:
            tangents.append(
                np.array(surface.GetPointData().GetArray("Tangents").GetTuple(point_id))
            )
            normals.append(
                np.array(surface.GetPointData().GetArray("Normals").GetTuple(point_id))
            )

    return tangents, normals


def _get_nearest_point_orientation_dataframe_3d(
    points: np.ndarray, surface: vtk.vtkPolyData
):
    locator = vtk.vtkKdTreePointLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()
    locator.Update()

    mean_normal = _mean_vtk_array(surface.GetPointData().GetArray("Normals"))
    mean_tangent = _mean_vtk_array(surface.GetPointData().GetArray("Tangents"))

    tangents = []
    normals = []
    for i in range(points.shape[0]):
        point_id = locator.FindClosestPoint(points[i, :].ravel().tolist())
        if point_id < 0:
            log.warning(
                "Closest point erroneous: setting normal and tangent to surface average"
            )
            tangents.append(mean_tangent)
            normals.append(mean_normal)
        else:
            tangents.append(
                np.array(surface.GetPointData().GetArray("Tangents").GetTuple(point_id))
            )
            normals.append(
                np.array(surface.GetPointData().GetArray("Normals").GetTuple(point_id))
            )

    return tangents, normals


def _get_nearest_point_orientation_dataframe_2d(
    points: np.ndarray, surface: vtk.vtkPolyData
):
    locator = vtk.vtkStaticPointLocator2D()
    locator.SetDataSet(surface)
    locator.AutomaticOn()
    locator.BuildLocator()
    locator.Update()

    mean_normal = _mean_vtk_array(surface.GetPointData().GetArray("Normals"))
    mean_tangent = _mean_vtk_array(surface.GetPointData().GetArray("Tangents"))
    tangents = []
    normals = []
    for i in range(points.shape[0]):
        point_id = locator.FindClosestPoint(points[i, :].ravel().tolist())
        if point_id < 0:
            log.warn(
                "Closest point erroneous: setting normal and tangent to surface average"
            )
            tangents.append(mean_tangent)
            normals.append(mean_normal)
        else:
            tangents.append(
                np.array(surface.GetPointData().GetArray("Tangents").GetTuple(point_id))
            )
            normals.append(
                np.array(surface.GetPointData().GetArray("Normals").GetTuple(point_id))
            )

    return tangents, normals


def _transform_data_arrays(
    data: Union[vtk.vtkRectilinearGrid, vtk.vtkImageData],
    rotation_matrices: list[np.ndarray],
) -> Union[vtk.vtkRectilinearGrid, vtk.vtkImageData]:
    for i in range(data.GetPointData().GetNumberOfArrays()):
        array_name = data.GetPointData().GetArrayName(i)
        data_array = data.GetPointData().GetArray(i)
        if (
            data_array.GetNumberOfComponents() == 3
            and "displacement" not in array_name.lower()
        ):
            log.info(f"Transforming {array_name} data array.")
            for j in range(data_array.GetNumberOfTuples()):
                data_array.SetTuple3(
                    j, *np.matmul(rotation_matrices[j].T, data_array.GetTuple(j))
                )
        if data_array.GetNumberOfComponents() == 9:
            log.info(f"Transforming {array_name} data array.")
            for j in range(data_array.GetNumberOfTuples()):
                value = np.array(data_array.GetTuple(j)).reshape(3, 3)
                data_array.SetTuple9(
                    j,
                    *np.matmul(
                        rotation_matrices[j].T,
                        np.matmul(value, rotation_matrices[j]),
                    ).ravel(),
                )
    return data


def _transform_dataframe(
    data: pds.DataFrame, rotation_matrices: list[np.ndarray]
) -> pds.DataFrame:
    for i, rmatrix in enumerate(rotation_matrices):
        for label in CELL_VECTOR_COLUMN_NAMES:
            data[list(label)] = np.einsum("ij,j", rmatrix, data[list(label)].iloc[i])
    return data


def transform_to_local_csys_3d(
    data: Union[vtk.vtkRectilinearGrid, vtk.vtkImageData], surface: vtk.vtkPolyData
) -> Union[vtk.vtkRectilinearGrid, vtk.vtkImageData]:
    """Rotate tensors stored on VTK grid to align with orientation
    of nearest point on provided surface.

    :param data: VTK grid with stored data arrays to rotate
    :param surface: Surface to align to.
    :return: VTK grid with rotated data.
    """
    tangents, normals = _get_nearest_point_orientation_3d(data, surface)
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
    """Rotate tensors stored on VTK grid to align with orientation
    of nearest point on provided surface.

    :param data: VTK grid with stored data arrays to rotate
    :param surface: Surface to align to.
    :return: VTK grid with rotated data.
    """
    tangents, normals = _get_nearest_point_orientation_2d(data, surface)
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

    rotated_data = _transform_data_arrays(data, rotation_matrices)
    return rotated_data


def globally_transform_3d(
    data: Union[vtk.vtkRectilinearGrid, vtk.vtkImageData], surface: vtk.vtkPolyData
) -> Union[vtk.vtkRectilinearGrid, vtk.vtkImageData]:
    """Rotate tensors stored on VTK grid to align with average orientation
    of provided surface.

    :param data: VTK grid with stored data arrays to rotate
    :param surface: Surface to align to.
    :return: VTK grid with rotated data.
    """
    mean_tangent = _mean_vtk_array(surface.GetPointData().GetArray("Tangents"))
    mean_normal = _mean_vtk_array(surface.GetPointData().GetArray("Normals"))

    rotation_matrix = _define_3d_rotation_matrix(
        np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), mean_tangent, mean_normal
    )

    rotated_data = _transform_data_arrays(
        data, [rotation_matrix] * data.GetNumberOfPoints()
    )

    return rotated_data


def globally_transform_2d(
    data: Union[vtk.vtkRectilinearGrid, vtk.vtkImageData], surface: vtk.vtkPolyData
) -> Union[vtk.vtkRectilinearGrid, vtk.vtkImageData]:
    """Rotate tensors stored on VTK grid to align with average orientation
    of provided surface.

    :param data: VTK grid with stored data arrays to rotate
    :param surface: Surface to align to.
    :return: VTK grid with rotated data.
    """
    tangents = surface.GetPointData().GetArray("Tangents")
    normals = surface.GetPointData().GetArray("Normals")
    mean_tangent = np.mean(tangents, axis=0)
    mean_normal = np.mean(normals, axis=0)
    rotation_matrix = _define_2d_rotation_matrix(
        np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), mean_tangent, mean_normal
    )

    rotated_data = _transform_data_arrays(
        data, [rotation_matrix] * data.GetNumberOfPoints()
    )

    return rotated_data


def transform_dataframe_to_local_csys_3d(data: dict, surface: vtk.vtkPolyData) -> dict:
    """Rotate tensors in dataframes to local coordinate systems constructed from the
    normals and tangent vectors of the nearest point on the provided surface.

    :param data: Dictionary of Pandas dataframes.
    :param surface: Surface to align to.
    :return: Dictionary of rotated dataframes.
    """
    for df in data.values():
        points = df[list(CELL_POSITION_COLUMNS)].values
        tangents, normals = _get_nearest_point_orientation_dataframe_3d(points, surface)
        rotation_matrices = []
        for tangent, normal in zip(tangents, normals):
            rotation_matrices.append(
                _define_3d_rotation_matrix(
                    np.array([1.0, 0.0, 0.0]),
                    np.array([0.0, 1.0, 0.0]),
                    np.array(tangent),
                    np.array(normal),
                )
            )

        df = _transform_dataframe(df, rotation_matrices)

    return data


def globally_transform_dataframe_3d(data: dict, surface: vtk.vtkPolyData) -> dict:
    """Rotate tensors to align with the average orientation of the provided surface
    PolyData.

    :param data: Dictionary of Pandas dataframes
    :param surface: Surface to align to.
    :return: Dictionary of rotated dataframes.
    """
    mean_normal = _mean_vtk_array(surface.GetPointData().GetArray("Normals"))
    mean_tangent = _mean_vtk_array(surface.GetPointData().GetArray("Tangents"))
    for df in data.values():
        rotation_matrix = _define_3d_rotation_matrix(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            mean_tangent,
            mean_normal,
        )

        for label in CELL_VECTOR_COLUMN_NAMES:
            df[list(label)] = np.einsum(
                "ij,...j", rotation_matrix, df[list(label)].values
            )

    return data


def globally_transform_polydata_coordinates_3d(
    data: list[vtk.vtkPolyData], surface: vtk.vtkPolyData
) -> list[vtk.vtkPolyData]:
    """Transform coordinates of each PolyData in list to align with average orientation of
    provided surface.

    :param data: list of PolyData objects
    :param surface: surface to align to.
    :return: list of rotated PolyData objects
    """
    mean_normal = _mean_vtk_array(surface.GetPointData().GetArray("Normals"))
    mean_tangent = _mean_vtk_array(surface.GetPointData().GetArray("Tangents"))
    rotation_matrix = _define_3d_rotation_matrix(
        np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), mean_tangent, mean_normal
    )
    vtk_matrix = np.eye(4)
    vtk_matrix[0:3, 0:3] = rotation_matrix.T
    tx = vtk.vtkTransform()
    tx.SetMatrix(vtk_matrix.ravel())
    tx.Update()
    transform = vtk.vtkTransformPolyDataFilter()
    transform.SetTransform(tx)
    rotated_data = []
    for datum in data:
        transform.SetInputData(datum)
        transform.Update()
        rotated_data.append(transform.GetOutput())
    return rotated_data


def write_to_vtk_polydata(data: vtk.vtkPolyData, name: str):
    """Write VTK PolyData to file.

    :param data:
    :param name:
    """
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(data)
    writer.SetFileName(f"{name}.vtp")
    writer.Write()


def write_to_vtk_image_data(data: vtk.vtkImageData, name: str):
    """Write VTK ImageData to file

    :param data:
    :param name:
    """
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(f"{name}.vti")
    writer.SetInputData(data)
    writer.Write()


def write_to_vtk_grid(data: vtk.vtkRectilinearGrid, name: str):
    """Write VTK RectilinearGrid to file.

    :param data:
    :param name:
    """
    writer = vtk.vtkXMLRectilinearGridWriter()
    writer.SetFileName(f"{name}.vtr")
    writer.SetInputData(data)
    writer.Write()


def write_dataframe_to_excel(data: Union[dict, pds.DataFrame], name: str):
    """Write Pandas dataframe or dictionary of dataframes to an Excel file.
    If dictionary provided each (key, value) will be a new sheet.

    :param data:
    :param name:
    """
    if isinstance(data, dict):
        with pds.ExcelWriter(f"{name}.xlsx") as writer:
            for k, df in data.items():
                df.to_excel(writer, sheet_name=k)
    else:
        data.to_excel(f"{name}.xlsx")


def read_dataframe_from_excel(name: str) -> dict:
    data = pds.read_excel(f"{name}.xlsx", sheet_name=None)
    return data


def read_vtk_grid(filename: str) -> Union[vtk.vtkRectilinearGrid, vtk.vtkImageData]:
    """Read VTK grid from file.

    :param filename: Path to file.
    :raises ValueError:
    :return:
    """
    if filename.endswith(".vtr"):
        reader = vtk.vtkXMLRectilinearGridReader()
    elif filename.endswith(".vti"):
        reader = vtk.vtkXMLImageDataReader()
    else:
        raise ValueError("File should be either vtkImageData or vtkRectilinearGrid")

    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def read_vtk_surface(filename: str) -> vtk.vtkPolyData:
    """Read vtkPolyData from file.

    :param filename: Path to file.
    :return:
    """
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def convert_vtk_to_dataframe(
    data: Union[vtk.vtkImageData, vtk.vtkRectilinearGrid]
) -> pds.DataFrame:
    """Converts VTK grid to a Pandas dataframe.

    :param data: VTK grid
    :return: Pandas dataframe with vertex coordinates and VTK data arrays.
    """
    point_array = np.zeros((data.GetNumberOfPoints(), 3))
    for i in range(point_array.shape[0]):
        point_array[i, :] = data.GetPoint(i)
    df = pds.DataFrame(point_array, columns=["x", "y", "z"])
    for i in range(data.GetPointData().GetNumberOfArrays()):
        array_name = data.GetPointData().GetArrayName(i)
        n_components = data.GetPointData().GetArray(i).GetNumberOfComponents()
        data_array = numpy_support.vtk_to_numpy(data.GetPointData().GetArray(i))
        if n_components > 1:
            tmp_df = pds.DataFrame(
                {
                    f"{array_name} Component {j + 1}": data_array[:, j]
                    for j in range(data_array.shape[1])
                }
            )
        else:
            tmp_df = pds.DataFrame({array_name: data_array})
        df = pds.concat([df, tmp_df], axis=1)
    return df
