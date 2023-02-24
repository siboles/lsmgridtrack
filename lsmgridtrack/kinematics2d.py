import logging
from dataclasses import dataclass, fields

import pandas as pds
from SimpleITK import Transform
import vtkmodules.all as vtk
from vtkmodules.util import numpy_support
import numpy as np
from .config import GridOptions, ImageOptions

log = logging.getLogger(__name__)


@dataclass
class Kinematics:
    x_coordinates: np.ndarray
    y_coordinates: np.ndarray
    displacements: np.ndarray
    deformation_gradients: np.ndarray
    strains: np.ndarray
    first_principal_strains: np.ndarray
    second_principal_strains: np.ndarray
    first_principal_strain_directions: np.ndarray
    second_principal_strain_directions: np.ndarray
    areal_strains: np.ndarray


def _create_vtk_grid(
    grid_options: GridOptions, image_options: ImageOptions
) -> vtk.vtkRectilinearGrid:
    physical_origin = [
        g * s for g, s in zip(grid_options.origin, image_options.spacing)
    ]
    physical_upper_bound = [
        g * s for g, s in zip(grid_options.upper_bound, image_options.spacing)
    ]

    x_domain = np.linspace(
        physical_origin[0], physical_upper_bound[0], grid_options.divisions[0]
    )
    y_domain = np.linspace(
        physical_origin[1], physical_upper_bound[1], grid_options.divisions[1]
    )
    z_domain = np.array([0.0])

    grid = vtk.vtkRectilinearGrid()
    grid.SetDimensions(x_domain.size, y_domain.size, z_domain.size)
    grid.SetXCoordinates(
        numpy_support.numpy_to_vtk(
            x_domain.ravel(), deep=True, array_type=vtk.VTK_FLOAT
        )
    )
    grid.SetYCoordinates(
        numpy_support.numpy_to_vtk(
            y_domain.ravel(), deep=True, array_type=vtk.VTK_FLOAT
        )
    )
    grid.SetZCoordinates(
        numpy_support.numpy_to_vtk(
            z_domain.ravel(), deep=True, array_type=vtk.VTK_FLOAT
        )
    )

    return grid


def _get_displacements(grid: vtk.vtkRectilinearGrid, transform: Transform):
    num_points = grid.GetNumberOfPoints()
    displacements = np.array(
        [
            np.array(transform.TransformPoint(grid.GetPoint(i)[0:2]))
            - np.array(grid.GetPoint(i)[0:2])
            for i in range(num_points)
        ]
    )

    return np.array(displacements)


def _get_deformation_gradients(
    grid: vtk.vtkRectilinearGrid, displacements: np.ndarray
) -> np.ndarray:
    num_cells = grid.GetNumberOfCells()
    dNdEta = (
        np.array(
            [
                [-1, -1],
                [1, -1],
                [1, 1],
                [-1, 1],
            ],
            float,
        )
        / 4.0
    )
    order = [0, 1, 3, 2]

    Farray = np.zeros((num_cells, 2, 2), float)
    for i in range(num_cells):
        nodeIDs = grid.GetCell(i).GetPointIds()
        X = numpy_support.vtk_to_numpy(grid.GetCell(i).GetPoints().GetData())
        X = X[order, 0:2]
        x = np.zeros_like(X)
        for j, k in enumerate(order):
            x[j, :] = X[j, :] + displacements[nodeIDs.GetId(k), :]
        dXdetaInvTrans = np.transpose(np.linalg.inv(np.einsum("ij,ik", X, dNdEta)))
        dNdX = np.einsum("ij,kj", dNdEta, dXdetaInvTrans)
        F = np.einsum("ij,ik", x, dNdX)
        Farray[i, :, :] = F
    vtkarray = numpy_support.numpy_to_vtk(
        np.transpose(Farray, axes=[0, 2, 1]).ravel(),
        deep=True,
        array_type=vtk.VTK_FLOAT,
    )
    vtkarray.SetName("deformation_gradients")
    vtkarray.SetNumberOfComponents(4)
    grid.GetCellData().AddArray(vtkarray)
    c2p = vtk.vtkCellDataToPointData()
    c2p.SetInputData(grid)
    c2p.Update()
    Farray = np.transpose(
        numpy_support.vtk_to_numpy(
            c2p.GetOutput().GetPointData().GetArray("deformation_gradients")
        ).reshape(-1, 2, 2),
        axes=[0, 2, 1],
    )
    return Farray


def _get_strains(deformation_gradients: np.ndarray) -> np.ndarray:
    strains = np.zeros_like(deformation_gradients)
    for i in range(deformation_gradients.shape[0]):
        F = deformation_gradients[i, :, :]
        strains[i, :, :] = 0.5 * (np.dot(F.T, F) - np.eye(2))
    return strains


def _get_principal_strains(strains: np.ndarray) -> tuple[np.ndarray, ...]:
    first_principal_strains = np.zeros(strains.shape[0], float)
    first_principal_strain_directions = np.zeros((strains.shape[0], 2), float)
    second_principal_strains = np.zeros(strains.shape[0], float)
    second_principal_strain_directions = np.zeros((strains.shape[0], 2), float)
    for i in range(strains.shape[0]):
        E = strains[i, :, :]
        l, v = np.linalg.eigh(E)
        first_principal_strains[i] = l[1]
        first_principal_strain_directions[i, :] = v[:, 1]
        second_principal_strains[i] = l[0]
        second_principal_strain_directions[i, :] = v[:, 0]

    for vec in (first_principal_strain_directions, second_principal_strain_directions):
        flip_array = np.sign(np.dot(vec, vec[0, :]))
        flip_array[np.abs(flip_array) < 1e-5] = 1.0
        vec *= flip_array.reshape(-1, 1)

    return (
        first_principal_strains,
        first_principal_strain_directions,
        second_principal_strains,
        second_principal_strain_directions,
    )


def _get_areal_strains(deformation_gradients: np.ndarray) -> np.ndarray:
    areal_strains = np.zeros(deformation_gradients.shape[0], float)
    for i in range(deformation_gradients.shape[0]):
        areal_strains[i] = np.linalg.det(deformation_gradients[i, :, :]) - 1.0
    return areal_strains


def get_kinematics(
    grid_options: GridOptions,
    image_options: ImageOptions,
    transform: Transform,
) -> Kinematics:
    """
    Args:
        grid_options: Options defining properties of the grid.
        image_options: Options defining properties of the registered images.
        transform: The transform calculated by the image registration.

    Returns:
        results: The kinematics of the grid after deforming with the supplied transform.

    """
    grid = _create_vtk_grid(grid_options, image_options)
    num_points = grid.GetNumberOfPoints()
    results = Kinematics(
        x_coordinates=numpy_support.vtk_to_numpy(grid.GetXCoordinates()),
        y_coordinates=numpy_support.vtk_to_numpy(grid.GetYCoordinates()),
        displacements=np.zeros((num_points, 2), float),
        deformation_gradients=np.zeros((num_points, 2, 2), float),
        strains=np.zeros((num_points, 2, 2), float),
        first_principal_strains=np.zeros(num_points, float),
        second_principal_strains=np.zeros(num_points, float),
        first_principal_strain_directions=np.zeros((num_points, 2), float),
        second_principal_strain_directions=np.zeros((num_points, 2), float),
        areal_strains=np.zeros(num_points, float),
    )

    results.displacements = _get_displacements(grid, transform)
    results.deformation_gradients = _get_deformation_gradients(
        grid, results.displacements
    )

    results.strains = _get_strains(results.deformation_gradients)
    (
        results.first_principal_strains,
        results.first_principal_strain_directions,
        results.second_principal_strains,
        results.second_principal_strain_directions,
    ) = _get_principal_strains(results.strains)
    results.areal_strains = _get_areal_strains(results.deformation_gradients)
    log.info("Calculated kinematics from provided transform and reference image.")
    return results


def convert_kinematics_to_vtk(kinematics: Kinematics) -> vtk.vtkRectilinearGrid:
    grid = vtk.vtkRectilinearGrid()
    grid.SetDimensions(kinematics.x_coordinates.size, kinematics.y_coordinates.size, 1)
    grid.SetXCoordinates(
        numpy_support.numpy_to_vtk(
            kinematics.x_coordinates, deep=True, array_type=vtk.VTK_FLOAT
        )
    )
    grid.SetYCoordinates(
        numpy_support.numpy_to_vtk(
            kinematics.y_coordinates, deep=True, array_type=vtk.VTK_FLOAT
        )
    )
    grid.SetZCoordinates(
        numpy_support.numpy_to_vtk(np.ravel([0.0]), deep=True, array_type=vtk.VTK_FLOAT)
    )
    num_points = grid.GetNumberOfPoints()
    num_cells = grid.GetNumberOfCells()
    for field in fields(kinematics):
        if "coordinates" in field.name:
            continue
        value = getattr(kinematics, field.name).copy()
        if len(value.shape) == 3:
            new_array = np.zeros((value.shape[0], 3, 3), float)
            new_array[:, 0:2, 0:2] = value
            if "gradient" in field.name:
                new_array[:, 2, 2] = 1.0
            value = np.transpose(new_array, axes=[0, 2, 1])
        elif len(value.shape) == 2:
            value = np.concatenate(
                [value, np.zeros((value.shape[0], 1), float)], axis=1
            )

        vtk_array = numpy_support.numpy_to_vtk(
            value.ravel(),
            deep=True,
            array_type=vtk.VTK_FLOAT,
        )
        vtk_array.SetName(field.name)
        if len(value.shape) > 1:
            vtk_array.SetNumberOfComponents(int(np.product(value.shape[1:])))
        else:
            vtk_array.SetNumberOfComponents(1)
        if value.shape[0] == num_points:
            grid.GetPointData().AddArray(vtk_array)
        elif value.shape[0] == num_cells:
            grid.GetCellData().AddArray(vtk_array)
        else:
            raise ValueError
    return grid


def write_to_vtk(data: vtk.vtkRectilinearGrid, name: str = "output"):
    writer = vtk.vtkXMLRectilinearGridWriter()
    writer.SetFileName(".".join([name, "vtr"]))
    writer.SetInputData(data)
    writer.Write()


def write_kinematics_to_vtk(data: Kinematics, name: str = "output"):
    grid = convert_kinematics_to_vtk(data)
    write_to_vtk(grid, name)


def convert_kinematics_to_pandas(results: Kinematics) -> pds.DataFrame:
    x, y = np.meshgrid(results.x_coordinates, results.y_coordinates)
    coordinates = np.zeros((x.size, 2), float)
    coordinates[:, 0] = x.ravel()
    coordinates[:, 1] = y.ravel()

    vector_suffixes = ("X", "Y")
    tensor_suffixes = ("XX", "XY", "YX", "YY")
    df = pds.DataFrame(coordinates, columns=["X", "Y"])
    for field in fields(results):
        if "coordinates" in field.name:
            continue
        value = getattr(results, field.name)
        if len(value.shape) == 1:
            df = pds.concat(
                [df, pds.DataFrame(value, columns=[field.name])],
                axis=1,
            )
        elif len(value.shape) == 2:
            df = pds.concat(
                [
                    df,
                    pds.DataFrame(
                        value,
                        columns=[
                            f"{field.name} ({suffix})" for suffix in vector_suffixes
                        ],
                    ),
                ],
                axis=1,
            )

        elif len(value.shape) == 3:
            view = value.reshape(-1, np.prod(value.shape[1:]))
            if "strain" in field.name:
                cols = [0, 1, 3]
                df = pds.concat(
                    [
                        df,
                        pds.DataFrame(
                            view[:, cols],
                            columns=[
                                f"{field.name} ({tensor_suffixes[col]})" for col in cols
                            ],
                        ),
                    ],
                    axis=1,
                )
            else:
                pds.concat(
                    [
                        df,
                        pds.DataFrame(
                            view,
                            columns=[
                                f"{field.name} ({suffix})" for suffix in tensor_suffixes
                            ],
                        ),
                    ],
                    axis=1,
                )
        else:
            raise ValueError

    return df


def write_kinematics_to_excel(results: Kinematics, name: str):
    df = convert_kinematics_to_pandas(results)
    df.to_excel(f"{name}.xlsx")
