import logging
from dataclasses import dataclass, fields

import numpy as np
import pandas as pds
import vtkmodules.all as vtk
from SimpleITK import ReadTransform, Transform
from vtkmodules.util import numpy_support

from .config import GridOptions, ImageOptions

log = logging.getLogger(__name__)


@dataclass
class Kinematics:
    """The calculated kinematics."""

    x_coordinates: np.ndarray
    y_coordinates: np.ndarray
    z_coordinates: np.ndarray
    displacements: np.ndarray
    deformation_gradients: np.ndarray
    strains: np.ndarray
    first_principal_strains: np.ndarray
    second_principal_strains: np.ndarray
    third_principal_strains: np.ndarray
    first_principal_strain_directions: np.ndarray
    second_principal_strain_directions: np.ndarray
    third_principal_strain_directions: np.ndarray
    volumetric_strains: np.ndarray


def _create_vtk_grid(
    grid_options: GridOptions, image_options: ImageOptions
) -> vtk.vtkRectilinearGrid:
    """

    :param grid_options:
    :param image_options:
    :return:
    """
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
    z_domain = np.linspace(
        physical_origin[2], physical_upper_bound[2], grid_options.divisions[2]
    )

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


def _get_displacements(
    grid: vtk.vtkRectilinearGrid, transform: Transform
) -> np.ndarray:
    """

    :param grid:
    :param transform:
    :return:
    """
    num_points = grid.GetNumberOfPoints()
    displacements = np.array(
        [
            np.array(transform.TransformPoint(grid.GetPoint(i)))
            - np.array(grid.GetPoint(i))
            for i in range(num_points)
        ]
    )

    return np.array(displacements)


def _get_deformation_gradients(
    grid: vtk.vtkRectilinearGrid, displacements: np.ndarray
) -> np.ndarray:
    """

    :param grid:
    :param displacements:
    :return:
    """
    num_cells = grid.GetNumberOfCells()
    dNdEta = (
        np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ],
            float,
        )
        / 8.0
    )
    order = [0, 1, 3, 2, 4, 5, 7, 6]

    Farray = np.zeros((num_cells, 3, 3), float)
    for i in range(num_cells):
        nodeIDs = grid.GetCell(i).GetPointIds()
        X = numpy_support.vtk_to_numpy(grid.GetCell(i).GetPoints().GetData())
        X = X[order, :]
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
    vtkarray.SetNumberOfComponents(9)
    grid.GetCellData().AddArray(vtkarray)
    c2p = vtk.vtkCellDataToPointData()
    c2p.SetInputData(grid)
    c2p.Update()
    Farray = np.transpose(
        numpy_support.vtk_to_numpy(
            c2p.GetOutput().GetPointData().GetArray("deformation_gradients")
        ).reshape(-1, 3, 3),
        axes=[0, 2, 1],
    )

    return Farray


def _get_strains(deformation_gradients: np.ndarray) -> np.ndarray:
    """

    :param deformation_gradients:
    :return:
    """
    strains = np.zeros_like(deformation_gradients)
    for i in range(deformation_gradients.shape[0]):
        F = deformation_gradients[i, :, :]
        strains[i, :, :] = 0.5 * (np.dot(F.T, F) - np.eye(3))
    return strains


def _get_principal_strains(strains: np.ndarray) -> tuple[np.ndarray, ...]:
    """

    :param strains:
    :return:
    """
    first_principal_strains = np.zeros(strains.shape[0], float)
    first_principal_strain_directions = np.zeros((strains.shape[0], 3), float)
    second_principal_strains = np.zeros(strains.shape[0], float)
    second_principal_strain_directions = np.zeros((strains.shape[0], 3), float)
    third_principal_strains = np.zeros(strains.shape[0], float)
    third_principal_strain_directions = np.zeros((strains.shape[0], 3), float)
    for i in range(strains.shape[0]):
        E = strains[i, :, :]
        l, v = np.linalg.eigh(E)
        first_principal_strains[i] = l[2]
        first_principal_strain_directions[i, :] = v[:, 2]
        second_principal_strains[i] = l[1]
        second_principal_strain_directions[i, :] = v[:, 1]
        third_principal_strains[i] = l[0]
        third_principal_strain_directions[i, :] = v[:, 0]

    for vec in (
        first_principal_strain_directions,
        second_principal_strain_directions,
        third_principal_strain_directions,
    ):
        flip_array = np.sign(np.dot(vec, vec[0, :]))
        flip_array[np.abs(flip_array) < 1e-5] = 1.0
        vec *= flip_array.reshape(-1, 1)

    return (
        first_principal_strains,
        first_principal_strain_directions,
        second_principal_strains,
        second_principal_strain_directions,
        third_principal_strains,
        third_principal_strain_directions,
    )


def _get_volumetric_strains(deformation_gradients: np.ndarray) -> np.ndarray:
    """

    :param deformation_gradients:
    :return:
    """
    volumetric_strains = np.zeros(deformation_gradients.shape[0], float)
    for i in range(deformation_gradients.shape[0]):
        volumetric_strains[i] = np.linalg.det(deformation_gradients[i, :, :]) - 1.0
    return volumetric_strains


def get_kinematics(
    grid_options: GridOptions,
    image_options: ImageOptions,
    transform: Transform,
) -> Kinematics:
    r"""
    Calculate kinematics based on registration transform.
    Calculates the deformation gradient at element centroids and interpolates to the VTK
    grid vertices. The Green-Lagrange strain can then be calculated from the
    deformation gradients. The principal strains are the eigenvalues of the
    strain tensors directed along the eigenvectors. Assuming Einstein's summation
    convention unless explicitly indicated, these calculations are as follows:

    .. note::
        Capital and lowercase letters imply reference and deformed configurations,
        respectively.

    .. math::

        F^i_{\,J} = \sum_{a=1}^{4} x^i_{\,a}\frac{\partial N_a}{\partial X^J}.

    We therefore need to determine :math:`\frac{\partial N_a}{\partial X^J}`.
    From the chain rule,

    .. math::

        \frac{\partial N_a}{\partial X^J} =
        \frac{\partial N_a}{\partial \eta^I}
        \left (\frac{\partial X^I}{\partial \eta^J} \right)^{-T}

    where

    .. math::

        \frac{\partial X^I}{\partial \eta^J} =
        \sum_{a=1}^{4} X^I_{\,a} \frac{\partial N_a}{\partial \eta^J}.

    The Green-Lagrange strain tensor then follows as,

    .. math::

        E_{IJ} = \frac{1}{2}\left(F_I^{\,i} g_{ij} F^j_{\,J} - G_{IJ}\right)

    where :math:`g_{ij}` is the spatial metric tensor and :math:`G_{IJ}` is the material
    metric tensor (both are the identity in Cartesian).

    The eigenvalues * eigenvectors of this tensor ordered decreasing by eigenvalue
    are the principal strains.

    The areal strain is,

    .. math::

        E_{areal} = \det{F^i_{\,J}} - 1.0.

    :param grid_options: Options defining properties of the grid.
    :param image_options: Options defining properties of the registered images.
    :param transform: The transform calculated by the image registration.
    :return: The kinematics of the grid after deforming with the supplied transform.
    """
    grid = _create_vtk_grid(grid_options, image_options)
    num_points = grid.GetNumberOfPoints()
    results = Kinematics(
        x_coordinates=numpy_support.vtk_to_numpy(grid.GetXCoordinates()),
        y_coordinates=numpy_support.vtk_to_numpy(grid.GetYCoordinates()),
        z_coordinates=numpy_support.vtk_to_numpy(grid.GetZCoordinates()),
        displacements=np.zeros((num_points, 3), float),
        deformation_gradients=np.zeros((num_points, 3, 3), float),
        strains=np.zeros((num_points, 3, 3), float),
        first_principal_strains=np.zeros(num_points, float),
        second_principal_strains=np.zeros(num_points, float),
        third_principal_strains=np.zeros(num_points, float),
        first_principal_strain_directions=np.zeros((num_points, 3), float),
        second_principal_strain_directions=np.zeros((num_points, 3), float),
        third_principal_strain_directions=np.zeros((num_points, 3), float),
        volumetric_strains=np.zeros(num_points, float),
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
        results.third_principal_strains,
        results.third_principal_strain_directions,
    ) = _get_principal_strains(results.strains)
    results.volumetric_strains = _get_volumetric_strains(results.deformation_gradients)
    log.info("Calculated kinematics from provided transform and reference image.")
    return results


def convert_kinematics_to_vtk(kinematics: Kinematics) -> vtk.vtkRectilinearGrid:
    """

    :param kinematics:
    :raises ValueError:
    :return:
    """
    grid = vtk.vtkRectilinearGrid()
    grid.SetDimensions(
        kinematics.x_coordinates.size,
        kinematics.y_coordinates.size,
        kinematics.z_coordinates.size,
    )
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
        numpy_support.numpy_to_vtk(
            kinematics.z_coordinates, deep=True, array_type=vtk.VTK_FLOAT
        )
    )
    for field in fields(kinematics):
        if "coordinates" in field.name:
            continue
        value = getattr(kinematics, field.name).copy()

        if len(value.shape) == 3:
            value = np.transpose(value, axes=[0, 2, 1])

        vtk_array = numpy_support.numpy_to_vtk(
            value.ravel(),
            deep=True,
            array_type=vtk.VTK_FLOAT,
        )
        vtk_array.SetName(field.name)
        if len(value.shape) > 1:
            vtk_array.SetNumberOfComponents(int(np.prod(value.shape[1:])))
        else:
            vtk_array.SetNumberOfComponents(1)
        grid.GetPointData().AddArray(vtk_array)
    return grid


def write_to_vtk(data: vtk.vtkRectilinearGrid, name: str = "output"):
    """

    :param data:
    :param name:
    """
    writer = vtk.vtkXMLRectilinearGridWriter()
    writer.SetFileName(f"{name}.vtr")
    writer.SetInputData(data)
    writer.Write()


def write_kinematics_to_vtk(data: Kinematics, name: str = "output"):
    """

    :param data:
    :param name:
    """
    grid = convert_kinematics_to_vtk(data)
    write_to_vtk(grid, name)


def convert_kinematics_to_pandas(results: Kinematics) -> pds.DataFrame:
    """

    :param results:
    :raises ValueError:
    :return:
    """
    x, y, z = np.meshgrid(
        results.x_coordinates, results.y_coordinates, results.z_coordinates
    )
    coordinates = np.zeros((x.size, 3), float)
    coordinates[:, 0] = x.ravel()
    coordinates[:, 1] = y.ravel()
    coordinates[:, 2] = z.ravel()

    vector_suffixes = ("X", "Y", "Z")
    tensor_suffixes = ("XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ")
    df = pds.DataFrame(coordinates, columns=["X", "Y", "Z"])
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
                cols = [0, 1, 2, 4, 5, 8]
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
    """
    Write the calculated kinematics to an excel file.

    :param results: Calculated Kinematics data object.

    :param name: Filename without extension.
    """
    df = convert_kinematics_to_pandas(results)
    df.to_excel(f"{name}.xlsx")


def read_transform(filepath: str) -> Transform:
    transform = ReadTransform(filepath)
    if transform.GetDimension() == 2:
        raise ValueError(
            f"Transform loaded from {filepath} has dimension 2. "
            "The kinematics2 module should be used instead."
        )
    return transform
