import logging
from SimpleITK import Transform
import vtkmodules.all as vtk
from vtkmodules.util import numpy_support
import numpy as np
from .config import GridOptions, ImageOptions
from dataclasses import dataclass, fields

log = logging.getLogger(__name__)


@dataclass
class Kinematics:
    x_coordinates: np.ndarray
    y_coordinates: np.ndarray
    z_coordinates: np.ndarray
    displacements: np.ndarray
    deformation_gradients: np.ndarray
    strains: np.ndarray
    principal_strains: np.ndarray
    volumetric_strains: np.ndarray


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
    if image_options.dimension == 2:
        z_domain = np.array([0.0])
    else:
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


def _get_displacements(grid: vtk.vtkRectilinearGrid, transform: Transform):
    num_points = grid.GetNumberOfPoints()
    displacements = [
        transform.TransformPoint(grid.GetPoint(i)) for i in range(num_points)
    ]
    return np.array(displacements)


def _get_deformation_gradients_2d(
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

    Farray = np.zeros((num_cells, 3, 3), float)
    for i in range(num_cells):
        nodeIDs = grid.GetCell(i).GetPointIds()
        X = numpy_support.vtk_to_numpy(grid.GetCell(i).GetPoints().GetData())
        X = X[order, 0:2]
        x = np.zeros_like(X)
        for j, k in enumerate(order):
            x[j, :] = X[j, :] + displacements[nodeIDs.GetId(k), 0:2]
        dXdetaInvTrans = np.transpose(np.linalg.inv(np.einsum("ij,ik", X, dNdEta)))
        dNdX = np.einsum("ij,kj", dNdEta, dXdetaInvTrans)
        F = np.einsum("ij,ik", x, dNdX)
        Farray[i, 0:2, 0:2] = F
        Farray[i, 2, 2] = 1.0
    return Farray


def _get_deformation_gradients(
    grid: vtk.vtkRectilinearGrid, displacements: np.ndarray
) -> np.ndarray:
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
    return Farray


def _get_strains(deformation_gradients: np.ndarray) -> np.ndarray:
    strains = np.zeros_like(deformation_gradients)
    for i in range(deformation_gradients.shape[0]):
        F = deformation_gradients[i, :, :]
        strains[i, :, :] = 0.5 * (np.dot(F.T, F) - np.eye(3))
    return strains


def _get_principal_strains(strains: np.ndarray) -> np.ndarray:
    principal_strains = np.zeros_like(strains)
    for i in range(strains.shape[0]):
        E = strains[i, :, :]
        l, v = np.linalg.eigh(E)
        principal_strains[i, :, :] = l[::-1] * v[:, ::-1]
    for i in np.arange(1, principal_strains.shape[0]):
        for j in range(3):
            if np.dot(principal_strains[0, :, j], principal_strains[i, :, j]) < 0:
                principal_strains[i, :, j] *= -1.0
    return principal_strains


def _get_volumetric_strains(deformation_gradients: np.ndarray) -> np.ndarray:
    volumetric_strains = np.zeros(deformation_gradients.shape[0], float)
    for i in range(deformation_gradients.shape[0]):
        volumetric_strains[i] = np.linalg.det(deformation_gradients[i, :, :])
    return volumetric_strains


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
    num_cells = grid.GetNumberOfCells()
    results = Kinematics(
        x_coordinates=numpy_support.vtk_to_numpy(grid.GetXCoordinates()),
        y_coordinates=numpy_support.vtk_to_numpy(grid.GetYCoordinates()),
        z_coordinates=numpy_support.vtk_to_numpy(grid.GetZCoordinates()),
        displacements=np.zeros((num_points, 3), float),
        deformation_gradients=np.zeros((num_cells, 3, 3), float),
        strains=np.zeros((num_cells, 3, 3), float),
        principal_strains=np.zeros((num_cells, 3, 3), float),
        volumetric_strains=np.zeros(num_cells, float),
    )

    results.displacements = _get_displacements(grid, transform)
    if grid.GetExtent()[-1] == 0:
        results.deformation_gradients = _get_deformation_gradients_2d(
            grid, results.displacements
        )
    else:
        results.deformation_gradients = _get_deformation_gradients(
            grid, results.displacements
        )

    results.strains = _get_strains(results.deformation_gradients)
    results.principal_strains = _get_principal_strains(results.strains)
    results.volumetric_strains = _get_volumetric_strains(results.deformation_gradients)
    log.info("Calculated kinematics from provided transform and reference image.")
    return results


def convert_kinematics_to_vtk(kinematics: Kinematics) -> vtk.vtkRectilinearGrid:
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
    num_points = grid.GetNumberOfPoints()
    num_cells = grid.GetNumberOfCells()
    for field in fields(kinematics):
        if "coordinates" in field.name:
            continue
        value = getattr(kinematics, field.name)
        vtk_array = numpy_support.numpy_to_vtk(
            value.ravel(),
            deep=True,
            array_type=vtk.VTK_FLOAT,
        )
        vtk_array.SetName(field.name)
        if len(value.shape) > 1:
            vtk_array.SetNumberOfComponents(np.product(value.shape[1:]))
        else:
            vtk_array.SetNumberOfComponents(1)
        if value.shape[0] == num_points:
            grid.GetPointData().AddArray(vtk_array)
        elif value.shape[0] == num_cells:
            grid.GetCellData().AddArray(vtk_array)
        else:
            raise ValueError
    return grid


def write_kinematics_to_vtk():
    pass
