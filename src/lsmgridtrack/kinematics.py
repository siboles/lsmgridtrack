import vtkmodules.all as vtk
import SimpleITK as sitk
from vtkmodules.util import numpy_support
import numpy as np
from .config import GridOptions
from dataclasses import dataclass


@dataclass
class Kinematics:
    displacements: np.ndarray
    deformation_gradients: np.ndarray
    strains: np.ndarray
    principal_strains: np.ndarray
    volumetric_strains: np.ndarray


def _create_vtk_grid(
    options: GridOptions, reference_image: sitk.Image
) -> vtk.vtkImageData:
    grid = vtk.vtkImageData()
    physical_origin = reference_image.TransformIndexToPhysicalPoint(options.origin)
    physical_upper_bound = reference_image.TransformIndexToPhysicalPoint(
        options.upper_bound
    )
    physical_spacing = [
        (u - o) / d if d > 0 else 1.0
        for (u, o, d) in zip(physical_upper_bound, physical_origin, options.divisions)
    ]
    grid.SetOrigin(physical_origin)
    grid.SetSpacing(physical_spacing)
    grid.SetExtent(
        0,
        options.divisions[0],
        0,
        options.divisions[1],
        0,
        options.divisions[2],
    )
    return grid


def _get_displacements(grid: vtk.vtkImageData, transform: sitk.Transform):
    num_points = grid.GetNumberOfPoints()
    displacements = [
        transform.TransformPoint(grid.GetPoint(i)) for i in range(num_points)
    ]
    return np.array(displacements)


def _get_deformation_gradients_2d(
    grid: vtk.vtkImageData, displacements: np.ndarray
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
    grid: vtk.vtkImageData, displacements: np.ndarray
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
    options: GridOptions, transform: sitk.Transform, reference_image: sitk.Image
) -> Kinematics:
    """
    Args:
        options: Options defining properties of the vtk.ImageData grid.
        transform: The transform calculated by the image registration.
        reference_image: The reference image used in the registration.

    Returns:
        results: The kinematics of the grid after deforming with the supplied transform.

    """
    grid = _create_vtk_grid(options, reference_image)
    num_points = grid.GetNumberOfPoints()
    num_cells = grid.GetNumberOfCells()
    results = Kinematics(
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
    return results
