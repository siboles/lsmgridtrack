import vtk
import SimpleITK as sitk
from vtk.util import numpy_support
import numpy as np
from .config import GridOptions
from dataclasses import dataclass


@dataclass
class Kinematics:
    displacements: np.ndarray
    deformation_gradients: np.ndarray
    strains: np.ndarray
    first_principal_strains: np.ndarray
    second_principal_strains: np.ndarray
    third_principal_strains: np.ndarray
    volumetric_strains: np.ndarray


def _create_vtk_grid(options: GridOptions, reference_image: sitk.Image):
    adjusted_size = np.array(options.size) * options.upsampling - (
        options.upsampling - 1
    )
    adjusted_spacing = np.array(options.spacing) / float(options.upsampling)
    grid = vtk.vtkImageData()
    grid.SetOrigin(reference_image.TransformIndexToPhysicalPoint(options.origin))
    grid.SetSpacing(reference_image.TransformIndexToPhysicalPoint(adjusted_spacing))
    grid.SetExtent(
        0, adjusted_size[0] - 1, 0, adjusted_size[1] - 1, 0, adjusted_size[2] - 1
    )
    return grid


def _get_displacements(grid: vtk.vtkImageData, transform: sitk.Transform):
    num_points = grid.GetNumberOfPoints()
    displacements = [
        transform.TransformPoint(grid.GetPoint(i)) for i in range(num_points)
    ]
    return np.array(displacements)


def _get_deformation_gradients_2d(grid: vtk.vtkImageData, displacements: np.ndarray):
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
        nodeIDs = grid.GetCell(i).GetPointIDs()
        X = numpy_support.vtk_to_numpy(grid.GetCell(i).GetPoints().Getdata())
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


def _get_deformation_gradients(grid: vtk.vtkImageData, displacements: np.ndarray):
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
        nodeIDs = grid.GetCell(i).GetPointIDs()
        X = numpy_support.vtk_to_numpy(grid.GetCell(i).GetPoints().Getdata())
        X = X[order, :]
        x = np.zeros_like(X)
        for j, k in enumerate(order):
            x[j, :] = X[j, :] + displacements[nodeIDs.GetId(k), :]
        dXdetaInvTrans = np.transpose(np.linalg.inv(np.einsum("ij,ik", X, dNdEta)))
        dNdX = np.einsum("ij,kj", dNdEta, dXdetaInvTrans)
        F = np.einsum("ij,ik", x, dNdX)
        Farray[i, :, :] = F
    return Farray


def _get_strains(deformation_gradients: np.ndarray):
    strains = np.zeros_like(deformation_gradients)
    for i in range(deformation_gradients.shape[0]):
        F = deformation_gradients[i, :, :]
        strains[i, :, :] = 0.5 * (np.dot(F.T, F) - np.eye(3))
    return strains


def get_kinematics(
    options: GridOptions, transform: sitk.Transform, reference_image: sitk.Image
):
    grid = _create_vtk_grid(options, reference_image)
    results = Kinematics(
        displacements=np.empty(),
        deformation_gradients=np.empty(),
        strains=np.empty(),
        first_principal_strains=np.empty(),
        second_principal_strains=np.empty(),
        third_principal_strains=np.empty(),
        volumetric_strains=np.empty(),
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
