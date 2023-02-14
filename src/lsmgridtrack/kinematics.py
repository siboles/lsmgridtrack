import vtk
import SimpleITK as sitk
from vtk.util import numpy_support
import numpy as np
from .config import GridOptions


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
    displacements = numpy_support.numpy_to_vtk(
        np.ravel(displacements), deep=True, array_type=vtk.vtkDouble
    )
    displacements.SetNumberOfComponents(3)
    displacements.SetName("Displacement")
    grid.GetPointData().AddArray(displacements)
    return grid


def _get_deformation_gradients(grid: vtk.vtkImageData):
    num_cells = grid.GetNumberOfCells()
    dNdEta = (
        np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
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


def get_kinematics(
    options: GridOptions, transform: sitk.Transform, reference_image: sitk.Image
):
    grid = _create_vtk_grid(options, reference_image)
    grid = _get_displacements(grid, transform)
