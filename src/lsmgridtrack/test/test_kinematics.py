import pytest
import numpy as np
import unittest
from .. import kinematics
import SimpleITK as sitk
import vtkmodules.all as vtk


@pytest.fixture(scope="module")
def create_3d_image():
    image = sitk.Image(10, 10, 10, sitk.sitkFloat32)
    image.SetOrigin([0, 0, 0])
    image.SetSpacing([1.0, 1.0, 1.0])
    return image


@pytest.fixture(scope="module")
def create_3d_grid_options():
    return kinematics.GridOptions(
        origin=[0, 0, 0], upper_bound=[9, 9, 9], divisions=[3, 3, 3]
    )


@pytest.fixture(scope="module")
def create_3d_grid_standard():
    grid = vtk.vtkImageData()
    grid.SetOrigin([0.0, 0.0, 0.0])
    grid.SetSpacing([3.0, 3.0, 3.0])
    grid.SetExtent(0, 3, 0, 3, 0, 3)
    return grid


@pytest.fixture(scope="module")
def create_3d_transform(create_3d_image):
    transform = sitk.BSplineTransformInitializer(create_3d_image, (3, 3, 3), 3)
    N = 648
    transform.SetParameters([np.random.uniform(-0.1, 0.1) for _ in range(N)])
    return transform


@pytest.fixture(scope="module")
def create_2d_image():
    image = sitk.Image(10, 10, 0, sitk.sitkFloat32)
    image.SetOrigin([0, 0, 0])
    image.SetSpacing([1.0, 1.0, 1.0])
    return image


@pytest.fixture(scope="module")
def create_2d_grid_options():
    return kinematics.GridOptions(
        origin=[0, 0, 0], upper_bound=[9, 9, 0], divisions=[3, 3, 0]
    )


@pytest.fixture(scope="module")
def create_2d_grid_standard():
    grid = vtk.vtkImageData()
    grid.SetOrigin([0.0, 0.0, 0.0])
    grid.SetSpacing([3.0, 3.0, 1.0])
    grid.SetExtent(0, 3, 0, 3, 0, 0)
    return grid


@pytest.fixture(scope="module")
def create_2d_transform(create_2d_image):
    transform = sitk.BSplineTransformInitializer(create_2d_image, (3, 3, 1), 3)
    N = 432
    transform.SetParameters([np.random.uniform(-0.1, 0.1) for _ in range(N)])
    return transform


def test_grid_creation_3d(
    create_3d_grid_options, create_3d_grid_standard, create_3d_image
):
    case = unittest.TestCase()
    grid = kinematics._create_vtk_grid(create_3d_grid_options, create_3d_image)
    case.assertTupleEqual(grid.GetOrigin(), create_3d_grid_standard.GetOrigin())
    case.assertTupleEqual(grid.GetSpacing(), create_3d_grid_standard.GetSpacing())
    case.assertTupleEqual(grid.GetExtent(), create_3d_grid_standard.GetExtent())


def test_get_kinematics_3d(
    create_3d_grid_options,
    create_3d_transform,
    create_3d_image,
    create_3d_grid_standard,
):
    results = kinematics.get_kinematics(
        create_3d_grid_options, create_3d_transform, create_3d_image
    )
    case = unittest.TestCase()
    num_cells = create_3d_grid_standard.GetNumberOfCells()
    num_points = create_3d_grid_standard.GetNumberOfPoints()
    case.assertTupleEqual(results.displacements.shape, (num_points, 3))
    case.assertTupleEqual(results.deformation_gradients.shape, (num_cells, 3, 3))
    case.assertTupleEqual(results.strains.shape, (num_cells, 3, 3))
    case.assertTupleEqual(results.principal_strains.shape, (num_cells, 3, 3))
    case.assertTupleEqual(results.volumetric_strains.shape, (num_cells,))


def test_grid_creation_2d(
    create_2d_grid_options, create_2d_grid_standard, create_2d_image
):
    case = unittest.TestCase()
    grid = kinematics._create_vtk_grid(create_2d_grid_options, create_2d_image)
    case.assertTupleEqual(grid.GetOrigin(), create_2d_grid_standard.GetOrigin())
    case.assertTupleEqual(grid.GetSpacing(), create_2d_grid_standard.GetSpacing())
    case.assertTupleEqual(grid.GetExtent(), create_2d_grid_standard.GetExtent())


def test_get_kinematics_2d(
    create_2d_grid_options,
    create_2d_transform,
    create_2d_image,
    create_2d_grid_standard,
):
    results = kinematics.get_kinematics(
        create_2d_grid_options, create_2d_transform, create_2d_image
    )
    case = unittest.TestCase()
    num_cells = create_2d_grid_standard.GetNumberOfCells()
    num_points = create_2d_grid_standard.GetNumberOfPoints()
    case.assertTupleEqual(results.displacements.shape, (num_points, 3))
    case.assertTupleEqual(results.deformation_gradients.shape, (num_cells, 3, 3))
    case.assertTupleEqual(results.strains.shape, (num_cells, 3, 3))
    case.assertTupleEqual(results.principal_strains.shape, (num_cells, 3, 3))
    case.assertTupleEqual(results.volumetric_strains.shape, (num_cells,))
