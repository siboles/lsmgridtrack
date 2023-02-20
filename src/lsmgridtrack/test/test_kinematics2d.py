import pytest
import numpy as np
import unittest
from .. import kinematics2d as kinematics
import SimpleITK as sitk
import vtkmodules.all as vtk
from vtkmodules.util import numpy_support


@pytest.fixture(scope="module")
def create_image():
    image = sitk.Image(10, 10, sitk.sitkFloat32)
    image.SetOrigin([0, 0])
    image.SetSpacing([1.0, 1.0])
    return image


@pytest.fixture(scope="module")
def create_image_options():
    return kinematics.ImageOptions(dimension=2, spacing=[1.0, 1.0])


@pytest.fixture(scope="module")
def create_grid_options():
    return kinematics.GridOptions(
        origin=[0, 0, 0], upper_bound=[9, 9, 0], divisions=[9, 9, 1]
    )


@pytest.fixture(scope="module")
def create_grid_standard():
    grid = vtk.vtkRectilinearGrid()
    grid.SetDimensions(9, 9, 1)
    coords = np.linspace(0.0, 9.0, 9)
    grid.SetXCoordinates(
        numpy_support.numpy_to_vtk(coords, deep=True, array_type=vtk.VTK_FLOAT)
    )
    grid.SetYCoordinates(
        numpy_support.numpy_to_vtk(coords, deep=True, array_type=vtk.VTK_FLOAT)
    )
    grid.SetZCoordinates(
        numpy_support.numpy_to_vtk([0.0], deep=True, array_type=vtk.VTK_FLOAT)
    )
    return grid


@pytest.fixture(scope="module")
def create_transform(create_image):
    transform = sitk.BSplineTransformInitializer(create_image, (3, 3, 1), 3)
    N = 432
    transform.SetParameters([np.random.uniform(-1.0, 1.0) for _ in range(N)])
    return transform


def test_grid_creation_2d(
    create_grid_options, create_image_options, create_grid_standard
):
    case = unittest.TestCase()
    grid = kinematics._create_vtk_grid(create_grid_options, create_image_options)
    case.assertTupleEqual(grid.GetDimensions(), create_grid_standard.GetDimensions())


def test_get_kinematics_2d(
    create_grid_options,
    create_image_options,
    create_transform,
    create_grid_standard,
):
    results = kinematics.get_kinematics(
        create_grid_options, create_image_options, create_transform
    )
    case = unittest.TestCase()
    num_points = create_grid_standard.GetNumberOfPoints()
    case.assertTupleEqual(results.displacements.shape, (num_points, 2))
    case.assertTupleEqual(results.deformation_gradients.shape, (num_points, 2, 2))
    case.assertTupleEqual(results.strains.shape, (num_points, 2, 2))
    case.assertTupleEqual(results.first_principal_strains.shape, (num_points,))
    case.assertTupleEqual(
        results.first_principal_strain_directions.shape, (num_points, 2)
    )
    case.assertTupleEqual(results.areal_strains.shape, (num_points,))

    results_grid = kinematics.convert_kinematics_to_vtk(results)
    kinematics.write_kinematics_to_vtk(results, "test_2d")
    kinematics.write_kinematics_to_excel(results, "test_2d")
