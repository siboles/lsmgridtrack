import pytest
import numpy as np
import unittest
from .. import kinematics
import SimpleITK as sitk
import vtkmodules.all as vtk
from vtkmodules.util import numpy_support


@pytest.fixture(scope="module")
def create_3d_image():
    image = sitk.Image(10, 10, 10, sitk.sitkFloat32)
    image.SetOrigin([0, 0, 0])
    image.SetSpacing([1.0, 1.0, 1.0])
    return image


@pytest.fixture(scope="module")
def create_3d_image_options():
    return kinematics.ImageOptions(dimension=3, spacing=[1.0, 1.0, 1.0])


@pytest.fixture(scope="module")
def create_3d_grid_options():
    return kinematics.GridOptions(
        origin=[0, 0, 0], upper_bound=[9, 9, 9], divisions=[9, 9, 9]
    )


@pytest.fixture(scope="module")
def create_3d_grid_standard():
    grid = vtk.vtkRectilinearGrid()
    grid.SetDimensions(9, 9, 9)
    coords = np.linspace(0.0, 9.0, 9)
    grid.SetXCoordinates(
        numpy_support.numpy_to_vtk(coords, deep=True, array_type=vtk.VTK_FLOAT)
    )
    grid.SetYCoordinates(
        numpy_support.numpy_to_vtk(coords, deep=True, array_type=vtk.VTK_FLOAT)
    )
    grid.SetZCoordinates(
        numpy_support.numpy_to_vtk(coords, deep=True, array_type=vtk.VTK_FLOAT)
    )
    return grid


@pytest.fixture(scope="module")
def create_3d_transform(create_3d_image):
    transform = sitk.BSplineTransformInitializer(create_3d_image, (3, 3, 3), 3)
    N = 648
    transform.SetParameters([np.random.uniform(-1.0, 1.0) for _ in range(N)])
    return transform


def test_grid_creation_3d(
    create_3d_grid_options, create_3d_image_options, create_3d_grid_standard
):
    case = unittest.TestCase()
    grid = kinematics._create_vtk_grid(create_3d_grid_options, create_3d_image_options)
    case.assertTupleEqual(grid.GetDimensions(), create_3d_grid_standard.GetDimensions())


def test_get_kinematics_3d(
    create_3d_grid_options,
    create_3d_image_options,
    create_3d_transform,
    create_3d_grid_standard,
):
    results = kinematics.get_kinematics(
        create_3d_grid_options, create_3d_image_options, create_3d_transform
    )
    case = unittest.TestCase()
    num_points = create_3d_grid_standard.GetNumberOfPoints()
    case.assertTupleEqual(results.displacements.shape, (num_points, 3))
    case.assertTupleEqual(results.deformation_gradients.shape, (num_points, 3, 3))
    case.assertTupleEqual(results.strains.shape, (num_points, 3, 3))
    case.assertTupleEqual(results.first_principal_strains.shape, (num_points,))
    case.assertTupleEqual(
        results.first_principal_strain_directions.shape, (num_points, 3)
    )
    case.assertTupleEqual(results.volumetric_strains.shape, (num_points,))

    kinematics.convert_kinematics_to_vtk(results)
    kinematics.write_kinematics_to_vtk(results, "test_3d")
    kinematics.write_kinematics_to_excel(results, "test_3d")
