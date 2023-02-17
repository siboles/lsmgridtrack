import pytest
import unittest
from .. import image
from ..config import ImageOptions
from . import data
import SimpleITK as sitk


@pytest.fixture(scope="module")
def image3d_options():
    return ImageOptions(
        dimension=3,
        spacing=[1.0, 1.0, 1.0],
        resampling=[1.0, 1.0, 1.0],
        surface_direction=[0, 0, -1],
    )


@pytest.fixture(scope="module")
def image3d_filepath():
    return data.get_image("reference - 1 layer")


@pytest.fixture(scope="module")
def image_seq3d_filepath():
    return data.get_image("reference image sequence")


@pytest.fixture(scope="module")
def image2d_options():
    return ImageOptions(
        dimension=2,
        spacing=[1.0, 1.0],
        resampling=[1.0, 1.0],
    )


@pytest.fixture(scope="module")
def image2d_filepath():
    return data.get_image("2d reference image")


@pytest.fixture(scope="module")
def image_standard_3d():
    img = sitk.Image(10, 10, 10, sitk.sitkFloat32)
    img.SetSpacing([1.0, 1.0, 1.0])
    return img


@pytest.fixture(scope="module")
def image_standard_2d():
    img = sitk.Image(10, 10, 0, sitk.sitkFloat32)
    img.SetSpacing([1.0, 1.0, 1.0])
    return img


def _get_minmax(img: sitk.Image):
    minmax_filter = sitk.MinimumMaximumImageFilter()
    minmax_filter.Execute(img)
    return (minmax_filter.GetMinimum(), minmax_filter.GetMaximum())


def test_read_3d_image_file(image3d_filepath, image3d_options):
    case = unittest.TestCase()
    img = image.parse_image_file(image3d_filepath, image3d_options)
    min_pixel, max_pixel = _get_minmax(img)
    case.assertTupleEqual(img.GetSpacing(), tuple(image3d_options.spacing))
    case.assertAlmostEqual(min_pixel, 0.0)
    case.assertAlmostEqual(max_pixel, 1.0)


def test_read_2d_image_file(image2d_filepath, image2d_options):
    case = unittest.TestCase()
    img = image.parse_image_file(image2d_filepath, image2d_options)
    min_pixel, max_pixel = _get_minmax(img)
    case.assertTupleEqual(img.GetSpacing(), tuple(image2d_options.spacing))
    case.assertAlmostEqual(min_pixel, 0.0)
    case.assertAlmostEqual(max_pixel, 1.0)


def test_read_3d_image_seq(image_seq3d_filepath, image3d_options):
    pass


def test_3d_image_to_vtk(image_standard_3d):
    case = unittest.TestCase()
    vtk_image = image.convert_image_to_vtk(image_standard_3d)
    case.assertTupleEqual(image_standard_3d.GetOrigin(), vtk_image.GetOrigin())
    case.assertTupleEqual(image_standard_3d.GetSpacing(), vtk_image.GetSpacing())
