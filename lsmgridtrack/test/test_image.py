import unittest

import numpy as np
import pytest
import SimpleITK as sitk
import vtkmodules.all as vtk
from hypothesis import event, given
from hypothesis import strategies as st

from .. import image
from ..config import ImageOptions, SurfaceAxis2D, SurfaceAxis3D
from . import data


@pytest.fixture(scope="module")
def image3d_options():
    return ImageOptions(
        spacing=[1.0, 1.0, 1.0],
        resampling=[1.0, 1.0, 1.0],
    )


@pytest.fixture(scope="module")
def image3d_filepath():
    return data.get_image("reference - 1 layer")


@pytest.fixture(scope="module")
def image_seq3d_filepath():
    return data.get_image("reference image sequence")


@pytest.fixture(scope="module")
def image2d_options():
    return ImageOptions(spacing=[1.0, 1.0], resampling=[1.0, 1.0])


@pytest.fixture(scope="module")
def image2d_filepath():
    return data.get_image("2d reference image")


@pytest.fixture(scope="module")
def image_standard_3d() -> dict:
    arr = np.zeros((15, 15, 15), dtype=float)
    arr[:, :, 3::] = 1.0
    arr[0:8, 0:8, 2::] = 1.0
    images = {}
    for surface_axis in SurfaceAxis3D:
        if surface_axis == SurfaceAxis3D.IN:
            img = sitk.GetImageFromArray(arr[:, :, ::-1])
        elif surface_axis == SurfaceAxis3D.JP:
            img = sitk.GetImageFromArray(arr.swapaxes(2, 1))
        elif surface_axis == SurfaceAxis3D.JN:
            img = sitk.GetImageFromArray(arr.swapaxes(2, 1)[:, ::-1, :])
        elif surface_axis == SurfaceAxis3D.KP:
            img = sitk.GetImageFromArray(arr.swapaxes(2, 0))
        elif surface_axis == SurfaceAxis3D.KN:
            img = sitk.GetImageFromArray(arr.swapaxes(2, 0)[::-1, :, :])
        else:
            img = sitk.GetImageFromArray(arr)
        img.SetSpacing([1.0, 1.0, 1.0])
        images[surface_axis.name] = img
    return images


@pytest.fixture(scope="module")
def image_standard_2d():
    arr = np.zeros((15, 15), dtype=float)
    arr[:, 3::] = 1.0
    arr[0:8, 2::] = 1.0
    images = {}
    for surface_axis in SurfaceAxis2D:
        if surface_axis == SurfaceAxis2D.IN:
            img = sitk.GetImageFromArray(arr[:, ::-1])
        elif surface_axis == SurfaceAxis2D.JP:
            img = sitk.GetImageFromArray(arr.swapaxes(1, 0))
        elif surface_axis == SurfaceAxis2D.JN:
            img = sitk.GetImageFromArray(arr.swapaxes(1, 0)[::-1, :])
        else:
            img = sitk.GetImageFromArray(arr)
        img.SetSpacing([1.0, 1.0, 1.0])
        images[surface_axis.name] = img
    return images


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
    case = unittest.TestCase()
    img = image.parse_image_sequence(image_seq3d_filepath, image3d_options)
    min_pixel, max_pixel = _get_minmax(img)
    case.assertTupleEqual(img.GetSpacing(), tuple(image3d_options.spacing))
    case.assertAlmostEqual(min_pixel, 0.0)
    case.assertAlmostEqual(max_pixel, 1.0)


def test_3d_image_to_vtk(image_standard_3d):
    case = unittest.TestCase()
    vtk_image = image.convert_image_to_vtk(image_standard_3d["IP"])
    case.assertTupleEqual(image_standard_3d["IP"].GetOrigin(), vtk_image.GetOrigin())
    case.assertTupleEqual(image_standard_3d["IP"].GetSpacing(), vtk_image.GetSpacing())


def test_2d_image_to_vtk(image_standard_2d):
    image.convert_image_to_vtk(image_standard_2d["IP"])


@given(
    surface_axis=st.sampled_from(SurfaceAxis3D),
)
def test_3d_find_surface(image_standard_3d, surface_axis):
    surface = image.get_sample_surface3d(
        image_standard_3d[surface_axis.name], surface_axis
    )
    event(f"Testing 3d find surface for axis: {surface_axis.name}")
    assert isinstance(surface, vtk.vtkPolyData)


@given(
    surface_axis=st.sampled_from(SurfaceAxis2D),
)
def test_2d_find_surface(image_standard_2d, surface_axis):
    surface = image.get_sample_surface2d(
        image_standard_2d[surface_axis.name], surface_axis
    )
    event(f"Testing 2d find surface for axis: {surface_axis.name}")
    assert isinstance(surface, vtk.vtkPolyData)
