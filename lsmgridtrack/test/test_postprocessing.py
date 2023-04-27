import tempfile

import numpy as np
import pytest
import vtkmodules.all as vtk
from vtkmodules.util import numpy_support

from .. import postprocessing


@pytest.fixture(scope="module")
def surface_3d() -> vtk.vtkPolyData:
    plane = vtk.vtkPlaneSource()
    plane.SetNormal(0.0, 0.0, 1.0)
    plane.SetXResolution(20)
    plane.SetYResolution(20)

    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(plane.GetOutputPort())

    spoints = vtk.vtkPoints()
    spoints.SetNumberOfPoints(16)
    tpoints = vtk.vtkPoints()
    tpoints.SetNumberOfPoints(16)
    xdiv, ydiv = np.meshgrid(np.linspace(-0.5, 0.5, 4), np.linspace(-0.5, 0.5, 4))
    for i in range(5):
        spoints.SetPoint(i, xdiv.ravel()[i], ydiv.ravel()[i], 0.0)
        tpoints.SetPoint(
            i, xdiv.ravel()[i], ydiv.ravel()[i], np.random.normal(0.0, 0.3)
        )

    thin = vtk.vtkThinPlateSplineTransform()
    thin.SetSourceLandmarks(spoints)
    thin.SetTargetLandmarks(tpoints)
    thin.SetBasisToR()

    tx = vtk.vtkGeneralTransform()
    tx.SetInput(thin)
    tx_polydata = vtk.vtkTransformPolyDataFilter()
    tx_polydata.SetInputConnection(tri.GetOutputPort())
    tx_polydata.SetTransform(tx)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(tx_polydata.GetOutputPort())

    texture_coords = vtk.vtkTextureMapToPlane()
    texture_coords.SetInputConnection(normals.GetOutputPort())
    texture_coords.SetNormal(0.0, 0.0, 1.0)

    tangents = vtk.vtkPolyDataTangents()
    tangents.SetInputConnection(texture_coords.GetOutputPort())
    tangents.Update()

    return tangents.GetOutput()


@pytest.fixture(scope="module")
def image_data_3d():
    data = vtk.vtkImageData()
    data.SetExtent(0, 5, 0, 5, 0, 5)
    data.SetOrigin(-0.5, -0.5, 0.0)
    data.SetSpacing(0.2, 0.2, 0.2)

    displacements = numpy_support.numpy_to_vtk(
        np.random.normal(0.0, 0.5, (data.GetNumberOfPoints(), 3)).ravel(),
        deep=True,
        array_type=vtk.VTK_DOUBLE,
    )
    displacements.SetName("Displacements")
    displacements.SetNumberOfComponents(3)

    strains = np.random.normal(0.0, 0.3, (data.GetNumberOfPoints(), 3, 3))
    strains = np.einsum("...jk,...jm", strains, strains)
    strains = numpy_support.numpy_to_vtk(
        strains.ravel(), deep=True, array_type=vtk.VTK_DOUBLE
    )
    strains.SetName("Strain")
    strains.SetNumberOfComponents(9)

    directions = np.random.uniform(0.0, 1.0, (data.GetNumberOfPoints(), 3))
    directions /= np.linalg.norm(directions)
    directions = numpy_support.numpy_to_vtk(
        directions.ravel(), deep=True, array_type=vtk.VTK_DOUBLE
    )
    directions.SetName("Direction")
    directions.SetNumberOfComponents(3)

    data.GetPointData().AddArray(displacements)
    data.GetPointData().AddArray(strains)
    data.GetPointData().AddArray(directions)

    return data


def test_transform_3d_data_to_local_csys(surface_3d, image_data_3d):
    postprocessing.transform_to_local_csys_3d(image_data_3d, surface_3d)


def test_vtk_image_roundtrip(image_data_3d):
    tf = tempfile.NamedTemporaryFile(suffix=".vti")
    postprocessing.write_to_vtk_image_data(image_data_3d, tf.name)
    data = postprocessing.read_vtk_grid(tf.name)
    assert isinstance(data, vtk.vtkImageData)


def test_read_surface(surface_3d):
    tf = tempfile.NamedTemporaryFile(suffix=".vtp")
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(tf.name)
    writer.SetInputData(surface_3d)
    writer.Write()
    surface = postprocessing.read_vtk_surface(tf.name)
    assert isinstance(surface, vtk.vtkPolyData)
