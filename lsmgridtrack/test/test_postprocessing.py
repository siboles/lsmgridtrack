import tempfile

import numpy as np
import pandas as pds
import pytest
import vtkmodules.all as vtk
from vtkmodules.util import numpy_support

from .. import postprocessing

CELL_3D_EXCEL = "data/example_3d_cell_data_excel.xlsx"


@pytest.fixture(scope="module")
def surface_3d() -> vtk.vtkPolyData:
    plane = vtk.vtkPlaneSource()
    plane.SetPoint1(0.6, -0.6, 0.5)
    plane.SetPoint2(-0.6, 0.6, 0.5)
    plane.SetXResolution(20)
    plane.SetYResolution(20)

    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(plane.GetOutputPort())

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(tri.GetOutputPort())

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


@pytest.fixture(scope="module")
def grid_data_3d():
    data = vtk.vtkRectilinearGrid()
    coordinates = np.meshgrid(
        np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5)
    )
    data.SetXCoordinates(
        numpy_support.numpy_to_vtk(
            coordinates[0].ravel(), deep=True, array_type=vtk.VTK_DOUBLE
        )
    )
    data.SetYCoordinates(
        numpy_support.numpy_to_vtk(
            coordinates[1].ravel(), deep=True, array_type=vtk.VTK_DOUBLE
        )
    )
    data.SetZCoordinates(
        numpy_support.numpy_to_vtk(
            coordinates[2].ravel(), deep=True, array_type=vtk.VTK_DOUBLE
        )
    )

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


@pytest.fixture(scope="module")
def image_data_2d():
    data = vtk.vtkImageData()
    data.SetExtent(0, 5, 0, 5, 0, 0)
    data.SetOrigin(-0.5, -0.5, 0)
    data.SetSpacing(0.2, 0.2, 1.0)
    displacements = numpy_support.numpy_to_vtk(
        np.random.normal(0.0, 0.5, (data.GetNumberOfPoints(), 3)).ravel(),
        deep=True,
        array_type=vtk.VTK_DOUBLE,
    )
    displacements.SetName("Displacements")
    displacements.SetNumberOfComponents(3)

    strains = np.random.normal(0.0, 0.3, (data.GetNumberOfPoints(), 3, 3))
    strains = np.einsum("...jk,...jm", strains, strains)
    strains[:, 2, :] = 0.0
    strains[:, :, 2] = 0.0
    strains = numpy_support.numpy_to_vtk(
        strains.ravel(), deep=True, array_type=vtk.VTK_DOUBLE
    )
    strains.SetName("Strain")
    strains.SetNumberOfComponents(9)

    directions = np.random.uniform(0.0, 1.0, (data.GetNumberOfPoints(), 3))
    directions[:, 2] = 0.0
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


@pytest.fixture(scope="module")
def surface_2d():
    N = 20
    points = vtk.vtkPoints()
    x = np.linspace(-0.5, 0.5, N)
    y = 0.5 + 0.1 * np.sin(2 * x)
    for i in range(N):
        points.InsertNextPoint(x[i], y[i], 0.0)

    lines = vtk.vtkCellArray()
    for i in range(1, points.GetNumberOfPoints()):
        lines.InsertNextCell(2)
        lines.InsertCellPoint(i - 1)
        lines.InsertCellPoint(i)

    surface = vtk.vtkPolyData()
    surface.SetPoints(points)
    surface.SetLines(lines)
    normals = vtk.vtkFloatArray()
    normals.SetNumberOfComponents(3)
    normals.SetName("Normals")
    tangents = vtk.vtkFloatArray()
    tangents.SetNumberOfComponents(3)
    tangents.SetName("Tangents")
    lines = surface.GetLines()
    for i in range(surface.GetNumberOfLines()):
        line = surface.GetCell(i)
        p1 = surface.GetPoint(line.GetPointId(0))
        p2 = surface.GetPoint(line.GetPointId(1))
        normal = np.array([-(p2[1] - p1[1]), p2[0] - p1[0], 0.0])
        normal /= np.linalg.norm(normal)
        tangent = np.array([p2[0] - p1[0], p2[1] - p1[1], 0.0])
        normals.InsertNextTuple3(normal[0], normal[1], normal[2])
        tangents.InsertNextTuple3(tangent[0], tangent[1], tangent[2])
    surface.GetCellData().AddArray(normals)
    surface.GetCellData().AddArray(tangents)
    c2p = vtk.vtkCellDataToPointData()
    c2p.SetInputData(surface)
    c2p.Update()

    return c2p.GetOutput()


@pytest.fixture(scope="module")
def cell_dataframe_3d():
    return pds.read_excel(CELL_3D_EXCEL, sheet_name=None)


def test_transform_3d_data_to_local_csys(surface_3d, image_data_3d):
    postprocessing.transform_to_local_csys_3d(image_data_3d, surface_3d)


def test_transform_2d_data_to_local_csys(surface_2d, image_data_2d):
    postprocessing.transform_to_local_csys_2d(image_data_2d, surface_2d)


def test_globally_transform_3d(image_data_3d, surface_3d):
    postprocessing.globally_transform_3d(image_data_3d, surface_3d)


def test_globally_transform_2d(image_data_2d, surface_2d):
    postprocessing.globally_transform_2d(image_data_2d, surface_2d)


def test_vtk_image_roundtrip(image_data_3d):
    tf = tempfile.NamedTemporaryFile()
    postprocessing.write_to_vtk_image_data(image_data_3d, tf.name)
    data = postprocessing.read_vtk_grid(f"{tf.name}.vti")
    assert isinstance(data, vtk.vtkImageData)
    assert data.GetNumberOfPoints() == image_data_3d.GetNumberOfPoints()


def test_vtk_grid_roundtrip(grid_data_3d):
    tf = tempfile.NamedTemporaryFile()
    postprocessing.write_to_vtk_grid(grid_data_3d, tf.name)
    data = postprocessing.read_vtk_grid(f"{tf.name}.vtr")
    assert isinstance(data, vtk.vtkRectilinearGrid)
    assert data.GetNumberOfPoints() == grid_data_3d.GetNumberOfPoints()


def test_not_vtk_file_read():
    tf = tempfile.NamedTemporaryFile(suffix=".foo")
    with pytest.raises(Exception):
        postprocessing.read_vtk_grid(tf.name)


def test_read_surface(surface_3d):
    tf = tempfile.NamedTemporaryFile(suffix=".vtp")
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(tf.name)
    writer.SetInputData(surface_3d)
    writer.Write()
    surface = postprocessing.read_vtk_surface(tf.name)
    assert isinstance(surface, vtk.vtkPolyData)


def test_dataframe_roundtrip(cell_dataframe_3d):
    tf = tempfile.NamedTemporaryFile()
    postprocessing.write_dataframe_to_excel(cell_dataframe_3d, tf.name)
    data = postprocessing.read_dataframe_from_excel(tf.name)
    for k in data.keys():
        assert k in cell_dataframe_3d.keys()


def test_locally_transform_cell_data_3d(cell_dataframe_3d, surface_3d):
    rotated_data = postprocessing.transform_dataframe_to_local_csys_3d(
        cell_dataframe_3d, surface_3d
    )
    for df in rotated_data.values():
        assert isinstance(df, pds.DataFrame)


def test_globally_transform_cell_data_3d(cell_dataframe_3d, surface_3d):
    rotated_data = postprocessing.globally_transform_dataframe_3d(
        cell_dataframe_3d, surface_3d
    )
    for df in rotated_data.values():
        assert isinstance(df, pds.DataFrame)


def test_convert_vtk_to_dataframe(image_data_3d):
    df = postprocessing.convert_vtk_to_dataframe(image_data_3d)
