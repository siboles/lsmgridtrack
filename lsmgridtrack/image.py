import logging
import pathlib

import numpy as np
import SimpleITK as sitk
import vtkmodules.all as vtk
from vtkmodules.util import numpy_support

from .config import (
    ImageOptions,
    SurfaceAxis2D,
    SurfaceAxis3D,
    _surface_axis2d_lut,
    _surface_axis3d_lut,
)

log = logging.getLogger(__name__)


def _rescale_intensity(
    img: sitk.Image, minimum: float = 0.0, maximum: float = 1.0
) -> sitk.Image:
    """

    :param img:
    :param minimum:
    :param maximum:
    :return:
    """
    filter = sitk.RescaleIntensityImageFilter()
    filter.SetOutputMinimum(minimum)
    filter.SetOutputMaximum(maximum)
    return filter.Execute(img)


def parse_image_sequence(filepath: str, options: ImageOptions) -> sitk.Image:
    """

    :param filepath:
    :param options:
    :return:
    """
    p = pathlib.Path(filepath)
    file_list = [f.as_posix() for f in p.glob("*.tif")]
    log.info(f"Parsing {len(file_list)} image slices from {p}")
    img = sitk.ReadImage(file_list, sitk.sitkFloat32)
    img.SetSpacing(options.spacing)
    return _rescale_intensity(img)


def parse_image_file(filepath: str, options: ImageOptions) -> sitk.Image:
    """

    :param filepath:
    :param options:
    :return:
    """
    img = sitk.ReadImage(filepath, sitk.sitkFloat32)
    log.info(f"Parsing image from {filepath}.")
    img.SetSpacing(options.spacing)
    return _rescale_intensity(img)


def convert_image_to_vtk(img: sitk.Image) -> vtk.vtkImageData:
    """

    :param img:
    :return:
    """
    image_array = numpy_support.numpy_to_vtk(
        sitk.GetArrayFromImage(img).ravel(), deep=True, array_type=vtk.VTK_FLOAT
    )
    origin = list(img.GetOrigin())
    spacing = list(img.GetSpacing())
    dimensions = list(img.GetSize())
    if len(origin) == 2:
        origin += [0.0]
        spacing += [1.0]
        dimensions += [1]
    vtk_image = vtk.vtkImageData()
    vtk_image.SetOrigin(origin)
    vtk_image.SetSpacing(spacing)
    vtk_image.SetDimensions(dimensions)
    vtk_image.GetPointData().SetScalars(image_array)
    return vtk_image


def write_image_as_vtk(img: sitk.Image, name: str = "image") -> None:
    """

    :param img:
    :param name:
    """
    vtk_image = convert_image_to_vtk(img)
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(f"{name}.vti")
    writer.SetInputData(vtk_image)
    writer.Write()
    log.info(f"Saved image as {name}.vti.")


def write_image_as_nii(img: sitk.Image, name: str = "image") -> None:
    """

    :param img:
    :param name:
    """
    sitk.WriteImage(img, f"{name}.nii")
    log.info(f"Saved image as {name}.nii")


def get_sample_surface3d(
    img: sitk.Image, direction: SurfaceAxis3D, threshold: float = 0.25, stride: int = 10
) -> vtk.vtkPolyData:
    """

    :param img:
    :param direction:
    :param threshold:
    :param stride:
    """
    if threshold > 0.9 or threshold < 0.1:
        raise ValueError("threshold should be in interval of [0.1, 0.9]")
    surface_direction = _surface_axis3d_lut[direction.value]
    smoothed_image = sitk.SmoothingRecursiveGaussian(img)
    image_array = sitk.GetArrayFromImage(smoothed_image)
    idx = [isinstance(v, int) for v in surface_direction[0:3]]
    ax_idx = np.argwhere(idx).ravel()
    N, M = np.array(image_array.shape)[idx]
    base_slice = list(surface_direction[0:3])
    surface_points = vtk.vtkPoints()
    for i in range(0, N, stride):
        for j in range(0, M, stride):
            s = base_slice.copy()
            s[ax_idx[0]] *= i
            s[ax_idx[1]] *= j
            arr = image_array[tuple(s)]
            surface_idx = np.array([-1, -1, -1], dtype=int)
            surface_idx[ax_idx[0]] = i
            surface_idx[ax_idx[1]] = j
            surface_idx[surface_idx < 0] = int(
                np.argwhere(arr >= threshold * arr.mean()).ravel()[
                    surface_direction[-1]
                ]
            )
            surface_points.InsertNextPoint(
                img.TransformIndexToPhysicalPoint(surface_idx[::-1].tolist())
            )
    surface = vtk.vtkPolyData()
    surface.SetPoints(surface_points)

    reconstruct = vtk.vtkSurfaceReconstructionFilter()
    reconstruct.SetInputData(surface)

    iso = vtk.vtkContourFilter()
    iso.SetInputConnection(reconstruct.GetOutputPort())
    iso.SetValue(0, 0.0)
    iso.Update()

    normals = numpy_support.vtk_to_numpy(
        iso.GetOutput().GetPointData().GetArray("Normals")
    )

    texture_coords = vtk.vtkTextureMapToPlane()
    texture_coords.SetInputConnection(iso.GetOutputPort())
    texture_coords.SetNormal(*normals.mean(axis=0).tolist())

    tangents = vtk.vtkPolyDataTangents()
    tangents.SetInputConnection(texture_coords.GetOutputPort())
    tangents.Update()

    return tangents.GetOutput()


def get_sample_surface2d(
    img: sitk.Image, direction: SurfaceAxis2D, threshold: float = 0.25, stride: int = 10
) -> vtk.vtkPolyData:
    """

    :param img:
    :param direction:
    :param threshold:
    :param stride:
    """
    if threshold > 0.9 or threshold < 0.1:
        raise ValueError("threshold should be in interval of [0.1, 0.9]")
    surface_direction = _surface_axis2d_lut[direction.value]
    smoothed_image = sitk.SmoothingRecursiveGaussian(img)
    image_array = sitk.GetArrayFromImage(smoothed_image)
    idx = [isinstance(v, int) for v in surface_direction[0:2]]
    ax_idx = np.argwhere(idx).ravel()[0]
    N = np.array(image_array.shape)[idx][0]
    base_slice = list(surface_direction[0:2])
    surface_points = vtk.vtkPoints()
    for i in range(0, N, stride):
        s = base_slice.copy()
        s[ax_idx] *= i
        arr = image_array[tuple(s)]
        surface_idx = np.array([-1, -1], dtype=int)
        surface_idx[ax_idx] = i
        surface_idx[surface_idx < 0] = int(
            np.argwhere(arr >= threshold * arr.mean()).ravel()[surface_direction[-1]]
        )
        point = img.TransformIndexToPhysicalPoint(surface_idx[::-1].tolist()) + (0.0,)
        surface_points.InsertNextPoint(point)

    lines = vtk.vtkCellArray()
    for i in range(1, surface_points.GetNumberOfPoints()):
        lines.InsertNextCell(2)
        lines.InsertCellPoint(i - 1)
        lines.InsertCellPoint(i)

    surface = vtk.vtkPolyData()
    surface.SetPoints(surface_points)
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


def write_surface_to_vtk(data: vtk.vtkPolyData, name: str):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(f"{name}.vtp")
    writer.SetInputData(data)
    writer.Write()
