import logging
import pathlib

import numpy as np
import SimpleITK as sitk
import vtkmodules.all as vtk
from vtkmodules.util import numpy_support

from .config import ImageOptions, SurfaceAxis2D, SurfaceAxis3D

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
    img: sitk.Image, direction: SurfaceAxis3D, threshold: float = 0.25
) -> vtk.vtkPolyData:
    """

    :param img:
    :param direction:
    :param threshold:
    """
    if threshold > 0.9 or threshold < 0.1:
        raise ValueError("threshold should be in interval of [0.1, 0.9]")
    smoothed_image = sitk.SmoothingRecursiveGaussian(img)
    image_array = sitk.GetArrayFromImage(smoothed_image)
    idx = [isinstance(v, int) for v in direction.value[0:3]]
    ax_idx = np.argwhere(idx).ravel()
    N, M = np.array(image_array.shape)[idx]
    base_slice = list(direction.value[0:3])
    surface_points = vtk.vtkPoints()
    for i in range(N):
        for j in range(M):
            s = base_slice.copy()
            s[ax_idx[0]] *= i
            s[ax_idx[1]] *= j
            arr = image_array[tuple(s)]
            surface_idx = np.array([-1, -1, -1], dtype=int)
            surface_idx[ax_idx[0]] = i
            surface_idx[ax_idx[1]] = j
            surface_idx[surface_idx < 0] = int(
                np.argwhere(arr >= threshold * arr.mean()).ravel()[direction.value[-1]]
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

    return iso.GetOutput()


def get_sample_surface2d(
    img: sitk.Image, direction: SurfaceAxis2D, threshold: float = 0.25
) -> vtk.vtkPolyData:
    """

    :param img:
    :param direction:
    :param threshold:
    """
    if threshold > 0.9 or threshold < 0.1:
        raise ValueError("threshold should be in interval of [0.1, 0.9]")
    smoothed_image = sitk.SmoothingRecursiveGaussian(img)
    image_array = sitk.GetArrayFromImage(smoothed_image)
    idx = [isinstance(v, int) for v in direction.value[0:2]]
    ax_idx = np.argwhere(idx).ravel()[0]
    N = np.array(image_array.shape)[idx][0]
    base_slice = list(direction.value[0:2])
    surface_points = vtk.vtkPoints()
    for i in range(N):
        s = base_slice.copy()
        s[ax_idx] *= i
        arr = image_array[tuple(s)]
        surface_idx = np.array([-1, -1], dtype=int)
        surface_idx[ax_idx] = i
        surface_idx[surface_idx < 0] = int(
            np.argwhere(arr >= threshold * arr.mean()).ravel()[direction.value[-1]]
        )
        point = img.TransformIndexToPhysicalPoint(surface_idx[::-1].tolist()) + (0.0,)
        surface_points.InsertNextPoint(point)

    surface_curve = vtk.vtkPolyLine()
    surface_curve.GetPointIds().SetNumberOfIds(surface_points.GetNumberOfPoints())
    for i in range(surface_points.GetNumberOfPoints()):
        surface_curve.GetPointIds().SetId(i, i)

    cells = vtk.vtkCellArray()
    cells.InsertNextCell(surface_curve)

    surface = vtk.vtkPolyData()
    surface.SetPoints(surface_points)
    surface.SetLines(cells)

    return surface
