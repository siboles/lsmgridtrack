import pathlib
import logging
import vtkmodules.all as vtk
from vtkmodules.util import numpy_support
import SimpleITK as sitk

from .config import ImageOptions

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
