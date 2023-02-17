import pathlib
import vtkmodules.all as vtk
from vtkmodules.util import numpy_support
import SimpleITK as sitk

from .config import ImageOptions


def _rescale_intensity(
    image: sitk.Image, minimum: float = 0.0, maximum: float = 1.0
) -> sitk.Image:
    filter = sitk.RescaleIntensityImageFilter()
    filter.SetOutputMinimum(minimum)
    filter.SetOutputMaximum(maximum)
    return filter.Execute(image)


def parse_image_sequence(filepath: str, options: ImageOptions):
    p = pathlib.Path(filepath)
    file_list = [f.as_posix() for f in p.glob("*.tif")]
    image = sitk.ReadImage(file_list, sitk.sitkFloat32)
    image.SetSpacing(options.spacing)
    return _rescale_intensity(image)


def parse_image_file(filepath: str, options: ImageOptions):
    image = sitk.ReadImage(filepath, sitk.sitkFloat32)
    image.SetSpacing(options.spacing)
    return _rescale_intensity(image)


def convert_image_to_vtk(image: sitk.Image) -> vtk.vtkImageData:
    image_array = numpy_support.numpy_to_vtk(
        sitk.GetArrayFromImage(image).ravel(), deep=True, array_type=vtk.VTK_FLOAT
    )
    origin = list(image.GetOrigin())
    spacing = list(image.GetSpacing())
    dimensions = list(image.GetSize())
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
