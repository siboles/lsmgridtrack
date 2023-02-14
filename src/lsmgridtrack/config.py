import pathlib
from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

import pydantic


class RegMethodEnum(str, Enum):
    BFGS = "bfgs"
    CONJUGATE_GRADIENT = "conjugate_gradient"
    GRADIENT_DESCENT = "gradient_descent"


class RegMetricEnum(str, Enum):
    CORRELATION = "correlation"
    HISTOGRAM = "histogram"


class RegSamplingEnum(str, Enum):
    RANDOM = "random"
    REGULAR = "regular"


class ImageOptions(BaseModel):
    """
    :param dimension: Dimension of images.
    :type dimension: int

    :param spacing: Physical voxel size of images.
    :type spacing: List[float]=[1.0, 1.0, 1.0], optional

    :param resampling: Target physical voxel sizes for resampling.
    :type resampling: List[float], optional

    :param surface_direction: Normal vector along which to search for sample surface.
    :type surface_direction: List[int], optional
    """

    class Config:
        validate_assignment = True

    dimension: int = 3
    spacing: List[float]
    resampling: Optional[List[float]]
    surface_direction: Optional[List[int]]


class GridOptions(BaseModel):
    """
    :param origin: Voxel coordinates of grid origin
    :type origin: List[int]

    :param spacing: Voxel spacing of grid lines
    :type spacing: List[int]

    :param size: Grid size in each direction.
    :type size: List[int]

    :param upsampling: Factor to upsample grid by when outputting results.
    :type upsampling: int, optional
    """

    class Config:
        validate_assignment = True

    origin: List[int]
    spacing: List[int]
    size: List[int]
    upsampling: int = 1


class RegistrationOptions(BaseModel):
    """
    :param method: Optimization method to employ for registration:
    :type method: RegMethodEnum; enum_values=[BFGS,
                                                  CONJUGATE_GRADIENT,
                                                  GRADIENT_DESCENT]

    :param metric: Metric to minimize.
    :type metric: RegMetricEnum; enum_values=[CORRELATION, HISTOGRAM]

    :param iterations: Maximum number of iterations to perform in optimization.
    :type iterations: int=20,optional

    :param sampling_fraction: Fraction of image to sample when evaluating metric.
    :type sampling_fraction: float=0.05,optional

    :param sampling_strategy: Volume sampling method to evaluate optimization metric.
    :type sampling_strategy: RegSamplingEnum; enum_values=[RANDOM, REGULAR]

    :param reference_landmarks: Fiducial voxel coordinates in reference image.
    :type reference_landmarks: List[List[int]]

    :param deformed landmarks: Fiducial voxel coordinates in deformed image.
    :type deformed landmarks: List[List[int]]
    """

    class Config:
        validate_assignment = True

    method: RegMethodEnum = RegMethodEnum.BFGS
    metric: RegMetricEnum = RegMetricEnum.CORRELATION
    iterations: int = 20
    sampling_fraction: float = 0.05
    sampling_strategy: RegSamplingEnum = RegSamplingEnum.RANDOM
    shrink_levels: List[float]
    sigma_levels: List[float]
    reference_landmarks: List[List[int]]
    deformed_landmarks: List[List[int]]


class Options(BaseModel):
    """
    Defines configuration parameters for registration and analysis.
    """

    image: ImageOptions
    grid: GridOptions
    registration: RegistrationOptions


def parse_config(configuration_file: str = "") -> Options:
    return pydantic.parse_file_as(path=pathlib.Path(configuration_file), type_=Options)
