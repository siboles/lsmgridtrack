import pathlib
from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

import pydantic


class RegMethodEnum(str, Enum):
    """
    Optimization method to employ in image registration.
    """

    BFGS = "bfgs"
    CONJUGATE_GRADIENT = "conjugate_gradient"
    GRADIENT_DESCENT = "gradient_descent"


class RegMetricEnum(str, Enum):
    """
    Metric to be minimized during registration.
    """

    CORRELATION = "correlation"
    HISTOGRAM = "histogram"


class RegSamplingEnum(str, Enum):
    """
    Image sampling strategy to be used when evaluating metric.
    """

    RANDOM = "random"
    REGULAR = "regular"


class ImageOptions(BaseModel):
    class Config:
        validate_assignment = True

    dimension: int = 3
    """Dimension of images."""
    spacing: List[float]
    """Physical voxel size of images."""
    resampling: Optional[List[float]]
    """Target physical voxel sizes for resampling."""
    surface_direction: Optional[List[int]]
    """Normal vector along which to search for sample surface."""


class GridOptions(BaseModel):
    class Config:
        validate_assignment = True

    origin: List[int]
    """Voxel coordinates of grid origin."""
    spacing: List[int]
    """Voxel spacing of grid lines."""
    size: List[int]
    """Grid size in each direction"""
    upsampling: Optional[int] = 1
    """Factor to upsample grid by when outputting results."""


class RegistrationOptions(BaseModel):
    class Config:
        validate_assignment = True

    method: RegMethodEnum = RegMethodEnum.BFGS
    """Optimization method to employ for registration"""
    metric: RegMetricEnum = RegMetricEnum.CORRELATION
    """Metric to minimize."""
    iterations: int = 20
    """Maximum number of iterations to perform in optimization."""
    sampling_fraction: float = 0.05
    """Fraction of image to sample when evaluating metric."""
    sampling_strategy: RegSamplingEnum = RegSamplingEnum.RANDOM
    """Volume sampling method to evaluate optimization metric."""
    shrink_levels: List[float]
    """Factors to resample image by in a pyramidal registration.
    If [1.0] then no pyramid is employed."""
    sigma_levels: List[float]
    """Gaussian variance to smooth by at each pyramid level."""
    reference_landmarks: List[List[int]]
    """Fiducial voxel coordinates in reference image."""
    deformed_landmarks: List[List[int]]
    """Fiducial voxel coordinates in deformed image."""


class Options(BaseModel):
    """
    Defines configuration parameters for registration and analysis.
    """

    image: ImageOptions
    grid: GridOptions
    registration: RegistrationOptions


def parse_config(configuration_file: str = "") -> Options:
    return pydantic.parse_file_as(path=pathlib.Path(configuration_file), type_=Options)
