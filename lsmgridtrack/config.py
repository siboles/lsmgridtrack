import pathlib
import json
from enum import Enum
from typing import List, Optional, Union

import pydantic
from pydantic import BaseModel


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


class SurfaceAxis3D(Enum):
    """
    Axis to search for sample surface along:
    P suffix indicates to search forwards.
    N suffix indicates to search backwards.
    """

    IP = (1, 1, slice(None, None, None), 0)
    JP = (1, slice(None, None, None), 1, 0)
    KP = (slice(None, None, None), 1, 1, 0)
    IN = (1, 1, slice(None, None, 1), -1)
    JN = (1, slice(None, None, 1), 1, -1)
    KN = (slice(None, None, 1), 1, 1, -1)


class SurfaceAxis2D(Enum):
    """
    Axis to search for sample surface along:
    P suffix indicates to search forwards.
    N suffix indicates to search backwards.
    """

    IP = (1, slice(None, None, None), 0)
    JP = (slice(None, None, None), 1, 0)
    IN = (1, slice(None, None, 1), -1)
    JN = (slice(None, None, 1), 1, -1)


class ImageOptions(BaseModel):
    """
    Options to set image attributes and behavior.
    """

    class Config:
        validate_assignment = True

    spacing: List[float]
    """Physical voxel size of images."""
    resampling: Optional[List[float]] = None
    """Target physical voxel sizes for resampling."""
    surface_axis: Optional[Union[SurfaceAxis2D, SurfaceAxis3D]] = None
    """Image axis to search for sample surface along"""


class GridOptions(BaseModel):
    """
    Options defining the grid used to output calculated results.
    """

    class Config:
        validate_assignment = True

    origin: List[int]
    """Voxel coordinates of grid origin."""
    upper_bound: List[int]
    """Voxel position of upper grid bound."""
    divisions: List[int]
    """Number of grid nodes in each direction"""


class RegistrationOptions(BaseModel):
    """
    Options defining the registration procedure.
    """

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
    shrink_levels: List[int]
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

def parse_config(filepath: str):
    with open(filepath, "r") as fid:
        contents = json.load(fid)
    try:
        m = Options.model_validate(contents)
        return m
    except ValidationError as e:
        raise ValidationError(e)
