import SimpleITK as sitk
import numpy as np
from typing import List
from .config import RegMethodEnum, RegMetricEnum, RegSamplingEnum, RegistrationOptions


def create_registration(
    options: RegistrationOptions,
    reference_image: sitk.Image,
    deformed_image: sitk.Image,
):
    """"""
    reg = sitk.ImageRegistrationMethod()

    # Optimizer settings
    if options.method == RegMethodEnum.BFGS:
        reg.SetOptimizerAsLBFGS2(numberOfIterations=options.iterations)
    elif options.method == RegMethodEnum.GRADIENT_DESCENT:
        reg.SetOptimizerAsGradientDescent(
            1.0, options.iterations, 1e-5, 20, reg.EachIteration
        )
    elif options.method == RegMethodEnum.CONJUGATE_GRADIENT:
        max_step_size = 0.5 * np.min(
            np.array(reference_image.GetSize()) * np.array(reference_image.GetSpacing())
        )
        reg.SetOptimizerAsConjugateGradientLineSearch(
            1.0,
            options.iterations,
            convergenceMinimumValue=1e-5,
            convergenceWindowSize=20,
            lineSearchUpperLimit=3.0,
            maximumStepSizeInPhysicalUnits=max_step_size,
        )

    # Metric settings
    if options.metric == RegMetricEnum.CORRELATION:
        reg.SetMetricAsCorrelation()
    elif options.metric == RegMetricEnum.HISTOGRAM:
        reg.SetMetricAsMattesMutualInformation()

    # Sampling Strategy
    if options.metric == RegSamplingEnum.RANDOM:
        reg.SetMetricSamplingStrategy(reg.RANDOM)

    fixed_points = _create_landmarks(reference_image, options.reference_landmarks)
    deformed_points = _create_landmarks(reference_image, options.reference_landmarks)


def _create_landmarks(reference_image: sitk.Image, landmarks: List[List[int]]):
    return np.ravel(
        [reference_image.TransformIndexToPhysicalPoint(point) for point in landmarks]
    )
