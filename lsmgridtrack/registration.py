import logging
import SimpleITK as sitk
import numpy as np
from typing import List
from .config import RegMethodEnum, RegMetricEnum, RegSamplingEnum, RegistrationOptions

log = logging.getLogger(__name__)


def _print_progress(reg):
    log.info(f"... Elapsed Iterations: {reg.GetOptimizerIteration()}")
    log.info(f"... Current Metric Value: {reg.GetMetricValue()}")


def _create_landmarks(reference_image: sitk.Image, landmarks: List[List[int]]):
    return np.ravel(
        [reference_image.TransformIndexToPhysicalPoint(point) for point in landmarks]
    )


def _create_landmark_transform(
    reference_image: sitk.Image, options: RegistrationOptions
):
    fixed_points = _create_landmarks(reference_image, options.reference_landmarks)
    deformed_points = _create_landmarks(reference_image, options.deformed_landmarks)
    if reference_image.GetDimension() == 2:
        tx = sitk.BSplineTransformInitializer(reference_image, (3, 3, 0), 3)
    else:
        tx = sitk.BSplineTransformInitializer(reference_image, (3, 3, 3), 3)
    landmark_tx = sitk.LandmarkBasedTransformInitializerFilter()
    landmark_tx.SetFixedLandmarks(fixed_points)
    landmark_tx.SetMovingLandmarks(deformed_points)
    landmark_tx.SetReferenceImage(reference_image)
    return landmark_tx.Execute(tx)


def create_registration(
    options: RegistrationOptions, reference_image: sitk.Image
) -> sitk.ImageRegistrationMethod:
    """

    Args:
        options: Options for the image registration.
        reference_image: The reference image that will be registered.

    Returns:
        A SimpleITK ImageRegistrationMethod

    """
    reg = sitk.ImageRegistrationMethod()

    # Optimizer settings
    if options.method == RegMethodEnum.BFGS:
        reg.SetOptimizerAsLBFGS2(numberOfIterations=options.iterations)
    elif options.method == RegMethodEnum.GRADIENT_DESCENT:
        reg.SetOptimizerAsGradientDescent(
            1.0, options.iterations, 1e-5, 20, reg.EachIteration
        )
        reg.SetOptimizerScalesFromPhysicalShift()
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
        reg.SetOptimizerScalesFromPhysicalShift()
    else:
        log.error("Optimizer was not set.")
        raise ValueError("Optimizer was not set.")

    # Metric settings
    if options.metric == RegMetricEnum.CORRELATION:
        reg.SetMetricAsCorrelation()
    elif options.metric == RegMetricEnum.HISTOGRAM:
        reg.SetMetricAsMattesMutualInformation()
    else:
        log.error("Metric was not set.")
        raise ValueError("Metric was not set.")

    # Sampling Strategy
    if options.sampling_strategy == RegSamplingEnum.RANDOM:
        reg.SetMetricSamplingStrategy(reg.RANDOM)
    elif options.sampling_strategy == RegSamplingEnum.REGULAR:
        reg.SetMetricSamplingStrategy(reg.REGULAR)
    else:
        log.error("Sampling strategy was not set.")
        raise ValueError("Sampling strategy was not set.")

    reg.SetMetricSamplingPercentagePerLevel(
        [options.sampling_fraction] * len(options.shrink_levels)
    )

    reg.SetShrinkFactorsPerLevel(options.shrink_levels)
    reg.SetSmoothingSigmasPerLevel(options.sigma_levels)
    reg.SetMetricUseFixedImageGradientFilter(False)
    reg.SetInterpolator(sitk.sitkBSpline)

    initial_transform = _create_landmark_transform(reference_image, options)
    reg.SetInitialTransform(initial_transform, True)
    return reg


def register(
    reg: sitk.ImageRegistrationMethod,
    reference_image: sitk.Image,
    deformed_image: sitk.Image,
) -> sitk.BSplineTransform:
    """

    Args:
        reg: The image registration method to execute.
        reference_image: The reference image to be registered.
        deformed_image: The deformed image to be registered.

    Returns:
        The resulting BSplineTransform from the registration.

    """
    log.info("Executing registration")
    reg.AddCommand(sitk.sitkIterationEvent, lambda: _print_progress(reg))

    transform = reg.Execute(reference_image, deformed_image)

    return transform