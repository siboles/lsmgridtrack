import logging
from copy import deepcopy
import SimpleITK as sitk
import numpy as np
from typing import List
from .config import RegMethodEnum, RegMetricEnum, RegSamplingEnum, RegistrationOptions

log = logging.getLogger(__name__)


def _print_progress(reg):
    log.info(f"... Elapsed Iterations: {reg.GetOptimizerIteration()}")
    log.info(f"... Current Metric Value: {reg.GetMetricValue()}")


def _create_landmarks(reference_image: sitk.Image, landmarks: List[List[int]]):
    return np.ravel([reference_image.TransformIndexToPhysicalPoint(point) for point in landmarks])


def _transform_landmarks(
    reference_image: sitk.Image, transform: sitk.Transform, landmarks: List[List[int]]
):
    physical_landmarks = _create_landmarks(reference_image, landmarks)
    if reference_image.GetDimension() == 2:
        reshaped = physical_landmarks.reshape((len(physical_landmarks) // 2, 2))
    else:
        reshaped = physical_landmarks.reshape((len(physical_landmarks) // 3, 3))
    return np.ravel([transform.TransformPoint(reshaped[i, :]) for i in range(reshaped.shape[0])])


def _create_landmark_transform(reference_image: sitk.Image, options: RegistrationOptions):
    fixed_points = _create_landmarks(reference_image, options.reference_landmarks)
    deformed_points = _create_landmarks(reference_image, options.deformed_landmarks)
    if reference_image.GetDimension() == 2:
        tx = sitk.BSplineTransformInitializer(
            reference_image, (options.control_points[0], options.control_points[1], 0), 3
        )
    else:
        tx = sitk.BSplineTransformInitializer(reference_image, options.control_points, 3)
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
            1.0, options.iterations, estimateLearningRate=reg.EachIteration
        )
        reg.SetOptimizerScalesFromPhysicalShift()
    elif options.method == RegMethodEnum.CONJUGATE_GRADIENT:
        reg.SetOptimizerAsConjugateGradientLineSearch(
            1.0, options.iterations, estimateLearningRate=reg.EachIteration
        )
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


def apply_final_landmark_transform(
    reference_image: sitk.Image,
    deformed_image: sitk.Image,
    transform: sitk.Transform,
    options: RegistrationOptions,
) -> sitk.Transform:
    """
    Applies the final landmark transform to the deformed image.

    Args:
        deformed_image: The deformed image to be transformed.
        transform: The transform to be applied.
        options: The registration options.

    Returns:
        The transformed image.
    """
    mapped_reference_landmarks = _transform_landmarks(
        reference_image, transform, options.reference_landmarks
    )
    new_transform = deepcopy(transform)
    deformed_landmarks = _create_landmarks(deformed_image, options.deformed_landmarks)
    landmark_tx = sitk.LandmarkBasedTransformInitializerFilter()
    landmark_tx.SetFixedLandmarks(mapped_reference_landmarks)
    landmark_tx.SetMovingLandmarks(deformed_landmarks)
    landmark_tx.SetReferenceImage(reference_image)
    tx = landmark_tx.Execute(new_transform)
    final_transform = sitk.CompositeTransform([tx, transform])
    return final_transform


def save_transform(transform: sitk.Transform, name: str = "transform") -> None:
    """

    :param transform:
    :param name:
    """
    sitk.WriteTransform(transform, f"{name}.tfm")
