import numpy as np
import pytest
import SimpleITK as sitk
from hypothesis import event, given, settings
from hypothesis import strategies as st

from .. import config, registration


@pytest.fixture(scope="module")
def reference_3d() -> sitk.Image:
    img = sitk.Image(4, 4, 4, sitk.sitkFloat32)
    img.SetOrigin([0.0, 0.0, 0.0])
    img.SetSpacing([1.0, 1.0, 1.0])
    img = sitk.AdditiveGaussianNoise(img)
    return img


@pytest.fixture(scope="module")
def create_3d_transform(reference_3d) -> sitk.Transform:
    transform = sitk.BSplineTransformInitializer(reference_3d, (3, 3, 3), 3)
    N = len(transform.GetParameters())
    transform.SetParameters([np.random.uniform(-0.1, 0.1) for _ in range(N)])
    return transform


@pytest.fixture(scope="module")
def deformed_3d(reference_3d, create_3d_transform) -> sitk.Image:
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(reference_3d)
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetTransform(create_3d_transform)
    img = resample.Execute(reference_3d)
    return img


@pytest.fixture(scope="module")
def registration_options_3d(create_3d_transform) -> config.RegistrationOptions:
    reference_landmarks = [
        [1, 1, 1],
        [1, 2, 1],
        [2, 2, 1],
        [2, 1, 1],
        [1, 1, 2],
        [1, 2, 2],
        [2, 2, 2],
        [2, 1, 2],
    ]

    deformed_landmarks = [create_3d_transform.TransformPoint(p) for p in reference_landmarks]
    options = config.RegistrationOptions(
        method=config.RegMethodEnum.BFGS,
        metric=config.RegMetricEnum.CORRELATION,
        iterations=1,
        shrink_levels=[1],
        sigma_levels=[0],
        reference_landmarks=reference_landmarks,
        deformed_landmarks=deformed_landmarks,
    )
    return options


@pytest.fixture(scope="module")
def reference_2d() -> sitk.Image:
    img = sitk.Image(10, 10, sitk.sitkFloat32)
    img.SetOrigin([0.0, 0.0])
    img.SetSpacing([1.0, 1.0])
    img = sitk.AdditiveGaussianNoise(img)
    return img


@pytest.fixture(scope="module")
def create_2d_transform(reference_2d) -> sitk.Transform:
    transform = sitk.BSplineTransformInitializer(reference_2d, (3, 3, 0), 3)
    N = len(transform.GetParameters())
    transform.SetParameters([np.random.uniform(-1.0, 1.0) for _ in range(N)])
    return transform


@pytest.fixture(scope="module")
def deformed_2d(reference_2d, create_2d_transform) -> sitk.Image:
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(reference_2d)
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetTransform(create_2d_transform)
    img = resample.Execute(reference_2d)
    return img


@pytest.fixture(scope="module")
def registration_options_2d(create_2d_transform):
    reference_landmarks = [[3, 3], [3, 9], [9, 9], [9, 3]]

    deformed_landmarks = [
        map(int, create_2d_transform.TransformPoint(p)) for p in reference_landmarks
    ]
    options = config.RegistrationOptions(
        method=config.RegMethodEnum.BFGS,
        metric=config.RegMetricEnum.CORRELATION,
        iterations=1,
        shrink_levels=[1],
        sigma_levels=[0],
        reference_landmarks=reference_landmarks,
        deformed_landmarks=deformed_landmarks,
    )
    return options


@settings(deadline=2000)
@given(
    method=st.sampled_from(config.RegMethodEnum),
    metric=st.sampled_from(config.RegMetricEnum),
    sampling=st.sampled_from(config.RegSamplingEnum),
)
def test_3d_registration(
    registration_options_3d, reference_3d, deformed_3d, method, metric, sampling
) -> None:
    tmp_options = registration_options_3d.copy()
    tmp_options.method = method
    tmp_options.metric = metric
    tmp_options.sampling_strategy = sampling
    event((f"Testing 3d registration with method={method}, metric={metric}, sampling={sampling}"))
    rx = registration.create_registration(tmp_options, reference_3d)
    registration.register(rx, reference_3d, deformed_3d)


@settings(deadline=2000)
@given(
    method=st.sampled_from(config.RegMethodEnum),
    metric=st.sampled_from(config.RegMetricEnum),
    sampling=st.sampled_from(config.RegSamplingEnum),
)
def test_2d_registration(
    registration_options_2d, reference_2d, deformed_2d, method, metric, sampling
):
    tmp_options = registration_options_2d.copy()
    tmp_options.method = method
    tmp_options.metric = metric
    tmp_options.sampling_strategy = sampling
    event((f"Testing 2d registration with method={method}, metric={metric}, sampling={sampling}"))
    rx = registration.create_registration(registration_options_2d, reference_2d)
    registration.register(rx, reference_2d, deformed_2d)
