import pytest
from hypothesis import given, event, strategies as st

import SimpleITK as sitk
import numpy as np
from .. import registration, config


@pytest.fixture(scope="module")
def reference_3d() -> sitk.Image:
    # create grid with 2 pixel wide stripes
    img = sitk.Image(20, 20, 20, sitk.sitkFloat32)
    for i in range(3, 20, 3):
        for j in range(3, 20, 3):
            for k in range(3, 20, 3):
                img[i, j, k] = 1.0
                img[i + 1, j + 1, k + 1] = 1.0

    img.SetOrigin([0.0, 0.0, 0.0])
    img.SetSpacing([1.0, 1.0, 1.0])
    return img


@pytest.fixture(scope="module")
def create_3d_transform(reference_3d) -> sitk.Transform:
    transform = sitk.BSplineTransformInitializer(reference_3d, (3, 3, 3), 3)
    N = len(transform.GetParameters())
    transform.SetParameters([np.random.uniform(-1.0, 1.0) for _ in range(N)])
    return transform


@pytest.fixture(scope="module")
def deformed_3d(reference_3d, create_3d_transform) -> sitk.Image:
    return sitk.Resample(reference_3d, create_3d_transform)


@pytest.fixture(scope="module")
def registration_options_3d(
    reference_3d, create_3d_transform
) -> config.RegistrationOptions:
    reference_landmarks = [
        [3, 3, 3],
        [3, 18, 3],
        [18, 18, 3],
        [18, 3, 3],
        [3, 3, 18],
        [3, 18, 18],
        [18, 18, 18],
        [18, 3, 18],
    ]

    deformed_landmarks = [
        create_3d_transform.TransformPoint(p) for p in reference_landmarks
    ]
    options = config.RegistrationOptions(
        method=config.RegMethodEnum.BFGS,
        metric=config.RegMetricEnum.CORRELATION,
        iterations=1,
        shrink_levels=[2, 1],
        sigma_levels=[0, 0],
        reference_landmarks=reference_landmarks,
        deformed_landmarks=deformed_landmarks,
    )
    return options


@pytest.fixture(scope="module")
def reference_2d():
    # create grid with 2 pixel wide stripes
    img = sitk.Image(20, 20, sitk.sitkFloat32)
    for i in range(3, 20, 3):
        for j in range(3, 20, 3):
            img[i, j] = 1.0
            img[i + 1, j + 1] = 1.0
    img.SetOrigin([0.0, 0.0])
    img.SetSpacing([1.0, 1.0])
    return img


@pytest.fixture(scope="module")
def create_2d_transform(reference_2d) -> sitk.Transform:
    transform = sitk.BSplineTransformInitializer(reference_2d, (3, 3, 0), 3)
    N = len(transform.GetParameters())
    transform.SetParameters([np.random.uniform(-1.0, 1.0) for _ in range(N)])
    return transform


@pytest.fixture(scope="module")
def deformed_2d(reference_2d, create_2d_transform) -> sitk.Image:
    return sitk.Resample(reference_2d, create_2d_transform)


@pytest.fixture(scope="module")
def registration_options_2d(reference_2d, create_2d_transform):
    reference_landmarks = [[3, 3], [3, 18], [18, 18], [18, 3]]

    deformed_landmarks = [
        create_2d_transform.TransformPoint(p) for p in reference_landmarks
    ]
    options = config.RegistrationOptions(
        method=config.RegMethodEnum.BFGS,
        metric=config.RegMetricEnum.CORRELATION,
        iterations=1,
        shrink_levels=[2, 1],
        sigma_levels=[0, 0],
        reference_landmarks=reference_landmarks,
        deformed_landmarks=deformed_landmarks,
    )
    return options


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
    event(
        (
            f"Testing 3d registration with method={method}"
            f", metric={metric}, sampling={sampling}"
        )
    )
    registration.create_registration(tmp_options, reference_3d)


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
    event(
        (
            f"Testing 2d registration with method={method}"
            f", metric={metric}, sampling={sampling}"
        )
    )
    registration.create_registration(registration_options_2d, reference_2d)
