import pytest
import logging

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
    N = 648
    transform.SetParameters([np.random.uniform(-1.0, 1.0) for _ in range(N)])
    return transform


@pytest.fixture(scope="module")
def deformed_3d(reference_3d, create_3d_transform) -> sitk.Image:
    return create_3d_transform.Execute(reference_3d)


@pytest.fixture
def registration_options_3d(reference_3d, create_3d_transform):
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
    transform = sitk.BSplineTransformInitializer(reference_2d, (3, 3, 1), 3)
    N = 432
    transform.SetParameters([np.random.uniform(-1.0, 1.0) for _ in range(N)])
    return transform


@pytest.fixture(scope="module")
def deformed_2d(reference_2d, create_2d_transform) -> sitk.Image:
    return create_2d_transform.Execute(reference_2d)


@pytest.fixture
def registration_options_2d(reference_2d, create_2d_transform):
    reference_landmarks = [[3, 3], [3, 18], [18, 18], [18, 3]]

    deformed_landmarks = [
        create_2d_transform.TransformPoint(p) for p in reference_landmarks
    ]
    options = config.RegistrationOptions(
        method=config.RegMethodEnum.BFGS,
        metric=config.RegMetricEnum.CORRELATION,
        shrink_levels=[2, 1],
        sigma_levels=[0, 0],
        reference_landmarks=reference_landmarks,
        deformed_landmarks=deformed_landmarks,
    )
    return options


def test_3d_registration(registration_options_3d, reference_3d):
    registration.create_registration(registration_options_3d, reference_3d)


def test_2d_registration(registration_options_2d, reference_2d):
    registration.create_registration(registration_options_2d, reference_2d)
