from __future__ import division

import os
import sys
from builtins import range

import SimpleITK as sitk
import numpy as np
import vtk
from past.utils import old_div
from vtk.util import numpy_support

from .context import lsmgridtrack as lsm


def main(repeats=1):
    repeats = int(repeats)
    path = os.path.dirname(__file__)
    reference_image = sitk.ReadImage(os.path.join(path, "data", "resonance_ref.nii"), sitk.sitkFloat32)
    reference_image.SetSpacing([0.249, 0.249, 1.0])
    # pad the image in z by +/- 5 zero voxels
    reference_image = sitk.ConstantPad(reference_image, (5, 5, 5), (5, 5, 5))
    reference_image.SetOrigin([0, 0, 0])
    sitk.WriteImage(reference_image, "reference.nii")
    np.random.seed(seed=1545281929)



    root_mean_square = np.zeros(repeats, float)
    for r in range(repeats):
        track = lsm.tracker(config=os.path.join(path, "data", "resonance.yaml"))
        for i in range(3):
            track.options["Grid"]["origin"][i] += 5
        x = []
        for i in range(3):
            start = track.options["Grid"]["origin"][i]
            stop = start + (track.options["Grid"]["size"][i] - 1) * track.options["Grid"]["spacing"][i]
            num_steps = track.options["Grid"]["size"][i] * track.options["Grid"]["upsampling"] - \
                        (track.options["Grid"]["upsampling"] - 1)
            x.append(np.linspace(start * reference_image.GetSpacing()[i],
                                 stop * reference_image.GetSpacing()[i],
                                 num_steps,
                                 endpoint=False))
        grid = np.meshgrid(x[0], x[1], x[2])
        bx = sitk.BSplineTransformInitializer(reference_image, (3, 3, 3), 2)
        perturb_magnitude = np.array(reference_image.GetSize()) * np.array(reference_image.GetSpacing()) / 20.0
        N = len(bx.GetParameters())
        displacements = np.zeros(N)
        for i in range(3):
            displacements[i * N // 3: (i + 1) * N // 3] = np.random.uniform(
                -perturb_magnitude[i], perturb_magnitude[i], N // 3)
        bx.SetParameters(displacements)
        deformed_image = sitk.Resample(reference_image, bx)
        sitk.WriteImage(deformed_image, "deformed_{:03d}.nii".format(r + 1))

        displacement = np.zeros((grid[0].size, 3))
        cnt = 0
        fixed_landmarks = np.array(track.options["Registration"]["reference landmarks"])
        landmarks = np.zeros(fixed_landmarks.shape, int)
        for i in range(landmarks.shape[0]):
            p = fixed_landmarks[i, :] * np.array(reference_image.GetSpacing())
            landmarks[i, :] = (2.0 * p - bx.TransformPoint(p)) / np.array(reference_image.GetSpacing())

        for k in range(grid[0].shape[2]):
            for i in range(grid[0].shape[0]):
                for j in range(grid[0].shape[1]):
                    p = np.array([grid[0][i, j, k],
                                  grid[1][i, j, k],
                                  grid[2][i, j, k]])
                    displacement[cnt, :] = p - bx.TransformPoint(p)
                    cnt += 1
        track.options["Registration"]["deformed landmarks"] = landmarks.tolist()
        vtk_disp = vtk.vtkImageData()
        vtk_disp.SetOrigin(np.array(track.options["Grid"]["origin"]) * np.array(reference_image.GetSpacing()))
        vtk_disp.SetSpacing(np.array(track.options["Grid"]["spacing"]) * np.array(reference_image.GetSpacing()) /
                            track.options["Grid"]["upsampling"])
        vtk_disp.SetDimensions(np.array(track.options["Grid"]["size"]) * track.options["Grid"]["upsampling"] -
                               (track.options["Grid"]["upsampling"] - 1))

        arr = numpy_support.numpy_to_vtk(displacement.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
        arr.SetNumberOfComponents(3)
        arr.SetName("Displacement")

        vtk_disp.GetPointData().SetVectors(arr)

        track.reference_path = "reference.nii"
        track.deformed_path = "deformed_{:03d}.nii".format(r + 1)

        track.execute()

        root_mean_square[r] = old_div(np.linalg.norm(displacement - track.results["Displacement"]),
                                      np.sqrt(displacement.shape[0]))

        track.writeResultsAsVTK("test_random_resonance_results_{:03d}".format(r + 1))

        if r == 0:
            track.writeImageAsVTK(track.ref_img, "test_random_resonance_reference")
        track.writeImageAsVTK(track.def_img, "test_random_resonance_deformed_{:03d}".format(r + 1))

        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName("resonance_true_disp_{:03d}.vti".format(r + 1))
        writer.SetInputData(vtk_disp)
        writer.Write()

    np.save('test_random_resonance_rmsd.npy', root_mean_square)


if __name__ == "__main__":
    main(sys.argv[-1])
