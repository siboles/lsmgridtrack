from __future__ import division
from builtins import range
from past.utils import old_div
from .context import lsmgridtrack as lsm
import sys
import os
import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util import numpy_support

def main(repeats=1):
    repeats = int(repeats)
    path = os.path.dirname(__file__)
    rimg = sitk.ReadImage(os.path.join(path, "data", "ref.nii"), sitk.sitkFloat32)
    rimg.SetSpacing((0.5, 0.5, 1.0))
    #pad the image in z by +/- 5 zero voxels
    rimg = sitk.ConstantPad(rimg, (0, 0, 5), (0, 0, 5))
    rimg.SetOrigin([0,0,0])
    sitk.WriteImage(rimg, "reference.nii")
    np.random.seed(seed=1545281929)

    rmsd = np.zeros(repeats, float)
    for r in range(repeats):
        track = lsm.tracker(config=os.path.join(path, "data", "testRandom.yaml"))
        if r == 0:
            x = []
            for i in range(3):
                x.append(np.arange(track.options["Grid"]["origin"][i] * rimg.GetSpacing()[i],
                                (track.options["Grid"]["origin"][i] + track.options["Grid"]["size"][i]*track.options["Grid"]["spacing"][i]) * rimg.GetSpacing()[i] ,
                                track.options["Grid"]["spacing"][i] * rimg.GetSpacing()[i]))
            grid = np.meshgrid(x[0], x[1], x[2])
        bx = sitk.BSplineTransformInitializer(rimg, (5, 5, 2), 2)
        displacements = np.random.uniform(-10, 10, len(bx.GetParameters()))
        # reduce displacements in z due to small volume depth
        displacements[np.arange(1, displacements.size + 1) % 3 == 0] /= 5.0
        bx.SetParameters(displacements)
        dimg = sitk.Resample(rimg, bx)
        sitk.WriteImage(dimg, "deformed_{:03d}.nii".format(r + 1))

        disp = np.zeros((grid[0].size, 3))
        cnt = 0
        edges = np.array(track.options["Grid"]["spacing"]) * (np.array(track.options["Grid"]["size"]) - 1)
        step = np.array([[0, 1, 0],
                        [1, 0, 0],
                        [0, -1, 0],
                        [-1, 0, 1],
                        [0, 1, 0],
                        [1, 0, 0],
                        [0, -1, 0]])
        fixedLandmarks = [np.array(track.options["Grid"]["origin"])]
        for i in range(step.shape[0]):
            fixedLandmarks.append(fixedLandmarks[-1] + edges * step[i,:])

        landmarks = np.zeros((8,3), int)
        for i, l in enumerate(fixedLandmarks):
            p = np.array(l, float) * np.array(rimg.GetSpacing())
            landmarks[i, :] = (old_div((2*p - bx.TransformPoint(p)), np.array(rimg.GetSpacing()))).astype(int)

        for k in range(grid[0].shape[2]):
            for i in range(grid[0].shape[0]):
                for j in range(grid[0].shape[1]):
                    p = np.array([grid[0][i,j,k],
                                  grid[1][i,j,k],
                                  grid[2][i,j,k]])
                    disp[cnt, :] = p - bx.TransformPoint(p)
                    cnt += 1
        track.options["Registration"]["landmarks"] = landmarks
        vtk_disp = vtk.vtkImageData()
        vtk_disp.SetOrigin(track.options["Grid"]["origin"] * np.array(rimg.GetSpacing()))
        vtk_disp.SetSpacing(track.options["Grid"]["spacing"] * np.array(rimg.GetSpacing()))
        vtk_disp.SetDimensions(track.options["Grid"]["size"])

        arr = numpy_support.numpy_to_vtk(disp.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
        arr.SetNumberOfComponents(3)
        arr.SetName("Displacement")

        vtk_disp.GetPointData().SetVectors(arr)

        track.reference_path="reference.nii"
        track.deformed_path="deformed_{:03d}.nii".format(r+1)
        track.execute()

        rmsd[r] = old_div(np.linalg.norm(disp - track.results["Displacement"]), np.sqrt(disp.shape[0]))

        track.writeResultsAsVTK("testRandom_results_{:03d}".format(r + 1))

        if r == 0:
            track.writeImageAsVTK(track.ref_img, "testRandom_reference")
        track.writeImageAsVTK(track.def_img, "testRandom_deformed_{:03d}".format(r + 1))

        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName("true_disp{:03d}.vti".format(r + 1))
        writer.SetInputData(vtk_disp)
        writer.Write()

    np.save('testRandom_rmsd.npy', rmsd)

if __name__ == "__main__":
    main(sys.argv[-1])
