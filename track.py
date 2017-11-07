import sys
import re
import fnmatch
import os
import ConfigParser
import SimpleITK as sitk
import numpy as np
import vtk
from vtk.util import numpy_support


def adjustContrastOverDepth(img, r):
    simg = []
    size = list(img.GetSize()[0:2]) + [0]
    for i in xrange(img.GetSize()[2]):
        simg.append(sitk.Extract(img, size, [0, 0, i]))
    nimg = []
    for i, im in enumerate(simg):
        if i < r:
            tmp = simg[2*r-i:0:-1] + simg[0:i+2*r + 1]
            index = [0, 0, 2*r]
        elif i > len(simg) - r:
            n = 2 * r - (len(simg) - i)
            tmp = simg[i-2*r:] + simg[-1:-n-1:-1]
            index = [0, 0, 2*r]
        else:
            tmp = simg[i-r:i+r]
            index = [0, 0, r]
        tmp = sitk.JoinSeries(tmp)
        tmp = sitk.RescaleIntensity(tmp, 0.0, 1.0)
        nimg.append(sitk.Extract(tmp, size, index))
    nimg = sitk.JoinSeries(nimg)
    return nimg

def removeBright(img, s):
    gimg = sitk.GetArrayFromImage(sitk.DiscreteGaussian(img, s, 32, 0.01, False))
    if np.isnan(np.sum(gimg.ravel())):
        print("WARNING: NaN arose when removing bright spots. Returning unaltered image instead.")
        return img
    arr = sitk.GetArrayFromImage(img)
    c = np.percentile(arr, 95)
    arr[arr > c] = gimg[arr > c]
    nimg = sitk.GetImageFromArray(arr)
    nimg.CopyInformation(img)
    return nimg

def parseImgs(directory, crop, r):
    if directory.endswith(".nii"):
        img = sitk.ReadImage(directory, sitk.sitkFloat32)
        arr = sitk.GetArrayFromImage(img)
        if crop > 0:
            arr[:, :, crop:] = 0.0
        img2 = sitk.GetImageFromArray(arr)
        img2.CopyInformation(img)
        #img2 = removeBright(img2, 10)
        #img2 = adjustContrastOverDepth(img2, r)
        return sitk.RescaleIntensity(img2, 0.0, 1.0)

    files = fnmatch.filter(sorted(os.listdir(directory)), "*.tif")
    counter = [re.search("[0-9]*\.tif", f).group() for f in files]
    for i, c in enumerate(counter):
        counter[i] = int(c.replace('.tif', ''))
    files = np.array(files, dtype=object)
    sorter = np.argsort(counter)
    files = files[sorter]
    img = []
    cnt = 0
    for fname in files:
        filename = os.path.join(directory, fname)
        img.append(sitk.ReadImage(filename, sitk.sitkFloat32))
        #img[-1] = removeBright(img[-1], img[-1].GetSize()[0] / 10)
        if cnt > crop and crop > 0:
            img[-1] *= 0.0
        cnt += 1
    img = sitk.JoinSeries(img)
    #nimg = adjustContrastOverDepth(img, r)
    print("\nImported 3D image stack ranging from {:s} to {:s}".format(files[0], files[-1]))
    return sitk.RescaleIntensity(img, 0.0, 1.0)

def parseCfg(config):
    reader = ConfigParser.SafeConfigParser()
    reader.read(config)
    values = {}
    values["img_spacing"] = np.array(reader.get("Image", "spacing").split(","), dtype=float)
    values["grid_origin"] = np.array(reader.get("Grid", "origin").split(","), dtype=int)
    values["grid_spacing"] = np.array(reader.get("Grid", "spacing").split(","), dtype=int)
    values["grid_size"] = np.array(reader.get("Grid", "size").split(","), dtype=int)
    values["def_crop"] = int(reader.get("Grid", "crop"))
    values["img_resampling"] = np.array(reader.get("Image", "factors").split(","), dtype=float)

    values["reg_landmarks"] = np.genfromtxt(reader.get("Registration", "landmarks"), dtype=int, delimiter=",")
    values["reg_usemask"] = bool(reader.get("Registration", "mask"))
    values["reg_optimizer"] = reader.get("Registration", "method")
    values["reg_iterations"] = int(reader.get("Registration", "iterations"))
    values["reg_shrink"] = np.array(reader.get("Registration", "shrink factors").split(","), dtype=int)
    values["reg_sigmas"] = np.array(reader.get("Registration", "sigmas").split(","), dtype=float)
    return values

def makeGrid(rimg, values):
    grid = sitk.GetArrayFromImage(rimg)*0
    grid = grid.astype(np.bool_)
    logic= [np.copy(grid), np.copy(grid), np.copy(grid)]

    for i in xrange(3):
        ind = np.arange(0, values["grid_size"][i] * values["grid_spacing"][i], values["grid_spacing"][i])
        ind += values["grid_origin"][i]
        if i == 0:
            logic[i][:, :, ind] = True
        elif i == 1:
            logic[i][:, ind, :] = True
        elif i == 2:
            logic[i][ind, :, :] = True
    grid = np.logical_or(logic[0], logic[1])
    grid = np.logical_and(grid, logic[2])
    for i in xrange(2):
        min = values["grid_origin"][i]
        max = values["grid_origin"][i] + (values["grid_size"][i]-1) * values["grid_spacing"][i]
        if i == 0:
            grid[:, :, 0:min] = False
            grid[:, :, max+1:] = False
        else:
            grid[:, 0:min, :] = False
            grid[:, max+1:, :] = False
    gimg = sitk.GetImageFromArray(grid.astype(np.uint8))
    gimg.CopyInformation(rimg)
    d = sitk.SignedMaurerDistanceMap(gimg, False, False, False)
    grid = sitk.BinaryThreshold(d, 0, 1)
    grid.CopyInformation(gimg)
    return grid

def makeBox(rimg, values):
    mask = (sitk.GetArrayFromImage(rimg) * 0).astype(np.bool_)
    upper = values["grid_origin"] + (values["grid_size"] - 1) * values["grid_spacing"] + 1
    mask[values["grid_origin"][2]:upper[2], values["grid_origin"][1]:upper[1], values["grid_origin"][0]:upper[0]] = True
    mask = sitk.GetImageFromArray(mask.astype(np.uint8))
    mask.CopyInformation(rimg)
    return mask

def resample(img, l):
    rs = sitk.ResampleImageFilter()
    rs.SetOutputOrigin(img.GetOrigin())
    rs.SetSize(l)
    spacing = np.array(img.GetSpacing()) * np.array(img.GetSize()) / np.array(l).astype(np.float32) 
    rs.SetOutputSpacing(spacing)
    rs.SetInterpolator(sitk.sitkLinear)
    return rs.Execute(img)

def printProgress(rx):
    print("... ... Elapsed Iterations: {:d}".format(rx.GetOptimizerIteration()))
    print("... ... Current Metric Value: {:.5E}".format(rx.GetMetricValue()))

def getStrain(vtkgrid, disp):
    # calculate Green-Lagrange Strain at element centroids
    dNdEta = np.array([[-1, -1, -1],
                       [1, -1, -1],
                       [1, 1, -1],
                       [-1, 1, -1],
                       [-1, -1, 1],
                       [1, -1, 1],
                       [1, 1, 1],
                       [-1, 1, 1]], float) / 8.0

    strains = np.zeros((vtkgrid.GetNumberOfCells(), 3, 3), float)
    pstrain1 = np.zeros((vtkgrid.GetNumberOfCells(), 3), float)
    pstrain2 = np.zeros((vtkgrid.GetNumberOfCells(), 3), float)
    pstrain3 = np.zeros((vtkgrid.GetNumberOfCells(), 3), float)
    vstrain = np.zeros(vtkgrid.GetNumberOfCells(), float)
    for i in xrange(vtkgrid.GetNumberOfCells()):
        nodeIDs = vtkgrid.GetCell(i).GetPointIds()
        X = numpy_support.vtk_to_numpy(vtkgrid.GetCell(i).GetPoints().GetData())
        order = [0, 1, 3, 2, 4, 5, 7, 6]
        X = X[order, :]
        x = np.zeros((8, 3), float)
        for j, k in enumerate(order):
            x[j, :] = X[j, :] + disp[nodeIDs.GetId(k), :]
        dXdetaInvTrans = np.transpose(np.linalg.inv(np.einsum('ij,ik', X, dNdEta)))
        dNdX = np.einsum('ij,kj', dNdEta, dXdetaInvTrans)
        F = np.einsum('ij,ik', x, dNdX)
        C = np.dot(F.T, F)
        strains[i, :, :] = (C - np.eye(3)) / 2.0
        l, v = np.linalg.eigh(strains[i, :, :])
        norder = np.argsort(l)
        pstrain1[i, :] = l[norder[2]] * v[:, norder[2]]
        pstrain2[i, :] = l[norder[1]] * v[:, norder[1]]
        pstrain3[i, :] = l[norder[0]] * v[:, norder[0]]
        vstrain[i] = np.linalg.det(F) - 1.0
    for i in np.arange(1, pstrain1.shape[0]):
        if np.dot(pstrain1[0,:], pstrain1[i,:]) < 0:
            pstrain1[i,:] *= -1.0
        if np.dot(pstrain2[0,:], pstrain2[i,:]) < 0:
            pstrain2[i,:] *= -1.0
        if np.dot(pstrain3[0,:], pstrain3[i,:]) < 0:
            pstrain3[i,:] *= -1.0

    vtk_strain = numpy_support.numpy_to_vtk(strains.ravel(), deep=1, array_type=vtk.VTK_FLOAT)
    vtk_strain.SetNumberOfComponents(9)
    vtk_strain.SetName("Strain")

    vtkgrid.GetCellData().SetTensors(vtk_strain)

    vtk_pstrain1 = numpy_support.numpy_to_vtk(pstrain1.ravel(), deep=1, array_type=vtk.VTK_FLOAT)
    vtk_pstrain1.SetNumberOfComponents(3)
    vtk_pstrain1.SetName("1st Principal Strain")

    vtk_pstrain2 = numpy_support.numpy_to_vtk(pstrain2.ravel(), deep=1, array_type=vtk.VTK_FLOAT)
    vtk_pstrain2.SetNumberOfComponents(3)
    vtk_pstrain2.SetName("2nd Principal Strain")

    vtk_pstrain3 = numpy_support.numpy_to_vtk(pstrain3.ravel(), deep=1, array_type=vtk.VTK_FLOAT)
    vtk_pstrain3.SetNumberOfComponents(3)
    vtk_pstrain3.SetName("3rd Principal Strain")

    vtk_vstrain = numpy_support.numpy_to_vtk(vstrain.ravel(), deep=1, array_type=vtk.VTK_FLOAT)
    vtk_vstrain.SetNumberOfComponents(1)
    vtk_vstrain.SetName("Volumetric Strain")

    vtkgrid.GetCellData().AddArray(vtk_pstrain1)
    vtkgrid.GetCellData().AddArray(vtk_pstrain2)
    vtkgrid.GetCellData().AddArray(vtk_pstrain3)
    vtkgrid.GetCellData().AddArray(vtk_vstrain)

    c2p = vtk.vtkCellDataToPointData()
    c2p.SetInputData(vtkgrid)
    c2p.Update()
    vtkgrid = c2p.GetOutput()
    return vtkgrid

def main(rdir, ddir, config):
    values = parseCfg(config)
    if values["def_crop"] > 0:
        ref_crop = values["grid_origin"][2] + values["grid_spacing"][2] * (values["grid_size"][2] + 1)
    else:
        ref_crop = -1
    rimg = parseImgs(rdir, ref_crop, values["grid_spacing"][2])
    rimg.SetSpacing(values["img_spacing"])
    size = np.array(rimg.GetSize()) * values["img_resampling"]
    rimg = resample(rimg, size.astype(int))
    values["grid_spacing"] = (values["grid_spacing"] * values["img_resampling"]).astype(int)
    values["grid_origin"] = (values["grid_origin"] * values["img_resampling"]).astype(int)
    values["reg_landmarks"] = (values["reg_landmarks"] * values["img_resampling"]).astype(int)
    dimg = parseImgs(ddir, values["def_crop"], values["grid_spacing"][2])
    dimg.SetSpacing(values["img_spacing"])
    dimg = resample(dimg, size.astype(int))

    mask = makeBox(rimg, values)
    sitk.WriteImage(mask, "grid.nii")
    print("... Starting Deformable Registration")

    # setup initial affine transform
    edges = values["grid_spacing"] * (values["grid_size"] - 1)
    step = np.array([[0, 1, 0],
                     [1, 0, 0],
                     [0, -1, 0],
                     [-1, 0, 1],
                     [0, 1, 0],
                     [1, 0, 0],
                     [0, -1, 0]])
    fixedLandmarks = [values["grid_origin"]]
    for i in xrange(step.shape[0]):
        fixedLandmarks.append(fixedLandmarks[-1] + edges * step[i,:])

    ix = sitk.AffineTransform(3)
    landmarkTx = sitk.LandmarkBasedTransformInitializerFilter()
    landmarkTx.SetFixedLandmarks(np.ravel(np.array(fixedLandmarks, float) * np.array(rimg.GetSpacing())))
    landmarkTx.SetMovingLandmarks(np.ravel(values["reg_landmarks"] * np.array(rimg.GetSpacing())))
    landmarkTx.SetBSplineNumberOfControlPoints(8)
    landmarkTx.SetReferenceImage(rimg)
    outTx = landmarkTx.Execute(ix)
    rx = sitk.ImageRegistrationMethod()
    rx.AddCommand(sitk.sitkIterationEvent, lambda: printProgress(rx))
    if values["reg_iterations"] > 0:
        print("... ... Finding optimal BSpline transform")
        bx = sitk.BSplineTransformInitializer(rimg, (5,5,5), 2)
        rx.SetMovingInitialTransform(outTx)
        rx.SetInitialTransform(bx, True)
        if values["reg_usemask"]:
            rx.SetMetricFixedMask(mask)
            rx.SetMetricSamplingStrategy(rx.REGULAR)
        else:
            rx.SetMetricSamplingStrategy(rx.RANDOM)
        rx.SetMetricSamplingPercentagePerLevel(0.01*values["reg_shrink"])
        rx.SetInterpolator(sitk.sitkLinear)
        rx.SetMetricAsCorrelation()
        rx.SetMetricUseFixedImageGradientFilter(False)
        rx.SetShrinkFactorsPerLevel(values["reg_shrink"])
        rx.SetSmoothingSigmasPerLevel(values["reg_sigmas"])
        if values["reg_optimizer"] == "ConjugateGradient":
            rx.SetOptimizerAsConjugateGradientLineSearch(1.0,
                                                        values["reg_iterations"],
                                                        1e-6,
                                                        10)
            rx.SetOptimizerScalesFromPhysicalShift()
        elif values["reg_optimizer"] == "GradientDescent":
            rx.SetOptimizerAsGradientDescent(1.0,
                                            values["reg_iterations"],
                                            1e-6,
                                            10,
                                            rx.EachIteration)
            rx.SetOptimizerScalesFromPhysicalShift()
        outTx.AddTransform(rx.Execute(rimg, dimg))
        print("... ... Optimal BSpline transform determined ")
        print("... ... ... Elapsed Iterations: {:d}\n... ... ... Final Metric Value: {:.5E}".format(rx.GetOptimizerIteration(),
                                                                                                    rx.GetMetricValue()))
    print("... Registration Complete")
    print("... Saving Results to VTK Grid")
    values["grid_origin"] = (values["grid_origin"] / values["img_resampling"]).astype(int)
    x = []
    for i in xrange(3):
        x.append(np.arange(values["grid_origin"][i] * np.array(rimg.GetSpacing())[i],
                           (values["grid_origin"][i] + values["grid_size"][i]*values["grid_spacing"][i]) * np.array(rimg.GetSpacing())[i] ,
                           values["grid_spacing"][i] * np.array(rimg.GetSpacing())[i]))
    grid = np.meshgrid(x[0], x[1], x[2])
    disp = np.zeros((grid[0].size, 3))
    cnt = 0
    for k in xrange(grid[0].shape[2]):
        for i in xrange(grid[0].shape[0]):
            for j in xrange(grid[0].shape[1]):
                p = np.array([grid[0][i,j,k],
                              grid[1][i,j,k],
                              grid[2][i,j,k]])
                disp[cnt, :] = outTx.TransformPoint(p) - p
                cnt += 1

    vtk_disp = vtk.vtkImageData()
    vtk_disp.SetOrigin(values["grid_origin"] * np.array(rimg.GetSpacing()))
    vtk_disp.SetSpacing(values["grid_spacing"] * np.array(rimg.GetSpacing()))
    vtk_disp.SetDimensions(values["grid_size"])


    vtk_disp = getStrain(vtk_disp, disp)

    arr = numpy_support.numpy_to_vtk(disp.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
    arr.SetNumberOfComponents(3)
    arr.SetName("Displacement")
    vtk_disp.GetPointData().SetVectors(arr)

    size = (np.array(dimg.GetSize()) / values["img_resampling"]).astype(int)
    dimg = resample(dimg, size)
    a = numpy_support.numpy_to_vtk(sitk.GetArrayFromImage(dimg).ravel(), deep=True, array_type=vtk.VTK_FLOAT)
    vtk_img = vtk.vtkImageData()
    vtk_img.SetOrigin(dimg.GetOrigin())
    vtk_img.SetSpacing(dimg.GetSpacing())
    vtk_img.SetDimensions(dimg.GetSize())
    vtk_img.GetPointData().SetScalars(a)

    if ddir.endswith(".nii"):
        name = ddir.replace(".nii", "")
    else:
        name = ddir
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName("disp_{:s}.vti".format(name))
    writer.SetInputData(vtk_disp)
    writer.Write()

    writer.SetFileName("image_{:s}.vti".format(name))
    writer.SetInputData(vtk_img)
    writer.Write()
    print("Analysis Complete!")

if __name__ == "__main__":
    main(*sys.argv[1:])
