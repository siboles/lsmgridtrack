import re
import fnmatch
import os
from collections import MutableMapping, OrderedDict
import yaml
import SimpleITK as sitk
import numpy as np
import vtk
from vtk.util import numpy_support
from openpyxl import Workbook

class FixedDict(MutableMapping):
    def __init__(self, data):
        self.__data = data

    def __len__(self):
        return len(self.__data)

    def __iter__(self):
        return iter(self.__data)

    def __setitem__(self, k, v):
        if k not in self.__data:
            raise KeyError("{:s} is not an acceptable key.".format(k))

        self.__data[k] = v

    def __delitem__(self, k):
        raise NotImplementedError

    def __getitem__(self, k):
        if k not in self.__data:
            raise KeyError("{:s} is not an acceptable key.".format(k))
        return self.__data[k]

    def __contains__(self, k):
        return k in self.__data

    def __repr__(self):
        return repr(self.__data)

class tracker(object):
    """
    Performs deformable image registration of laser scanning microscopy images
    with a photobleached grid.

    Parameters
    ----------
    reference_path : str
        The path to either a TIFF stack or 3D Nifti (.nii) image in the reference configuration.

        .. note::
            If *ref_img* is also specified this will be overridden.

    deformed_path : str
        The path to either a TIFF stack or 3D Nifti (.nii) image in a deformed configuration.

       latex linespacing .. note::
            If *def_img* is also specified this will be overridden.

    ref_img : SimpleITK.Image()
        Reference image stored in memory rather than read from disk.

        .. note::
            If *reference_path* is also specified, *ref_img* will supercede.

    def_img : SimpleITK.Image()
        Deformed image stored in memory rather than read from disk.

        .. note::
            If *deformed_path* is also specified, *def_img* will supercede.

    config : str, optional
        The path to a configuration file in YAML format where analysis options are set. If this
        is not provided, default options will be set.

        .. note::
            "Grid" key, value pairs must be set prior to running *execute()*.

    options: dict, optional
        Alternatively to using a configuration file, *options* can be set by providing a dictionary of
        key, value pairs to change. Likewise, the *options* attribute of the object can be modified.

        .. note::
            *options* keys are immutable.

        .. note::
            If both *options* and *config* are provided during object instantiation, values in *config* will
            supercede those in *options*.

    Attributes
    ----------
    transform : SimpleITK.Transform()
        The composite transform calculated from the deformable registration.
    results : dict
        Results at N grid vertices

        * Coordinates - ndarray(N, 3) -- undeformed grid vertex locations
        * Displacement - ndarray(N, 3) -- grid vertex displacements
        * Strain - ndarray(N, 3, 3) -- Green-Lagrange strain tensors at grid vertices
        * 1st Principal Strain -- ndarray(N, 3) - 1st (Maximum) Principal Strain vectors at grid vertices
        * 2nd Principal Strain -- ndarray(N, 3) - 2nd (Median) Principal Strain vectors at grid vertices
        * 3rd Principal Strain -- ndarray(N, 3) - 3rd (Minimum) Principal Strain vectors at grid vertices
        * Maximum Shear Strain -- ndarray(N) - Maximum Shear Strain at grid vertices
        * Volumetric Strain -- ndarray(N) - Volumetric Strain at grid vertices
    vtkgrid : vtk.ImageData()
        A VTK image of the tracked grid with displacements and strains stored at vertices.
    """
    def __init__(self, **kwargs):
        # Default values
        self.options = FixedDict({
            "Image": FixedDict({
                "spacing": [1.0, 1.0, 1.0],
                "resampling": [1.0, 1.0, 1.0]}),
            "Grid": FixedDict({
                "origin": False,
                "spacing": False,
                "size": False,
                "crop": False}),
            "Registration": FixedDict({
                "method": "BFGS",
                "iterations": 100,
                "usemask": False,
                "landmarks": False,
                "shrink_levels": [1.0],
                "sigma_levels": [0.0]})})

        self.reference_path = None
        self.deformed_path = None
        self.ref_img = None
        self.def_img = None

        self.config = None

        for key, value in kwargs.iteritems():
            if key == "options":
                for k, v in value.iteritems():
                    for k2, v2 in v.iteritems():
                        self.options[k][k2] = v2
            else:
                setattr(self, key, value)

        if self.config is not None:
            self.parseConfig()

    def execute(self):
        """
        Executes the deformable image registration and post-analysis.
        """
        self._castOptions()
        self.results = OrderedDict()

        if self.ref_img is None and self.reference_path is not None:
            if self.options["Grid"]["crop"]:
                crop = self.options["Grid"]["origin"][2] + self.options["Grid"]["spacing"][2] * self.options["Grid"]["size"][2] + 2
            else:
                crop = self.options["Grid"]["crop"]
            self.ref_img = self.parseImg(self.reference_path, crop, self.options["Image"]["spacing"])
        if self.def_img is None and self.deformed_path is not None:
            self.def_img = self.parseImg(self.deformed_path, self.options["Grid"]["crop"], self.options["Image"]["spacing"])
        if np.abs(self.options["Image"]["resampling"] - 1.0).sum() > 1e-7:
            self.ref_img = self._resampleImage(self.ref_img, self.options["Image"]["resampling"])
            self.def_img = self._resampleImage(self.def_img, self.options["Image"]["resampling"])

        if self.options["Registration"]["usemask"]:
            self._makeMask()
        print("... Starting Deformable Registration")

        if np.any(self.options["Registration"]["landmarks"]):
            fixed_pts = self._defineFixedLandmarks()
            moving_pts = (self.options["Registration"]["landmarks"] * np.array(self.ref_img.GetSpacing())).ravel()
            if moving_pts.size % 3 != 0:
                raise("ERROR: deformed image landmark index arrays must all be length 3.")
            if moving_pts.size != 24:
                raise ("ERROR: {:d} deformed image landmarks were provided. Initialization with landmarks requires the 8 corners of the grid domain order counter-clockwise".format(moving_pts.size / 3))
            # setup initial affine transform
            ix = sitk.AffineTransform(3)
            ix = sitk.BSplineTransformInitializer(self.ref_img, (3, 3, 3), 3)
            landmarkTx = sitk.LandmarkBasedTransformInitializerFilter()
            landmarkTx.SetFixedLandmarks(fixed_pts)
            landmarkTx.SetMovingLandmarks(moving_pts)
            landmarkTx.SetReferenceImage(self.ref_img)
            outTx = landmarkTx.Execute(ix)
        else:
            outTx = sitk.BSplineTransformInitializer(self.ref_img, (3, 3, 3), 3)

        rx = sitk.ImageRegistrationMethod()
        rx.AddCommand(sitk.sitkIterationEvent, lambda: self._printProgress(rx))
        if self.options["Registration"]["iterations"] > 0:
            print("... ... Finding optimal BSpline transform")
            rx.SetInitialTransform(outTx, True)
            if self.options["Registration"]["usemask"]:
                rx.SetMetricFixedMask(self._mask)
                rx.SetMetricSamplingStrategy(rx.REGULAR)
            else:
                rx.SetMetricSamplingStrategy(rx.RANDOM)
            rx.SetMetricSamplingPercentagePerLevel(0.02*self.options["Registration"]["shrink_levels"])
            rx.SetInterpolator(sitk.sitkLinear)
            rx.SetMetricAsCorrelation()
            rx.SetMetricUseFixedImageGradientFilter(False)
            rx.SetShrinkFactorsPerLevel(self.options["Registration"]["shrink_levels"])
            rx.SetSmoothingSigmasPerLevel(self.options["Registration"]["sigma_levels"])
            if self.options["Registration"]["method"] == "ConjugateGradient":
                maximumStepSize = np.min(np.array(self.ref_img.GetSize()) * np.array(self.ref_img.GetSpacing())) / 2.0
                rx.SetOptimizerAsConjugateGradientLineSearch(1.0,
                                                             self.options["Registration"]["iterations"],
                                                             1e-5,
                                                             20,
                                                             lineSearchUpperLimit = 3.0,
                                                             maximumStepSizeInPhysicalUnits = maximumStepSize)
                rx.SetOptimizerScalesFromPhysicalShift()
            elif self.options["Registration"]["method"] == "GradientDescent":
                rx.SetOptimizerAsGradientDescent(1.0,
                                                 self.options["Registration"]["iterations"],
                                                 1e-5,
                                                 20,
                                                 rx.EachIteration)
                rx.SetOptimizerScalesFromPhysicalShift()
            elif self.options["Registration"]["method"] == "BFGS":
                rx.SetOptimizerAsLBFGSB(numberOfIterations = self.options["Registration"]["iterations"])
            outTx = rx.Execute(self.ref_img, self.def_img)
            print("... ... Optimal BSpline transform determined ")
            print("... ... ... Elapsed Iterations: {:d}\n... ... ... Final Metric Value: {:.5E}".format(rx.GetOptimizerIteration(),
                                                                                                        rx.GetMetricValue()))
        print("... Registration Complete")
        self.transform = outTx
        self.getGridDisplacements()
        self.getStrain()
        print("Analysis Complete!")

    def parseConfig(self):
        """
        Parse configuration file in YAML format.
        """
        with open(self.config) as user:
            user_settings = yaml.load(user)

        for k, v in user_settings.items():
            for k2, v2 in v.items():
                self.options[k][k2] = v2

    def parseImg(self, p, crop, spacing):
        """
        Parse image from disk. Supports either a sequence of TIFF images or
        a NifTI (.nii) 3D image.
        """
        if p.endswith(".nii"):
            img = sitk.ReadImage(p, sitk.sitkFloat32)
            img.SetSpacing(spacing)
            arr = sitk.GetArrayFromImage(img)
            if crop:
                arr[:, :, crop:] = 0.0
            img2 = sitk.GetImageFromArray(arr)
            img2.CopyInformation(img)
            return sitk.RescaleIntensity(img2, 0.0, 1.0)

        files = fnmatch.filter(sorted(os.listdir(p)), "*.tif")
        counter = [re.search("[0-9]*\.tif", f).group() for f in files]
        for i, c in enumerate(counter):
            counter[i] = int(c.replace('.tif', ''))
        files = np.array(files, dtype=object)
        sorter = np.argsort(counter)
        files = files[sorter]
        img = []
        cnt = 0
        for fname in files:
            filename = os.path.join(p, fname)
            img.append(sitk.ReadImage(filename, sitk.sitkFloat32))
            if crop and cnt > crop:
                img[-1] *= 0.0
            cnt += 1
        img = sitk.JoinSeries(img)
        img.SetSpacing(spacing)
        print("\nImported 3D image stack ranging from {:s} to {:s}".format(files[0], files[-1]))
        return sitk.RescaleIntensity(img, 0.0, 1.0)

    def getGridDisplacements(self):
        r"""
        Using calculated transform from deformed image to reference image.
        """
        origin = (self.options["Grid"]["origin"] / self.options["Image"]["resampling"]).astype(int)
        x = []
        for i in xrange(3):
            x.append(np.arange(origin[i] * self.ref_img.GetSpacing()[i],
                               (origin[i] + self.options["Grid"]["size"][i]*self.options["Grid"]["spacing"][i]) * self.ref_img.GetSpacing()[i],
            self.options["Grid"]["spacing"][i] * self.ref_img.GetSpacing()[i]))
        grid = np.meshgrid(x[0], x[1], x[2])
        self.results["Coordinates"] = np.zeros((grid[0].size, 3))
        self.results["Displacement"] = np.zeros((grid[0].size, 3))
        cnt = 0
        for k in xrange(grid[0].shape[2]):
            for i in xrange(grid[0].shape[0]):
                for j in xrange(grid[0].shape[1]):
                    p = np.array([grid[0][i,j,k],
                                  grid[1][i,j,k],
                                  grid[2][i,j,k]])
                    self.results["Coordinates"][cnt, :] = p
                    self.results["Displacement"][cnt, :] = self.transform.TransformPoint(p) - p
                    cnt += 1

    def getStrain(self):
        r"""
        Calculates the Green-Lagrange Strain at element centroids and interpolates to VTK
        grid vertices. Likewise, calculates principal and volumetric strains. For these values
        we need the deformation gradient, which assuming Einstein's summation convention unless explicitly
        indicated, follows as:

        .. note::
            Capital and lowercase letters imply reference and deformed configurations, respectively.

        Notation:

        .. math::

           F^i_{\,J} = \sum_{a=1}^{8} x^i_{\,a}\frac{\partial N_a}{\partial X^J}.

        We therefore need to determine :math:`\frac{\partial N_a}{\partial X^J}`.
        From the chain rule,

        .. math::

           \frac{\partial N_a}{\partial X^J} = \frac{\partial N_a}{\partial \eta^I} \left (\frac{\partial X^I}{\partial \eta^J} \right)^{-T}

        where

        .. math::

           \frac{\partial X^I}{\partial \eta^J} = \sum_{a=1}^{8} X^I_{\,a} \frac{\partial N_a}{\partial \eta^J}.

        The Green-Lagrange strain tensor then follows as,

        .. math::

           E_{IJ} = \frac{1}{2}\left(\left(F_I^{\,i}\right) g_{ij} F^j_{\,J} - I_{IJ}\right)

        where :math:`g_{ij}` is the metric tensor and :math:`I_{IJ}` is the identity.

        The eigevalues * eigenvectors of this tensor ordered decreasing by eigenvalue are the principal strains.
        The volumetric strain is,

        .. math::

           E_{volumetric} = \det{F^i_{\,J}} - 1.0.

        Returns
        -------
        vtkgrid : vtk.ImageData()
            VTK image grid with all results stored at grid vertices. 
        """
        vtkgrid = vtk.vtkImageData()
        vtkgrid.SetOrigin(np.array(self.options["Grid"]["origin"]) * np.array(self.ref_img.GetSpacing()))
        vtkgrid.SetSpacing(np.array(self.options["Grid"]["spacing"]) * np.array(self.ref_img.GetSpacing()))
        vtkgrid.SetDimensions(self.options["Grid"]["size"])

        arr = numpy_support.numpy_to_vtk(self.results["Displacement"].ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
        arr.SetNumberOfComponents(3)
        arr.SetName("Displacement")
        vtkgrid.GetPointData().SetVectors(arr)

        cells = vtkgrid.GetNumberOfCells()

        dNdEta = np.array([[-1, -1, -1],
                           [1, -1, -1],
                           [1, 1, -1],
                           [-1, 1, -1],
                           [-1, -1, 1],
                           [1, -1, 1],
                           [1, 1, 1],
                           [-1, 1, 1]], float) / 8.0

        strain = np.zeros((cells, 3, 3), float)
        pstrain1 = np.zeros((cells, 3), float)
        pstrain2 = np.zeros((cells, 3), float)
        pstrain3 = np.zeros((cells, 3), float)
        vstrain = np.zeros(cells, float)
        maxshear = np.zeros(cells, float)
        order = [0, 1, 3, 2, 4, 5, 7, 6]
        for i in xrange(cells):
            nodeIDs = vtkgrid.GetCell(i).GetPointIds()
            X = numpy_support.vtk_to_numpy(vtkgrid.GetCell(i).GetPoints().GetData())
            X = X[order, :]
            x = np.zeros((8, 3), float)
            for j, k in enumerate(order):
                x[j, :] = X[j, :] + self.results["Displacement"][nodeIDs.GetId(k), :]
            dXdetaInvTrans = np.transpose(np.linalg.inv(np.einsum('ij,ik', X, dNdEta)))
            dNdX = np.einsum('ij,kj', dNdEta, dXdetaInvTrans)
            F = np.einsum('ij,ik', x, dNdX)
            C = np.dot(F.T, F)
            strain[i, :, :] = (C - np.eye(3)) / 2.0
            l, v = np.linalg.eigh(strain[i, :, :])
            pstrain1[i, :] = l[2] * v[:, 2]
            pstrain2[i, :] = l[1] * v[:, 1]
            pstrain3[i, :] = l[0] * v[:, 0]
            vstrain[i] = np.linalg.det(F) - 1.0
            maxshear[i] = np.abs(np.linalg.norm(pstrain1[i,:]) - np.linalg.norm(pstrain3[i,:]))
        for i in np.arange(1, pstrain1.shape[0]):
            if np.dot(pstrain1[0,:], pstrain1[i,:]) < 0:
                pstrain1[i,:] *= -1.0
            if np.dot(pstrain2[0,:], pstrain2[i,:]) < 0:
                pstrain2[i,:] *= -1.0
            if np.dot(pstrain3[0,:], pstrain3[i,:]) < 0:
                pstrain3[i,:] *= -1.0


        vtk_strain = numpy_support.numpy_to_vtk(strain.ravel(), deep=1, array_type=vtk.VTK_FLOAT)
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

        vtk_maxshear = numpy_support.numpy_to_vtk(maxshear.ravel(), deep=1, array_type=vtk.VTK_FLOAT)
        vtk_maxshear.SetNumberOfComponents(1)
        vtk_maxshear.SetName("Maximum Shear Strain")

        vtkgrid.GetCellData().AddArray(vtk_pstrain1)
        vtkgrid.GetCellData().AddArray(vtk_pstrain2)
        vtkgrid.GetCellData().AddArray(vtk_pstrain3)
        vtkgrid.GetCellData().AddArray(vtk_vstrain)
        vtkgrid.GetCellData().AddArray(vtk_maxshear)

        c2p = vtk.vtkCellDataToPointData()
        c2p.SetInputData(vtkgrid)
        c2p.Update()
        self.vtkgrid = c2p.GetOutput()
        names = ("Strain", "1st Principal Strain", "2nd Principal Strain",
                 "3rd Principal Strain", "Maximum Shear Strain", "Volumetric Strain")
        for a in names:
            if a != "Strain":
                self.results[a] = numpy_support.vtk_to_numpy(self.vtkgrid.GetPointData().GetArray(a))
            else:
                self.results[a] = numpy_support.vtk_to_numpy(self.vtkgrid.GetPointData().GetArray(a)).reshape(-1, 3, 3)


    def writeImageAsVTK(self, img, name, sampling_factor=[1.0, 1.0, 1.0]):
        """
        Save image to disk as a .vti file. Image will be resampled such that it has spacing equal
        to that specified in *options["Image"]["spacing"]*.

        Parameters
        ----------
        img : SimpleITK.Image(), required
            Image in memory to save to disk. This is assumed to be either *ref_img* or *def_img*.
        name : str, required
            Name of file to save to disk without the file suffix
        """
        print("... Saving Image to {:s}.vti".format(name))
        factor = self.options["Image"]["spacing"] / np.array(img.GetSpacing(), float) * np.array(sampling_factor) 
        img = self._resampleImage(img, factor)
        a = numpy_support.numpy_to_vtk(sitk.GetArrayFromImage(img).ravel(), deep=True, array_type=vtk.VTK_FLOAT)
        vtk_img = vtk.vtkImageData()
        vtk_img.SetOrigin(img.GetOrigin())
        vtk_img.SetSpacing(img.GetSpacing())
        vtk_img.SetDimensions(img.GetSize())
        vtk_img.GetPointData().SetScalars(a)
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName("{:s}.vti".format(name))
        writer.SetInputData(vtk_img)
        writer.Write()

    def writeResultsAsVTK(self, name):
        """
        Save *vtkgrid* to disk as .vti file.

        Parameters
        ----------
        name : str, required
            Name of file to save to disk without the file suffix
        """
        print("... Saving Results to {:s}.vti".format(name))
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName("{:s}.vti".format(name))
        writer.SetInputData(self.vtkgrid)
        writer.Write()

    def writeResultsAsExcel(self, name):
        """
        Save *results* to disk as an xlsx file. Each key, value pair is saved on a separate sheet.

        Parameters
        ----------
        name : str, required
            Name of file to save to disk without the file suffix
        """
        print("... Saving Results to {:s}.xlsx".format(name))
        wb = Workbook()
        titles = ("Coordinates",
                  "Displacement",
                  "Strain",
                  "1st Principal Strain",
                  "2nd Principal Strain",
                  "3rd Principal Strain",
                  "Maximum Shear Strain",
                  "Volumetric Strain")
        ws = []
        for i, t in enumerate(titles):
            if i == 0:
                ws.append(wb.active)
                ws[-1].title = t
            else:
                ws.append(wb.create_sheet(title=t))
            if t == "Strain":
                ws[i].append(["XX", "YY", "ZZ", "XY", "XZ", "YZ"])
                data = self.results[t].reshape(-1, 9)[:, [0, 4, 8, 1, 2, 5]]
            else:
                data = self.results[t]
            if len(data.shape) > 1:
                for j in xrange(data.shape[0]):
                    ws[i].append(list(data[j, :]))
            else:
                for j in xrange(data.shape[0]):
                    ws[i].append([data[j]])

        wb.save(filename="{:s}.xlsx".format(name))

    def writeResultsAsNumpy(self, name):
        """
        Save *results* to disk as an .npz file, which is an uncompressed zipped archive of
        multiple numpy arrays in .npy format. This can easily be loaded into memory from disk
        using numpy.load(filename).

        Parameters
        ----------
        name : str, required
            Name of file to save to disk without the file suffix
        """
        print("... Saving file as numpy archive {:s}.npz".format(name))
        np.savez_compressed("{:s}.npz".format(name), **self.results)

    def _resampleImage(self, img, factor):
        rs = sitk.ResampleImageFilter()
        rs.SetOutputOrigin(img.GetOrigin())
        rs.SetSize((np.array(factor) * np.array(img.GetSize())).astype(int))
        spacing = (np.array(img.GetSpacing()) /
                   np.array(factor).astype(np.float32))
        rs.SetOutputSpacing(spacing)
        rs.SetInterpolator(sitk.sitkLinear)
        return rs.Execute(img)

    def _makeMask(self):
        mask = (sitk.GetArrayFromImage(self.ref_img) * 0).astype(np.bool_)
        upper = self.options["Grid"]["origin"] + (self.options["Grid"]["size"] - 1) * self.options["Grid"]["spacing"] + 1
        mask[self.options["Grid"]["origin"][2]:upper[2], self.options["Grid"]["origin"][1]:upper[1], self.options["Grid"]["origin"][0]:upper[0]] = True
        self._mask = sitk.GetImageFromArray(mask.astype(np.uint8))
        self._mask.CopyInformation(self.ref_img)

    def _defineFixedLandmarks(self):
        edges = np.array(self.options["Grid"]["spacing"]) * np.array((self.options["Grid"]["size"] - 1))
        step = np.array([[0, 1, 0],
                        [1, 0, 0],
                        [0, -1, 0],
                        [-1, 0, 1],
                        [0, 1, 0],
                        [1, 0, 0],
                        [0, -1, 0]])
        fixedLandmarks = [self.options["Grid"]["origin"]]
        for i in xrange(step.shape[0]):
            fixedLandmarks.append(fixedLandmarks[-1] + edges * step[i,:])
        return (np.array(fixedLandmarks) * np.array(self.ref_img.GetSpacing())).ravel()

    def saveTransform(self, name):
        sitk.WriteTransform(self.transform, "{:s}.tfm")

    def _castOptions(self):
        arrays = (("Image", "spacing", "float"),
                  ("Image", "resampling", "float"),
                  ("Grid", "origin", "int"),
                  ("Grid", "spacing", "int"),
                  ("Grid", "size", "int"),
                  ("Registration", "landmarks", "int"),
                  ("Registration", "shrink_levels", "int"),
                  ("Registration", "sigma_levels", "float"))
        for k1, k2, v in arrays:
            if k1 == "Grid" and k2 in ("spacing", "origin", "size"):
                if self.options[k1][k2]:
                    pass
                else:
                    raise SystemError("Values for Grid spacing, origin, and size must be provided before executing analysis.")
            if v == "float":
                self.options[k1][k2] = np.array(self.options[k1][k2], dtype=float)
            else:
                self.options[k1][k2] = np.array(self.options[k1][k2], dtype=int)

    def _printProgress(self, rx):
        print("... ... Elapsed Iterations: {:d}".format(rx.GetOptimizerIteration()))
        print("... ... Current Metric Value: {:.5E}".format(rx.GetMetricValue()))