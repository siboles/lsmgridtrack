import sys
import os
import operator
import SimpleITK as sitk
import numpy as np
from numba import njit, prange
import vtk
from vtk.util import numpy_support

@njit(parallel=True)
def _getAxes(s, cutoff, Rz):
    axes = np.zeros_like(s)
    for i in prange(s.shape[0]):
        for j in prange(s.shape[1]):
            for k in prange(s.shape[2]):
                w, v = np.linalg.eigh(s[i,j,k,:,:])
                w = np.abs(w)
                idx = w.argsort()
                w = w[idx]
                confidence = 1 - np.exp(-np.sum(w**2) / (2 * cutoff**2))
                a = 0.1 + np.sqrt(confidence * np.exp(-w[0] / w[2]))
                b = 0.1
                scale = (3.0 / (4.0 * np.pi * a * b**2))**(1.0 / 3.0)
                v1 = v[:, idx[0]]
                flipper = np.dot(v1, np.array([1.0, 0.0, 0.0], dtype=np.float32))
                if flipper < 0.0:
                    flipper = -1.0
                else:
                    flipper = 1.0
                v1 *= flipper

                v2 = np.dot(v1, Rz.T)
                # cross product v1, v2
                v2 = np.array([
                    v1[1] * v2[2] - v1[2] * v2[1],
                    -v1[0] * v2[2] + v1[2] * v2[0],
                    v1[0] * v2[1] - v1[1] * v2[0]
                ])
                #check for case of v1=Z axis
                l = np.linalg.norm(v2)
                if l < 1e-7:
                    v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                v3 = np.array([
                    v2[1] * v1[2] - v2[2] * v1[1],
                    -v2[0] * v1[2] + v2[2] * v1[0],
                    v2[0] * v1[1] - v2[1] * v1[0]
                ])

                v1 *= scale * a
                v2 *= scale * b
                v3 *= scale * b

                axes[i, j, k, :, 0] = v1
                axes[i, j, k, :, 1] = v2
                axes[i, j, k, :, 2] = v3
    return axes

class orientation():
    def __init__(self, data=None, spacing=None, featureScale=1.0):
        self.__supportedImageTypes = (".tif",
                                      ".tiff",
                                      ".png",
                                      ".jpg",
                                      ".jpeg",
                                      ".nii")

        self.featureScale = featureScale

        if data is None:
            raise ValueError('data must be a file, directory, SimpleITK image, or ndarray')
        elif isinstance(data, sitk.Image):
            self.image = data
        elif isinstance(data, np.ndarray):
            self.image = sitk.GetImageFromArray(data)
        elif os.path.isfile(data):
            self._parseImageFile(data)
        elif os.path.isdir(data):
            self._parseImageSequence(data)

        self.spacing = spacing

        print(self.image.GetSpacing())
        self.vtkimage = None

    @property
    def spacing(self):
        return self.__spacing

    @spacing.setter
    def spacing(self, spacing):
        if spacing is None:
            print(':::WARNING::: Image spacing was not specified.')
            self.__spacing = self.image.GetSpacing()
            self.image.SetSpacing(self.__spacing)
        else:
            self.__spacing = spacing
            self.image.SetSpacing(self.__spacing)


    def _parseImageFile(self, p):
        filename, file_extension = os.path.splitext(p)
        if file_extension.lower() in self.__supportedImageTypes:
            self.image = sitk.ReadImage(p, sitk.sitkFloat32)
            print('Imported image {:s}'.format(p))
        else:
            raise ValueError('Unsupported file type with extension, {:s}, detected.'.format(file_extension)+
                             '\n'.join('{:s}'.format(t) for t in self.__supportedImageTypes))

    def _parseImageSequence(self, p):
        files = sorted(os.listdir(p))
        file_extensions = [os.path.splitext(f)[1].lower() for f in files]
        ftypes = []
        for t in self.__supportedImageTypes:
            if t in file_extensions:
                ftypes.append(t)
        if len(ftypes) > 1:
            raise RuntimeError('The following file types were detected in {:s}:'.format(p)+
                               '\n'.join('{:s}'.format(t) for t in ftypes)+
                               '\nPlease only include files of one image type.')
        elif len(ftypes) == 0:
            raise RuntimeError('No supported files were found in {:s}'.format(p))

        files = fnmatch.filter(files, '*{:s}'.format(ftypes[0]))

        if len(files) > 1:
            counter = [re.search('[0-9]*\{:s}'.format(ftypes[0]), f).group() for f in files]
            for i, c in enumerate(counter):
                counter[i] = int(c.replace('{:s}'.format(ftypes[0]), ''))
            files = np.array(files, dtype=object)
            sorter = np.argsort(counter)
            files = files[sorter]
            img = [sitk.ReadImage(os.path.join(p, f), sitk.sitkFloat32) for f in files]
            img = sitk.JoinSeries(img)
            print('\nImported 3D image stack ranging from {:s} to {:s}'.format(files[0], files[-1]))
        else:
            img = sitk.ReadImage(os.path.join(p, files[0]), sitk.sitkFloat32)
            print('\nImported 2D image {:s}'.format(files[0]))
        self.image = img

    def _getHessian(self):
        gx = sitk.RecursiveGaussian(self.image, sigma=self.featureScale,
                                    order=1, direction=0, normalizeAcrossScale=True)
        gx = sitk.RecursiveGaussian(self.image, sigma=self.featureScale,
                                    order=1, direction=0, normalizeAcrossScale=True)
        gy = sitk.RecursiveGaussian(self.image, sigma=self.featureScale,
                                    order=1, direction=1, normalizeAcrossScale=True)
        gz = sitk.RecursiveGaussian(self.image, sigma=self.featureScale,
                                    order=1, direction=2, normalizeAcrossScale=True)

        gx2 = sitk.RecursiveGaussian(gx, sigma=self.featureScale, order=1, direction=0, normalizeAcrossScale=True)
        gy2 = sitk.RecursiveGaussian(gy, sigma=self.featureScale, order=1, direction=1, normalizeAcrossScale=True)
        gz2 = sitk.RecursiveGaussian(gz, sigma=self.featureScale, order=1, direction=2, normalizeAcrossScale=True)
        gxgy = sitk.RecursiveGaussian(gx, sigma=self.featureScale, order=1, direction=1, normalizeAcrossScale=True)
        gxgz = sitk.RecursiveGaussian(gx, sigma=self.featureScale, order=1, direction=2, normalizeAcrossScale=True)
        gygz = sitk.RecursiveGaussian(gy, sigma=self.featureScale, order=1, direction=2, normalizeAcrossScale=True)

        del gx
        del gy
        del gz

        gx2 = sitk.GetArrayFromImage(gx2).astype(np.float32)
        gy2 = sitk.GetArrayFromImage(gy2).astype(np.float32)
        gz2 = sitk.GetArrayFromImage(gz2).astype(np.float32)
        gxgy = sitk.GetArrayFromImage(gxgy).astype(np.float32)
        gxgz = sitk.GetArrayFromImage(gxgz).astype(np.float32)
        gygz = sitk.GetArrayFromImage(gygz).astype(np.float32)

        s = np.zeros((*gx2.shape, 3, 3), dtype=np.float32)
        s[:, :, :, 0, 0] = gx2
        s[:, :, :, 1, 1] = gy2
        s[:, :, :, 2, 2] = gz2
        s[:, :, :, 0, 1] = gxgy
        s[:, :, :, 0, 2] = gxgz
        s[:, :, :, 1, 0] = gxgy
        s[:, :, :, 1, 2] = gygz
        s[:, :, :, 2, 0] = gxgz
        s[:, :, :, 2, 1] = gygz

        return s

    def _makeVTK(self):
        vtkImage = vtk.vtkImageData()
        vtkImage.SetOrigin([i for i in self.image.GetOrigin()])
        vtkImage.SetSpacing([i for i in self.image.GetSpacing()])
        vtkImage.SetDimensions([i for i in self.image.GetSize()])

        intensities = numpy_support.numpy_to_vtk(sitk.GetArrayFromImage(self.image).ravel(), deep=True, array_type=vtk.VTK_FLOAT)
        intensities.SetNumberOfComponents(1)
        intensities.SetName("Intensity")
        vtkImage.GetPointData().SetScalars(intensities)

        arr = numpy_support.numpy_to_vtk(np.transpose(self.axes, axes=[0, 1, 2, 4, 3]).ravel(), deep=True, array_type=vtk.VTK_FLOAT)
        arr.SetNumberOfComponents(9)
        arr.SetName("Orientation Distribution Function")
        vtkImage.GetPointData().SetTensors(arr)
        self.vtkimage = vtkImage

    def writeToVTK(self, name=None):
        if name is None:
            name = "output"

        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName("{}.vti".format(name))
        writer.SetInputData(self.vtkimage)
        writer.Write()

    def execute(self):
        s = self._getHessian()
        Rz = np.array([[0.0, -1.0, 0],
                       [1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0]], dtype=np.float32)
        cutoff = np.max(np.linalg.norm(s, axis=(3,4)).ravel())
        self.axes = _getAxes(s, cutoff, Rz)
        self._makeVTK()

