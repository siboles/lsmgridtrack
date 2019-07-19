import unittest
import tempfile
import shutil
import os

import lsmgridtrack as lsm
import SimpleITK as sitk

from lsmgridtrack.test import data

class IOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tiffRootDir = tempfile.mkdtemp()
        cls._outputDir = tempfile.mkdtemp()
        img = sitk.ReadImage(data.get_image('reference - 2 layers'), sitk.sitkFloat32)
        for i in range(img.GetSize()[2]):
            im = sitk.Extract(img, list(img.GetSize()[0:2]) + [0], [0, 0, i])
            sitk.WriteImage(im, os.path.join(cls._tiffRootDir, 'slice.{:03d}.tif'.format(i)))
        cls._niiRootDir = data.get_image('reference - 2 layers')
        cls._tracker = lsm.tracker(reference_path=cls._niiRootDir, deformed_path=cls._niiRootDir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tiffRootDir)
        shutil.rmtree(cls._outputDir)

    def test_parse_config(self):
        self._tracker.config = 'bfgs.yaml'
        self._tracker.parseConfig()

    def test_read_nii(self):
        self._tracker.ref_img = self._tracker.parseImg(self._niiRootDir,
                                                       self._tracker.options['Grid']['crop'],
                                                       self._tracker.options['Image']['spacing'])
        self.assertIsInstance(self._tracker.ref_img, sitk.Image)

    def test_read_tiffs(self):
        self._tracker.ref_img = self._tracker.parseImg(self._tiffRootDir,
                                                       self._tracker.options['Grid']['crop'],
                                                       self._tracker.options['Image']['spacing'])
        self.assertIsInstance(self._tracker.ref_img, sitk.Image)

    def test_write_image_to_vtk(self):
        self._tracker.writeImageAsVTK(self._tracker.ref_img,
                                      name=os.path.join(self._outputDir, 'reference'))

class ProcessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._outputDir = tempfile.mkdtemp()
        cls._niiRootDir = data.get_image('reference - 2 layers')

    def setUp(self):
        self.tracker = lsm.tracker(reference_path=self._niiRootDir, deformed_path=self._niiRootDir)
        self.tracker.options["Image"]["spacing"] = [0.5, 0.5, 1.0]
        self.tracker.options["Grid"]["origin"] = [69, 72, 5]
        self.tracker.options["Grid"]["spacing"] = [20, 20, 10]
        self.tracker.options["Grid"]["size"] = [20, 20, 3]
        self.tracker.options["Registration"]["iterations"] = 1

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._outputDir)

    def test_registration_gradient_descent_histogram(self):
        self.tracker.options["Registration"]["method"] = "GradientDescent"
        self.tracker.options["Registration"]["metric"] = "correlation"
        self.tracker.execute()

    def test_registration_gradient_descent_correlation(self):
        self.tracker.options["Registration"]["method"] = "GradientDescent"
        self.tracker.options["Registration"]["metric"] = "correlation"
        self.tracker.execute()

    def test_registration_conjugate_gradient_histogram(self):
        self.tracker.options["Registration"]["method"] = "ConjugateGradient"
        self.tracker.options["Registration"]["metric"] = "correlation"
        self.tracker.execute()

    def test_registration_conjugate_gradient_correlation(self):
        self.tracker.options["Registration"]["method"] = "ConjugateGradient"
        self.tracker.options["Registration"]["metric"] = "correlation"
        self.tracker.execute()

    def test_registration_bfgs_histogram(self):
        self.tracker.options["Registration"]["method"] = "BFGS"
        self.tracker.options["Registration"]["metric"] = "histogram"
        self.tracker.execute()

    def test_registration_bfgs_correlation(self):
        self.tracker.options["Registration"]["method"] = "GradientDescent"
        self.tracker.options["Registration"]["metric"] = "correlation"
        self.tracker.execute()
