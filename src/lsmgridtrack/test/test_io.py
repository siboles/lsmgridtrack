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