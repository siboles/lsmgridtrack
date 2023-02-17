import unittest
import tempfile
import shutil

from ..tracker import tracker

from . import data


class ProcessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._outputDir = tempfile.mkdtemp()
        cls._niiRootDir = data.get_image("reference - 2 layers")

    def setUp(self):
        self.tracker = tracker(
            reference_path=self._niiRootDir, deformed_path=self._niiRootDir
        )
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
