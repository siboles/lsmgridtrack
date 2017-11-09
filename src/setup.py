from setuptools import setup, find_packages
from codecs import open
import os

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(os.path.join(here, os.path.pardir), "README.md"), encoding="utf-8") as f:
    long_description = f.read()

class BinaryDistribution(Distribution):
    def is_pure(self):
        return False
setup(
    name = 'lsmgridtrack',
    version = '0.0',
    description = 'Provides tools for easy generation of hexahedral meshes of primitive shapes: boxes, elliptical cylinders, and ellipsoids, for use in finite element models.',
    packages = find_packages('.')
    long_description = long_description,
    url = "https://github.com/siboles/hexapy",
    author = 'Scott Sibole',
    author_email = 'scott.sibole@gmail.com',
    license = 'BSD 3-Clause',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: BSD 3-Clause',

        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ]
    python_requires='>=3.5',
    package_data={
        'lsmgridtrack': ['test/data/*.nii', 'test/data/testRandom.yaml'] 
    }
    install_requires = ['future', 'numpy', 'SimpleITK', 'vtk', 'openpyxl'],
)
