from setuptools import setup, find_packages
from setuptools.dist import Distribution

class BinaryDistribution(Distribution):
    def is_pure(self):
        return False
setup(
    name = 'lsmgridtrack',
    version = '0.2',
    description = 'A Python module providing a framework for deformable image registration of 3D images from multiphoton laser scanning microscopy. It is aimed at a technique involving the photobleaching of a 3D grid onto the image and then observing this grid region in unloaded and loaded states.',
    packages = find_packages('.'),
    url = "https://github.com/siboles/lsmgridtrack",
    author = 'Scott Sibole',
    author_email = 'scott.sibole@gmail.com',
    license = 'BSD 3-Clause',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: BSD 3-Clause',

        'Programming Language :: Python :: 3.6',
    ],
    python_requires='>=3.5',
    package_data={
        'lsmgridtrack': ['test/data/*.nii',
                         'test/data/testRandom.yaml',]
    },
    distclass=BinaryDistribution,
)
