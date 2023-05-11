from setuptools import find_packages, setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    def is_pure(self):
        return False


setup(
    name="lsmgridtrack",
    version="0.4.2",
    description="A Python module providing a framework for deformable image registration of 3D images from multiphoton laser scanning microscopy. It is aimed at a technique involving the photobleaching of a 3D grid onto the image and then observing this grid region in unloaded and loaded states.",
    packages=find_packages("."),
    url="https://github.com/siboles/lsmgridtrack",
    author="Scott Sibole",
    author_email="scott.sibole@gmail.com",
    license="BSD 3-Clause",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: BSD 3-Clause",
        "Programming Language :: Python",
    ],
    package_data={
        "lsmgridtrack": [
            "test/data/*.nii",
            "test/data/testRandom.yaml",
            "test/data/resonance.yaml",
            "test/data/ref_seq/*.tif",
            "test/data/def_seq/*.tif",
        ]
    },
    distclass=BinaryDistribution,
)
