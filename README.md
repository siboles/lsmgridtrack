[![codecov](https://codecov.io/github/siboles/lsmgridtrack/branch/master/graph/badge.svg?token=1QM2LSL8X5)](https://codecov.io/github/siboles/lsmgridtrack)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1205560.svg)](https://doi.org/10.5281/zenodo.1205560)

Please find the official documentation [here](http://lsmgridtrack.readthedocs.io/en/master/).

OVERVIEW
==========

lsmgridtrack is a Python module providing a framework for deformable image registration of 3D images from multiphoton laser scanning microscopy. It is aimed at a technique involving the photobleaching of a 3D grid onto the image and then observing this grid region in unloaded and loaded states.

Unlike confocal microscopes, multiphoton laser scanning microscopes can target a precise excitation volume. This allows for local photobleaching not only in the x-y plane but also the z, and thus a 3D grid can be drawn on fluorostained tissue. An example of such a grid photobleached onto a microscopic region (200x200x100 microns) of articular cartilage is shown below.


![image](doc/reference.png)

By using custom mechanical testing systems that allow for sample imaging while under known applied mechanics, this grid can be observed in the tissue under different deformed states. Registering the undeformed and deformed images then allows for the determination of the grid deformation. Below, the calculated grid deformation between the undeformed tissue state shown above to a 30% applied nominal compressive strain is shown.

![image](doc/deforming.gif)

From the grid vertex displacements, lsmgridtrack automatically calculates the deformation gradient at each grid cell centroid. From the deformation gradient, any kinematic variable can be determined. The Green-Lagrange strains, principal strains, maximum shear strains, and volumetric strains are calculated automatically. These can then be saved in different formats, such as a VTK image. 

TESTING
=======

Resonanced-scanned Images
-------------------------
To test the application of lsmgridtrack registration on lower quality images acquired
using a resonance scanning protocol first download the test image from

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6423385.svg)](https://doi.org/10.5281/zenodo.6423385)

In a directory where writing multiple output files is acceptable execute the command:

```
python -m lsmgridtrack.test.test_random_resonance PATH_TO_IMAGE_FILE N
```

substituting the path and filename of the downloaded test image for PATH_TO_IMAGE_FILE and
how many random deformations of the image to test for N.
