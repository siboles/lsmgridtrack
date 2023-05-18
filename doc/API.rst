.. _api-label:

API
===

The autodocumentation for the lsmgridtrack package is provided below.

.. toctree::
   :maxdepth: 1
   :glob:


Module: config
~~~~~~~~~~~~~~

The config module contains functions and classes for defining analysis settings.

.. automodule:: lsmgridtrack.config
   :members:

Module: image
~~~~~~~~~~~~~

The image module contains functions and classes for reading and writing images,
preprocessing the images for use in lsmgridtrack, and quantifying features like
the the sample surface.

.. automodule:: lsmgridtrack.image
   :members:

Module: registration
~~~~~~~~~~~~~~~~~~~~

The registration module contains functions for defining and executing deformable
image registration and saving the resulting transform.

.. automodule:: lsmgridtrack.registration
   :members:

Module: kinematics
~~~~~~~~~~~~~~~~~~

The kinematics module provides functions for calculating the 3-D displacement field
and subsequent kinematic variables on a specified rectilinear grid. These include
the deformation gradient, Green-Lagrange strain, principal Green-Lagrange strains,
and volumetric strains.

.. automodule:: lsmgridtrack.kinematics
   :members:

Module: kinematics2d
~~~~~~~~~~~~~~~~~~~~

The kinematics2d module provides functions for calculating the 2-D displacement field
and subsequent kinematic variables on a specified rectilinear grid. These include
the deformation gradient, Green-Lagrange strain, principal Green-Lagrange strains,
and volumetric strains.

.. automodule:: lsmgridtrack.kinematics2d
   :members:

Module: postprocessing
~~~~~~~~~~~~~~~~~~~~~~

The postprocessing module provides functions to rotate data to new coordinate systems 
aligned with a provided sample surface geometry.

.. automodule:: lsmgridtrack.postprocessing
   :members:
