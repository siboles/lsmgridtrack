Usage
=====

.. toctree::
   :maxdepth: 2
   :glob:

*lsmgridtrack* has been refactored, and its usage has changed. For older versions, please refer to :ref:`Old Versions`  

Two helper modules are provided to streamline the typical registration and analysis process. These are *run3d* and *run2d* for 3-D and 2-D analyses, respectively.
They can be executed directly from the command-line with necessary arguments.

.. code-block:: bash

   python -m lsmgridtrack.run3d --help

or

.. code-block:: bash

   python -m lsmgridtrack.run2d --help

To see a list of required and optional arguments as follows:

::

  usage: run2d.py [-h] [--config CONFIG] [--reference REFERENCE] [--deformed DEFORMED] [--vtk [VTK]]
                  [--excel [EXCEL]] [--ref2vtk [REF2VTK]] [--def2vtk [DEF2VTK]]

  options:
    -h, --help            show this help message and exit
    --config CONFIG       Path to configuration file.
    --reference REFERENCE Path to reference image file or image file sequence.
    --deformed DEFORMED   Path to deformed image file or image file sequence.
    --vtk [VTK]           Base name of file to write vtk grid.
    --excel [EXCEL]       Base name excel file to write.
    --ref2vtk [REF2VTK]   Write reference image to vtk file with provided name.
    --def2vtk [DEF2VTK]   Write deformed image to vtk file with provided name.

The configuration file is in JSON format with the following definition:

.. code-block:: json

  {
    "image": {
      "spacing": [1.25, 1.25]
    },
    "grid": {
      "origin": [86, 148],
      "upper_bound": [324, 385],
      "divisions": [5, 5]
    },
    "registration": {
      "metric": "histogram",
      "sampling_fraction": 0.05,
      "method": "conjugate_gradient",
      "iterations": 30,
      "shrink_levels": [2, 1],
      "sigma_levels": [0, 0],
      "reference_landmarks": [
        [86, 148],
        [85, 384],
        [324, 385],
        [324, 146],
        [145, 208],
        [144, 326],
        [264, 326],
        [264, 206]
      ],
      "deformed_landmarks": [
        [176, 143],
        [171, 401],
        [349, 429],
        [359, 161],
        [224, 203],
        [217, 340],
        [304, 354],
        [312, 215]
      ]
    }
  }

The refactor takes a more functional/less object-oriented approach. The current modules are config, image, registration, and kinematics.
Import them as follows.

.. code:: python

   from lsmgridtrack import config, image, registration, kinematics

Now, let's set some options using the pydantic classes from the config module. These provide excellent type assignment and validation.

.. code:: python

  image_options = config.ImageOptions(spacing=[0.5, 0.5, 1.0])

  grid_options = config.GridOptions(
      origin=[69, 72, 5], upper_bound=[469, 472, 35], divisions=[20, 20, 3]
  )

  registration_options = config.RegistrationOptions(
      metric=config.RegMetricEnum.HISTOGRAM,
      sampling_fraction=0.05,
      method=config.RegMethodEnum.CONJUGATE_GRADIENT,
      iterations=30,
      shrink_levels=[2, 1],
      sigma_levels=[0.0, 0.0],
  )

For more details on these options consult the :ref:`api-label`

Let's also set some landmark coordinates for the reference and deformed images. We left these blank
when we first created *registration_options*, but we can set them now.

.. code:: python

  registration_options.reference_landmarks = [
      [69, 72, 5],
      [69, 472, 5],
      [469, 472, 5],
      [469, 72, 5],
      [69, 72, 35],
      [69, 472, 35],
      [469, 472, 35],
      [469, 72, 35],
  ]

  registration_options.deformed_landmarks = [
      [72, 81, 5],
      [71, 467, 5],
      [457, 468, 5],
      [455, 82, 5],
      [71, 80, 20],
      [72, 468, 20],
      [458, 466, 20],
      [457, 80, 20],
  ]

Options are set now, so we can import the reference and deformed images. We will be using some example images that ship with lsmgridtrack.
A helper module is included to access these data.

.. code:: python

   from lsmgridtrack.test import data

   # Get the path to reference image file
   reference_path = data.get_image("reference - 2 layers")

   # And get the path to deformed image file
   deformed_path = data.get_image("10 percent strain - 2 layers")

   # Parse these files into SimpleITK images

   reference_image = image.parse_image_file(reference_path, image_options)
   deformed_image = image.parse_image_file(deformed_path, image_options)

These images will automatically have the spacing and resampling specified in *image_options* set. Also,
they will be converted to 32-bit float format and rescaled to have intensities ranging from 0.0 to 1.0.

Next, we will setup the registration.

.. code:: python

  # Create a SimpleITK ImageRegistrationMethod
  registration_method = registration.create_registration(
      registration_options, reference_image
  )

  # Execute the registration method on the reference and deformed images
  # This returns a 3rd order basis spline transform
  transform = registration.register(registration_method, reference_image, deformed_image)

We have now successfully registered our images. Next, we will calculate kinematics
at all nodes of a grid we define with *grid_options*

.. code:: python

  # All kinematics are calculated with a single function call
  results = kinematics.get_kinematics(grid_options, image_options, transform)

  # We can write these to a VTK rectilinear grid file "vtk_output.vtr"
  kinematics.write_kinematics_to_vtk(results, "vtk_output")

  # We can also write the results to an excel file, "excel_output.xlsx"
  kinematics.write_kinematics_to_excel(results, "vtk_excel")



.. _Old Versions:

Old Versions
************

Example 1
---------

Here, we demonstrate a registration of two included test images, ref_2layers.nii and 10_2layers.nii. This approach starts with a default "tracker" object with no specified options or images. `Example 2`_ demonstrates the same analysis using a configuration file for options and providing the image paths during object creation.


Initial Setup
~~~~~~~~~~~~~

Firstly, we will import the lsmgridtrack core module and create a
tracker object with default options.

.. code:: python

    import lsmgridtrack as lsm
    
    t = lsm.tracker()

Since we didn’t provide an options or config keyword argument this
tracker object has the default options. We can view these.

.. code:: python

    print(t.options)


.. parsed-literal::

    {'Image': {'spacing': [1.0, 1.0, 1.0], 'resampling': [1.0, 1.0, 1.0]}, 'Grid': {'origin': False, 'spacing': False, 'size': False, 'crop': False}, 'Registration': {'method': 'BFGS', 'iterations': 100, 'sampling_fraction': 0.05, 'sampling_strategy': 'RANDOM', 'usemask': False, 'landmarks': False, 'shrink_levels': [1], 'sigma_levels': [0.0]}}


These are a custom class built on the normal Python dictionary, but with
immutable keys. If we try to introduce a new key, an error will be
raised. This will help prevent spelling typos from causing runtime bugs.

.. code:: python

    t.options['foo'] = True


::


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-3-a002a83e06f0> in <module>()
    ----> 1 t.options['foo'] = True
    

    ~/anaconda3/envs/testlsm/lib/python3.6/site-packages/lsmgridtrack/tracker.py in __setitem__(self, k, v)
         27     def __setitem__(self, k, v):
         28         if k not in self.__data:
    ---> 29             raise KeyError("{:s} is not an acceptable key.".format(k))
         30 
         31         self.__data[k] = v


    KeyError: 'foo is not an acceptable key.'


Indicate paths to image files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We didn’t provide a reference or deformed image to register. Let’s do
that now with the two images included in the test module.

.. code:: python

    from lsmgridtrack.test import data
    
    # Names of available images
    print(data.get_image_names())



.. parsed-literal::

    ['reference - 1 layer', '10 percent strain - 1 layer', 'reference - 2 layers', '10 percent strain - 2 layers']


.. code:: python

    # path to reference image
    reference = data.get_image('reference - 2 layers')
    
    # path to deformed image
    deformed = data.get_image('10 percent strain - 2 layers')
    
    # assign these image paths to tracker object
    t.reference_path = reference
    t.deformed_path = deformed


Now, we have images to analyze, but the default options are not correct
for these. Let’s modify these directly.

.. code:: python

    # Change the image spacing
    t.options['Image']['spacing'] = [0.5, 0.5, 1.0]
    # Change the grid origin, spacing, and size
    t.options['Grid']['origin'] = [69, 72, 5]
    t.options['Grid']['spacing'] = [20, 20, 10]
    t.options['Grid']['size'] = [20, 20, 3]
    
    # Set the registration method to BFGS
    t.options['Registration']['method'] = 'BFGS'
    
    
    #See the options are now changed
    print(t.options)


.. parsed-literal::

    {'Image': {'spacing': [0.5, 0.5, 1.0], 'resampling': [1.0, 1.0, 1.0]}, 'Grid': {'origin': [69, 72, 5], 'spacing': [20, 20, 10], 'size': [20, 20, 3], 'crop': False}, 'Registration': {'method': 'BFGS', 'iterations': 100, 'sampling_fraction': 0.05, 'sampling_strategy': 'RANDOM', 'usemask': False, 'landmarks': False, 'shrink_levels': [1], 'sigma_levels': [0.0]}}


Running the registration and analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now, let’s perform the registration and post-processing.

.. code:: python

    t.execute()


.. parsed-literal::

    ... Starting Deformable Registration
    ... ... Finding optimal BSpline transform
    ... ... Elapsed Iterations: 0
    ... ... Current Metric Value: -4.49300E-02
    ... ... Elapsed Iterations: 0
    ... ... Current Metric Value: -5.40839E-02
    ... ... Elapsed Iterations: 0
    ... ... Current Metric Value: -7.82703E-02
    ... ... Elapsed Iterations: 0
    ... ... Current Metric Value: -7.82703E-02
    ... ... Elapsed Iterations: 1
    ... ... Current Metric Value: -5.33780E-02
    ... ... Elapsed Iterations: 1
    ... ... Current Metric Value: -7.28831E-03
    ... ... Elapsed Iterations: 1
    ... ... Current Metric Value: -8.40190E-02
    ... ... Elapsed Iterations: 1
    ... ... Current Metric Value: -8.40190E-02
    ... ... Elapsed Iterations: 2
    ... ... Current Metric Value: -8.56601E-02
    ... ... Elapsed Iterations: 2
    ... ... Current Metric Value: -8.56601E-02
    ... ... Elapsed Iterations: 3
    ... ... Current Metric Value: -8.78699E-02
    ... ... Elapsed Iterations: 3
    ... ... Current Metric Value: -8.78699E-02
    ... ... Elapsed Iterations: 4
    ... ... Current Metric Value: -1.21636E-01
    ... ... Elapsed Iterations: 4
    ... ... Current Metric Value: -1.21636E-01
    ... ... Elapsed Iterations: 5
    ... ... Current Metric Value: -1.13094E-01
    ... ... Elapsed Iterations: 5
    ... ... Current Metric Value: -1.73892E-01
    ... ... Elapsed Iterations: 5
    ... ... Current Metric Value: -1.73892E-01
    ... ... Elapsed Iterations: 6
    ... ... Current Metric Value: -2.13524E-01
    ... ... Elapsed Iterations: 6
    ... ... Current Metric Value: -2.13524E-01
    ... ... Elapsed Iterations: 7
    ... ... Current Metric Value: -5.62144E-02
    ... ... Elapsed Iterations: 7
    ... ... Current Metric Value: -2.37666E-01
    ... ... Elapsed Iterations: 7
    ... ... Current Metric Value: -2.37666E-01
    ... ... Elapsed Iterations: 8
    ... ... Current Metric Value: -3.35179E-01
    ... ... Elapsed Iterations: 8
    ... ... Current Metric Value: -3.35179E-01
    ... ... Elapsed Iterations: 9
    ... ... Current Metric Value: -3.55780E-01
    ... ... Elapsed Iterations: 9
    ... ... Current Metric Value: -3.55780E-01
    ... ... Elapsed Iterations: 10
    ... ... Current Metric Value: -3.93469E-01
    ... ... Elapsed Iterations: 10
    ... ... Current Metric Value: -3.93469E-01
    ... ... Elapsed Iterations: 11
    ... ... Current Metric Value: -4.52675E-01
    ... ... Elapsed Iterations: 11
    ... ... Current Metric Value: -4.52675E-01
    ... ... Elapsed Iterations: 12
    ... ... Current Metric Value: -4.88912E-01
    ... ... Elapsed Iterations: 12
    ... ... Current Metric Value: -4.88912E-01
    ... ... Elapsed Iterations: 13
    ... ... Current Metric Value: -5.27685E-01
    ... ... Elapsed Iterations: 13
    ... ... Current Metric Value: -5.27685E-01
    ... ... Elapsed Iterations: 14
    ... ... Current Metric Value: -5.52841E-01
    ... ... Elapsed Iterations: 14
    ... ... Current Metric Value: -5.52841E-01
    ... ... Elapsed Iterations: 15
    ... ... Current Metric Value: -5.78553E-01
    ... ... Elapsed Iterations: 15
    ... ... Current Metric Value: -5.78553E-01
    ... ... Elapsed Iterations: 16
    ... ... Current Metric Value: -6.07158E-01
    ... ... Elapsed Iterations: 16
    ... ... Current Metric Value: -6.07158E-01
    ... ... Elapsed Iterations: 17
    ... ... Current Metric Value: -6.42384E-01
    ... ... Elapsed Iterations: 17
    ... ... Current Metric Value: -6.42384E-01
    ... ... Elapsed Iterations: 18
    ... ... Current Metric Value: -6.59279E-01
    ... ... Elapsed Iterations: 18
    ... ... Current Metric Value: -6.59279E-01
    ... ... Elapsed Iterations: 19
    ... ... Current Metric Value: -6.64448E-01
    ... ... Elapsed Iterations: 19
    ... ... Current Metric Value: -6.64448E-01
    ... ... Elapsed Iterations: 20
    ... ... Current Metric Value: -6.69215E-01
    ... ... Elapsed Iterations: 20
    ... ... Current Metric Value: -6.69215E-01
    ... ... Elapsed Iterations: 21
    ... ... Current Metric Value: -6.76915E-01
    ... ... Elapsed Iterations: 21
    ... ... Current Metric Value: -6.76915E-01
    ... ... Elapsed Iterations: 22
    ... ... Current Metric Value: -6.74770E-01
    ... ... Elapsed Iterations: 22
    ... ... Current Metric Value: -6.77933E-01
    ... ... Elapsed Iterations: 22
    ... ... Current Metric Value: -6.77933E-01
    ... ... Elapsed Iterations: 23
    ... ... Current Metric Value: -6.81256E-01
    ... ... Elapsed Iterations: 23
    ... ... Current Metric Value: -6.81256E-01
    ... ... Elapsed Iterations: 24
    ... ... Current Metric Value: -6.81313E-01
    ... ... Elapsed Iterations: 24
    ... ... Current Metric Value: -6.81313E-01
    ... ... Elapsed Iterations: 25
    ... ... Current Metric Value: -6.84426E-01
    ... ... Elapsed Iterations: 25
    ... ... Current Metric Value: -6.84426E-01
    ... ... Elapsed Iterations: 26
    ... ... Current Metric Value: -6.86446E-01
    ... ... Elapsed Iterations: 26
    ... ... Current Metric Value: -6.86446E-01
    ... ... Elapsed Iterations: 27
    ... ... Current Metric Value: -6.87593E-01
    ... ... Elapsed Iterations: 27
    ... ... Current Metric Value: -6.87593E-01
    ... ... Elapsed Iterations: 28
    ... ... Current Metric Value: -6.89631E-01
    ... ... Elapsed Iterations: 28
    ... ... Current Metric Value: -6.89631E-01
    ... ... Elapsed Iterations: 29
    ... ... Current Metric Value: -6.88889E-01
    ... ... Elapsed Iterations: 29
    ... ... Current Metric Value: -6.89986E-01
    ... ... Elapsed Iterations: 29
    ... ... Current Metric Value: -6.89986E-01
    ... ... Elapsed Iterations: 30
    ... ... Current Metric Value: -6.90509E-01
    ... ... Elapsed Iterations: 30
    ... ... Current Metric Value: -6.90509E-01
    ... ... Elapsed Iterations: 31
    ... ... Current Metric Value: -6.91024E-01
    ... ... Elapsed Iterations: 31
    ... ... Current Metric Value: -6.91024E-01
    ... ... Elapsed Iterations: 32
    ... ... Current Metric Value: -6.93241E-01
    ... ... Elapsed Iterations: 32
    ... ... Current Metric Value: -6.93241E-01
    ... ... Elapsed Iterations: 33
    ... ... Current Metric Value: -6.93557E-01
    ... ... Elapsed Iterations: 33
    ... ... Current Metric Value: -6.93557E-01
    ... ... Elapsed Iterations: 34
    ... ... Current Metric Value: -6.94442E-01
    ... ... Elapsed Iterations: 34
    ... ... Current Metric Value: -6.94442E-01
    ... ... Elapsed Iterations: 35
    ... ... Current Metric Value: -6.95494E-01
    ... ... Elapsed Iterations: 35
    ... ... Current Metric Value: -6.95494E-01
    ... ... Elapsed Iterations: 36
    ... ... Current Metric Value: -6.96220E-01
    ... ... Elapsed Iterations: 36
    ... ... Current Metric Value: -6.96220E-01
    ... ... Elapsed Iterations: 37
    ... ... Current Metric Value: -6.98191E-01
    ... ... Elapsed Iterations: 37
    ... ... Current Metric Value: -6.98191E-01
    ... ... Elapsed Iterations: 38
    ... ... Current Metric Value: -6.97982E-01
    ... ... Elapsed Iterations: 38
    ... ... Current Metric Value: -6.98650E-01
    ... ... Elapsed Iterations: 38
    ... ... Current Metric Value: -6.98650E-01
    ... ... Elapsed Iterations: 39
    ... ... Current Metric Value: -6.99334E-01
    ... ... Elapsed Iterations: 39
    ... ... Current Metric Value: -6.99334E-01
    ... ... Elapsed Iterations: 40
    ... ... Current Metric Value: -6.98763E-01
    ... ... Elapsed Iterations: 40
    ... ... Current Metric Value: -6.99328E-01
    ... ... Elapsed Iterations: 40
    ... ... Current Metric Value: -6.99334E-01
    ... ... Elapsed Iterations: 40
    ... ... Current Metric Value: -6.99334E-01
    ... ... Elapsed Iterations: 40
    ... ... Current Metric Value: -6.99334E-01
    ... ... Elapsed Iterations: 40
    ... ... Current Metric Value: -6.99334E-01
    ... ... Elapsed Iterations: 40
    ... ... Current Metric Value: -6.99334E-01
    ... ... Elapsed Iterations: 40
    ... ... Current Metric Value: -6.99334E-01
    ... ... Elapsed Iterations: 40
    ... ... Current Metric Value: -6.99334E-01
    ... ... Elapsed Iterations: 40
    ... ... Current Metric Value: -6.99334E-01
    ... ... Elapsed Iterations: 40
    ... ... Current Metric Value: -6.99334E-01
    ... ... Elapsed Iterations: 40
    ... ... Current Metric Value: -6.99334E-01
    ... ... Elapsed Iterations: 40
    ... ... Current Metric Value: -6.99334E-01
    ... ... Elapsed Iterations: 40
    ... ... Current Metric Value: -6.99334E-01
    ... ... Elapsed Iterations: 40
    ... ... Current Metric Value: -6.99334E-01
    ... ... Optimal BSpline transform determined 
    ... ... ... Elapsed Iterations: 41
    ... ... ... Final Metric Value: -6.99334E-01
    ... Registration Complete
    Analysis Complete!


We did not provide any initialization to the registration algorithm in
the above execution. If we provide the indices of the 8 grid corners
(ordered counter-clockwise) in the deformed image, the registration can
be better initialized. Let’s see if this initialization changes the
convergence behaviour. We determined these voxel indices using ImageJ.

.. code:: python

    t.options['Registration']['landmarks'] = [[72, 81, 5],
                                              [71, 467, 5],
                                              [457, 468, 5],
                                              [455, 82, 5],
                                              [71, 80, 20],
                                              [72, 468, 20],
                                              [458, 466, 20],
                                              [457, 80, 20]]

And re-executing the registration.

.. code:: python

    t.execute()


.. parsed-literal::

    ... Starting Deformable Registration
    ... ... Finding optimal BSpline transform
    ... ... Elapsed Iterations: 0
    ... ... Current Metric Value: -2.32286E-01
    ... ... Elapsed Iterations: 0
    ... ... Current Metric Value: -2.62576E-01
    ... ... Elapsed Iterations: 0
    ... ... Current Metric Value: -3.85003E-01
    ... ... Elapsed Iterations: 0
    ... ... Current Metric Value: -3.85003E-01
    ... ... Elapsed Iterations: 1
    ... ... Current Metric Value: -1.67176E-01
    ... ... Elapsed Iterations: 1
    ... ... Current Metric Value: -3.80853E-01
    ... ... Elapsed Iterations: 1
    ... ... Current Metric Value: -5.01466E-01
    ... ... Elapsed Iterations: 1
    ... ... Current Metric Value: -5.01466E-01
    ... ... Elapsed Iterations: 2
    ... ... Current Metric Value: -5.62287E-01
    ... ... Elapsed Iterations: 2
    ... ... Current Metric Value: -5.62287E-01
    ... ... Elapsed Iterations: 3
    ... ... Current Metric Value: -5.52476E-01
    ... ... Elapsed Iterations: 3
    ... ... Current Metric Value: -6.67195E-01
    ... ... Elapsed Iterations: 3
    ... ... Current Metric Value: -6.67195E-01
    ... ... Elapsed Iterations: 4
    ... ... Current Metric Value: -6.81427E-01
    ... ... Elapsed Iterations: 4
    ... ... Current Metric Value: -6.81427E-01
    ... ... Elapsed Iterations: 5
    ... ... Current Metric Value: -6.95327E-01
    ... ... Elapsed Iterations: 5
    ... ... Current Metric Value: -6.95327E-01
    ... ... Elapsed Iterations: 6
    ... ... Current Metric Value: -7.01208E-01
    ... ... Elapsed Iterations: 6
    ... ... Current Metric Value: -7.01208E-01
    ... ... Elapsed Iterations: 7
    ... ... Current Metric Value: -7.04607E-01
    ... ... Elapsed Iterations: 7
    ... ... Current Metric Value: -7.04607E-01
    ... ... Elapsed Iterations: 8
    ... ... Current Metric Value: -7.05672E-01
    ... ... Elapsed Iterations: 8
    ... ... Current Metric Value: -7.05672E-01
    ... ... Elapsed Iterations: 9
    ... ... Current Metric Value: -7.07048E-01
    ... ... Elapsed Iterations: 9
    ... ... Current Metric Value: -7.07048E-01
    ... ... Elapsed Iterations: 10
    ... ... Current Metric Value: -7.05148E-01
    ... ... Elapsed Iterations: 10
    ... ... Current Metric Value: -7.07038E-01
    ... ... Elapsed Iterations: 10
    ... ... Current Metric Value: -7.07043E-01
    ... ... Elapsed Iterations: 10
    ... ... Current Metric Value: -7.07045E-01
    ... ... Elapsed Iterations: 10
    ... ... Current Metric Value: -7.07048E-01
    ... ... Elapsed Iterations: 10
    ... ... Current Metric Value: -7.07048E-01
    ... ... Elapsed Iterations: 10
    ... ... Current Metric Value: -7.07048E-01
    ... ... Elapsed Iterations: 10
    ... ... Current Metric Value: -7.07048E-01
    ... ... Elapsed Iterations: 10
    ... ... Current Metric Value: -7.07048E-01
    ... ... Elapsed Iterations: 10
    ... ... Current Metric Value: -7.07048E-01
    ... ... Elapsed Iterations: 10
    ... ... Current Metric Value: -7.07048E-01
    ... ... Elapsed Iterations: 10
    ... ... Current Metric Value: -7.07048E-01
    ... ... Elapsed Iterations: 10
    ... ... Current Metric Value: -7.07048E-01
    ... ... Elapsed Iterations: 10
    ... ... Current Metric Value: -7.07048E-01
    ... ... Elapsed Iterations: 10
    ... ... Current Metric Value: -7.07048E-01
    ... ... Elapsed Iterations: 10
    ... ... Current Metric Value: -7.07048E-01
    ... ... Elapsed Iterations: 10
    ... ... Current Metric Value: -7.07048E-01
    ... ... Elapsed Iterations: 10
    ... ... Current Metric Value: -7.07048E-01
    ... ... Elapsed Iterations: 10
    ... ... Current Metric Value: -7.07048E-01
    ... ... Elapsed Iterations: 10
    ... ... Current Metric Value: -7.07048E-01
    ... ... Elapsed Iterations: 10
    ... ... Current Metric Value: -7.07048E-01
    ... ... Optimal BSpline transform determined 
    ... ... ... Elapsed Iterations: 11
    ... ... ... Final Metric Value: -7.07048E-01
    ... Registration Complete
    Analysis Complete!


We converged in less iterations of BFGS than the uninitialized
registration. The final metric value (negative cross-correlation) was
quite close. This suggests the objective function may be near convex
since the determined minima are nearly equal; although, this cannot be
proven. Qualitative inspection of the two results suggests these
particular images can be registered well without initialization. The
reader is encouraged to do this inspection by outputting results from
each execution.

Saving the results
~~~~~~~~~~~~~~~~~~

We can write the results in different formats such as a VTK image,

.. code:: python

    t.writeResultsAsVTK('example1')


.. parsed-literal::

    ... Saving Results to example1.vti


an Excel workbook,

.. code:: python

    t.writeResultsAsExcel('example1')


.. parsed-literal::

    ... Saving Results to example1.xlsx


and a NumPy binary.

.. code:: python

    t.writeResultsAsNumpy('example1')


.. parsed-literal::

    ... Saving file as numpy archive example1.npz


Saving the 3D images for later visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To view the original images in the open source 3D visualization
software, ParaView, we can save the images as a VTK image.

.. code:: python

    # Write the reference image to VTK image
    t.writeImageAsVTK(t.ref_img, 'reference')
    
    # WRite the deformed image to VTK image
    t.writeImageAsVTK(t.def_img, 'deformed')


.. parsed-literal::

    ... Saving Image to reference.vti
    ... Saving Image to deformed.vti


Example 2
---------

An alternative and often more preferable method for setting the registration options is by creating a configuration file. This file adopts the popular YAML format, and the user can set any number of the options within it. The modifications made to the default in the previous example are instead indicated in the contents of example2.yaml:

.. code:: yaml

    Image:
      spacing: [0.5, 0.5, 1.0]
    Grid:
      origin: [69, 72, 5]
      spacing: [20, 20, 10]
      size: [20, 20, 3]
    Registration:
      method: BFGS
      landmarks: [[72, 81, 5],
                  [71, 467, 5],
                  [457, 468, 5],
                  [455, 82, 5],
                  [71, 80, 20],
                  [72, 468, 20],
                  [458, 466, 20],
                  [457, 80, 20]]


Initial Setup
~~~~~~~~~~~~~

.. code:: python

    from lsmgridtrack.test import data
    import lsmgridtrack as lsm
    
    # Get paths to reference and deformed images
    reference_path = data.get_image('reference - 2 layers')
    deformed_path = data.get_image('10 percent strain - 2 layers')
    
    # Instantiate a tracker object with image paths and configuration file specified as arguments
    t = lsm.tracker(reference_path = reference_path,
                    deformed_path = deformed_path,
                    config = "example2.yaml")
    
    # Print the options to show they are changed by the config file
    print(t.options)



.. parsed-literal::

    {'Image': {'spacing': [0.5, 0.5, 1.0], 'resampling': [1.0, 1.0, 1.0]}, 'Grid': {'origin': [69, 72, 5], 'spacing': [20, 20, 10], 'size': [20, 20, 3], 'crop': False}, 'Registration': {'method': 'BFGS', 'iterations': 100, 'sampling_fraction': 0.05, 'sampling_strategy': 'RANDOM', 'usemask': False, 'landmarks': [[72, 81, 5], [71, 467, 5], [457, 468, 5], [455, 82, 5], [71, 80, 20], [72, 468, 20], [458, 466, 20], [457, 80, 20]], 'shrink_levels': [1], 'sigma_levels': [0.0]}}


Execution
~~~~~~~~~

.. code:: python

    t.execute()


.. parsed-literal::

    ... Starting Deformable Registration
    ... ... Finding optimal BSpline transform
    ... ... Elapsed Iterations: 0
    ... ... Current Metric Value: -2.34740E-01
    ... ... Elapsed Iterations: 0
    ... ... Current Metric Value: -2.65271E-01
    ... ... Elapsed Iterations: 0
    ... ... Current Metric Value: -3.87429E-01
    ... ... Elapsed Iterations: 0
    ... ... Current Metric Value: -3.87429E-01
    ... ... Elapsed Iterations: 1
    ... ... Current Metric Value: -1.61627E-01
    ... ... Elapsed Iterations: 1
    ... ... Current Metric Value: -4.17114E-01
    ... ... Elapsed Iterations: 1
    ... ... Current Metric Value: -5.04918E-01
    ... ... Elapsed Iterations: 1
    ... ... Current Metric Value: -5.04918E-01
    ... ... Elapsed Iterations: 2
    ... ... Current Metric Value: -5.66540E-01
    ... ... Elapsed Iterations: 2
    ... ... Current Metric Value: -5.66540E-01
    ... ... Elapsed Iterations: 3
    ... ... Current Metric Value: -5.77310E-01
    ... ... Elapsed Iterations: 3
    ... ... Current Metric Value: -5.77310E-01
    ... ... Elapsed Iterations: 4
    ... ... Current Metric Value: -6.71897E-01
    ... ... Elapsed Iterations: 4
    ... ... Current Metric Value: -6.71897E-01
    ... ... Elapsed Iterations: 5
    ... ... Current Metric Value: -6.87199E-01
    ... ... Elapsed Iterations: 5
    ... ... Current Metric Value: -6.87199E-01
    ... ... Elapsed Iterations: 6
    ... ... Current Metric Value: -6.96427E-01
    ... ... Elapsed Iterations: 6
    ... ... Current Metric Value: -6.96427E-01
    ... ... Elapsed Iterations: 7
    ... ... Current Metric Value: -7.00449E-01
    ... ... Elapsed Iterations: 7
    ... ... Current Metric Value: -7.00449E-01
    ... ... Elapsed Iterations: 8
    ... ... Current Metric Value: -7.03506E-01
    ... ... Elapsed Iterations: 8
    ... ... Current Metric Value: -7.03506E-01
    ... ... Elapsed Iterations: 9
    ... ... Current Metric Value: -7.05269E-01
    ... ... Elapsed Iterations: 9
    ... ... Current Metric Value: -7.05269E-01
    ... ... Elapsed Iterations: 10
    ... ... Current Metric Value: -7.07828E-01
    ... ... Elapsed Iterations: 10
    ... ... Current Metric Value: -7.07828E-01
    ... ... Elapsed Iterations: 11
    ... ... Current Metric Value: -7.07464E-01
    ... ... Elapsed Iterations: 11
    ... ... Current Metric Value: -7.07819E-01
    ... ... Elapsed Iterations: 11
    ... ... Current Metric Value: -7.07836E-01
    ... ... Elapsed Iterations: 11
    ... ... Current Metric Value: -7.07831E-01
    ... ... Elapsed Iterations: 11
    ... ... Current Metric Value: -7.07836E-01
    ... ... Elapsed Iterations: 11
    ... ... Current Metric Value: -7.07836E-01
    ... ... Elapsed Iterations: 11
    ... ... Current Metric Value: -7.07836E-01
    ... ... Elapsed Iterations: 12
    ... ... Current Metric Value: -7.07222E-01
    ... ... Elapsed Iterations: 12
    ... ... Current Metric Value: -7.07831E-01
    ... ... Elapsed Iterations: 12
    ... ... Current Metric Value: -7.07834E-01
    ... ... Elapsed Iterations: 12
    ... ... Current Metric Value: -7.07835E-01
    ... ... Elapsed Iterations: 12
    ... ... Current Metric Value: -7.07836E-01
    ... ... Elapsed Iterations: 12
    ... ... Current Metric Value: -7.07836E-01
    ... ... Elapsed Iterations: 12
    ... ... Current Metric Value: -7.07836E-01
    ... ... Elapsed Iterations: 12
    ... ... Current Metric Value: -7.07836E-01
    ... ... Elapsed Iterations: 12
    ... ... Current Metric Value: -7.07836E-01
    ... ... Elapsed Iterations: 12
    ... ... Current Metric Value: -7.07836E-01
    ... ... Elapsed Iterations: 12
    ... ... Current Metric Value: -7.07836E-01
    ... ... Elapsed Iterations: 12
    ... ... Current Metric Value: -7.07836E-01
    ... ... Elapsed Iterations: 12
    ... ... Current Metric Value: -7.07836E-01
    ... ... Elapsed Iterations: 12
    ... ... Current Metric Value: -7.07836E-01
    ... ... Elapsed Iterations: 12
    ... ... Current Metric Value: -7.07836E-01
    ... ... Elapsed Iterations: 12
    ... ... Current Metric Value: -7.07836E-01
    ... ... Elapsed Iterations: 12
    ... ... Current Metric Value: -7.07836E-01
    ... ... Elapsed Iterations: 12
    ... ... Current Metric Value: -7.07836E-01
    ... ... Elapsed Iterations: 12
    ... ... Current Metric Value: -7.07836E-01
    ... ... Elapsed Iterations: 12
    ... ... Current Metric Value: -7.07836E-01
    ... ... Elapsed Iterations: 12
    ... ... Current Metric Value: -7.07836E-01
    ... ... Optimal BSpline transform determined 
    ... ... ... Elapsed Iterations: 13
    ... ... ... Final Metric Value: -7.07836E-01
    ... Registration Complete
    Analysis Complete!


Outputting Results
~~~~~~~~~~~~~~~~~~

.. code:: python

    t.writeResultsAsVTK('example2')
    t.writeResultsAsExcel('example2')
    t.writeResultsAsNumpy('example2')


.. parsed-literal::

    ... Saving Results to example2.vti
    ... Saving Results to example2.xlsx
    ... Saving file as numpy archive example2.npz


Post-analysis
~~~~~~~~~~~~~

.. code:: python

    data = lsm.utils.readVTK("example2.vti")

Here *data* is an OrderedDict that works with the post-processing
functions included in lsm.utils. All the reader functions return this
type, so we could also have read the excel file we output above:

.. code:: python

    excel_data = lsm.utils.readExcel("example2.xlsx")

Now if we have deformation data from images with grids of the same size,
we can do one-to-one comparisons. Since in this example we only have
results from one analysis, we will simulate a second data set by
randomly perturbing a copy of our *data*. Note that this deepcopy is
important, so nothing is copied by reference.

.. code:: python

    import numpy as np
    from copy import deepcopy
    data_perturbed = deepcopy(data)

Let’s add some random pertubation to the values stored in our copy of
the data.

.. code:: python

    for k, v in data_perturbed.items():
        if k == "Coordinates":
            continue
        mu = np.mean(v.ravel())
        sigma = np.std(v.ravel())
        data_perturbed[k] += np.random.normal(loc=mu, scale=sigma, size=v.shape)

A nice aggregate measure is the root-mean-square difference. We can
easily calculate this for all data variables.

.. code:: python

    rmsd = lsm.utils.calculateRMSDifference(x=data, y=data_perturbed, variables=data.keys())
    print(rmsd)


.. parsed-literal::

    OrderedDict([('Coordinates', 0.0), ('Displacement', 4.3950385988639971), ('Strain', 0.099083840270760157), ('1st Principal Strain', 0.013016456504172057), ('2nd Principal Strain', 0.0070212611135075983), ('3rd Principal Strain', 0.13202721704346612), ('Volumetric Strain', 0.18459445802160926), ('Maximum Shear Strain', 0.1692308333647991)])


However, all spatial information is lost here. Instead we can calculate
the difference at every grid vertex, and save it to a variable.

.. code:: python

    strain_difference = lsm.utils.calculateDifference(x=data, y=data_perturbed, variable="Strain")

.. code:: python

    data["Strain Difference"] = strain_difference

Of course this could be done in one line.

.. code:: python

    data["1st Principal Strain Difference"] = lsm.utils.calculateDifference(x=data, y=data_perturbed, variable="1st Principal Strain")

Now, it would be nice to visualize this again in ParaView (or other
software that can handle VTK image format). We supply writer functions
to VTK, NumPy, and Excel formats as well. Let’s write to a VTK image.

.. code:: python

    lsm.utils.writeAsVTK(data=data, name="example2_processed")


.. parsed-literal::

    ... Wrote grid data to example2_processed.vti

