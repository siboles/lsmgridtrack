Examples
========

.. toctree::
   :maxdepth: 1
   :glob:

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

    {'Image': {'spacing': [1.0, 1.0, 1.0], 'resampling': [1.0, 1.0, 1.0]}, 'Grid': {'origin': False, 'spacing': False, 'size': False, 'crop': False}, 'Registration': {'method': 'BFGS', 'iterations': 100, 'sampling_fraction': 0.05, 'usemask': False, 'landmarks': False, 'shrink_levels': [1], 'sigma_levels': [0.0]}}


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

    {'Image': {'spacing': [0.5, 0.5, 1.0], 'resampling': [1.0, 1.0, 1.0]}, 'Grid': {'origin': [69, 72, 5], 'spacing': [20, 20, 10], 'size': [20, 20, 3], 'crop': False}, 'Registration': {'method': 'BFGS', 'iterations': 100, 'sampling_fraction': 0.05, 'usemask': False, 'landmarks': False, 'shrink_levels': [1], 'sigma_levels': [0.0]}}


Running the registration and analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now, let’s perform the registration and post-processing.

.. code:: python

    t.execute()


.. parsed-literal::

    ... Starting Deformable Registration
    ... ... Finding optimal BSpline transform
    ... ... Elapsed Iterations: 0
    ... ... Current Metric Value: -4.47128E-02
    ... ... Elapsed Iterations: 0
    ... ... Current Metric Value: -5.38580E-02
    ... ... Elapsed Iterations: 0
    ... ... Current Metric Value: -7.79082E-02
    ... ... Elapsed Iterations: 0
    ... ... Current Metric Value: -7.79082E-02
    ... ... Elapsed Iterations: 1
    ... ... Current Metric Value: -5.67148E-02
    ... ... Elapsed Iterations: 1
    ... ... Current Metric Value: -7.74855E-03
    ... ... Elapsed Iterations: 1
    ... ... Current Metric Value: -8.36789E-02
    ... ... Elapsed Iterations: 1
    ... ... Current Metric Value: -8.36789E-02
    ... ... Elapsed Iterations: 2
    ... ... Current Metric Value: -8.65976E-02
    ... ... Elapsed Iterations: 2
    ... ... Current Metric Value: -8.65976E-02
    ... ... Elapsed Iterations: 3
    ... ... Current Metric Value: -7.37156E-02
    ... ... Elapsed Iterations: 3
    ... ... Current Metric Value: -9.95214E-02
    ... ... Elapsed Iterations: 3
    ... ... Current Metric Value: -9.95214E-02
    ... ... Elapsed Iterations: 4
    ... ... Current Metric Value: -5.62963E-02
    ... ... Elapsed Iterations: 4
    ... ... Current Metric Value: -1.10608E-01
    ... ... Elapsed Iterations: 4
    ... ... Current Metric Value: -1.10608E-01
    ... ... Elapsed Iterations: 5
    ... ... Current Metric Value: -1.20648E-01
    ... ... Elapsed Iterations: 5
    ... ... Current Metric Value: -1.20648E-01
    ... ... Elapsed Iterations: 6
    ... ... Current Metric Value: -1.56926E-01
    ... ... Elapsed Iterations: 6
    ... ... Current Metric Value: -2.78771E-01
    ... ... Elapsed Iterations: 6
    ... ... Current Metric Value: -2.78771E-01
    ... ... Elapsed Iterations: 7
    ... ... Current Metric Value: -2.21146E-01
    ... ... Elapsed Iterations: 7
    ... ... Current Metric Value: -3.04320E-01
    ... ... Elapsed Iterations: 7
    ... ... Current Metric Value: -3.04320E-01
    ... ... Elapsed Iterations: 8
    ... ... Current Metric Value: -3.44805E-01
    ... ... Elapsed Iterations: 8
    ... ... Current Metric Value: -3.44805E-01
    ... ... Elapsed Iterations: 9
    ... ... Current Metric Value: -3.98011E-01
    ... ... Elapsed Iterations: 9
    ... ... Current Metric Value: -3.98011E-01
    ... ... Elapsed Iterations: 10
    ... ... Current Metric Value: -4.30215E-01
    ... ... Elapsed Iterations: 10
    ... ... Current Metric Value: -4.30215E-01
    ... ... Elapsed Iterations: 11
    ... ... Current Metric Value: -4.55702E-01
    ... ... Elapsed Iterations: 11
    ... ... Current Metric Value: -4.55702E-01
    ... ... Elapsed Iterations: 12
    ... ... Current Metric Value: -4.87163E-01
    ... ... Elapsed Iterations: 12
    ... ... Current Metric Value: -4.87163E-01
    ... ... Elapsed Iterations: 13
    ... ... Current Metric Value: -5.25781E-01
    ... ... Elapsed Iterations: 13
    ... ... Current Metric Value: -5.25781E-01
    ... ... Elapsed Iterations: 14
    ... ... Current Metric Value: -5.78780E-01
    ... ... Elapsed Iterations: 14
    ... ... Current Metric Value: -5.78780E-01
    ... ... Elapsed Iterations: 15
    ... ... Current Metric Value: -6.27303E-01
    ... ... Elapsed Iterations: 15
    ... ... Current Metric Value: -6.27303E-01
    ... ... Elapsed Iterations: 16
    ... ... Current Metric Value: -6.28957E-01
    ... ... Elapsed Iterations: 16
    ... ... Current Metric Value: -6.42487E-01
    ... ... Elapsed Iterations: 16
    ... ... Current Metric Value: -6.42487E-01
    ... ... Elapsed Iterations: 17
    ... ... Current Metric Value: -6.53138E-01
    ... ... Elapsed Iterations: 17
    ... ... Current Metric Value: -6.53138E-01
    ... ... Elapsed Iterations: 18
    ... ... Current Metric Value: -6.63182E-01
    ... ... Elapsed Iterations: 18
    ... ... Current Metric Value: -6.63182E-01
    ... ... Elapsed Iterations: 19
    ... ... Current Metric Value: -6.68434E-01
    ... ... Elapsed Iterations: 19
    ... ... Current Metric Value: -6.68434E-01
    ... ... Elapsed Iterations: 20
    ... ... Current Metric Value: -6.72653E-01
    ... ... Elapsed Iterations: 20
    ... ... Current Metric Value: -6.72653E-01
    ... ... Elapsed Iterations: 21
    ... ... Current Metric Value: -6.76842E-01
    ... ... Elapsed Iterations: 21
    ... ... Current Metric Value: -6.76842E-01
    ... ... Elapsed Iterations: 22
    ... ... Current Metric Value: -6.81208E-01
    ... ... Elapsed Iterations: 22
    ... ... Current Metric Value: -6.81208E-01
    ... ... Elapsed Iterations: 23
    ... ... Current Metric Value: -6.80984E-01
    ... ... Elapsed Iterations: 23
    ... ... Current Metric Value: -6.81642E-01
    ... ... Elapsed Iterations: 23
    ... ... Current Metric Value: -6.81642E-01
    ... ... Elapsed Iterations: 24
    ... ... Current Metric Value: -6.83337E-01
    ... ... Elapsed Iterations: 24
    ... ... Current Metric Value: -6.83337E-01
    ... ... Elapsed Iterations: 25
    ... ... Current Metric Value: -6.85477E-01
    ... ... Elapsed Iterations: 25
    ... ... Current Metric Value: -6.85477E-01
    ... ... Elapsed Iterations: 26
    ... ... Current Metric Value: -6.88208E-01
    ... ... Elapsed Iterations: 26
    ... ... Current Metric Value: -6.88208E-01
    ... ... Elapsed Iterations: 27
    ... ... Current Metric Value: -6.89079E-01
    ... ... Elapsed Iterations: 27
    ... ... Current Metric Value: -6.89079E-01
    ... ... Elapsed Iterations: 28
    ... ... Current Metric Value: -6.90247E-01
    ... ... Elapsed Iterations: 28
    ... ... Current Metric Value: -6.90247E-01
    ... ... Elapsed Iterations: 29
    ... ... Current Metric Value: -6.92602E-01
    ... ... Elapsed Iterations: 29
    ... ... Current Metric Value: -6.92602E-01
    ... ... Elapsed Iterations: 30
    ... ... Current Metric Value: -6.94249E-01
    ... ... Elapsed Iterations: 30
    ... ... Current Metric Value: -6.94249E-01
    ... ... Elapsed Iterations: 31
    ... ... Current Metric Value: -6.94165E-01
    ... ... Elapsed Iterations: 31
    ... ... Current Metric Value: -6.94373E-01
    ... ... Elapsed Iterations: 31
    ... ... Current Metric Value: -6.94373E-01
    ... ... Elapsed Iterations: 32
    ... ... Current Metric Value: -6.95214E-01
    ... ... Elapsed Iterations: 32
    ... ... Current Metric Value: -6.95214E-01
    ... ... Elapsed Iterations: 33
    ... ... Current Metric Value: -6.94936E-01
    ... ... Elapsed Iterations: 33
    ... ... Current Metric Value: -6.95299E-01
    ... ... Elapsed Iterations: 33
    ... ... Current Metric Value: -6.95299E-01
    ... ... Elapsed Iterations: 34
    ... ... Current Metric Value: -6.95973E-01
    ... ... Elapsed Iterations: 34
    ... ... Current Metric Value: -6.95973E-01
    ... ... Elapsed Iterations: 35
    ... ... Current Metric Value: -6.97031E-01
    ... ... Elapsed Iterations: 35
    ... ... Current Metric Value: -6.97031E-01
    ... ... Elapsed Iterations: 36
    ... ... Current Metric Value: -6.98673E-01
    ... ... Elapsed Iterations: 36
    ... ... Current Metric Value: -6.98673E-01
    ... ... Elapsed Iterations: 37
    ... ... Current Metric Value: -6.98719E-01
    ... ... Elapsed Iterations: 37
    ... ... Current Metric Value: -6.99118E-01
    ... ... Elapsed Iterations: 37
    ... ... Current Metric Value: -6.99118E-01
    ... ... Elapsed Iterations: 38
    ... ... Current Metric Value: -7.00089E-01
    ... ... Elapsed Iterations: 38
    ... ... Current Metric Value: -7.00089E-01
    ... ... Elapsed Iterations: 39
    ... ... Current Metric Value: -7.00070E-01
    ... ... Elapsed Iterations: 39
    ... ... Current Metric Value: -7.00124E-01
    ... ... Elapsed Iterations: 39
    ... ... Current Metric Value: -7.00124E-01
    ... ... Elapsed Iterations: 40
    ... ... Current Metric Value: -7.00162E-01
    ... ... Elapsed Iterations: 40
    ... ... Current Metric Value: -7.00162E-01
    ... ... Elapsed Iterations: 41
    ... ... Current Metric Value: -6.99007E-01
    ... ... Elapsed Iterations: 41
    ... ... Current Metric Value: -7.00150E-01
    ... ... Elapsed Iterations: 41
    ... ... Current Metric Value: -7.00161E-01
    ... ... Elapsed Iterations: 41
    ... ... Current Metric Value: -7.00161E-01
    ... ... Elapsed Iterations: 41
    ... ... Current Metric Value: -7.00162E-01
    ... ... Elapsed Iterations: 41
    ... ... Current Metric Value: -7.00162E-01
    ... ... Elapsed Iterations: 41
    ... ... Current Metric Value: -7.00162E-01
    ... ... Elapsed Iterations: 41
    ... ... Current Metric Value: -7.00162E-01
    ... ... Elapsed Iterations: 41
    ... ... Current Metric Value: -7.00162E-01
    ... ... Elapsed Iterations: 41
    ... ... Current Metric Value: -7.00162E-01
    ... ... Elapsed Iterations: 41
    ... ... Current Metric Value: -7.00162E-01
    ... ... Elapsed Iterations: 41
    ... ... Current Metric Value: -7.00162E-01
    ... ... Elapsed Iterations: 41
    ... ... Current Metric Value: -7.00162E-01
    ... ... Elapsed Iterations: 41
    ... ... Current Metric Value: -7.00162E-01
    ... ... Elapsed Iterations: 41
    ... ... Current Metric Value: -7.00162E-01
    ... ... Elapsed Iterations: 41
    ... ... Current Metric Value: -7.00162E-01
    ... ... Elapsed Iterations: 41
    ... ... Current Metric Value: -7.00162E-01
    ... ... Elapsed Iterations: 41
    ... ... Current Metric Value: -7.00162E-01
    ... ... Optimal BSpline transform determined 
    ... ... ... Elapsed Iterations: 42
    ... ... ... Final Metric Value: -7.00162E-01
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
    ... ... Current Metric Value: -2.34990E-01
    ... ... Elapsed Iterations: 0
    ... ... Current Metric Value: -2.65505E-01
    ... ... Elapsed Iterations: 0
    ... ... Current Metric Value: -3.88065E-01
    ... ... Elapsed Iterations: 0
    ... ... Current Metric Value: -3.88065E-01
    ... ... Elapsed Iterations: 1
    ... ... Current Metric Value: -1.62432E-01
    ... ... Elapsed Iterations: 1
    ... ... Current Metric Value: -4.11585E-01
    ... ... Elapsed Iterations: 1
    ... ... Current Metric Value: -5.03522E-01
    ... ... Elapsed Iterations: 1
    ... ... Current Metric Value: -5.03522E-01
    ... ... Elapsed Iterations: 2
    ... ... Current Metric Value: -5.65625E-01
    ... ... Elapsed Iterations: 2
    ... ... Current Metric Value: -5.65625E-01
    ... ... Elapsed Iterations: 3
    ... ... Current Metric Value: -5.77515E-01
    ... ... Elapsed Iterations: 3
    ... ... Current Metric Value: -5.77515E-01
    ... ... Elapsed Iterations: 4
    ... ... Current Metric Value: -6.74862E-01
    ... ... Elapsed Iterations: 4
    ... ... Current Metric Value: -6.74862E-01
    ... ... Elapsed Iterations: 5
    ... ... Current Metric Value: -6.90037E-01
    ... ... Elapsed Iterations: 5
    ... ... Current Metric Value: -6.90037E-01
    ... ... Elapsed Iterations: 6
    ... ... Current Metric Value: -6.98297E-01
    ... ... Elapsed Iterations: 6
    ... ... Current Metric Value: -6.98297E-01
    ... ... Elapsed Iterations: 7
    ... ... Current Metric Value: -7.02514E-01
    ... ... Elapsed Iterations: 7
    ... ... Current Metric Value: -7.02514E-01
    ... ... Elapsed Iterations: 8
    ... ... Current Metric Value: -7.05001E-01
    ... ... Elapsed Iterations: 8
    ... ... Current Metric Value: -7.05001E-01
    ... ... Elapsed Iterations: 9
    ... ... Current Metric Value: -7.08849E-01
    ... ... Elapsed Iterations: 9
    ... ... Current Metric Value: -7.08849E-01
    ... ... Elapsed Iterations: 10
    ... ... Current Metric Value: -7.04038E-01
    ... ... Elapsed Iterations: 10
    ... ... Current Metric Value: -7.08915E-01
    ... ... Elapsed Iterations: 10
    ... ... Current Metric Value: -7.08915E-01
    ... ... Elapsed Iterations: 11
    ... ... Current Metric Value: -7.08871E-01
    ... ... Elapsed Iterations: 11
    ... ... Current Metric Value: -7.09039E-01
    ... ... Elapsed Iterations: 11
    ... ... Current Metric Value: -7.09039E-01
    ... ... Elapsed Iterations: 12
    ... ... Current Metric Value: -7.08931E-01
    ... ... Elapsed Iterations: 12
    ... ... Current Metric Value: -7.09178E-01
    ... ... Elapsed Iterations: 12
    ... ... Current Metric Value: -7.09178E-01
    ... ... Elapsed Iterations: 13
    ... ... Current Metric Value: -7.08856E-01
    ... ... Elapsed Iterations: 13
    ... ... Current Metric Value: -7.09276E-01
    ... ... Elapsed Iterations: 13
    ... ... Current Metric Value: -7.09276E-01
    ... ... Elapsed Iterations: 14
    ... ... Current Metric Value: -7.08786E-01
    ... ... Elapsed Iterations: 14
    ... ... Current Metric Value: -7.09363E-01
    ... ... Elapsed Iterations: 14
    ... ... Current Metric Value: -7.09363E-01
    ... ... Elapsed Iterations: 15
    ... ... Current Metric Value: -7.08550E-01
    ... ... Elapsed Iterations: 15
    ... ... Current Metric Value: -7.09421E-01
    ... ... Elapsed Iterations: 15
    ... ... Current Metric Value: -7.09421E-01
    ... ... Elapsed Iterations: 16
    ... ... Current Metric Value: -7.08371E-01
    ... ... Elapsed Iterations: 16
    ... ... Current Metric Value: -7.09519E-01
    ... ... Elapsed Iterations: 16
    ... ... Current Metric Value: -7.09519E-01
    ... ... Elapsed Iterations: 17
    ... ... Current Metric Value: -7.04025E-01
    ... ... Elapsed Iterations: 17
    ... ... Current Metric Value: -7.09488E-01
    ... ... Elapsed Iterations: 17
    ... ... Current Metric Value: -7.09506E-01
    ... ... Elapsed Iterations: 17
    ... ... Current Metric Value: -7.09519E-01
    ... ... Elapsed Iterations: 17
    ... ... Current Metric Value: -7.09519E-01
    ... ... Elapsed Iterations: 17
    ... ... Current Metric Value: -7.09519E-01
    ... ... Elapsed Iterations: 17
    ... ... Current Metric Value: -7.09519E-01
    ... ... Elapsed Iterations: 17
    ... ... Current Metric Value: -7.09519E-01
    ... ... Elapsed Iterations: 17
    ... ... Current Metric Value: -7.09519E-01
    ... ... Elapsed Iterations: 17
    ... ... Current Metric Value: -7.09519E-01
    ... ... Elapsed Iterations: 17
    ... ... Current Metric Value: -7.09519E-01
    ... ... Elapsed Iterations: 17
    ... ... Current Metric Value: -7.09519E-01
    ... ... Elapsed Iterations: 17
    ... ... Current Metric Value: -7.09519E-01
    ... ... Elapsed Iterations: 17
    ... ... Current Metric Value: -7.09519E-01
    ... ... Elapsed Iterations: 17
    ... ... Current Metric Value: -7.09519E-01
    ... ... Elapsed Iterations: 17
    ... ... Current Metric Value: -7.09519E-01
    ... ... Optimal BSpline transform determined 
    ... ... ... Elapsed Iterations: 18
    ... ... ... Final Metric Value: -7.09519E-01
    ... Registration Complete
    Analysis Complete!


We converged in 18 iterations of BFGS rather than the 42 iterations the
uninitialized registration took. The final metric value (negative
cross-correlation) was quite close: -0.70 vs -0.71. This suggests the
objective function may be near convex since the determined minima are
very close; although, this cannot be proven. Qualitative inspection of
the two results suggests these particular images can be registered well
without initialization. The reader is encouraged to do this inspection
by outputting results from each execution.

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


a NumPy binary,

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

Example 2
---------

lorem ipsum
