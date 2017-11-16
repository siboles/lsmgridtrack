Examples
========

.. toctree::
   :maxdepth: 1
   :glob:

Example 1
---------

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

    <ipython-input-15-a002a83e06f0> in <module>()
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
    
    # Provide the positions of the 8 corner nodes of the deformed grid to initialize transform
    t.options['Registration']['landmarks'] = False
    
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
    ... ... Current Metric Value: -4.50453E-02
    ... ... Elapsed Iterations: 0
    ... ... Current Metric Value: -5.43574E-02
    ... ... Elapsed Iterations: 0
    ... ... Current Metric Value: -7.83617E-02
    ... ... Elapsed Iterations: 0
    ... ... Current Metric Value: -1.60110E-02
    ... ... Elapsed Iterations: 0
    ... ... Current Metric Value: -8.51354E-02
    ... ... Elapsed Iterations: 0
    ... ... Current Metric Value: -8.51354E-02
    ... ... Elapsed Iterations: 1
    ... ... Current Metric Value: -4.05880E-02
    ... ... Elapsed Iterations: 1
    ... ... Current Metric Value: -8.61641E-02
    ... ... Elapsed Iterations: 1
    ... ... Current Metric Value: -8.61641E-02
    ... ... Elapsed Iterations: 2
    ... ... Current Metric Value: -8.98502E-02
    ... ... Elapsed Iterations: 2
    ... ... Current Metric Value: -8.98502E-02
    ... ... Elapsed Iterations: 3
    ... ... Current Metric Value: -1.02397E-01
    ... ... Elapsed Iterations: 3
    ... ... Current Metric Value: -1.02397E-01
    ... ... Elapsed Iterations: 4
    ... ... Current Metric Value: -1.31571E-01
    ... ... Elapsed Iterations: 4
    ... ... Current Metric Value: -1.31571E-01
    ... ... Elapsed Iterations: 5
    ... ... Current Metric Value: -1.92294E-01
    ... ... Elapsed Iterations: 5
    ... ... Current Metric Value: -2.30642E-01
    ... ... Elapsed Iterations: 5
    ... ... Current Metric Value: -2.30642E-01
    ... ... Elapsed Iterations: 6
    ... ... Current Metric Value: -3.19663E-01
    ... ... Elapsed Iterations: 6
    ... ... Current Metric Value: -3.19663E-01
    ... ... Elapsed Iterations: 7
    ... ... Current Metric Value: -3.31261E-01
    ... ... Elapsed Iterations: 7
    ... ... Current Metric Value: -3.31261E-01
    ... ... Elapsed Iterations: 8
    ... ... Current Metric Value: -3.78521E-01
    ... ... Elapsed Iterations: 8
    ... ... Current Metric Value: -3.78521E-01
    ... ... Elapsed Iterations: 9
    ... ... Current Metric Value: -4.09511E-01
    ... ... Elapsed Iterations: 9
    ... ... Current Metric Value: -4.09511E-01
    ... ... Elapsed Iterations: 10
    ... ... Current Metric Value: -4.28550E-01
    ... ... Elapsed Iterations: 10
    ... ... Current Metric Value: -4.28550E-01
    ... ... Elapsed Iterations: 11
    ... ... Current Metric Value: -4.57928E-01
    ... ... Elapsed Iterations: 11
    ... ... Current Metric Value: -4.57928E-01
    ... ... Elapsed Iterations: 12
    ... ... Current Metric Value: -4.91014E-01
    ... ... Elapsed Iterations: 12
    ... ... Current Metric Value: -4.91014E-01
    ... ... Elapsed Iterations: 13
    ... ... Current Metric Value: -5.46429E-01
    ... ... Elapsed Iterations: 13
    ... ... Current Metric Value: -5.46429E-01
    ... ... Elapsed Iterations: 14
    ... ... Current Metric Value: -5.91795E-01
    ... ... Elapsed Iterations: 14
    ... ... Current Metric Value: -5.91795E-01
    ... ... Elapsed Iterations: 15
    ... ... Current Metric Value: -6.21534E-01
    ... ... Elapsed Iterations: 15
    ... ... Current Metric Value: -6.21534E-01
    ... ... Elapsed Iterations: 16
    ... ... Current Metric Value: -6.42160E-01
    ... ... Elapsed Iterations: 16
    ... ... Current Metric Value: -6.42160E-01
    ... ... Elapsed Iterations: 17
    ... ... Current Metric Value: -6.54203E-01
    ... ... Elapsed Iterations: 17
    ... ... Current Metric Value: -6.54203E-01
    ... ... Elapsed Iterations: 18
    ... ... Current Metric Value: -6.57690E-01
    ... ... Elapsed Iterations: 18
    ... ... Current Metric Value: -6.57690E-01
    ... ... Elapsed Iterations: 19
    ... ... Current Metric Value: -6.66482E-01
    ... ... Elapsed Iterations: 19
    ... ... Current Metric Value: -6.66482E-01
    ... ... Elapsed Iterations: 20
    ... ... Current Metric Value: -6.73026E-01
    ... ... Elapsed Iterations: 20
    ... ... Current Metric Value: -6.73026E-01
    ... ... Elapsed Iterations: 21
    ... ... Current Metric Value: -6.76938E-01
    ... ... Elapsed Iterations: 21
    ... ... Current Metric Value: -6.78362E-01
    ... ... Elapsed Iterations: 21
    ... ... Current Metric Value: -6.78362E-01
    ... ... Elapsed Iterations: 22
    ... ... Current Metric Value: -6.83180E-01
    ... ... Elapsed Iterations: 22
    ... ... Current Metric Value: -6.83180E-01
    ... ... Elapsed Iterations: 23
    ... ... Current Metric Value: -6.83598E-01
    ... ... Elapsed Iterations: 23
    ... ... Current Metric Value: -6.83598E-01
    ... ... Elapsed Iterations: 24
    ... ... Current Metric Value: -6.84963E-01
    ... ... Elapsed Iterations: 24
    ... ... Current Metric Value: -6.84963E-01
    ... ... Elapsed Iterations: 25
    ... ... Current Metric Value: -6.85955E-01
    ... ... Elapsed Iterations: 25
    ... ... Current Metric Value: -6.85955E-01
    ... ... Elapsed Iterations: 26
    ... ... Current Metric Value: -6.88375E-01
    ... ... Elapsed Iterations: 26
    ... ... Current Metric Value: -6.88375E-01
    ... ... Elapsed Iterations: 27
    ... ... Current Metric Value: -6.90440E-01
    ... ... Elapsed Iterations: 27
    ... ... Current Metric Value: -6.90440E-01
    ... ... Elapsed Iterations: 28
    ... ... Current Metric Value: -6.91065E-01
    ... ... Elapsed Iterations: 28
    ... ... Current Metric Value: -6.91065E-01
    ... ... Elapsed Iterations: 29
    ... ... Current Metric Value: -6.91744E-01
    ... ... Elapsed Iterations: 29
    ... ... Current Metric Value: -6.91744E-01
    ... ... Elapsed Iterations: 30
    ... ... Current Metric Value: -6.92618E-01
    ... ... Elapsed Iterations: 30
    ... ... Current Metric Value: -6.92618E-01
    ... ... Elapsed Iterations: 31
    ... ... Current Metric Value: -6.94498E-01
    ... ... Elapsed Iterations: 31
    ... ... Current Metric Value: -6.94498E-01
    ... ... Elapsed Iterations: 32
    ... ... Current Metric Value: -6.92047E-01
    ... ... Elapsed Iterations: 32
    ... ... Current Metric Value: -6.94533E-01
    ... ... Elapsed Iterations: 32
    ... ... Current Metric Value: -6.94533E-01
    ... ... Elapsed Iterations: 33
    ... ... Current Metric Value: -6.95340E-01
    ... ... Elapsed Iterations: 33
    ... ... Current Metric Value: -6.95340E-01
    ... ... Elapsed Iterations: 34
    ... ... Current Metric Value: -6.96607E-01
    ... ... Elapsed Iterations: 34
    ... ... Current Metric Value: -6.96607E-01
    ... ... Elapsed Iterations: 35
    ... ... Current Metric Value: -6.99010E-01
    ... ... Elapsed Iterations: 35
    ... ... Current Metric Value: -6.99010E-01
    ... ... Elapsed Iterations: 36
    ... ... Current Metric Value: -6.99886E-01
    ... ... Elapsed Iterations: 36
    ... ... Current Metric Value: -6.99886E-01
    ... ... Elapsed Iterations: 37
    ... ... Current Metric Value: -7.00176E-01
    ... ... Elapsed Iterations: 37
    ... ... Current Metric Value: -7.00176E-01
    ... ... Elapsed Iterations: 38
    ... ... Current Metric Value: -7.00065E-01
    ... ... Elapsed Iterations: 38
    ... ... Current Metric Value: -7.00210E-01
    ... ... Elapsed Iterations: 38
    ... ... Current Metric Value: -7.00210E-01
    ... ... Elapsed Iterations: 39
    ... ... Current Metric Value: -7.00919E-01
    ... ... Elapsed Iterations: 39
    ... ... Current Metric Value: -7.00919E-01
    ... ... Elapsed Iterations: 40
    ... ... Current Metric Value: -7.00090E-01
    ... ... Elapsed Iterations: 40
    ... ... Current Metric Value: -7.00899E-01
    ... ... Elapsed Iterations: 40
    ... ... Current Metric Value: -7.00916E-01
    ... ... Elapsed Iterations: 40
    ... ... Current Metric Value: -7.00919E-01
    ... ... Elapsed Iterations: 40
    ... ... Current Metric Value: -7.00919E-01
    ... ... Elapsed Iterations: 40
    ... ... Current Metric Value: -7.00919E-01
    ... ... Elapsed Iterations: 40
    ... ... Current Metric Value: -7.00919E-01
    ... ... Elapsed Iterations: 40
    ... ... Current Metric Value: -7.00919E-01
    ... ... Elapsed Iterations: 40
    ... ... Current Metric Value: -7.00919E-01
    ... ... Elapsed Iterations: 40
    ... ... Current Metric Value: -7.00919E-01
    ... ... Elapsed Iterations: 40
    ... ... Current Metric Value: -7.00919E-01
    ... ... Elapsed Iterations: 40
    ... ... Current Metric Value: -7.00919E-01
    ... ... Elapsed Iterations: 40
    ... ... Current Metric Value: -7.00919E-01
    ... ... Elapsed Iterations: 40
    ... ... Current Metric Value: -7.00919E-01
    ... ... Elapsed Iterations: 40
    ... ... Current Metric Value: -7.00919E-01
    ... ... Optimal BSpline transform determined 
    ... ... ... Elapsed Iterations: 41
    ... ... ... Final Metric Value: -7.00919E-01
    ... Registration Complete
    Analysis Complete!


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

a NumPy binary,

.. code:: python

    t.writeResultsAsNumpy('example1')

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

