Installation
============

.. toctree::
   :maxdepth: 1
   :glob:

Conda
-----

lsmgridtrack is designed for use with conda package manager, or its optimized fork mamba. There are multiple packages that provide conda. 
Currently, the smallest, fastest, and thus recommended package is `<mambaforge https://github.com/conda-forge/miniforge>`__.

.. note::
   lsmgridtrack is only released for Python 3.5 and 3.6. If you install an Anaconda with a Python version that is different, you must follow the installation `In a conda environment`_ instructions. 

To install the current release
------------------------------

lsmgridtrack depends on packages outside of the default conda channels, so we need to add these. The order also matters, because this determines channel priority.

.. code-block:: bash

   conda config --add channels SimpleITK
   conda config --add channels siboles
   conda config --add channels conda-forge

.. note::
   If mambaforge or miniforge is used, conda-forge, will be added by default, in which case the last line will have no effect. 

In a conda environment
~~~~~~~~~~~~~~~~~~~~~~

By installing lsmgridtrack into its own conda environment it is guaranteed to not interfere with your system or other environments. Therefore, this approach is the most recommended. This can be accomplished in one command line entry.

.. code-block:: guess

   mamba create -n [ENVIRONMENT_NAME] lsmgridtrack

.. note::
   This assumes mambaforge. Substitute conda for mamba if another manager was installed.

This creates an environment named whatever you replace [ENVIRONMENT_NAME] with. By providing the last argument, lsmgridtrack, the package will be installed into the newly created
environment along with all the dependencies it requires.

To use *lsmgridtrack* this environment will need to be active. In a command terminal, type:

.. code-block:: bash

   conda activate [ENVIRONMENT_NAME]

You should see the command prompt change to include *(ENVIRONMENT_NAME)* in parentheses.

To change to a different environment type the above command with the new environment name. To deactivate the environment type:

.. code-block:: bash
   conda deactivate


To install the latest code from the github repository
-----------------------------------------------------

The source code can be downloaded from `GitHub <https://github.com/siboles/lsmgridtrack/archive/master.zip>`_ or if git is installed cloned with:

.. code-block:: bash

   git clone https://github.com/siboles/lsmgridtrack.git

It is recommended to follow the prior instructions to create a conda environment from the released version first. With this environment active,
navigate to the root level of the cloned repository where *setup.py* is located. Then install with:

.. code-block:: bash

   python -m pip install -vv


If you have previously cloned the git repository, it can be updated to the latest version with:

.. code-block:: bash

   git pull

