Installation
============
The SDCC has relatively few dependencies and can be installed easily using pip.

First, clone the repository:

.. code-block:: bash
   
   git clone https://github.com/bcych/SDCC.git

Navigate to the directory and install the `build` module with pip.

.. code-block:: bash
   
   python3 -m pip install --upgrade build

Build the project

.. code-block:: bash
  
   python3 -m build

Install the package:

.. code-block:: bash
   
   python3 -m pip install ./

To test whether the installation worked, try calculating the zeeman energy as a unit test and check whether you get the correct answer

.. code-block:: python3

   >>> from sdcc.energy import Ez
   >>> Ez(0,0,1,1,1,1)
   Array(-0.29192658, dtype=float64)

If you encounter any problems with installing `jax`, check out the `jax installation page <https://jax.readthedocs.io/en/latest/installation.html>`_
