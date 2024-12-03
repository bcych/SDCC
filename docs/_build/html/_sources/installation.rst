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
   
   python3 -m install build

To test whether the installation worked, try plotting an energy surface

.. code-block:: python3

   from sdcc.plotting import plot_energy_surface
   plot_energy_surface(0,'hard',1.00,1.00)

If you encounter any problems with installing `jax`, check out the `jax installation page <https://jax.readthedocs.io/en/latest/installation.html>`_
