# Single Domain Comprehensive Calculator

 This is a first github upload of the SDCC package. It contains several modules detailed here.
 Required packages: jax, numpy, scipy, scikit-image, matplotlib, pickle.

### Installation
 From within the directory containing this readme:

 `python3 -m pip install --upgrade build`

 `python3 -m build`

 `python3 -m pip install ./`

 Prerequisites: pip and setuptools

### materials.py
 This module contains "material" objects which could be used for extending the SDCC to other materials and other anisotropy forms (e.g. hematite). It's not currently used by anything else.

### energy.py
 This module contains functions to calculate the energy density surface for an SD grain as a function of composition, shape and anisotropy form. Could be extended by using materials.py.

### barriers.py
 This module contains functions to calculate energy barriers for an SD grain. It additionally has code to calculate these barriers at all temperatures, track which barriers are which across temperatures, and then fit coefficients to this data. There's several things that could be improved here.

 - The module contains a load of deprecated functions which I was testing. It could be cleaned up a lot.
 - The function to calculate barriers at all temperatures could be parallelized so more than one temperature can be calculated at a time.
 - The merge_barriers() function is used to account for when the energy barriers change across temperatures - it involves remaking a lot of the arrays to have consistent dimensions, and there's probably a much simpler way of doing it.
 - The GEL class could probably inherit from some base class which can also do hysteresis. This might then make things easier when converting simulations.py routines to be able to do both (you need some flag to be able to switch on/off the small-field approximation when generating Q-matrices, but that's the only real difference).

### simulation.py
This module deals with running thermally activated models. This is probably what you're interested in if you want to use this with Merrill models. In particular, the matrix exponentiation function is pretty unstable here which sometimes leads to weird results (especially when delta T is big). It would be great to combine this with Les' code for his cooling rate paper to make things work a bit better.

### treatment.py
This module deals with treatment routines that can be fed into simulation.py - currently this is done through lists of ThermalStep class objects. ThermalStep could be generalized to just Step and each type of step could be a class that inherits from it. Currently, everything's done as a big switch statement which doesn't make much sense once the number of steps gets larger.

### plotting.py
This module contains code for all types of plots that might be used by the SDCC.

