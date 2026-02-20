==========    
pyzentropy
==========

**pyzentropy** is an open-source Python package for first-principles thermodynamic calculations 
based on the recursion property of entropy. For a system consisting of multiple configurations, 
each with its own intrinsic entropy, the total entropy can be written as

.. math::

   S = -k_B \sum_{k=1}^{N} p_k \ln p_k + \sum_{k=1}^{N} p_k S_k

where :math:`k` indexes the :math:`N` configurations and :math:`p_k` is the probability 
of configuration :math:`k`. The first term corresponds to the *configurational entropy*, 
while the second term is the probability-weighted sum of the *intra-configurational entropies*.

This expression is based on the recursion property of entropy, well known in information 
theory, which allows the system’s microstates to be coarse-grained into partitions indexed 
by :math:`k`. We also refer to this as the *zentropy* method, which combines *Zustandssumme* 
("sum over states") with *entropy*, emphasizing that the total entropy is constructed by 
summing over configurations that themselves possess intrinsic entropy.

The zentropy method allows microstates sharing similar physical characteristics to be 
grouped into configurations, making the total entropy easier to compute by first calculating
the entropies of individual configurations. In practice, the intra-configurational entropies 
:math:`S_k` can be obtained using established open-source tools such as Phonopy and DFTTK.

Key Features
------------  

Input Data
^^^^^^^^^^

For each configuration :math:`k`:

- Helmholtz free energy :math:`F_k(T,V)` and its first and second derivatives with respect to :math:`V`
- Entropy :math:`S_k(T,V)` and heat capacity at constant volume :math:`C_{V,k}(T,V)`

Capabilities
^^^^^^^^^^^^

**pyzentropy** enables the following analyses and visualizations:

- Generate 2D plots of configuration-dependent properties as functions of :math:`T` or :math:`V`.
- Compute system-level properties as a function of temperature :math:`T` and volume :math:`V`.
- Calculate thermodynamic properties at fixed pressure :math:`P`.
- Construct :math:`T\!-\!V` and :math:`P\!-\!T` phase diagrams *(currently limited to a single common tangent construction)*.
- Generate 2D plots of system properties as a function of :math:`T` or :math:`V`.
- Generate 2D plots of system properties at fixed :math:`P` as a function of :math:`T`.
- Generate 3D thermodynamic landscapes (in development).
