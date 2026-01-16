Installation
============

.. note::

   **pyzentropy** is not yet available on PyPI. For now, please install from source as described below. PyPI support will be added in a future release.

It is recommended to first set up a virtual environment using Conda:

.. code-block:: bash

    conda create -n pyzentropy python=3.12      
    conda activate pyzentropy

Clone the main branch of the repository:

.. code-block:: bash

    git clone https://github.com/nhew1994/pyzentropy.git

Or clone a specific branch:

.. code-block:: bash

    git clone -b <branch_name> https://github.com/nhew1994/pyzentropy.git

Then move to the `pyzentropy` directory and install in editable (`-e`) mode:

.. code-block:: bash

    cd pyzentropy
    python -m pip install -e .
