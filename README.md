
<p align="left">
    <img src="docs/latest_pyzentropy.drawio.png" alt="pyzentropy Logo" width="450">
</p>

[![Testing](https://github.com/nhew1994/pyzentropy/actions/workflows/test.yml/badge.svg)](https://github.com/nhew1994/pyzentropy/actions/workflows/test.yml)

## Installation
It is recommended to first set up a virtual environment using Conda:

    conda create -n pyzentropy python=3.12      
    conda activate pyzentropy

Clone the main branch of the repository:
    
    git clone https://github.com/nhew1994/pyzentropy.git

Or clone a specific branch:
    
    git clone -b <branch_name> https://github.com/nhew1994/pyzentropy.git

  Then move to `pyzentropy` directory and install in editable (`-e`) mode.

    cd pyzentropy
    python -m pip install -e .

## Example notebooks
Click the badge below to open the project in GitHub Codespaces.  
Then, browse the `examples/codespace` folder to explore and run the example notebooks:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/PhasesResearchLab/pyzentropy?quickstart=1)

| Notebooks    | Description |
|--------------|-------------|
| Fe<sub>3</sub>Pt | Example of applying zentropy to the 3 lowest energy configurations in Fe<sub>3</sub>Pt 12-atom supercell. The Helmholtz energies were calculated using the energy–volume curves and the Debye–Grüneisen model. |

