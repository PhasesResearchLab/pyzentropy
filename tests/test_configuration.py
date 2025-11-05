# Standard Library Imports
import os
import pickle

# Third-Party Library Imports
import numpy as np
import pytest

# PyZentropy Imports
from pyzentropy.configuration import Configuration


def test_configuration_shape_check():
    """Test that Configuration raises ValueError for mismatched array shapes."""
    temperatures = np.arange(3)
    volumes = np.arange(4)
    correct_shape = (3, 4)
    wrong_shape = (2, 4)
    arr = np.zeros(correct_shape)
    arr_wrong = np.zeros(wrong_shape)

    with pytest.raises(ValueError, match="must have shape"):
        Configuration(
            name="test",
            multiplicity=1,
            number_of_atoms=1,
            volumes=volumes,
            temperatures=temperatures,
            helmholtz_energies=arr_wrong,  # wrong shape here
            helmholtz_energies_dV=arr,
            helmholtz_energies_d2V2=arr,
            entropies=arr,
            heat_capacities=arr,
        )


# Load test data once for all tests
# Contains FM, SF28, and SF22 using DFTTK EV and Debye
test_data_path = os.path.join(os.path.dirname(__file__), "test_data", "test_configs.pkl")
with open(test_data_path, "rb") as f:
    config_data = pickle.load(f)


@pytest.mark.parametrize("config_key", ["FM", "SF28"])
def test_configuration_param(config_key):
    """Test Configuration internal energy and partition function calculations."""
    config = config_data[config_key]
    reference_helmholtz_energies = config_data["FM"].helmholtz_energies
    instance = Configuration(
        name=config.name,
        multiplicity=config.multiplicity,
        number_of_atoms=config.number_of_atoms,
        volumes=config.volumes,
        temperatures=config.temperatures,
        helmholtz_energies=config.helmholtz_energies,
        helmholtz_energies_dV=config.helmholtz_energies_dV,
        helmholtz_energies_d2V2=config.helmholtz_energies_d2V2,
        entropies=config.entropies,
        heat_capacities=config.heat_capacities,
    )
    instance.calculate_internal_energies()
    assert np.allclose(instance.internal_energies, config.internal_energies)
    instance.calculate_partition_functions(reference_helmholtz_energies)
    assert np.allclose(instance.partition_functions, config.partition_functions, equal_nan=True)


def test_configuration_plot_smoke():
    """Smoke test: Ensure all plot types run without error."""
    config = config_data["FM"]
    plot_types = [
        "helmholtz_energy_vs_volume",
        "helmholtz_energy_vs_temperature",
        "internal_energy_vs_volume",
        "internal_energy_vs_temperature",
        "entropy_vs_volume",
        "entropy_vs_temperature",
        "heat_capacity_vs_volume",
        "heat_capacity_vs_temperature",
    ]
    # Default plot
    for plot_type in plot_types:
        config.plot(plot_type)
    # With selected temperatures/volumes
    selected_temperatures = np.array([300, 400, 500, 600, 700, 800, 900, 1000])
    selected_volumes = np.array([100, 150, 200])
    for plot_type in plot_types:
        config.plot(plot_type, selected_temperatures=selected_temperatures, selected_volumes=selected_volumes)
    # With custom width and height
    for plot_type in plot_types:
        config.plot(plot_type, width=800, height=600)
