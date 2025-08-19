# Standard Library Imports
import os
import pickle
import re
import copy

# Third-Party Library Imports
import numpy as np
import pytest

# PyZentropy Imports
from pyzentropy.configuration import Configuration
from pyzentropy.system import System


# Helper Functions
def make_config(name, number_of_atoms=1, volumes=None, temperatures=None):
    """Create a Configuration object with zeroed arrays for testing."""
    nT = len(temperatures)
    nV = len(volumes)
    shape = (nT, nV)
    arr = np.zeros(shape)
    return Configuration(
        name=name,
        multiplicity=1,
        number_of_atoms=number_of_atoms,
        volumes=volumes,
        temperatures=temperatures,
        helmholtz_energies=arr,
        helmholtz_energies_dV=arr,
        helmholtz_energies_d2V2=arr,
        entropies=arr,
        heat_capacities=arr,
    )


def make_system(configs, reference_helmholtz_energies):
    """Create a System and run all calculations up to heat capacities."""
    system = System(configs)
    system.calculate_partition_functions()
    system.calculate_probabilities()
    system.calculate_helmholtz_energies(reference_helmholtz_energies)
    system.calculate_entropies()
    system.calculate_helmholtz_energies_dV()
    system.calculate_bulk_moduli()
    system.calculate_helmholtz_energies_d2V2()
    system.calculate_heat_capacities()
    return system


# Consistency Checks
def test_system_inconsistent_number_of_atoms():
    """Test error if configs have different number of atoms."""
    config1 = make_config("A", number_of_atoms=1, volumes=np.arange(2), temperatures=np.arange(3))
    config2 = make_config("B", number_of_atoms=2, volumes=np.arange(2), temperatures=np.arange(3))
    with pytest.raises(ValueError, match="Number of atoms for configurations are not the same"):
        System({"A": config1, "B": config2})


def test_system_inconsistent_volumes():
    """Test error if configs have different volumes."""
    config1 = make_config("A", volumes=np.arange(2), temperatures=np.arange(3))
    config2 = make_config("B", volumes=np.arange(3), temperatures=np.arange(3))
    with pytest.raises(ValueError, match="Volumes for configurations are not the same"):
        System({"A": config1, "B": config2})


def test_system_inconsistent_temperatures():
    """Test error if configs have different temperatures."""
    config1 = make_config("A", volumes=np.arange(2), temperatures=np.arange(3))
    config2 = make_config("B", volumes=np.arange(2), temperatures=np.arange(4))
    with pytest.raises(ValueError, match="Temperatures for configurations are not the same"):
        System({"A": config1, "B": config2})


# Load Test Data
# Contains config_0, config_28, and config_22 using Nigel's EV and Shang's Debye using his MATLAB code
test_data_path = os.path.join(os.path.dirname(__file__), "test_data", "test_configs.pkl")
with open(test_data_path, "rb") as f:
    config_data = pickle.load(f)
reference_helmholtz_energies = config_data["config_0"].helmholtz_energies
for name, config in config_data.items():
    config.calculate_internal_energies()
    config.calculate_partition_functions(reference_helmholtz_energies)

expected_results_path = os.path.join(os.path.dirname(__file__), "test_data", "system_object.pkl")
with open(expected_results_path, "rb") as f:
    expected_results = pickle.load(f)


# Calculation Tests
def test_partition_functions():
    """Test partition functions and error on missing config partition functions."""
    local_config_data = copy.deepcopy(config_data)
    system = make_system(local_config_data, reference_helmholtz_energies)
    assert np.allclose(system.partition_functions, expected_results.partition_functions, equal_nan=True)
    # Test error if config partition_functions is None
    for config in system.configurations.values():
        config.partition_functions = None
    with pytest.raises(ValueError, match=f"partition_functions not set for configuration 'config_0'"):
        system.calculate_partition_functions()


def test_probabilities():
    """Test probabilities and error on missing system partition functions."""
    local_config_data = copy.deepcopy(config_data)
    system = make_system(local_config_data, reference_helmholtz_energies)
    for config in system.configurations.values():
        assert np.allclose(config.probabilities, expected_results.configurations[config.name].probabilities, equal_nan=True)
    # Probabilities should sum to 1 (ignoring NaN)
    total_probabilities = np.zeros_like(system.configurations["config_0"].probabilities)
    for config in system.configurations.values():
        total_probabilities += config.probabilities
    mask = ~np.isnan(total_probabilities)
    assert np.allclose(total_probabilities[mask], 1.0)
    # Test error if system partition_functions is None
    system.partition_functions = None
    with pytest.raises(ValueError, match=re.escape("Partition functions not calculated. Call calculate_partition_functions() first.")):
        system.calculate_probabilities()


def test_helmholtz_energies():
    """Test Helmholtz energies and error on missing system partition functions."""
    local_config_data = copy.deepcopy(config_data)
    system = make_system(local_config_data, reference_helmholtz_energies)
    assert np.allclose(system.helmholtz_energies, expected_results.helmholtz_energies, equal_nan=True)
    system.partition_functions = None
    with pytest.raises(ValueError, match=re.escape("Partition functions not calculated. Call calculate_partition_functions() first.")):
        system.calculate_helmholtz_energies(reference_helmholtz_energies)


def test_helmholtz_energies_dV():
    """Test dF/dV and error on missing config dF/dV or probabilities."""
    local_config_data = copy.deepcopy(config_data)
    system = make_system(local_config_data, reference_helmholtz_energies)
    assert np.allclose(system.helmholtz_energies_dV, expected_results.helmholtz_energies_dV, equal_nan=True)
    # Error if config dF/dV is None
    for config in system.configurations.values():
        config.helmholtz_energies_dV = None
    with pytest.raises(ValueError, match=re.escape(f"helmholtz_energies_dV not set for configuration 'config_0'.")):
        system.calculate_helmholtz_energies_dV()
    # Error if config probabilities is None
    for config in system.configurations.values():
        config.probabilities = None
    with pytest.raises(
        ValueError,
        match=re.escape(f"Probabilities not set for configuration 'config_0'. Call calculate_probabilities() first."),
    ):
        system.calculate_helmholtz_energies_dV()


def test_entropies():
    """Test entropies and error on missing config/system data."""
    local_config_data = copy.deepcopy(config_data)
    system = make_system(local_config_data, reference_helmholtz_energies)
    assert np.allclose(system.entropies, expected_results.entropies, equal_nan=True)
    assert np.allclose(system.configurational_entropies, expected_results.configurational_entropies, equal_nan=True)
    # Error if config helmholtz_energies is None
    for config in system.configurations.values():
        config.helmholtz_energies = None
    with pytest.raises(ValueError, match=re.escape("Helmholtz energies not set for configuration 'config_0'.")):
        system.calculate_entropies()
    # Error if config internal_energies is None
    for config in system.configurations.values():
        config.internal_energies = None
    with pytest.raises(ValueError, match=re.escape("Internal energies not set for configuration 'config_0'.")):
        system.calculate_entropies()
    # Error if config probabilities is None
    for config in system.configurations.values():
        config.probabilities = None
    with pytest.raises(
        ValueError,
        match=re.escape("Probabilities not set for configuration 'config_0'. Call calculate_probabilities() first."),
    ):
        system.calculate_entropies()
    # Error if system helmholtz_energies is None
    system.helmholtz_energies = None
    with pytest.raises(ValueError, match=re.escape("Helmholtz energies not calculated. Call calculate_helmholtz_energies() first.")):
        system.calculate_entropies()


def test_bulk_moduli():
    """Test bulk moduli and error on missing config data."""
    local_config_data = copy.deepcopy(config_data)
    system = make_system(local_config_data, reference_helmholtz_energies)
    assert np.allclose(system.bulk_moduli, expected_results.bulk_moduli, equal_nan=True)
    # Restore configs before error tests
    for name, config in system.configurations.items():
        config.probabilities = expected_results.configurations[name].probabilities.copy()
        config.helmholtz_energies_dV = expected_results.configurations[name].helmholtz_energies_dV.copy()
    # Error if config d2F/dV2 is None
    for config in system.configurations.values():
        config.helmholtz_energies_d2V2 = None
    with pytest.raises(ValueError, match=re.escape(f"helmholtz_energies_d2V2 not set for configuration 'config_0'.")):
        system.calculate_bulk_moduli()
    # Error if config dF/dV is None
    for config in system.configurations.values():
        config.helmholtz_energies_dV = None
    with pytest.raises(ValueError, match=re.escape(f"helmholtz_energies_dV not set for configuration 'config_0'.")):
        system.calculate_bulk_moduli()
    # Error if config probabilities is None
    for config in system.configurations.values():
        config.probabilities = None
    with pytest.raises(
        ValueError,
        match=re.escape(f"Probabilities not set for configuration 'config_0'. Call calculate_probabilities() first."),
    ):
        system.calculate_bulk_moduli()


def test_helmholtz_energies_d2V2():
    """Test d2F/dV2 and error on missing bulk moduli."""
    local_config_data = copy.deepcopy(config_data)
    system = make_system(local_config_data, reference_helmholtz_energies)
    assert np.allclose(system.helmholtz_energies_d2V2, expected_results.helmholtz_energies_d2V2, equal_nan=True)
    # Error if bulk_moduli is None
    system.bulk_moduli = None
    with pytest.raises(ValueError, match=re.escape("Bulk moduli not calculated. Call calculate_bulk_moduli() first.")):
        system.calculate_helmholtz_energies_d2V2()


def test_heat_capacities():
    """Test heat capacities and error on missing config data."""
    local_config_data = copy.deepcopy(config_data)
    system = make_system(local_config_data, reference_helmholtz_energies)
    assert np.allclose(system.heat_capacities, expected_results.heat_capacities, equal_nan=True)
    # Error if config internal_energies is None
    for config in system.configurations.values():
        config.internal_energies = None
    with pytest.raises(ValueError, match=re.escape(f"Internal energies not set for configuration 'config_0'.")):
        system.calculate_heat_capacities()
    # Error if config heat_capacities is None
    for config in system.configurations.values():
        config.heat_capacities = None
    with pytest.raises(ValueError, match=re.escape(f"Heat capacities not set for configuration 'config_0'.")):
        system.calculate_heat_capacities()
    # Error if config probabilities is None
    for config in system.configurations.values():
        config.probabilities = None
    with pytest.raises(
        ValueError,
        match=re.escape(f"Probabilities not set for configuration 'config_0'. Call calculate_probabilities() first."),
    ):
        system.calculate_heat_capacities()


def test_calculate_pressure_properties():
    """Test pressure properties and edge cases (all entropies None, bulk_moduli None, helmholtz_energies None, helmholtz_energies_dV None)."""
    local_config_data = copy.deepcopy(config_data)
    system = make_system(local_config_data, reference_helmholtz_energies)
    system.calculate_pressure_properties(P=0)
    # Test values against expected results
    assert np.allclose(system.pt_properties["0.00_GPa"]["V0"], expected_results.pt_properties["0.00_GPa"]["V0"], equal_nan=True)
    assert np.allclose(system.pt_properties["0.00_GPa"]["G0"], expected_results.pt_properties["0.00_GPa"]["G0"], equal_nan=True)
    assert np.allclose(system.pt_properties["0.00_GPa"]["Sconf"], expected_results.pt_properties["0.00_GPa"]["Sconf"], equal_nan=True)
    assert np.allclose(system.pt_properties["0.00_GPa"]["S0"], expected_results.pt_properties["0.00_GPa"]["S0"], equal_nan=True)
    assert np.allclose(system.pt_properties["0.00_GPa"]["B0"], expected_results.pt_properties["0.00_GPa"]["B0"], equal_nan=True)
    assert np.allclose(system.pt_properties["0.00_GPa"]["CTE"], expected_results.pt_properties["0.00_GPa"]["CTE"], equal_nan=True)
    assert np.allclose(system.pt_properties["0.00_GPa"]["LCTE"], expected_results.pt_properties["0.00_GPa"]["LCTE"], equal_nan=True)
    assert np.allclose(system.pt_properties["0.00_GPa"]["Cp"], expected_results.pt_properties["0.00_GPa"]["Cp"], equal_nan=True)
    for config in system.configurations.values():
        assert np.allclose(config.probabilities_at_P["0.00_GPa"], expected_results.configurations[config.name].probabilities_at_P["0.00_GPa"], equal_nan=True)

    # Test with all entropies set to None
    system2 = make_system(local_config_data, reference_helmholtz_energies)
    system2.configurational_entropies = None
    system2.entropies = None
    system2.calculate_pressure_properties(P=0)
    assert np.allclose(system2.pt_properties["0.00_GPa"]["V0"], expected_results.pt_properties["0.00_GPa"]["V0"], equal_nan=True)
    assert np.allclose(system2.pt_properties["0.00_GPa"]["G0"], expected_results.pt_properties["0.00_GPa"]["G0"], equal_nan=True)
    assert np.all(np.isnan(system2.pt_properties["0.00_GPa"]["Sconf"]))
    assert np.all(np.isnan(system2.pt_properties["0.00_GPa"]["S0"]))
    assert np.allclose(system2.pt_properties["0.00_GPa"]["B0"], expected_results.pt_properties["0.00_GPa"]["B0"], equal_nan=True)
    assert np.allclose(system2.pt_properties["0.00_GPa"]["CTE"], expected_results.pt_properties["0.00_GPa"]["CTE"], equal_nan=True)
    assert np.allclose(system2.pt_properties["0.00_GPa"]["LCTE"], expected_results.pt_properties["0.00_GPa"]["LCTE"], equal_nan=True)
    assert np.all(np.isnan(system2.pt_properties["0.00_GPa"]["Cp"]))
    for config in system2.configurations.values():
        assert np.allclose(config.probabilities_at_P["0.00_GPa"], expected_results.configurations[config.name].probabilities_at_P["0.00_GPa"], equal_nan=True)

    # Test with bulk_moduli set to None
    system3 = make_system(local_config_data, reference_helmholtz_energies)
    system3.bulk_moduli = None
    system3.calculate_pressure_properties(P=0)
    assert np.allclose(system3.pt_properties["0.00_GPa"]["V0"], expected_results.pt_properties["0.00_GPa"]["V0"], equal_nan=True)
    assert np.allclose(system3.pt_properties["0.00_GPa"]["G0"], expected_results.pt_properties["0.00_GPa"]["G0"], equal_nan=True)
    assert np.allclose(system3.pt_properties["0.00_GPa"]["Sconf"], expected_results.pt_properties["0.00_GPa"]["Sconf"], equal_nan=True)
    assert np.allclose(system3.pt_properties["0.00_GPa"]["S0"], expected_results.pt_properties["0.00_GPa"]["S0"], equal_nan=True)
    assert np.all(np.isnan(system3.pt_properties["0.00_GPa"]["B0"]))
    assert np.allclose(system3.pt_properties["0.00_GPa"]["CTE"], expected_results.pt_properties["0.00_GPa"]["CTE"], equal_nan=True)
    assert np.allclose(system3.pt_properties["0.00_GPa"]["LCTE"], expected_results.pt_properties["0.00_GPa"]["LCTE"], equal_nan=True)
    assert np.allclose(system3.pt_properties["0.00_GPa"]["Cp"], expected_results.pt_properties["0.00_GPa"]["Cp"], equal_nan=True)
    for config in system3.configurations.values():
        assert np.allclose(config.probabilities_at_P["0.00_GPa"], expected_results.configurations[config.name].probabilities_at_P["0.00_GPa"], equal_nan=True)

    # Test with helmholtz_energies set to None
    system4 = make_system(local_config_data, reference_helmholtz_energies)
    system4.helmholtz_energies = None
    with pytest.raises(ValueError, match=re.escape("Helmholtz energies not calculated. Call calculate_helmholtz_energies() first.")):
        system4.calculate_pressure_properties(P=0)

    # Test with helmholtz_energies_dV set to None
    system5 = make_system(local_config_data, reference_helmholtz_energies)
    system5.helmholtz_energies_dV = None
    with pytest.raises(
        ValueError,
        match=re.escape("Helmholtz energies derivatives not calculated. Call calculate_helmholtz_energies_dV() first."),
    ):
        system5.calculate_pressure_properties(P=0)


# Plotting Tests
@pytest.mark.parametrize(
    "plot_type",
    [
        "helmholtz_energy_vs_volume",
        "helmholtz_energy_vs_temperature",
        "helmholtz_energy_dV_vs_volume",
        "helmholtz_energy_dV_vs_temperature",
        "helmholtz_energy_d2V2_vs_volume",
        "helmholtz_energy_d2V2_vs_temperature",
        "entropy_vs_volume",
        "entropy_vs_temperature",
        "configurational_entropy_vs_volume",
        "configurational_entropy_vs_temperature",
        "heat_capacity_vs_volume",
        "heat_capacity_vs_temperature",
        "bulk_modulus_vs_volume",
        "bulk_modulus_vs_temperature",
    ],
)
def test_plot_vt_smoke(plot_type):
    """Test that plot_vt runs without error for all supported plot types."""
    local_config_data = copy.deepcopy(config_data)
    system = make_system(local_config_data, reference_helmholtz_energies)
    system.plot_vt(plot_type)


def test_plot_vt_invalid_type():
    """Test that an invalid plot type raises ValueError."""
    local_config_data = copy.deepcopy(config_data)
    system = make_system(local_config_data, reference_helmholtz_energies)
    with pytest.raises(ValueError, match="Invalid plot type"):
        system.plot_vt("not_a_real_plot_type")


@pytest.mark.parametrize(
    "plot_type, attr",
    [
        ("helmholtz_energy_vs_volume", "helmholtz_energies"),
        ("helmholtz_energy_vs_temperature", "helmholtz_energies"),
        ("helmholtz_energy_dV_vs_volume", "helmholtz_energies_dV"),
        ("helmholtz_energy_dV_vs_temperature", "helmholtz_energies_dV"),
        ("helmholtz_energy_d2V2_vs_volume", "helmholtz_energies_d2V2"),
        ("helmholtz_energy_d2V2_vs_temperature", "helmholtz_energies_d2V2"),
        ("entropy_vs_volume", "entropies"),
        ("entropy_vs_temperature", "entropies"),
        ("configurational_entropy_vs_volume", "configurational_entropies"),
        ("configurational_entropy_vs_temperature", "configurational_entropies"),
        ("heat_capacity_vs_volume", "heat_capacities"),
        ("heat_capacity_vs_temperature", "heat_capacities"),
        ("bulk_modulus_vs_volume", "bulk_moduli"),
        ("bulk_modulus_vs_temperature", "bulk_moduli"),
    ],
)
def test_plot_vt_missing_data(plot_type, attr):
    """Test that missing required data for each plot type raises ValueError."""
    local_config_data = copy.deepcopy(config_data)
    system = System(local_config_data)
    setattr(system, attr, None)
    with pytest.raises(ValueError):
        system.plot_vt(plot_type)


@pytest.mark.parametrize(
    "plot_type",
    [
        "helmholtz_energy_vs_volume",
        "helmholtz_energy_dV_vs_volume",
        "helmholtz_energy_d2V2_vs_volume",
        "entropy_vs_volume",
        "configurational_entropy_vs_volume",
        "heat_capacity_vs_volume",
        "bulk_modulus_vs_volume",
    ],
)
def test_plot_vt_selected_temperatures(plot_type):
    """Test that plot_vt works with selected_temperatures argument for relevant plot types."""
    local_config_data = copy.deepcopy(config_data)
    system = make_system(local_config_data, reference_helmholtz_energies)
    system.plot_vt(plot_type, selected_temperatures=np.array([300, 600, 900]))


@pytest.mark.parametrize(
    "plot_type",
    [
        "helmholtz_energy_vs_temperature",
        "helmholtz_energy_dV_vs_temperature",
        "helmholtz_energy_d2V2_vs_temperature",
        "entropy_vs_temperature",
        "configurational_entropy_vs_temperature",
        "heat_capacity_vs_temperature",
        "bulk_modulus_vs_temperature",
    ],
)
def test_plot_vt_selected_volumes(plot_type):
    """Test that plot_vt works with selected_volumes argument."""
    local_config_data = copy.deepcopy(config_data)
    system = make_system(local_config_data, reference_helmholtz_energies)
    system.plot_vt(plot_type, selected_volumes=np.array([100, 150, 200]))


@pytest.mark.parametrize(
    "plot_type",
    [
        "helmholtz_energy_pv_vs_volume",
        "volume_vs_temperature",
        "CTE_vs_temperature",
        "LCTE_vs_temperature",
        "entropy_vs_temperature",
        "configurational_entropy_vs_temperature",
        "heat_capacity_vs_temperature",
        "gibbs_energy_vs_temperature",
        "bulk_modulus_vs_temperature",
        "probability_vs_temperature",
    ],
)
def test_plot_pt_smoke(plot_type):
    """Test that plot_pt runs without error for all supported plot types."""
    local_config_data = copy.deepcopy(config_data)
    system = make_system(local_config_data, reference_helmholtz_energies)
    system.calculate_pressure_properties(P=0)
    system.plot_pt(plot_type)


def test_plot_pt_properties_at_P_missing():
    """Test that plot_pt raises ValueError if pressure properties are not calculated."""
    local_config_data = copy.deepcopy(config_data)
    system = make_system(local_config_data, reference_helmholtz_energies)
    with pytest.raises(ValueError, match="Properties at 0.00 GPa not calculated. Run calculate_pressure_properties() first."):
        system.plot_pt("helmholtz_energy_pv_vs_volume")


def test_plot_pt_invalid_type():
    """Test that an invalid plot type raises ValueError."""
    local_config_data = copy.deepcopy(config_data)
    system = make_system(local_config_data, reference_helmholtz_energies)
    system.calculate_pressure_properties(P=0)
    with pytest.raises(ValueError, match="Invalid plot type"):
        system.plot_pt("not_a_real_plot_type")


@pytest.mark.parametrize(
    "plot_type",
    [
        "helmholtz_energy_pv_vs_volume",
        "volume_vs_temperature",
        "CTE_vs_temperature",
        "LCTE_vs_temperature",
        "entropy_vs_temperature",
        "configurational_entropy_vs_temperature",
        "heat_capacity_vs_temperature",
        "gibbs_energy_vs_temperature",
        "bulk_modulus_vs_temperature",
        "probability_vs_temperature",
    ],
)
def test_plot_pt_missing_data(plot_type):
    """Test that missing required data for each plot type raises ValueError."""
    local_config_data = copy.deepcopy(config_data)
    system = make_system(local_config_data, reference_helmholtz_energies)
    with pytest.raises(ValueError):
        system.plot_pt(plot_type)


@pytest.mark.parametrize(
    "plot_type",
    ["helmholtz_energy_pv_vs_volume"],
)
def test_plot_pt_selected_temperatures(plot_type):
    """Test that plot_pt works with selected_temperatures argument for relevant plot types."""
    local_config_data = copy.deepcopy(config_data)
    system = make_system(local_config_data, reference_helmholtz_energies)
    system.calculate_pressure_properties(P=0)
    system.plot_pt(plot_type, selected_temperatures=np.array([300, 600, 900]))


def test_plot_pt_probabilities_ground_state():
    """Test that plot_pt works with ground_state argument."""
    local_config_data = copy.deepcopy(config_data)
    system = make_system(local_config_data, reference_helmholtz_energies)
    system.calculate_pressure_properties(P=0)
    system.plot_pt("probability_vs_temperature", ground_state="config_0")
