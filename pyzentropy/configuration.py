# Standard Library Imports
import scipy.constants

# Third-Party Library Imports
import numpy as np
import plotly.graph_objects as go

# Local Imports
from pyzentropy.plotly_utils import format_plot

BOLTZMANN_CONSTANT = scipy.constants.Boltzmann / scipy.constants.electron_volt  # The Boltzmann constant in eV/K


class Configuration:
    """
    Represents a single configuration and its thermodynamic properties,
    dependent on temperature and volume.
    
    This class stores Helmholtz free energies and related thermodynamic
    quantities for a given configuration, and provides methods for computing
    derived properties such as internal energy. It is used as a component of
    a larger thermodynamic `System`.

    Notes:
        - All array-valued thermodynamic quantities follow the shape ``(n_temperatures, n_volumes)``.
        - Temperatures are in Kelvin.
        - Volumes are in Å³.
        - Energies are in eV and are extensive with respect to the configuration size (`number_of_atoms`).
        - Entropies and heat capacities are in eV/K and are extensive with respect to the configuration size (`number_of_atoms`).

    Args:
        name (str):
            Name or label of the configuration.
        multiplicity (int):
            Degeneracy factor used in statistical weighting.
        number_of_atoms (int):
            Number of atoms in the configuration.
        volumes (np.ndarray):
            Volume grid of shape ``(n_volumes,)``.
        temperatures (np.ndarray):
            Temperature grid of shape ``(n_temperatures,)``.
        helmholtz_energies (np.ndarray):
            Helmholtz free energies :math:`F_k(T, V)`.
        helmholtz_energies_dV (np.ndarray):
            First volume derivatives :math:`\partial F_k / \partial V`.
        helmholtz_energies_d2V2 (np.ndarray):
            Second volume derivatives :math:`\partial^2 F_k / \partial V^2`.
        reference_helmholtz_energies (np.ndarray): 
            Reference Helmholtz free energies to shift by.
        entropies (np.ndarray):
            Entropies :math:`S_k(T, V)`. Defaults to None.
        heat_capacities (np.ndarray):
            Heat capacities at constant volume :math:`C_{V,k}(T, V)`. Defaults to None.
            
    Raises:
        ValueError: If any input array does not match the expected shape.
            
    Attributes:
        name (str):
            Name or label of the configuration.
        multiplicity (int):
            Degeneracy factor used in statistical weighting.
        number_of_atoms (int):
            Number of atoms in the configuration.
        volumes (np.ndarray):
            Volume grid of shape ``(n_volumes,)``.
        temperatures (np.ndarray):
            Temperature grid of shape ``(n_temperatures,)``.
        
        helmholtz_energies (np.ndarray):
            Helmholtz free energies :math:`F_k(T, V)`.
        helmholtz_energies_dV (np.ndarray):
            First volume derivatives :math:`\partial F_k / \partial V`.
        helmholtz_energies_d2V2 (np.ndarray):
            Second volume derivatives :math:`\partial^2 F_k / \partial V^2`.
        entropies (np.ndarray):
            Entropies :math:`S_k(T, V)`.
        heat_capacities (np.ndarray):
            Heat capacities at constant volume :math:`C_{V,k}(T, V)`.
        
        internal_energies (np.ndarray):
            Internal energies :math:`E_k(T, V) = F_k(T, V) + T S_k(T, V)`, computed from
            stored thermodynamic data.
        partition_functions (np.ndarray):
            Configuration partition functions :math:`Z_k(T, V) = \exp(-F_k(T, V)/(k_B T))`,
            computed by a `System` object during ensemble calculations.
        probabilities (np.ndarray):
            Configuration probabilities :math:`p_k(T, V) = Z_k(T, V) / Z(T, V)`,
            computed by a `System` object during ensemble calculations.
    """

    def __init__(
        self,
        name: str,
        multiplicity: int,
        number_of_atoms: int,
        volumes: np.ndarray,
        temperatures: np.ndarray,
        helmholtz_energies: np.ndarray,
        helmholtz_energies_dV: np.ndarray,
        helmholtz_energies_d2V2: np.ndarray,
        entropies: np.ndarray = None,
        heat_capacities: np.ndarray = None,
    ):

        expected_shape = (len(temperatures), len(volumes))
        for arr, arr_name in [
            (helmholtz_energies, "helmholtz_energies"),
            (helmholtz_energies_dV, "helmholtz_energies_dV"),
            (helmholtz_energies_d2V2, "helmholtz_energies_d2V2"),
            (entropies, "entropies"),
            (heat_capacities, "heat_capacities"),
        ]:
            if arr is not None and arr.shape != expected_shape:
                raise ValueError(
                    f"{arr_name} must have shape {expected_shape} (n_temperatures, n_volumes), "
                    f"where n_temperatures = len(temperatures) = {len(temperatures)} and "
                    f"n_volumes = len(volumes) = {len(volumes)}. "
                    f"Received array with shape {arr.shape}. "
                    "Please ensure your input array matches the expected dimensions."
                )

        self.name = name
        self.multiplicity = multiplicity
        self.number_of_atoms = number_of_atoms
        self.volumes = volumes
        self.temperatures = temperatures
        self.helmholtz_energies = helmholtz_energies
        self.helmholtz_energies_dV = helmholtz_energies_dV
        self.helmholtz_energies_d2V2 = helmholtz_energies_d2V2
        self.entropies = entropies
        self.heat_capacities = heat_capacities
        self.internal_energies = None
        self.partition_functions = None
        self.probabilities = None
        self.probabilities_at_P = {}

        if self.entropies is not None:
            self.calculate_internal_energies()

    def calculate_internal_energies(self) -> None:
        """
        Calculate internal energies using the formula: E = F + T*S.
        """
        self.internal_energies = self.helmholtz_energies + self.temperatures[:, np.newaxis] * self.entropies

    def plot_vt(
        self,
        type: str,
        selected_temperatures: np.ndarray = None,
        selected_volumes: np.ndarray = None,
        width: int = 650,
        height: int = 600,
    ):
        """
        Plot thermodynamic properties as a function of temperature or volume.

        Args:
            type (str): Type of plot to generate.
            selected_temperatures (np.ndarray, optional): Temperatures to plot (for fixed temperature plots).
            selected_volumes (np.ndarray, optional): Volumes to plot (for fixed volume plots).
            width (int, optional): Plot width in pixels.
            height (int, optional): Plot height in pixels.

        Raises:
            ValueError: If an invalid plot type is provided.
            ValueError: If internal_energies is None when calling internal energy plots
            ValueError: If entropies is None when calling entropy plots
            ValueError: If heat_capacities is None when calling heat capacity plots

        Returns:
            plotly.graph_objects.Figure: The generated plotly figure.
        """

        plot_data = {
            "helmholtz_energy_vs_volume": {
                "x": self.volumes,
                "y": self.helmholtz_energies,
                "fixed": "temperature",
                "ylabel": "F (eV)",
            },
            "helmholtz_energy_vs_temperature": {
                "x": self.temperatures,
                "y": self.helmholtz_energies,
                "fixed": "volume",
                "ylabel": "F (eV)",
            },
            "internal_energy_vs_volume": {
                "x": self.volumes,
                "y": self.internal_energies,
                "fixed": "temperature",
                "ylabel": "E (eV)",
            },
            "internal_energy_vs_temperature": {
                "x": self.temperatures,
                "y": self.internal_energies,
                "fixed": "volume",
                "ylabel": "E (eV)",
            },
            "entropy_vs_volume": {
                "x": self.volumes,
                "y": self.entropies,
                "fixed": "temperature",
                "ylabel": "S (eV/K)",
            },
            "entropy_vs_temperature": {
                "x": self.temperatures,
                "y": self.entropies,
                "fixed": "volume",
                "ylabel": "S (eV/K)",
            },
            "heat_capacity_vs_volume": {
                "x": self.volumes,
                "y": self.heat_capacities,
                "fixed": "temperature",
                "ylabel": "C<sub>v</sub> (eV/K)",
            },
            "heat_capacity_vs_temperature": {
                "x": self.temperatures,
                "y": self.heat_capacities,
                "fixed": "volume",
                "ylabel": "C<sub>v</sub> (eV/K)",
            },
        }

        if type not in plot_data:
            raise ValueError(f"Invalid plot type. Choose from: {', '.join(plot_data.keys())}")

        data = plot_data[type]

        # Raise if required data is None for the selected plot type
        if type in ["internal_energy_vs_volume", "internal_energy_vs_temperature"] and self.internal_energies is None:
            raise ValueError(
                "internal_energies is None. Please call calculate_internal_energies() before plotting internal energy."
            )
        if type in ["entropy_vs_volume", "entropy_vs_temperature"] and self.entropies is None:
            raise ValueError("entropies is None. Please provide entropy data before plotting entropy.")
        if type in ["heat_capacity_vs_volume", "heat_capacity_vs_temperature"] and self.heat_capacities is None:
            raise ValueError(
                "heat_capacities is None. Please provide heat capacity data before plotting heat capacity."
            )

        x_data = data["x"]
        y_data = data["y"]
        fixed_by = data["fixed"]
        y_label_template = data["ylabel"]

        # Select indices and labels
        if fixed_by == "temperature":
            if selected_temperatures is None:
                selected = np.linspace(self.temperatures.min(), self.temperatures.max(), 10)
            else:
                selected = selected_temperatures
            indices = self._get_closest_indices(self.temperatures, selected)
            legend_vals = self.temperatures[indices]
            legend_fmt = lambda t: f"{int(t)} K" if t % 1 == 0 else f"{t} K"
        elif fixed_by == "volume":
            if selected_volumes is None:
                selected = np.linspace(self.volumes.min(), self.volumes.max(), 10)
            else:
                selected = selected_volumes
            indices = self._get_closest_indices(self.volumes, selected)
            legend_vals = self.volumes[indices]
            legend_fmt = lambda v: f"{v:.2f} Å³"

        # Create plot
        fig = go.Figure()
        for i, val in zip(indices, legend_vals):
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data[i, :] if fixed_by == "temperature" else y_data[:, i],
                    mode="lines",
                    name=legend_fmt(val),
                )
            )

        x_label = "Temperature (K)" if "temperature" in type else f"Volume (Å³)"
        y_label = y_label_template

        format_plot(fig, x_label, y_label, width=width, height=height)
        return fig

    def _get_closest_indices(self, values: np.ndarray, targets: np.ndarray) -> list:
        """
        Find indices of the closest matches in `values` for each target in `targets`.

        Args:
            values (np.ndarray): Array to search.
            targets (np.ndarray): Target values to match.

        Returns:
            list: Indices of closest matches.
        """
        return [np.argmin(np.abs(values - target)) for target in targets]
