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
    Represents a thermodynamic configuration with methods for calculating
    internal energies, partition functions, and plotting thermodynamic properties.

    Attributes:
        name (str): Name of the configuration.
        multiplicity (int): Multiplicity of the configuration.
        number_of_atoms (int): Number of atoms in the configuration.
        volumes (np.ndarray): Array of volumes (shape: [n_volumes]).
        temperatures (np.ndarray): Array of temperatures (shape: [n_temperatures]).
        helmholtz_energies (np.ndarray): Helmholtz energies (shape: [n_temperatures, n_volumes]).
        helmholtz_energies_dV (np.ndarray): First derivatives of Helmholtz energies (shape: [n_temperatures, n_volumes]).
        helmholtz_energies_d2V2 (np.ndarray): Second derivatives of Helmholtz energies (shape: [n_temperatures, n_volumes]).
        entropies (np.ndarray): Entropies (shape: [n_temperatures, n_volumes]).
        heat_capacities (np.ndarray): Heat capacities (shape: [n_temperatures, n_volumes]).
        internal_energies (np.ndarray): Internal energies (calculated, shape: [n_temperatures, n_volumes]).
        partition_functions (np.ndarray): Partition functions (calculated, shape: [n_temperatures, n_volumes]).
        probabilities (np.ndarray): Probabilities (calculated via System class, shape: [n_temperatures, n_volumes]).
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
        entropies: np.ndarray,
        heat_capacities: np.ndarray,
    ):
        """
        Initialize a Configuration object and check input array shapes.

        Args:
            name (str): Name of the configuration.
            multiplicity (int): Multiplicity of the configuration.
            number_of_atoms (int): Number of atoms in the configuration.
            volumes (np.ndarray): Array of volumes (shape: [n_volumes]).
            temperatures (np.ndarray): Array of temperatures (shape: [n_temperatures]).
            helmholtz_energies (np.ndarray): Helmholtz energies (shape: [n_temperatures, n_volumes]).
            helmholtz_energies_dV (np.ndarray): First derivatives of Helmholtz energies (shape: [n_temperatures, n_volumes]).
            helmholtz_energies_d2V2 (np.ndarray): Second derivatives of Helmholtz energies (shape: [n_temperatures, n_volumes]).
            entropies (np.ndarray): Entropies (shape: [n_temperatures, n_volumes]).
            heat_capacities (np.ndarray): Heat capacities (shape: [n_temperatures, n_volumes]).
        Raises:
            ValueError: If any input array does not match the expected shape.
        """
        expected_shape = (len(temperatures), len(volumes))
        for arr, arr_name in [
            (helmholtz_energies, "helmholtz_energies"),
            (helmholtz_energies_dV, "helmholtz_energies_dV"),
            (helmholtz_energies_d2V2, "helmholtz_energies_d2V2"),
            (entropies, "entropies"),
            (heat_capacities, "heat_capacities"),
        ]:
            if arr.shape != expected_shape:
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
        self.probabilities_at_P = None

        self.calculate_internal_energies()

    def calculate_internal_energies(self) -> None:
        """
        Calculate internal energies using the formula: E = F + T*S.
        """
        self.internal_energies = self.helmholtz_energies + self.temperatures[:, np.newaxis] * self.entropies

    def calculate_partition_functions(self, reference_helmholtz_energies: np.ndarray) -> None:
        """
        Calculate the partition function for each state, using a reference Helmholtz energy.

        Args:
            reference_helmholtz_energies (np.ndarray): Reference Helmholtz energies to shift by.
        """
        helmholtz_energy_shifted = self.helmholtz_energies - reference_helmholtz_energies
        exponent = -helmholtz_energy_shifted / (BOLTZMANN_CONSTANT * self.temperatures[:, np.newaxis])
        with np.errstate(over="ignore"):
            partition_functions = np.exp(exponent)
        partition_functions[np.isinf(partition_functions)] = np.nan
        self.partition_functions = partition_functions

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

    def plot(
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

        Returns:
            plotly.graph_objects.Figure: The generated plotly figure.
        """
        plot_data = {
            "helmholtz_energy_vs_volume": {
                "x": self.volumes,
                "y": self.helmholtz_energies,
                "fixed": "temperature",
                "ylabel": "F (eV/{unit})",
            },
            "helmholtz_energy_vs_temperature": {
                "x": self.temperatures,
                "y": self.helmholtz_energies,
                "fixed": "volume",
                "ylabel": "F (eV/{unit})",
            },
            "internal_energy_vs_volume": {
                "x": self.volumes,
                "y": self.internal_energies,
                "fixed": "temperature",
                "ylabel": "E (eV/{unit})",
            },
            "internal_energy_vs_temperature": {
                "x": self.temperatures,
                "y": self.internal_energies,
                "fixed": "volume",
                "ylabel": "E (eV/{unit})",
            },
            "entropy_vs_volume": {
                "x": self.volumes,
                "y": self.entropies,
                "fixed": "temperature",
                "ylabel": "S (eV/K/{unit})",
            },
            "entropy_vs_temperature": {
                "x": self.temperatures,
                "y": self.entropies,
                "fixed": "volume",
                "ylabel": "S (eV/K/{unit})",
            },
            "heat_capacity_vs_volume": {
                "x": self.volumes,
                "y": self.heat_capacities,
                "fixed": "temperature",
                "ylabel": "C<sub>v</sub> (eV/K/{unit})",
            },
            "heat_capacity_vs_temperature": {
                "x": self.temperatures,
                "y": self.heat_capacities,
                "fixed": "volume",
                "ylabel": "C<sub>v</sub> (eV/K/{unit})",
            },
        }

        if type not in plot_data:
            raise ValueError(f"Invalid plot type. Choose from: {', '.join(plot_data.keys())}")

        data = plot_data[type]
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

        unit = "atom" if self.number_of_atoms == 1 else f"{self.number_of_atoms} atoms"
        x_label = "Temperature (K)" if "temperature" in type else f"Volume (Å³/{unit})"
        y_label = y_label_template.format(unit=unit)

        format_plot(fig, x_label, y_label, width=width, height=height)
        return fig
