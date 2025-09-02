# Third-Party Library Imports
import numpy as np
import plotly.graph_objects as go
import scipy.constants
from scipy.optimize import root_scalar
from scipy.interpolate import PchipInterpolator

# PyZentropy Imports
from pyzentropy.plotly_utils import format_plot
from pyzentropy.configuration import Configuration

EV_PER_CUBIC_ANGSTROM_TO_GPA = 160.21766208  # 1 eV/Å^3  = 160.21766208 GPa
BOLTZMANN_CONSTANT = scipy.constants.Boltzmann / scipy.constants.electron_volt  # The Boltzmann constant in eV/K


class System:
    """
    Represents a thermodynamic system composed of multiple configurations.
    Provides methods to calculate ensemble properties, thermodynamic quantities,
    and to generate various plots for analysis.
    """

    def __init__(self, configurations: dict[str, Configuration]):
        """
        Initialize the System with a dictionary of Configuration objects.

        Args:
            configurations (dict[str, Configuration]): Dictionary mapping configuration names to Configuration objects.

        Raises:
            ValueError: If configurations have inconsistent number of atoms, volumes, or temperatures.
        """
        self.configurations = configurations

        # Get reference values from the first configuration
        first_config = next(iter(configurations.values()))
        ref_num_atoms = first_config.number_of_atoms
        ref_volumes = first_config.volumes
        ref_temperatures = first_config.temperatures

        # Ensure all configurations are consistent in number of atoms, temperatures, and volumes
        for name, config in configurations.items():
            if config.number_of_atoms != ref_num_atoms:
                raise ValueError("Number of atoms for configurations are not the same.")
            if not np.array_equal(config.volumes, ref_volumes):
                raise ValueError("Volumes for configurations are not the same.")
            if not np.array_equal(config.temperatures, ref_temperatures):
                raise ValueError("Temperatures for configurations are not the same.")

        self.number_of_atoms = ref_num_atoms
        self.temperatures = ref_temperatures
        self.volumes = ref_volumes

        # Dimensions
        self._n_temps = len(self.temperatures)
        self._n_vols = len(self.volumes)

        # Calculated properties (V, T)
        vt_properties = [
            "partition_functions",
            "helmholtz_energies",
            "helmholtz_energies_dV",
            "helmholtz_energies_d2V2",
            "entropies",
            "configurational_entropies",
            "bulk_moduli",
            "heat_capacities",
        ]
        for attr in vt_properties:
            setattr(self, attr, None)

        # Calculated properties (P, T)
        # Dictionary to store pressure-temperature dependent properties
        # Format: {f"{P:.2f}_GPa": {"helmholtz_energy_pv", "V0", "G0", "S0", "Sconf", "B0", "CTE", "LCTE", "Cp"}}
        self.pt_properties = {}

        # Initialize pressure-temperature and volume-temperature diagrams
        self.pt_phase_diagram = {}
        self.vt_phase_diagram = {}

    def calculate_partition_functions(self) -> None:
        """
        Calculate the total partition function for the system by summing over all configurations.
        Each configuration's partition function is weighted by its multiplicity.

        Raises:
            ValueError: If any configuration is missing its partition function.
        """
        # Initialize partition functions array
        partition_functions = np.zeros((self._n_temps, self._n_vols))

        # Check that the partition functions are calculated for each configuration
        with np.errstate(invalid="ignore", over="ignore"):
            for config in self.configurations.values():
                if config.partition_functions is None:
                    raise ValueError(f"partition_functions not set for configuration '{config.name}'.")
                partition_functions += config.partition_functions * config.multiplicity

            # Replace inf values with nan
            partition_functions[np.isinf(partition_functions)] = np.nan
        self.partition_functions = partition_functions

    def calculate_probabilities(self) -> None:
        """
        Calculate the probability of each configuration at every temperature and volume,
        based on the partition function.

        Raises:
            ValueError: If the system partition function is not calculated.
        """
        # Check that system partition functions are calculated
        if self.partition_functions is None:
            raise ValueError("Partition functions not calculated. Call calculate_partition_functions() first.")

        # Calculate probabilities for each configuration
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            for config in self.configurations.values():
                probabilities = config.multiplicity * config.partition_functions / self.partition_functions
                config.probabilities = probabilities

    def calculate_helmholtz_energies(self, reference_helmholtz_energies: np.ndarray) -> None:
        """
        Calculate the Helmholtz energy for the system at each temperature and volume.

        Args:
            reference_helmholtz_energies (np.ndarray): Reference Helmholtz energies to shift the calculated values.

        Raises:
            ValueError: If the system partition function is not calculated.
        """
        # Check that system partition functions are calculated
        if self.partition_functions is None:
            raise ValueError("Partition functions not calculated. Call calculate_partition_functions() first.")

        # Calculate Helmholtz energies
        self.helmholtz_energies = -BOLTZMANN_CONSTANT * self.temperatures[:, np.newaxis] * np.log(self.partition_functions) + reference_helmholtz_energies

    def calculate_helmholtz_energies_dV(self) -> None:
        """
        Calculate the first derivative of the Helmholtz energy with respect to volume.

        Raises:
            ValueError: If probabilities or dF/dV are missing for any configuration.
        """
        # Initialize dF/dV array
        self.helmholtz_energies_dV = np.zeros((self._n_temps, self._n_vols))

        # Check that probabilities and dF/dV are calculated for each configuration
        for config in self.configurations.values():
            if config.probabilities is None:
                raise ValueError(f"Probabilities not set for configuration '{config.name}'. Call calculate_probabilities() first.")
            if config.helmholtz_energies_dV is None:
                raise ValueError(f"helmholtz_energies_dV not set for configuration '{config.name}'.")
            self.helmholtz_energies_dV += config.probabilities * config.helmholtz_energies_dV

        # Replace inf values with nan
        self.helmholtz_energies_dV[np.isinf(self.helmholtz_energies_dV)] = np.nan

    # TODO: Change to use Sk and pk only
    def calculate_entropies(self) -> None:
        """
        Calculate the configurational entropy, total entropy, and internal energy for the system.

        Raises:
            ValueError: If any of the following are not calculated:
                - Helmholtz energies for the system (self.helmholtz_energies)
                - Probabilities for any configuration (config.probabilities)
                - Internal energies for any configuration (config.internal_energies)
                - Helmholtz energies for any configuration (config.helmholtz_energies)
        """
        # Initialize arrays
        self.configurational_entropies = np.zeros((self._n_temps, self._n_vols))
        self.entropies = np.zeros((self._n_temps, self._n_vols))
        self.internal_energies = np.zeros((self._n_temps, self._n_vols))
        average_F = np.zeros((self._n_temps, self._n_vols))

        # Check that the system has Helmholtz energies calculated
        if self.helmholtz_energies is None:
            raise ValueError("Helmholtz energies not calculated. Call calculate_helmholtz_energies() first.")

        # Check that the probabilities, internal energies, and helmholtz energies are calculated for each configuration
        # Accumulate contributions from each configuration
        for config in self.configurations.values():
            if config.probabilities is None:
                raise ValueError(f"Probabilities not set for configuration '{config.name}'. Call calculate_probabilities() first.")
            if config.internal_energies is None:
                raise ValueError(f"Internal energies not set for configuration '{config.name}'.")
            if config.helmholtz_energies is None:
                raise ValueError(f"Helmholtz energies not set for configuration '{config.name}'.")
            self.internal_energies += config.probabilities * config.internal_energies
            average_F += config.probabilities * config.helmholtz_energies

        # Final entropy calculations
        T = self.temperatures[:, np.newaxis]
        self.configurational_entropies = (average_F - self.helmholtz_energies) / T
        self.entropies = (-self.helmholtz_energies / T) + (self.internal_energies / T)

    def calculate_bulk_moduli(self) -> None:
        """
        Calculate the bulk modulus for the system at each temperature and volume.

        Raises:
            ValueError: If probabilities, dF/dV, or d2F/dV2 are missing for any configuration.
        """
        # Initialize arrays
        bulkeach = np.zeros((self._n_temps, self._n_vols))
        bulkconf = np.zeros((self._n_temps, self._n_vols))
        self.bulk_moduli = np.zeros((self._n_temps, self._n_vols))

        # Loop over temperatures
        for i in range(self._n_temps):
            ened2_sum = np.zeros(self._n_vols)
            ened1_sum = np.zeros(self._n_vols)
            ened1_sq_sum = np.zeros(self._n_vols)

            # Check that the probabilities, dF/dV, and d2F/dV2 are calculated for each configuration
            for config in self.configurations.values():
                if config.probabilities is None:
                    raise ValueError(f"Probabilities not set for configuration '{config.name}'. Call calculate_probabilities() first.")
                if config.helmholtz_energies_dV is None:
                    raise ValueError(f"helmholtz_energies_dV not set for configuration '{config.name}'.")
                if config.helmholtz_energies_d2V2 is None:
                    raise ValueError(f"helmholtz_energies_d2V2 not set for configuration '{config.name}'.")

                frac = config.probabilities[i]
                ened1 = config.helmholtz_energies_dV[i]
                ened2 = config.helmholtz_energies_d2V2[i]
                vv = self.volumes
                ened2_sum += frac * vv * ened2
                ened1_sum += frac * ened1
                ened1_sq_sum += frac * ened1**2

            bulkeach[i] = ened2_sum
            bulkconf[i] = self.volumes * (ened1_sum**2 - ened1_sq_sum) / (BOLTZMANN_CONSTANT * self.temperatures[i])
            self.bulk_moduli[i] = (bulkeach[i] + bulkconf[i]) * EV_PER_CUBIC_ANGSTROM_TO_GPA

    def calculate_helmholtz_energies_d2V2(self) -> None:
        """
        Calculate the second derivative of the Helmholtz free energy with respect to volume.

        Raises:
            ValueError: If the system bulk moduli are not calculated.
        """
        # Check that the system bulk moduli are calculated
        if self.bulk_moduli is None:
            raise ValueError("Bulk moduli not calculated. Call calculate_bulk_moduli() first.")
        self.helmholtz_energies_d2V2 = self.bulk_moduli / EV_PER_CUBIC_ANGSTROM_TO_GPA / self.volumes

    def calculate_heat_capacities(self) -> None:
        """
        Calculate the heat capacity for the system at each temperature and volume.

        Raises:
            ValueError: If probabilities, heat capacities, or internal energies are missing for any configuration.
        """
        # Initialize terms
        first_term = np.zeros((self._n_temps, self._n_vols))
        second_term = np.zeros((self._n_temps, self._n_vols))
        third_term = np.zeros((self._n_temps, self._n_vols))
        factor = 1 / (BOLTZMANN_CONSTANT * self.temperatures[:, np.newaxis] ** 2)

        # Check that the probabilities, heat capacities, and internal energies are calculated for each configuration
        # Accumulate contributions from each configuration
        for config in self.configurations.values():
            if config.probabilities is None:
                raise ValueError(f"Probabilities not set for configuration '{config.name}'. Call calculate_probabilities() first.")
            if config.heat_capacities is None:
                raise ValueError(f"Heat capacities not set for configuration '{config.name}'.")
            if config.internal_energies is None:
                raise ValueError(f"Internal energies not set for configuration '{config.name}'.")
            first_term += config.probabilities * config.heat_capacities
            second_term += config.probabilities * config.internal_energies**2
            third_term += config.probabilities * config.internal_energies

        # Final heat capacity calculation
        self.heat_capacities = first_term + factor * second_term - factor * third_term**2

    def calculate_pressure_properties(self, P: float) -> None:
        """
        Calculate pressure-dependent properties (V0, G0, S0, Sconf, B0, CTE, LCTE, Cp) at a given pressure.

        Args:
            P (float): Pressure in GPa.

        Raises:
            ValueError: If any of the following are not calculated:
                - Helmholtz energies (self.helmholtz_energies)
                - Helmholtz energies first derivative (self.helmholtz_energies_dV)
                - Probabilities for any configuration (config.probabilities)
        """

        # Check required attributes
        if self.helmholtz_energies is None:
            raise ValueError("Helmholtz energies not calculated. Call calculate_helmholtz_energies() first.")
        if self.helmholtz_energies_dV is None:
            raise ValueError("Helmholtz energies derivatives not calculated. Call calculate_helmholtz_energies_dV() first.")
        self.pt_properties[f"{P:.2f}_GPa"] = {
            "helmholtz_energy_pv": None,
            "V0": None,
            "G0": None,
            "S0": None,
            "Sconf": None,
            "B0": None,
            "CTE": None,
            "LCTE": None,
            "Cp": None,
        }
        P_eV_per_A3 = P / EV_PER_CUBIC_ANGSTROM_TO_GPA  # Convert pressure from GPa to eV/Å^3

        V0_array = []
        G0_array = []
        S0_array = []
        Sconf_array = []
        B0_array = []

        for i in range(self._n_temps):
            volumes = self.volumes
            derivatives = self.helmholtz_energies_dV[i, :]
            helmholtz_energies = self.helmholtz_energies[i, :]
            configurational_entropies = self.configurational_entropies[i, :] if self.configurational_entropies is not None else np.full_like(volumes, np.nan)
            entropies = self.entropies[i, :] if self.entropies is not None else np.full_like(volumes, np.nan)
            bulk_moduli = self.bulk_moduli[i, :] if self.bulk_moduli is not None else np.full_like(volumes, np.nan)

            # Filter valid data
            valid_helmholtz_indices = ~np.isnan(derivatives) & ~np.isnan(helmholtz_energies)
            filtered_helmholtz_volumes = volumes[valid_helmholtz_indices]
            filtered_derivatives = derivatives[valid_helmholtz_indices]
            filtered_helmholtz_energies = helmholtz_energies[valid_helmholtz_indices]

            valid_entropy_indices = ~np.isnan(configurational_entropies) & ~np.isnan(entropies)
            filtered_entropy_volumes = volumes[valid_entropy_indices]
            filtered_configurational_entropies = configurational_entropies[valid_entropy_indices]
            filtered_entropies = entropies[valid_entropy_indices]

            valid_bulk_indices = ~np.isnan(bulk_moduli)
            filtered_bulk_volumes = volumes[valid_bulk_indices]
            filtered_bulk_moduli = bulk_moduli[valid_bulk_indices]

            # Skip if not enough valid points
            if len(filtered_helmholtz_volumes) < 5:
                V0_array.append(np.nan)
                G0_array.append(np.nan)
                Sconf_array.append(np.nan)
                S0_array.append(np.nan)
                B0_array.append(np.nan)
                continue

            # Interpolators
            dfdv_interpolator = PchipInterpolator(filtered_helmholtz_volumes, filtered_derivatives + P_eV_per_A3, extrapolate=True)
            helmholtz_energy_interpolator = PchipInterpolator(
                filtered_helmholtz_volumes,
                filtered_helmholtz_energies + P_eV_per_A3 * filtered_helmholtz_volumes,  # F + PV
                extrapolate=True,
            )

            # Find minimum of F + PV (root of dF/dV + P)
            try:
                result = root_scalar(
                    dfdv_interpolator,
                    bracket=(filtered_helmholtz_volumes.min(), filtered_helmholtz_volumes.max()),
                    method="brentq",
                )
                min_x = result.root
                V0_array.append(min_x)
                G0_array.append(helmholtz_energy_interpolator(min_x))
            except ValueError as e:
                V0_array.append(np.nan)
                G0_array.append(np.nan)
                Sconf_array.append(np.nan)
                S0_array.append(np.nan)
                B0_array.append(np.nan)
                continue

            # Interpolate entropy at V0
            if len(filtered_entropy_volumes) >= 5:
                try:
                    Sconf_interp = PchipInterpolator(filtered_entropy_volumes, filtered_configurational_entropies, extrapolate=True)
                    S0_interp = PchipInterpolator(filtered_entropy_volumes, filtered_entropies, extrapolate=True)
                    Sconf_array.append(Sconf_interp(min_x))
                    S0_array.append(S0_interp(min_x))
                except Exception:
                    Sconf_array.append(np.nan)
                    S0_array.append(np.nan)
            else:
                Sconf_array.append(np.nan)
                S0_array.append(np.nan)

            # Interpolate bulk modulus at V0
            if len(filtered_bulk_volumes) >= 5:
                try:
                    B0_interp = PchipInterpolator(filtered_bulk_volumes, filtered_bulk_moduli, extrapolate=True)
                    B0_array.append(B0_interp(min_x))
                except Exception:
                    B0_array.append(np.nan)
            else:
                B0_array.append(np.nan)

        # Store results in pt_properties
        self.pt_properties[f"{P:.2f}_GPa"] = {
            "helmholtz_energy_pv": self.helmholtz_energies + P_eV_per_A3 * self.volumes,
            "V0": np.array(V0_array),
            "G0": np.array(G0_array),
            "S0": np.array(S0_array),
            "Sconf": np.array(Sconf_array),
            "B0": np.array(B0_array),
        }

        # Calculate CTE and LCTE
        dV_dT = np.diff(self.pt_properties[f"{P:.2f}_GPa"]["V0"]) / np.diff(self.temperatures)
        self.pt_properties[f"{P:.2f}_GPa"]["CTE"] = 1 / self.pt_properties[f"{P:.2f}_GPa"]["V0"][:-1] * dV_dT * 1e6
        self.pt_properties[f"{P:.2f}_GPa"]["LCTE"] = self.pt_properties[f"{P:.2f}_GPa"]["CTE"] / 3

        # Calculate heat capacity
        if np.all(np.isnan(self.pt_properties[f"{P:.2f}_GPa"]["S0"])):
            self.pt_properties[f"{P:.2f}_GPa"]["Cp"] = np.full(self._n_temps - 1, np.nan)
        else:
            dS_dT = np.diff(self.pt_properties[f"{P:.2f}_GPa"]["S0"]) / np.diff(self.temperatures)
            self.pt_properties[f"{P:.2f}_GPa"]["Cp"] = self.temperatures[:-1] * dS_dT

        # Calculate probabilities at P for each configuration
        for config in self.configurations.values():
            if config.probabilities is None:
                raise ValueError(f"Probabilities not set for configuration '{config.name}'. Call calculate_probabilities() first.")
            prob_at_V0 = np.full(self._n_temps, np.nan)
            for i in range(self._n_temps):
                probabilities = config.probabilities[i, :]
                valid_probability_indices = ~np.isnan(probabilities)
                filtered_volumes = self.volumes[valid_probability_indices]
                filtered_probabilities = probabilities[valid_probability_indices]
                if len(filtered_volumes) < 2:
                    continue
                probabilities_interpolator = PchipInterpolator(filtered_volumes, filtered_probabilities, extrapolate=True)
                try:
                    prob_at_V0[i] = probabilities_interpolator(self.pt_properties[f"{P:.2f}_GPa"]["V0"][i])
                except Exception:
                    prob_at_V0[i] = np.nan
            config.probabilities_at_P[f"{P:.2f}_GPa"] = prob_at_V0

    def calculate_phase_diagrams(self, ground_state: str, dP: float = 0.2, volume_step_size: float = 1e-4, atol: float = 1e-6) -> None:
        """
        Calculate pressure-temperature and volume-temperature phase diagrams for the system.

        Args:
            ground_state (str): Name of the ground state configuration.
            dP (float): Pressure increment in GPa. Default is 0.2 GPa.
            volume_step_size (float): Step size for volume when searching for the miscibility gap. Default is 1e-4.
            atol (float): Absolute tolerance for convergence in the common tangent search. Default is 1e-6.

        Raises:
            ValueError: If any of the following are not calculated:
                - Helmholtz energies second derivative (helmholtz_energies_d2V2)
                - Helmholtz energies first derivative (helmholtz_energies_dV)
                - Helmholtz energies (helmholtz_energies)
                - Probabilities for any configuration
        """

        if self.helmholtz_energies_d2V2 is None:
            raise ValueError("Helmholtz energies second derivative not calculated. Call calculate_helmholtz_energies_d2V2() first.")
        if self.helmholtz_energies_dV is None:
            raise ValueError("Helmholtz energies first derivative not calculated. Call calculate_helmholtz_energies_dV() first.")
        if self.helmholtz_energies is None:
            raise ValueError("Helmholtz energies not calculated. Call calculate_helmholtz_energies() first.")
        for name, config in self.configurations.items():
            if config.probabilities is None:
                raise ValueError(f"Probabilities not set for configuration '{name}'. Call calculate_probabilities() first.")

        # Initialize phase diagram containers
        self.pt_phase_diagram = {
            "second_order": {"P": np.array([]), "T": np.array([])},
        }
        self.vt_phase_diagram = {
            "first_order": {"V_left": np.array([]), "V_right": np.array([]), "T": np.array([])},
            "second_order": {"V": np.array([]), "T": np.array([])},
        }

        # Find second order phase transition points (where ground state probability crosses 50%)
        P = 0.0
        while True:
            try:
                self.calculate_pressure_properties(P)
                gs_probabilities = self.configurations[ground_state].probabilities_at_P[f"{P:.2f}_GPa"]
                V0 = self.pt_properties[f"{P:.2f}_GPa"]["V0"]

                # Create interpolators for smooth curves
                interp_probabilities = PchipInterpolator(self.temperatures, gs_probabilities)
                interp_V0 = PchipInterpolator(self.temperatures, V0)

                # Find where probability crosses 50% (phase transition)
                roots = []
                for i in range(len(self.temperatures) - 1):
                    # Check if probability crosses 0.5 between consecutive points
                    if (gs_probabilities[i] - 0.5) * (gs_probabilities[i + 1] - 0.5) < 0:  # True if there is a change of sign
                        res = root_scalar(
                            lambda T: interp_probabilities(T) - 0.5,
                            bracket=(self.temperatures[i], self.temperatures[i + 1]),
                            method="brentq",
                            xtol=1e-10,
                            rtol=1e-12,
                        )
                        if res.converged:
                            roots.append(res.root)

                # Record transition point in phase diagrams
                if roots:
                    temp_50 = roots[0]  # Take first crossing
                    V0_at_T50 = interp_V0(temp_50)
                    self.pt_phase_diagram["second_order"]["P"] = np.append(self.pt_phase_diagram["second_order"]["P"], P)
                    self.pt_phase_diagram["second_order"]["T"] = np.append(self.pt_phase_diagram["second_order"]["T"], temp_50)
                    self.vt_phase_diagram["second_order"]["V"] = np.append(self.vt_phase_diagram["second_order"]["V"], V0_at_T50)
                    self.vt_phase_diagram["second_order"]["T"] = np.append(self.vt_phase_diagram["second_order"]["T"], temp_50)

                P += dP

            except Exception:
                # If we hit an error (e.g., no crossing found), we stop the loop
                break

        # Find first order phase transition points (miscibility gap) using the common tangent method
        for index, temperature in enumerate(self.temperatures):
            V0 = self.pt_properties[f"{0.0:.2f}_GPa"]["V0"][index]  # Only consider 0 GPa for miscibility gap for T-V diagram
            min_V0 = 0.9 * V0  # Set a threshold for minimum V0
            max_V0 = 1.1 * V0  # Set a threshold for maximum V0

            # Find roots where d^2F/dV^2 = 0 between min_V0 and max_V0
            filtered_indices = np.where((self.volumes >= min_V0) & (self.volumes <= max_V0))[0]
            filtered_volumes = self.volumes[filtered_indices]
            filtered_helmholtz_energies_dV = self.helmholtz_energies_dV[index, filtered_indices]
            filtered_helmholtz_energies_d2V2 = self.helmholtz_energies_d2V2[index, filtered_indices]

            # Interpolators for dF/dV and d^2F/dV^2
            d2V2_interpolator = PchipInterpolator(filtered_volumes, filtered_helmholtz_energies_d2V2)
            dV_interpolator = PchipInterpolator(filtered_volumes, filtered_helmholtz_energies_dV)

            # Find sign changes in d^2F/dV^2 (roots)
            roots = []
            for i in range(len(filtered_volumes) - 1):
                if (filtered_helmholtz_energies_d2V2[i] * filtered_helmholtz_energies_d2V2[i + 1]) < 0:  # True if there is a change of sign
                    res = root_scalar(
                        d2V2_interpolator,
                        bracket=(filtered_volumes[i], filtered_volumes[i + 1]),
                        method="brentq",
                        xtol=1e-10,
                        rtol=1e-12,
                    )
                    if res.converged:
                        roots.append(res.root)

            # Move left from the first root and right from the second root until dF/dV values are approximately equal (miscibility gap volumes)
            if roots:
                left_root = roots[0]
                right_root = roots[-1]

                left_volume = left_root - volume_step_size
                right_volume = right_root + volume_step_size
                left_helmholtz_energy_dV = dV_interpolator(left_volume)
                right_helmholtz_energy_dV = dV_interpolator(right_volume)

                while ~np.isclose(left_helmholtz_energy_dV, right_helmholtz_energy_dV, atol=atol):
                    left_volume -= volume_step_size
                    right_volume += volume_step_size
                    left_helmholtz_energy_dV = dV_interpolator(left_volume)
                    right_helmholtz_energy_dV = dV_interpolator(right_volume)

                self.vt_phase_diagram["first_order"]["V_left"] = np.append(self.vt_phase_diagram["first_order"]["V_left"], left_volume)
                self.vt_phase_diagram["first_order"]["V_right"] = np.append(self.vt_phase_diagram["first_order"]["V_right"], right_volume)
                self.vt_phase_diagram["first_order"]["T"] = np.append(self.vt_phase_diagram["first_order"]["T"], temperature)

        # Remove second order points that fall within the miscibility gap region
        max_T_first_order = np.max(self.vt_phase_diagram["first_order"]["T"])
        mask = self.vt_phase_diagram["second_order"]["T"] > max_T_first_order
        self.vt_phase_diagram["second_order"]["V"] = self.vt_phase_diagram["second_order"]["V"][mask]
        self.vt_phase_diagram["second_order"]["T"] = self.vt_phase_diagram["second_order"]["T"][mask]
        self.pt_phase_diagram["second_order"]["T"] = np.where(mask, self.pt_phase_diagram["second_order"]["T"], np.nan)

        # Add the last second order point to the end of first order arrays for continuity in the diagram
        self.vt_phase_diagram["first_order"]["V_left"] = np.append(self.vt_phase_diagram["first_order"]["V_left"], self.vt_phase_diagram["second_order"]["V"][-1])
        self.vt_phase_diagram["first_order"]["V_right"] = np.append(self.vt_phase_diagram["first_order"]["V_right"], self.vt_phase_diagram["second_order"]["V"][-1])
        self.vt_phase_diagram["first_order"]["T"] = np.append(self.vt_phase_diagram["first_order"]["T"], self.vt_phase_diagram["second_order"]["T"][-1])

        return None

    def _get_closest_indices(self, values: np.ndarray, targets: np.ndarray) -> list:
        """
        Find indices of the closest matches in `values` for each target in `targets`.

        Args:
            values (np.ndarray): Array of values to search.
            targets (np.ndarray): Array of target values.

        Returns:
            list: List of indices in `values` closest to each target.
        """
        return [np.argmin(np.abs(values - target)) for target in targets]

    # TODO: write tests for vt diagram
    def plot_vt(
        self,
        type: str,
        selected_temperatures: np.ndarray = None,
        selected_volumes: np.ndarray = None,
        width: int = 650,
        height: int = 600,
    ):
        """
        Generate a plot vs. volume or temperature for the specified thermodynamic quantity.

        Args:
            type (str): The type of plot to generate (e.g., "helmholtz_energy_vs_volume").
            selected_temperatures (np.ndarray, optional): Temperatures to highlight in the plot.
            selected_volumes (np.ndarray, optional): Volumes to highlight in the plot.
            width (int, optional): Width of the plot in pixels.
            height (int, optional): Height of the plot in pixels.

        Returns:
            plotly.graph_objects.Figure: The generated plotly figure.

        Raises:
            ValueError: If:
                - The plot type is invalid.
                - Required data for the plot (e.g., Helmholtz energies, probabilities, volumes, pressure, etc.) is missing or not calculated.
        """

        # Central dictionary for plot behavior
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
            "helmholtz_energy_dV_vs_volume": {
                "x": self.volumes,
                "y": self.helmholtz_energies_dV,
                "fixed": "temperature",
                "ylabel": "dF/dV (eV/Å³)",
            },
            "helmholtz_energy_dV_vs_temperature": {
                "x": self.temperatures,
                "y": self.helmholtz_energies_dV,
                "fixed": "volume",
                "ylabel": "dF/dV (eV/Å³)",
            },
            "helmholtz_energy_d2V2_vs_volume": {
                "x": self.volumes,
                "y": self.helmholtz_energies_d2V2,
                "fixed": "temperature",
                "ylabel": "d²F/dV² ({unit}.eV/Å⁶)",
            },
            "helmholtz_energy_d2V2_vs_temperature": {
                "x": self.temperatures,
                "y": self.helmholtz_energies_d2V2,
                "fixed": "volume",
                "ylabel": "d²F/dV² ({unit}.eV/Å⁶)",
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
            "configurational_entropy_vs_volume": {
                "x": self.volumes,
                "y": self.configurational_entropies,
                "fixed": "temperature",
                "ylabel": "S<sub>conf</sub> (eV/K/{unit})",
            },
            "configurational_entropy_vs_temperature": {
                "x": self.temperatures,
                "y": self.configurational_entropies,
                "fixed": "volume",
                "ylabel": "S<sub>conf</sub> (eV/K/{unit})",
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
            "bulk_modulus_vs_volume": {
                "x": self.volumes,
                "y": self.bulk_moduli,
                "fixed": "temperature",
                "ylabel": "B (GPa)",
            },
            "bulk_modulus_vs_temperature": {
                "x": self.temperatures,
                "y": self.bulk_moduli,
                "fixed": "volume",
                "ylabel": "B (GPa)",
            },
            "vt_phase_diagram": {
                "x": (
                    {
                        "V_left_first_order": self.vt_phase_diagram["first_order"]["V_left"],
                        "V_right_first_order": self.vt_phase_diagram["first_order"]["V_right"],
                        "V_second_order": self.vt_phase_diagram["second_order"]["V"],
                    }
                    if len(self.vt_phase_diagram) > 0
                    else None
                ),
                "y": (
                    {
                        "T_first_order": self.vt_phase_diagram["first_order"]["T"],
                        "T_second_order": self.vt_phase_diagram["second_order"]["T"],
                    }
                    if len(self.vt_phase_diagram) > 0
                    else None
                ),
                "fixed": None,
                "ylabel": "Temperature (K)",
            },
        }

        if type not in plot_data:
            raise ValueError(f"Invalid plot type. Choose from: {', '.join(plot_data.keys())}")

        data = plot_data[type]
        x_data = data["x"]
        fixed_by = data["fixed"]
        y_label_template = data["ylabel"]
        y_data = data["y"]

        # Check for missing y_data (e.g., not calculated yet)
        if y_data is None:
            raise ValueError(f"{type} data not calculated. Run the appropriate calculation method first.")

        # Select indices and labels for fixed_by
        if fixed_by == "temperature":
            selected = selected_temperatures if selected_temperatures is not None else np.linspace(self.temperatures.min(), self.temperatures.max(), 10)
            indices = self._get_closest_indices(self.temperatures, selected)
            legend_vals = self.temperatures[indices]
            legend_fmt = lambda t: f"{int(t)} K" if t % 1 == 0 else f"{t} K"
        elif fixed_by == "volume":
            selected = selected_volumes if selected_volumes is not None else np.linspace(self.volumes.min(), self.volumes.max(), 10)
            indices = self._get_closest_indices(self.volumes, selected)
            legend_vals = self.volumes[indices]
            legend_fmt = lambda v: f"{v:.2f} Å³"

        fig = go.Figure()
        if type != "vt_phase_diagram":
            for i, val in zip(indices, legend_vals):
                fig.add_trace(
                    go.Scatter(
                        x=x_data,
                        y=y_data[i, :] if fixed_by == "temperature" else y_data[:, i],
                        mode="lines",
                        showlegend=True,
                        name=legend_fmt(val),
                        legendgroup=legend_fmt(val),
                    )
                )
        else:
            # Add miscibility gap traces
            for showlegend, x_key in zip([True, False], ["V_left_first_order", "V_right_first_order"]):
                fig.add_trace(go.Scatter(x=x_data[x_key], y=y_data["T_first_order"], mode="lines", line=dict(color="#636efa", dash="solid"), name="Misc. Gap", legendgroup="misc_gap", showlegend=showlegend))
            # Add second order trace
            fig.add_trace(
                go.Scatter(
                    x=x_data["V_second_order"],
                    y=y_data["T_second_order"],
                    mode="lines",
                    line=dict(color="#636efa", dash="6px,4px"),
                    name="2<sup>nd</sup> Order",
                )
            )
            # Plot open circle for the last point of second order
            fig.add_trace(go.Scatter(x=[x_data["V_second_order"][-1]], y=[y_data["T_second_order"][-1]], mode="markers", marker=dict(color="red", size=20, symbol="circle-open"), name="Critical Point", showlegend=False))
            fig.update_layout(yaxis=dict(range=[0, max(y_data["T_second_order"])]))

        unit = "atom" if self.number_of_atoms == 1 else f"{self.number_of_atoms} atoms"
        x_label = "Temperature (K)" if "temperature" in type else f"Volume (Å³/{unit})"
        y_label = y_label_template.format(unit=unit)
        format_plot(fig, x_label, y_label, width=width, height=height)
        return fig

    def plot_pt(
        self,
        type: str,
        P: float = 0.00,
        selected_temperatures: np.ndarray = None,
        ground_state: str = None,
        width: int = 650,
        height: int = 600,
    ):
        """
        Generate a plot for the specified thermodynamic quantity.

        Args:
            type (str): The type of plot to generate (e.g., "helmholtz_energy_pv_vs_volume").
            P (float): Pressure in GPa for the plot. Default is 0.00 GPa.
            selected_temperatures (np.ndarray, optional): Temperatures to highlight in the helmholtz_energy_pv_vs_volume plot.
            ground_state (str, optional): Name of the ground state configuration for probability plots.
            width (int, optional): Width of the plot in pixels.
            height (int, optional): Height of the plot in pixels.

        Returns:
            plotly.graph_objects.Figure: The generated plotly figure.

        Raises:
            ValueError: If:
                - Properties at P are not calculated.
                - The plot type is invalid.
                - Required data for the plot (e.g., Helmholtz energies, probabilities, volumes, pressure, etc.) is missing or not calculated.
        """
        # Raise ValueError if properties at P are not calculated
        if f"{P:.2f}_GPa" not in self.pt_properties:
            raise ValueError(f"Properties at {P:.2f} GPa not calculated. Run calculate_pressure_properties first.")

        # Central dictionary for plot behavior
        plot_data = {
            "helmholtz_energy_pv_vs_volume": {
                "x": self.volumes,
                "y": self.pt_properties[f"{P:.2f}_GPa"]["helmholtz_energy_pv"],
                "fixed": "temperature",
                "ylabel": "F + PV (eV/{unit})",
            },
            "volume_vs_temperature": {
                "x": self.temperatures,
                "y": self.pt_properties[f"{P:.2f}_GPa"]["V0"],
                "fixed": "pressure",
                "ylabel": "V (Å³/{unit})",
            },
            "CTE_vs_temperature": {
                "x": self.temperatures[:-1],
                "y": self.pt_properties[f"{P:.2f}_GPa"]["CTE"],
                "fixed": "pressure",
                "ylabel": "CTE (10<sup>-6</sup> K<sup>-1</sup>)",
            },
            "LCTE_vs_temperature": {
                "x": self.temperatures[:-1],
                "y": self.pt_properties[f"{P:.2f}_GPa"]["LCTE"],
                "fixed": "pressure",
                "ylabel": "LCTE (10<sup>-6</sup> K<sup>-1</sup>)",
            },
            "entropy_vs_temperature": {
                "x": self.temperatures,
                "y": self.pt_properties[f"{P:.2f}_GPa"]["S0"],
                "fixed": "pressure",
                "ylabel": "S (eV/K/{unit})",
            },
            "configurational_entropy_vs_temperature": {
                "x": self.temperatures,
                "y": self.pt_properties[f"{P:.2f}_GPa"]["Sconf"],
                "fixed": "pressure",
                "ylabel": "S<sub>conf</sub> (eV/K/{unit})",
            },
            "heat_capacity_vs_temperature": {
                "x": self.temperatures[:-1],
                "y": self.pt_properties[f"{P:.2f}_GPa"]["Cp"],
                "fixed": "pressure",
                "ylabel": "C<sub>p</sub> (eV/K/{unit})",
            },
            "gibbs_energy_vs_temperature": {
                "x": self.temperatures,
                "y": self.pt_properties[f"{P:.2f}_GPa"]["G0"],
                "fixed": "pressure",
                "ylabel": "G (eV/{unit})",
            },
            "bulk_modulus_vs_temperature": {
                "x": self.temperatures,
                "y": self.pt_properties[f"{P:.2f}_GPa"]["B0"],
                "fixed": "pressure",
                "ylabel": "B (GPa)",
            },
            "probability_vs_temperature": {
                "x": self.temperatures,
                "y": {name: config.probabilities_at_P[f"{P:.2f}_GPa"] for name, config in self.configurations.items()},
                "fixed": "pressure",
                "ylabel": "Probability",
            },
            "pt_phase_diagram": {
                "x": self.pt_phase_diagram["second_order"]["P"] if len(self.pt_phase_diagram) > 0 else None,
                "y": self.pt_phase_diagram["second_order"]["T"] if len(self.pt_phase_diagram) > 0 else None,
                "fixed": None,
                "ylabel": "Pressure (GPa)",
            },
        }

        if type not in plot_data:
            raise ValueError(f"Invalid plot type. Choose from: {', '.join(plot_data.keys())}")

        data = plot_data[type]
        x_data = data["x"]
        fixed_by = data["fixed"]
        y_label_template = data["ylabel"]
        y_data = data["y"]

        # Check for missing y_data (e.g., not calculated yet)
        if y_data is None:
            raise ValueError(f"{type} data not calculated. Run the appropriate calculation method first.")

        # Select indices and labels for fixed_by
        if fixed_by == "temperature":
            selected = selected_temperatures if selected_temperatures is not None else np.linspace(self.temperatures.min(), self.temperatures.max(), 10)
            indices = self._get_closest_indices(self.temperatures, selected)
            legend_vals = self.temperatures[indices]
            legend_fmt = lambda t: f"{int(t)} K" if t % 1 == 0 else f"{t} K"
        else:
            indices = None
            legend_vals = None

        fig = go.Figure()

        if type == "probability_vs_temperature":
            traces = []
            # Add ground state first if present
            if ground_state is not None and ground_state in y_data:
                traces.append(
                    go.Scatter(
                        x=x_data,
                        y=y_data[ground_state],
                        mode="lines",
                        name=ground_state,
                    )
                )
                gs_probs = np.array(y_data[ground_state])
                # Interpolate to find temperature where probability crosses 0.5
                try:
                    interp = PchipInterpolator(x_data, gs_probs)
                    roots = []
                    for i in range(len(x_data) - 1):
                        # Check if the probability crosses 0.5 between x_data[i] and x_data[i+1]
                        if (gs_probs[i] - 0.5) * (gs_probs[i + 1] - 0.5) < 0:
                            res = root_scalar(
                                lambda T: interp(T) - 0.5,
                                bracket=(x_data[i], x_data[i + 1]),
                                method="brentq",
                                xtol=1e-10,
                                rtol=1e-12,
                            )
                            if res.converged:
                                roots.append(res.root)
                except Exception:
                    pass
                # Sum all probabilities except the ground state
                excited_probs = np.zeros_like(x_data, dtype=float)
                for name, probs in y_data.items():
                    if name != ground_state:
                        excited_probs += probs
                traces.append(
                    go.Scatter(
                        x=x_data,
                        y=excited_probs,
                        mode="lines",
                        name="Sum (Non-GS)",
                        line=dict(color="black"),
                    )
                )
                # Add the rest (non-ground states)
                for name, probs in y_data.items():
                    if name != ground_state:
                        traces.append(
                            go.Scatter(
                                x=x_data,
                                y=probs,
                                mode="lines",
                                name=name,
                            )
                        )

            else:
                for name, probs in y_data.items():
                    traces.append(
                        go.Scatter(
                            x=x_data,
                            y=probs,
                            mode="lines",
                            name=name,
                        )
                    )
            for trace in traces:
                fig.add_trace(trace)

            x_label = "Temperature (K)"
            y_label = "Probability"
            fig.update_layout(
                title=dict(
                    text=f"P = {P:.2f} GPa" if P is not None else "P = None",
                    font=dict(size=22, color="rgb(0,0,0)"),
                ),
                yaxis=dict(range=[0, None]),
            )
        # TODO: write tests for these
        elif type == "pt_phase_diagram":
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode="lines",
                    line=dict(color="#636efa", dash="6px,4px"),
                    name="2<sup>nd</sup> Order",
                    showlegend=True,
                )
            )
            fig.update_layout(xaxis=dict(autorange="reversed"), yaxis=dict(range=[0, max(y_data)]))
            x_label = "Pressure (GPa)"
            y_label = "Temperature (K)"

            # Plot open circle for the last point of second order
            last_valid_index = np.where(~np.isnan(y_data))[0][-1] if np.any(~np.isnan(y_data)) else None
            fig.add_trace(go.Scatter(x=[x_data[last_valid_index]], y=[y_data[last_valid_index]], mode="markers", marker=dict(color="red", size=20, symbol="circle-open"), name="Critical Point", showlegend=False))
        else:
            if fixed_by == "temperature":
                for i, val in zip(indices, legend_vals):
                    fig.add_trace(
                        go.Scatter(
                            x=x_data,
                            y=y_data[i, :] if fixed_by == "temperature" else y_data[:, i],
                            mode="lines",
                            showlegend=True,
                            name=legend_fmt(val),
                            legendgroup=legend_fmt(val),
                        )
                    )
                    if type == "helmholtz_energy_pv_vs_volume" and fixed_by == "temperature":
                        fig.add_trace(
                            go.Scatter(
                                x=[self.pt_properties[f"{P:.2f}_GPa"]["V0"][i]],
                                y=[self.pt_properties[f"{P:.2f}_GPa"]["G0"][i]],
                                mode="markers",
                                marker=dict(color="black", size=10, symbol="cross"),
                                showlegend=False,
                                name=f"minimum",
                                legendgroup=legend_fmt(val),
                            )
                        )
            elif fixed_by == "pressure" and type != "probability_vs_temperature":
                fig.add_trace(
                    go.Scatter(
                        x=x_data,
                        y=y_data,
                        mode="lines",
                    )
                )
            if fixed_by == "pressure" or type == "helmholtz_energy_pv_vs_volume":
                title_text = f"P = {P:.2f} GPa" if P is not None else "P = None"
                fig.update_layout(title=dict(text=title_text, font=dict(size=22, color="rgb(0,0,0)")))

            unit = "atom" if self.number_of_atoms == 1 else f"{self.number_of_atoms} atoms"
            x_label = "Temperature (K)" if "temperature" in type else f"Volume (Å³/{unit})"
            y_label = y_label_template.format(unit=unit)

        format_plot(fig, x_label, y_label, width=width, height=height)
        return fig
