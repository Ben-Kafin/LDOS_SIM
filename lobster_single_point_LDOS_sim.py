import sys
import os
from os.path import exists
import numpy as np
from numpy import pi, sqrt, exp
import matplotlib.pyplot as plt
from pymatgen.io.lobster import Doscar
from pymatgen.core import Structure
from pymatgen.electronic_structure.core import Spin


def tunneling_factor(V, E, phi):
    """
    Calculate the tunneling factor based on the applied voltage (V),
    energy relative to Fermi level (E), and the workfunction (phi).
    """
    V *= 1.60218e-19  # Convert from eV to Joules
    E *= 1.60218e-19  # Convert from eV to Joules
    phi *= 1.60218e-19  # Convert from eV to Joules

    m_e = 9.11e-31  # Electron mass (kg)
    hbar = 6.626e-34  # Planck's constant (J·s)

    prefactor = (8 / (3 * V)) * pi * sqrt(2 * m_e) / hbar
    barrier = (phi - E + V)**(3/2) - (phi - E)**(3/2)

    return prefactor * barrier


def parse_poscar(ifile):
    """
    Parses the POSCAR file and extracts lattice vectors, atomic positions, and atom types.
    """
    with open(ifile, 'r') as file:
        lines = file.readlines()
        sf = float(lines[1])
        latticevectors = [float(lines[i].split()[j]) * sf for i in range(2, 5) for j in range(3)]
        latticevectors = np.array(latticevectors).reshape(3, 3)
        atomtypes = lines[5].split()
        atomnums = [int(i) for i in lines[6].split()]
        if 'Direct' in lines[7] or 'Cartesian' in lines[7]:
            start = 8
            mode = lines[7].split()[0]
        else:
            mode = lines[8].split()[0]
            start = 9
        coord = np.array([[float(lines[i].split()[j]) for j in range(3)] for i in range(start, sum(atomnums) + start)])
        if mode != 'Cartesian':
            for i in range(sum(atomnums)):
                for j in range(3):
                    while coord[i][j] > 1.0 or coord[i][j] < 0.0:
                        if coord[i][j] > 1.0:
                            coord[i][j] -= 1.0
                        elif coord[i][j] < 0.0:
                            coord[i][j] += 1.0
                coord[i] = np.dot(coord[i], latticevectors)
    return latticevectors, coord, atomtypes, atomnums


class ldos_single_point:
    def __init__(self, filepath):
        """
        Initialize the single-point LDOS calculator.

        Parameters:
            filepath (str): Path to the folder containing output files.
        """
        self.filepath = filepath
        self.lv = None
        self.coord = None
        self.atomtypes = None
        self.atomnums = None
        self.energies = None
        self.ef = None
        self.pdos = None  # Projected DOS for all orbitals
        self.tip_disp = 15.0  # Default tip displacement
        self.estart = None
        self.eend = None

    def load_files(self):
        """
        Load and parse POSCAR and LOBSTER DOSCAR files.
        """
        doscar_path = f"{self.filepath}/DOSCAR.lobster"
        poscar_path = f"{self.filepath}/POSCAR"
    
        if not exists(doscar_path):
            raise FileNotFoundError(f"DOSCAR.lobster file not found in: {doscar_path}")
        if not exists(poscar_path):
            if exists(f"{self.filepath}/CONTCAR"):
                poscar_path = f"{self.filepath}/CONTCAR"
            else:
                raise FileNotFoundError(f"POSCAR file not found in: {self.filepath}")
    
        # Parse POSCAR using the provided parse_poscar method
        self.lv, self.coord, self.atomtypes, self.atomnums = parse_poscar(poscar_path)
    
        # Parse DOSCAR using pymatgen's LOBSTER module and save as a class attribute
        self.doscar = Doscar(doscar_path, structure_file=poscar_path)
        self.energies = np.array(self.doscar.energies)  # Energies relative to E_fermi
        self.ef = self.doscar.tdos.efermi
        self.pdos = self.doscar.pdos  # Orbital-projected DOS data


    def calculate_single_point_ldos(self, position, emin, emax, phi, V):
        """
        Calculate LDOS at a single spatial position using the tunneling factor and TDOS.
    
        Parameters:
            position (np.array): 3D spatial position of the tip (x, y, z).
            emin (float): Minimum energy (eV) for LDOS integration.
            emax (float): Maximum energy (eV) for LDOS integration.
            phi (float): Workfunction (eV) of the material.
            V (float): Bias voltage (eV).
    
        Returns:
            dict: LDOS values for each orbital and spin component.
        """
        tip_pos = np.array([position[0], position[1], np.mean(self.coord[:, 2]) + self.tip_disp])
    
        if emax > max(self.energies):
            emax = max(self.energies)
        if emin < min(self.energies):
            emin = min(self.energies)
    
        self.estart = np.where(self.energies >= emin)[0][0]
        self.eend = np.where(self.energies <= emax)[0][-1] + 1
        energy_range = self.energies[self.estart:self.eend]
    
        # Check if spin-polarized data is available
        if Spin.down in self.doscar.tdos.densities:
            tdos_values = self.doscar.tdos.densities[Spin.up] + self.doscar.tdos.densities[Spin.down]
        else:
            tdos_values = self.doscar.tdos.densities[Spin.up]
    
        # Extract the TDOS weights for the energy range
        tdos_weights = tdos_values[self.estart:self.eend]
        #print(f"TDOS Weights: {tdos_weights}")
    
        ldos = {}
        for atom_idx, atom_pdos in enumerate(self.pdos):
            for orbital, spin_data in atom_pdos.items():
                if orbital not in ldos:
                    ldos[orbital] = {Spin.up: np.zeros_like(energy_range)}
                    if Spin.down in spin_data:  # Only include Spin.down if present
                        ldos[orbital][Spin.down] = np.zeros_like(energy_range)
    
                for spin, dos_values in spin_data.items():
                    energy_filtered_dos = dos_values[self.estart:self.eend]
                    distance = np.linalg.norm(tip_pos - self.coord[atom_idx])
    
                    # Calculate tunneling weights
                    tunneling_weights = np.array(
                        [tunneling_factor(V, E, phi)  * exp(-distance) for E in energy_range]
                    )
    
                    # Energy-resolved LDOS contribution
                    ldos_contrib = energy_filtered_dos * tunneling_weights
                    ldos[orbital][spin] += ldos_contrib  # Accumulate energy-resolved LDOS
    
                    #print(f"Atom {atom_idx}, Orbital {orbital}, Spin {spin}, LDOS Contribution: {ldos_contrib}")
    
        return ldos

    def plot_ldos_curve(self, ldos, emin, emax):
        """
        Plot the LDOS curve.
    
        Parameters:
            ldos (dict): LDOS values for each orbital and spin component.
            emin (float): Minimum energy (eV) for the plot range.
            emax (float): Maximum energy (eV) for the plot range.
        """
        # Get the energy range for plotting
        energy_range = self.energies[self.estart:self.eend]
    
        # Initialize total LDOS array
        total_ldos = np.zeros_like(energy_range)
    
        # Sum contributions across orbitals and spins
        for orbital, spin_data in ldos.items():
            for spin, ldos_values in spin_data.items():
                if spin == Spin.down and Spin.down not in ldos[orbital]:
                    continue  # Skip Spin.down if not present
                total_ldos += ldos_values
    
        # Plot the LDOS curve
        plt.figure(figsize=(8, 5))
        plt.plot(energy_range, total_ldos, label="LDOS")
        plt.xlabel("Energy (eV)")
        plt.ylabel("LDOS (states/eV)")
        plt.title("Local Density of States (LDOS)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def print_top_contributions(self, ldos, percentage=10):
        """
        Print the most relevant states based on the top contributions.
        """
        contributions = []
    
        # Aggregate LDOS contributions
        for atom_idx, atom_pdos in enumerate(self.pdos):
            if isinstance(atom_pdos, dict):
                for orbital, spin_data in atom_pdos.items():
                    spin_up = np.sum(ldos[orbital].get(Spin.up, 0.0))
                    spin_down = np.sum(ldos[orbital].get(Spin.down, 0.0))
                    total_contribution = spin_up + spin_down
                    if total_contribution > 0:
                        contributions.append((total_contribution, atom_idx, orbital))
    
        # Ensure contributions are not empty
        if not contributions:
            raise ValueError("No valid contributions found. Ensure LDOS data is correctly calculated.")
    
        # Sort and print contributions
        contributions.sort(reverse=True, key=lambda x: x[0])
        top_count = max(1, int(len(contributions) * percentage / 100))
    
        print("\nTop Contributions to LDOS:")
        for i in range(top_count):
            contribution, atom_idx, orbital = contributions[i]
            print(f"Atom #{atom_idx}, Orbital {orbital}: Contribution = {contribution}")








# Example Usage
if __name__ == "__main__":
    filepath = 'C:/Users/Benjamin Kafin/Documents/VASP/NHC_Cu/freeCu1/freeCu2/lobster/'  # Directory containing output files
    spatial_position = np.array([5.00807, 8.66844, 23])  # Tip position
    emin, emax = -2.0, 2.0  # Energy range (eV)
    phi=4.5
    V=2.0

    # Initialize and load files
    try:
        ldos_calc = ldos_single_point(filepath)
        ldos_calc.load_files()
    except FileNotFoundError as e:
        print(e)
        sys.exit()

    # Calculate LDOS at a single point
    ldos = ldos_calc.calculate_single_point_ldos(spatial_position, emin, emax, phi, V)
    
    # Call print_top_contributions method
    ldos_calc.print_top_contributions(ldos, percentage=5)
    
    # Plot the LDOS curve
    ldos_calc.plot_ldos_curve(ldos, emin, emax)


