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
        tip_pos = np.array([position[0], position[1], np.mean(self.coord[:, 2]) + self.tip_disp])
    
        if emax > max(self.energies):
            emax = max(self.energies)
        if emin < min(self.energies):
            emin = min(self.energies)
    
        self.estart = np.where(self.energies >= emin)[0][0]
        self.eend = np.where(self.energies <= emax)[0][-1] + 1
        energy_range = self.energies[self.estart:self.eend]
    
        ldos = {atom_idx: {} for atom_idx in range(len(self.coord))}  # Separate by atom
    
        for atom_idx, atom_pdos in enumerate(self.pdos):
            for orbital, spin_data in atom_pdos.items():
                if orbital not in ldos[atom_idx]:
                    ldos[atom_idx][orbital] = {Spin.up: np.zeros_like(energy_range)}
                    if Spin.down in spin_data:
                        ldos[atom_idx][orbital][Spin.down] = np.zeros_like(energy_range)
    
                for spin, dos_values in spin_data.items():
                    energy_filtered_dos = dos_values[self.estart:self.eend]
                    distance = np.linalg.norm(tip_pos - self.coord[atom_idx])
    
                    # Tunneling weights
                    tunneling_weights = np.array(
                        [tunneling_factor(V, E, phi) * exp(-distance*5e-2) for E in energy_range]
                    )
    
                    # Calculate LDOS contribution
                    ldos_contrib = energy_filtered_dos * tunneling_weights
                    ldos[atom_idx][orbital][spin] += ldos_contrib
    
        return ldos



    def plot_ldos_curve(self, ldos, emin, emax):
        """
        Plot the total LDOS and filtered atomic orbital LDOS, normalizing the total LDOS
        to the sum of the areas of significant atomic orbital LDOS contributions.
        
        Parameters:
            ldos (dict): LDOS values for each atom, orbital, and spin component.
            emin (float): Minimum energy (eV) for the plot range.
            emax (float): Maximum energy (eV) for the plot range.
        """
        # Get the energy range for plotting
        energy_range = self.energies[self.estart:self.eend]
        
        # Initialize total LDOS array
        total_ldos = np.zeros_like(energy_range)
        
        # Dictionary to store areas for normalization
        areas = {}
    
        # Generate distinct colors for each atom
        atom_colors = plt.cm.tab10(np.linspace(0, 1, len(self.coord)))
    
        # Calculate total LDOS and individual contributions
        for atom_idx, atom_pdos in ldos.items():
            for orbital, spin_data in atom_pdos.items():
                orbital_ldos = np.zeros_like(energy_range)
                
                # Combine contributions from both spins
                for spin, ldos_values in spin_data.items():
                    orbital_ldos += ldos_values
                
                # Integrate area under the curve (using trapezoidal rule)
                area = np.trapz(orbital_ldos, energy_range)
                areas[(atom_idx, orbital)] = area
                
                # Add to total LDOS
                total_ldos += orbital_ldos
    
        # Normalize areas so the largest is 1
        max_area = max(areas.values())
        normalized_areas = {key: area / max_area for key, area in areas.items()}
        
        # Filter to include only curves with normalized area > 0.75
        filtered_areas = {key: area for key, area in normalized_areas.items() if area > 0.9}
        
        # Calculate the sum of the areas of the filtered atomic orbital LDOS
        significant_area_sum = sum(areas[key] for key in filtered_areas.keys())
        
        # Scale the total LDOS to match the sum of the significant areas
        total_area = np.trapz(total_ldos, energy_range)
        if total_area > 0:  # Prevent division by zero
            total_ldos *= significant_area_sum / total_area
        
        # Prepare the plot
        plt.figure(figsize=(12, 8))
        
        # Plot the scaled total LDOS
        plt.plot(
            energy_range, total_ldos, label="Total LDOS (Normalized)", 
            color="black", linewidth=2, linestyle="--"
        )
        
        # Plot the filtered contributions
        for (atom_idx, orbital), area in filtered_areas.items():
            orbital_ldos = np.zeros_like(energy_range)
            for spin, ldos_values in ldos[atom_idx][orbital].items():
                orbital_ldos += ldos_values
            
            plt.plot(
                energy_range, orbital_ldos, 
                label=f"Atom {atom_idx + 1}, Orbital {orbital} (Normalized Area: {normalized_areas[(atom_idx, orbital)]:.2f})",
                color=atom_colors[atom_idx],
                alpha=0.7
            )
        
        # Add labels, title, and legend
        plt.xlabel("Energy (eV)")
        plt.ylabel("LDOS (states/eV)")
        plt.title("Filtered and Normalized LDOS (Total LDOS Scaled)")
        plt.legend(fontsize="small", loc="best", ncol=2, frameon=True)
        plt.grid(True)
        plt.tight_layout()
        plt.show()



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

    # Plot the LDOS curve
    ldos_calc.plot_ldos_curve(ldos, emin, emax)


