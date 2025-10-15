# stm_simulator.py

import numpy as np
import sys
import matplotlib.pyplot as plt
import argparse
import copy
from os.path import exists, getsize, join, isdir
from os import chdir, makedirs
from time import time
from pathos.multiprocessing import ProcessPool
from numpy import array, dot, exp, zeros, sqrt, shape
from numpy.linalg import norm, inv
from math import pi
from scipy.integrate import simpson
from scipy.ndimage import map_coordinates, gaussian_filter

# #############################################################################
# HELPER FUNCTIONS
# #############################################################################

def _parse_vasp_locpot(ifile):
    """
    Parses a raw VASP LOCPOT file by correctly reading the header
    to determine the number of atom coordinate lines to skip.
    """
    with open(ifile,'r') as f:
        f.readline(); f.readline()
        [f.readline() for _ in range(3)]
        f.readline()
        atom_counts = [int(i) for i in f.readline().split()]
        total_atoms = sum(atom_counts)
        f.readline()
        [f.readline() for _ in range(total_atoms)]
        f.readline()
        dim = [int(i) for i in f.readline().split()]
        pot = np.zeros(dim)
        data_lines = f.readlines()
        all_data_points = ' '.join(data_lines).split()
        num_grid_points = dim[0] * dim[1] * dim[2]
        data = [float(p) for p in all_data_points[:num_grid_points]]
        idx = 0
        for z in range(dim[2]):
            for y in range(dim[1]):
                for x in range(dim[0]):
                    if idx < len(data):
                        pot[x, y, z] = data[idx]
                        idx += 1
    return pot

def parse_doscar(filepath):
    """Reads a VASP DOSCAR file."""
    with open(filepath, 'r') as f:
        atomnum = int(f.readline().split()[0])
        [f.readline() for _ in range(4)]
        line = f.readline().split()
        nedos, ef = int(line[2]), float(line[3])
        dos, energies = [], []
        for i in range(atomnum + 1):
            if i != 0: f.readline()
            for j in range(nedos):
                line = f.readline().split()
                if not line: continue
                if i == 0: energies.append(float(line[0]))
                if j == 0: temp_dos = [[] for _ in range(len(line) - 1)]
                for k in range(len(line) - 1): temp_dos[k].append(float(line[k+1]))
            dos.append(temp_dos)
    energies = array(energies) - ef
    num_columns = shape(dos[1:])[1] if len(dos) > 1 else 0
    if num_columns == 3: orbitals = ['s', 'p', 'd']
    elif num_columns == 9: orbitals = ['s','p_y','p_z','p_x','d_xy','d_yz','d_z2','d_xz','d_x2-y2']
    else: orbitals = [f'orb_{i}' for i in range(num_columns)]
    return dos, energies, ef, orbitals

def parse_poscar(ifile):
    """Reads a VASP POSCAR/CONTCAR file."""
    with open(ifile, 'r') as f:
        lines = f.readlines()
        sf = float(lines[1])
        lv = array([float(c) for c in ' '.join(lines[2:5]).split()]).reshape(3,3) * sf
        atomtypes = lines[5].split()
        atomnums = [int(i) for i in lines[6].split()]
        start_line = 7
        if lines[start_line].strip().lower()[0] not in ['d','c']: start_line = 8
        mode = lines[start_line].strip().lower()
        coord_start = start_line + 1
        coord = array([[float(c) for c in line.split()[:3]] for line in lines[coord_start:sum(atomnums)+coord_start]])
        if 'direct' in mode: coord = dot(coord, lv)
    return lv, coord, atomtypes, atomnums

# #############################################################################
# MAIN STM_Simulator CLASS
# #############################################################################

class STM_Simulator:
    def __init__(self, filepath):
        self.filepath = filepath
        self.nprocs = 4
        self.unit_cell_num = 4
        
        if not exists(filepath): raise FileNotFoundError(f"Directory does not exist: {filepath}")
        chdir(filepath)

    def _find_bulk_surface_z(self, bin_width=0.515, filter_atom_type=None):
        """
        Robustly detects the Z-height of the bulk surface layer, with an option
        to filter by a specific atom type (e.g., the substrate material).
        """
        print("Analyzing atomic coordinates to find bulk surface height...")
        z_coords_all = self.coord[:, 2]

        if filter_atom_type:
            print(f"Filtering for atom type '{filter_atom_type}' for surface detection.")
            all_atom_types = np.repeat(self.atomtypes, self.atomnums)
            type_mask = (all_atom_types == filter_atom_type)
            if not np.any(type_mask):
                print(f"Warning: Atom type '{filter_atom_type}' not found. Defaulting to all atoms.")
                z_coords_to_scan = z_coords_all
            else:
                z_coords_to_scan = z_coords_all[type_mask]
        else:
            z_coords_to_scan = z_coords_all

        z_cell_midpoint = self.lv[2, 2] / 2.0
        z_min = np.min(z_coords_to_scan)
        if z_cell_midpoint <= z_min:
            print("Warning: Lowest atom is above cell midpoint. Falling back to highest atom mode.")
            return np.max(z_coords_all)
        
        print(f"Searching for bulk surface below cell midpoint ({z_cell_midpoint:.2f} Å) with bin width {bin_width} Å.")
        bins = np.arange(z_min, z_cell_midpoint + bin_width, bin_width)
        counts, bin_edges = np.histogram(z_coords_to_scan[z_coords_to_scan < z_cell_midpoint], bins=bins)
        
        if len(counts) == 0:
             print("Warning: No atomic layers found for the specified criteria. Falling back to highest atom mode.")
             return np.max(z_coords_all)

        max_count = np.max(counts)
        indices_of_max_counts = np.where(counts == max_count)[0]
        highest_z_bin_idx = np.max(indices_of_max_counts)
        surface_z = bin_edges[highest_z_bin_idx + 1]
        print(f"Automatic bulk surface detection complete. Z_surface set to {surface_z:.3f} Å.")
        return surface_z

    def parse_vasp_outputs(self, locpot_path, surface_mode='bulk', bulk_atom_type=None):
        """Reads all VASP files and handles LOCPOT to .npy conversion."""
        poscar_path = './CONTCAR' if exists('./CONTCAR') and getsize('./CONTCAR') > 0 else './POSCAR'
        doscar_path = './DOSCAR'
        try:
            self.lv, self.coord, self.atomtypes, self.atomnums = parse_poscar(poscar_path)
            self.dos, self.energies, self.ef, self.orbitals = parse_doscar(doscar_path)
            self.z_highest_atom = np.max(self.coord[:, 2])
            
            if surface_mode == 'bulk':
                self.z_bulk_surface = self._find_bulk_surface_z(filter_atom_type=bulk_atom_type)
            else:
                self.z_bulk_surface = self.z_highest_atom

            locpot_npy_path = join(self.filepath, "LOCPOT.npy")
            if exists(locpot_npy_path):
                print(f"Loading pre-converted potential from {locpot_npy_path}...")
                self.locpot_data = np.load(locpot_npy_path)
            else:
                print(f"LOCPOT.npy not found. Parsing raw LOCPOT file: {locpot_path}...")
                self.locpot_data = _parse_vasp_locpot(locpot_path)
                print(f"Saving potential data to {locpot_npy_path} for future runs...")
                np.save(locpot_npy_path, self.locpot_data)
            self.locpot_dims = np.array(self.locpot_data.shape)
            self.inv_lv = inv(self.lv)
            print("Local potential data loaded successfully.")
        except Exception as e:
            print(f"Error reading VASP files: {e}"); sys.exit()

    def _calculate_ldos_at_points(self, tip_positions, emin, emax, target_energies=None):
        """
        Core workhorse function. Calculates full spectra or specific energy slices.
        """
        num_points = tip_positions.shape[0]

        if target_energies is not None:
            energy_indices = [np.argmin(np.abs(self.energies - E)) for E in target_energies]
            calculation_energies = self.energies[energy_indices]
        else:
            estart = np.searchsorted(self.energies, emin)
            eend = np.searchsorted(self.energies, emax, side='right')
            energy_indices = list(range(estart, eend))
            calculation_energies = self.energies[energy_indices]
        
        num_energies = len(energy_indices)
        if num_energies == 0:
            print("Warning: No energy points found in the specified range.")
            return [], np.array([])
        
        if not hasattr(self, 'periodic_coord'):
            coords, indices, base_indices = [], [], np.arange(len(self.coord))
            for i in range(-self.unit_cell_num, self.unit_cell_num + 1):
                for j in range(-self.unit_cell_num, self.unit_cell_num + 1):
                    coords.append(self.coord + self.lv[0] * i + self.lv[1] * j)
                    indices.append(base_indices)
            self.periodic_coord, self.atom_indices_periodic = np.concatenate(coords), np.concatenate(indices)
        
        print(f"Integrating LDOS for {num_points} points over {num_energies} energies...")
        start_time = time()
        
        all_ldos_spectra = []
        for i in range(num_points):
            pos = tip_positions[i]
            frac_coords = np.dot(pos, self.inv_lv)
            grid_indices = (frac_coords % 1.0) * self.locpot_dims
            potential_local = map_coordinates(self.locpot_data, grid_indices.reshape(3, 1), order=1, mode='wrap')[0]
            phi_local = potential_local - self.ef
            
            if phi_local <= 0: kappa = 0 
            else: kappa = 0.512 * np.sqrt(phi_local)
            
            pos_diffs = norm(pos - self.periodic_coord, axis=1)
            sf = exp(-2.0 * kappa * pos_diffs)
            
            point_spectrum = np.zeros((len(self.orbitals), num_energies))
            for atom_idx in range(sum(self.atomnums)):
                mask = (self.atom_indices_periodic == atom_idx)
                total_sf_for_atom = np.sum(sf[mask])
                dos_slices = np.array([self.dos[atom_idx + 1][orb_idx] for orb_idx in range(len(self.orbitals))])[:, energy_indices]
                point_spectrum += dos_slices * total_sf_for_atom
                
            all_ldos_spectra.append(point_spectrum)

        print(f"Integration complete. Time elapsed: {time() - start_time:.2f} seconds.")
        return all_ldos_spectra, calculation_energies

    def _normalize_spectra(self, spectra_list, energy_axis):
        """Normalizes each spectrum in a list to have a total integrated area of 1."""
        normalized_spectra = []
        for spec in spectra_list:
            total_intensity_curve = np.sum(spec, axis=0)
            area = simpson(total_intensity_curve, x=energy_axis)
            if area > 1e-9: normalized_spectra.append(spec / area)
            else: normalized_spectra.append(spec)
        return normalized_spectra

    def run_point_scan(self, position, erange, locpot_path, surface_mode, bulk_atom_type):
        print("--- Running Single Point LDOS Simulation ---")
        self.parse_vasp_outputs(locpot_path, surface_mode, bulk_atom_type)
        ldos_spectra, energy_axis = self._calculate_ldos_at_points(np.array([position]), erange[0], erange[1])
        final_spectrum = np.sum(ldos_spectra[0], axis=0)
        fig, ax = plt.subplots(); ax.plot(energy_axis, final_spectrum)
        ax.set(title=f'LDOS Spectrum at ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}) Å',
               xlabel='Energy - $E_f$ (eV)', ylabel='LDOS (arb. units)'); ax.grid(True); plt.show()

    def run_line_scan(self, path_coords, npts, erange, topo_bias, const_height, scaler, base_dist, locpot_path, surface_mode, bulk_atom_type, cmap_ldos, cmap_topo, normalize_spectra, ldos_smoothing_sigma):
        from matplotlib.collections import LineCollection
        print("--- Running Variable Height Line Scan (Cartesian Path) ---")
        self.parse_vasp_outputs(locpot_path, surface_mode, bulk_atom_type)
        
        start_point_xy = np.array(path_coords[:2])
        end_point_xy = np.array(path_coords[2:])
        path_coords_xy = np.array([np.linspace(start_point_xy[i], end_point_xy[i], npts) for i in range(2)]).T
        path_distance = np.linspace(0, norm(end_point_xy - start_point_xy), npts)
        
        print(f"\n[STEP 1] Topography Scan (Bias = {topo_bias:.2f}V)...")
        const_z = self.z_highest_atom + const_height
        tip_positions_const = np.hstack([path_coords_xy, np.full((npts, 1), const_z)])
        topo_emin, topo_emax = min(0.0, topo_bias), max(0.0, topo_bias)
        topo_spectra, topo_energies = self._calculate_ldos_at_points(tip_positions_const, topo_emin, topo_emax)
        topography_data = np.array([simpson(np.sum(spec, axis=0), x=topo_energies) for spec in topo_spectra])
        topography_data /= np.max(topography_data) if np.max(topography_data) > 0 else 1.0

        print("\n[STEP 2] Spectroscopic Scan...")
        variable_z = self.z_bulk_surface + base_dist + (topography_data * scaler)
        tip_positions_variable = np.hstack([path_coords_xy, variable_z[:, np.newaxis]])
        final_spectra, final_energies = self._calculate_ldos_at_points(tip_positions_variable, erange[0], erange[1])
        
        if normalize_spectra:
            print("\nNormalizing final LDOS spectra...")
            final_spectra = self._normalize_spectra(final_spectra, final_energies)

        # --- MODIFIED PLOTTING ---
        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 5], height_ratios=[3, 1.5], wspace=0.3, hspace=0.35)
        
        gs_top = gs[0, :].subgridspec(1, 2, width_ratios=[0.2, 3], wspace=0.05)
        ax_topo_stripe = fig.add_subplot(gs_top[0])
        ax_ldos = fig.add_subplot(gs_top[1], sharey=ax_topo_stripe)
        
        gs_bottom = gs[1, :].subgridspec(1, 2, wspace=0.3)
        ax_path_view = fig.add_subplot(gs_bottom[0])
        ax_topo_line = fig.add_subplot(gs_bottom[1])
        
        ldos_plot_data = np.sum(np.array(final_spectra), axis=1)
        if ldos_smoothing_sigma > 0:
            ldos_plot_data = gaussian_filter(ldos_plot_data, sigma=ldos_smoothing_sigma)
        
        energies_mesh, distance_mesh = np.meshgrid(final_energies, path_distance)
        ldos_map = ax_ldos.pcolormesh(energies_mesh, distance_mesh, ldos_plot_data, cmap=cmap_ldos, shading='nearest', rasterized=True)
        fig.colorbar(ldos_map, ax=ax_ldos, label='Normalized LDOS' if normalize_spectra else 'LDOS (arb. units)')
        ax_ldos.set_xlabel("Energy (eV)"); ax_ldos.yaxis.set_ticks_position('right')
        
        points = np.array([np.zeros_like(path_distance), path_distance]).T
        segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
        color_values = (variable_z[:-1] + variable_z[1:]) / 2.0
        norm_topo = plt.Normalize(variable_z.min(), variable_z.max())
        lc = LineCollection(segments, cmap=cmap_topo, norm=norm_topo, linewidth=40)
        lc.set_array(color_values); ax_topo_stripe.add_collection(lc)
        ax_topo_stripe.set(xlim=(-0.1, 0.1), ylim=(path_distance.min(), path_distance.max()), xticks=[], ylabel="Position (Å)", title="Topo")
        cax = fig.add_axes([0.05, 0.4, 0.02, 0.5])
        fig.colorbar(lc, cax=cax, label='Tip Z-Height (Å)'); cax.yaxis.set_ticks_position('left'); cax.yaxis.set_label_position('left')
        
        ax_path_view.set_aspect('equal')
        for i in range(-1, 2):
            for j in range(-1, 2):
                offset = i * self.lv[0, :2] + j * self.lv[1, :2]
                all_atom_types = np.repeat(self.atomtypes, self.atomnums)
                colors = [plt.cm.tab10(k % 10) for k in range(len(self.atomtypes))]
                atom_colors = [colors[self.atomtypes.index(t)] for t in all_atom_types]
                ax_path_view.scatter(self.coord[:, 0] + offset[0], self.coord[:, 1] + offset[1], s=30, c=atom_colors, alpha=0.3)
        ax_path_view.plot(path_coords_xy[:, 0], path_coords_xy[:, 1], color='red', lw=2)
        ax_path_view.set(xlabel='X (Å)', ylabel='Y (Å)', title='Scan Path')
        ax_path_view.grid(True)
        
        ax_topo_line.plot(path_distance, variable_z)
        ax_topo_line.set(ylabel=r'Tip Z-Height ($\AA$)', xlabel=r'Distance Along Path ($\AA$)', title='Calculated Tip Topography', ylim=(0, self.lv[2, 2]))
        ax_topo_line.grid(True)
        
        title_str = (f"Path: ({path_coords[0]:.2f}, {path_coords[1]:.2f}) -> ({path_coords[2]:.2f}, {path_coords[3]:.2f}) | "
                     f"Const Height: {const_height:.1f} Å | Topo Bias: {topo_bias:.2f} V\n"
                     f"Offset: {base_dist:.1f} Å | Scaler: {scaler:.1f}")
        fig.suptitle(title_str, y=0.99)
        plt.show()

    def run_map_scan(self, npts_2d, erange, num_energy_points, topo_bias, const_height, scaler, base_dist, locpot_path, plot_cells, surface_mode, bulk_atom_type, cmap_ldos, cmap_topo, normalize_spectra):
        print("--- Running Variable Height 2D Map Scan ---")
        self.parse_vasp_outputs(locpot_path, surface_mode, bulk_atom_type)
        nx, ny = npts_2d
        u, v = np.linspace(0, 1, nx, endpoint=False), np.linspace(0, 1, ny, endpoint=False)
        u_grid, v_grid = np.meshgrid(u, v)
        map_coords_xy = u_grid.ravel()[:, np.newaxis] * self.lv[0, :2] + v_grid.ravel()[:, np.newaxis] * self.lv[1, :2]
        
        print(f"\n[STEP 1] Topography Scan (Bias = {topo_bias:.2f}V)...")
        const_z = self.z_highest_atom + const_height
        tip_positions_const = np.hstack([map_coords_xy, np.full((nx*ny, 1), const_z)])
        topo_emin, topo_emax = min(0.0, topo_bias), max(0.0, topo_bias)
        topo_spectra, topo_energies = self._calculate_ldos_at_points(tip_positions_const, topo_emin, topo_emax)
        topography_data = np.array([simpson(np.sum(spec, axis=0), x=topo_energies) for spec in topo_spectra]).reshape((ny, nx))
        
        print("\n[STEP 2] Spectroscopic Scan...")
        variable_z = self.z_bulk_surface + base_dist + ((topography_data.ravel() / np.max(topography_data)) * scaler)
        tip_positions_variable = np.hstack([map_coords_xy, variable_z[:, np.newaxis]])
        
        target_energies = np.linspace(erange[0], erange[1], num_energy_points)
        final_spectra_list, final_energies = self._calculate_ldos_at_points(tip_positions_variable, erange[0], erange[1], target_energies=target_energies)
        
        if normalize_spectra:
            print("\nNormalizing final LDOS spectra...")
            final_spectra_list = self._normalize_spectra(final_spectra_list, final_energies)

        num_energies = len(final_energies)
        final_ldos_map = np.sum(np.array(final_spectra_list), axis=1).reshape((ny, nx, num_energies))
        
        title_str = (f"Const Height: {const_height:.1f} Å | Topo Bias: {topo_bias:.2f} V | "
                     f"Offset: {base_dist:.1f} Å | Scaler: {scaler:.1f}")

        fig_topo_map, ax_topo_map = plt.subplots(figsize=(10, 10))
        topo_plot_data = np.tile(topography_data, (2*plot_cells+1, 2*plot_cells+1))
        all_x, all_y = [], []
        for i in range(-plot_cells, plot_cells + 1):
            for j in range(-plot_cells, plot_cells + 1):
                offset = i * self.lv[0, :2] + j * self.lv[1, :2]
                all_x.append(map_coords_xy[:,0] + offset[0]); all_y.append(map_coords_xy[:,1] + offset[1])
        contour = ax_topo_map.tricontourf(np.concatenate(all_x), np.concatenate(all_y), topo_plot_data.ravel(), levels=100, cmap=cmap_topo)
        fig_topo_map.colorbar(contour, ax=ax_topo_map, label='Integrated LDOS (arb. units)')
        ax_topo_map.set_aspect('equal')
        ax_topo_map.set(xlabel='X ($\AA$)', ylabel='Y ($\AA$)'); ax_topo_map.set_title(title_str, y=1.02)
        plt.show()

        output_dir = "ldos_map_energy_slices"; makedirs(output_dir, exist_ok=True)
        print(f"\nSaving {num_energies} energy slice plots to '{output_dir}/' directory...")
        for e_idx, energy in enumerate(final_energies):
            fig, ax = plt.subplots(figsize=(10, 10))
            plot_data = np.tile(final_ldos_map[:, :, e_idx], (2*plot_cells+1, 2*plot_cells+1))
            ax.tricontourf(np.concatenate(all_x), np.concatenate(all_y), plot_data.ravel(), levels=100, cmap=cmap_ldos)
            ax.set_aspect('equal')
            ax.set(title=f'LDOS Map at E = {energy:.3f} eV', xlabel='X ($\AA$)', ylabel='Y ($\AA$)')
            filename = join(output_dir, f"ldos_map_E_{energy:.3f}eV.png"); fig.savefig(filename, dpi=150); plt.close(fig)
        print("--- All plots saved. Simulation complete. ---")

# =============================================================================
# --- USER CONFIGURATION AND EXECUTION ---
# =============================================================================
if __name__ == "__main__":
    
    simulation_mode = 'line'

    vasp_directory = r'E:/VASP/NHC/IPR/SAM/NHC2Au_complexes/p2/spinpol/freegold1/freegold2/freegold3/kpoints551'
    #    vasp_directory = r'C:/Users/Benjamin Kafin/Documents/VASP/SAM/zigzag/kpoints551'
    locpot_file_path = "LOCPOT"
    surface_detection_mode = 'bulk'
    bulk_atom_type = 'Au' 
    normalize_final_spectra = True
    
    cmap_ldos = 'jet'
    cmap_topo = 'copper'
    
    if simulation_mode == 'point':
        tip_position = [1.0, 2.5, 20.0]
        energy_range = [-1.0, 1.0]

    elif simulation_mode == 'line':
        scan_path_coords = [34.87265,  30.65221,  -6.56552,  -7.22864]
        #scan_path_coords = [17.05115,  -8.99863,  0.32571,  24.06844]
        number_of_points = 64
        energy_range = [-2.25, 0.936]
        topo_bias_voltage = 0.75
        const_scan_height = 2.5
        base_tip_distance = 1.5
        tip_height_scaler = 11.25
        ldos_smoothing_sigma = 0.7

    elif simulation_mode == 'map':
        grid_points = [64, 64]
        number_of_energy_points = 50
        energy_range = [-1.0, 1.0]
        topo_bias_voltage = 0.2
        const_scan_height = 15.0
        base_tip_distance = 7.5
        tip_height_scaler = 1.2
        plot_supercell = 1

    # =============================================================================
    sim = STM_Simulator(vasp_directory)
    
    if simulation_mode == 'point':
        sim.run_point_scan(position=tip_position, erange=energy_range, 
                           locpot_path=locpot_file_path, surface_mode=surface_detection_mode,
                           bulk_atom_type=bulk_atom_type)
    elif simulation_mode == 'line':
        sim.run_line_scan(path_coords=scan_path_coords, npts=number_of_points, erange=energy_range, 
                          topo_bias=topo_bias_voltage, const_height=const_scan_height, 
                          scaler=tip_height_scaler, base_dist=base_tip_distance,
                          locpot_path=locpot_file_path, surface_mode=surface_detection_mode,
                          bulk_atom_type=bulk_atom_type,
                          cmap_ldos=cmap_ldos, cmap_topo=cmap_topo, 
                          normalize_spectra=normalize_final_spectra, ldos_smoothing_sigma=ldos_smoothing_sigma)
    elif simulation_mode == 'map':
        sim.run_map_scan(npts_2d=grid_points, erange=energy_range, num_energy_points=number_of_energy_points,
                         topo_bias=topo_bias_voltage, const_height=const_scan_height, 
                         scaler=tip_height_scaler, base_dist=base_tip_distance, plot_cells=plot_supercell,
                         locpot_path=locpot_file_path, surface_mode=surface_detection_mode,
                         bulk_atom_type=bulk_atom_type,
                         cmap_ldos=cmap_ldos, cmap_topo=cmap_topo, normalize_spectra=normalize_final_spectra)