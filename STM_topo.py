import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp  # GPU Acceleration
import cupyx.scipy.ndimage as cp_ndimage # GPU Interpolation
from os.path import exists, getsize, join, isdir
from os import chdir, makedirs
from numpy.linalg import inv
from scipy.integrate import simpson
from matplotlib.colors import LinearSegmentedColormap
from time import time

# --- ROBUST DLL SEARCH ---
try:
    import nvidia
    nvidia_path = os.path.dirname(nvidia.__file__)
    dll_found = False
    for root, dirs, files in os.walk(nvidia_path):
        for f in files:
            if f.startswith('nvrtc64_') and f.endswith('.dll'):
                os.add_dll_directory(root)
                print(f"--- Found and registered CUDA library: {f} ---")
                dll_found = True
                break
        if dll_found: break
except Exception:
    pass

# #############################################################################
# HELPER FUNCTIONS (VASP Parsers)
# #############################################################################

def _parse_vasp_locpot(ifile):
    """Parses a raw VASP LOCPOT file."""
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
    return dos, np.array(energies) - ef, ef, [f'orb_{i}' for i in range(len(dos[1]))]

def parse_poscar(ifile):
    """Original POSCAR parser logic."""
    with open(ifile, 'r') as f:
        lines = f.readlines()
        sf = float(lines[1])
        lv = np.array([float(c) for c in ' '.join(lines[2:5]).split()]).reshape(3,3) * sf
        atomtypes = lines[5].split()
        atomnums = [int(i) for i in lines[6].split()]
        start_line = 7
        if lines[start_line].strip().lower()[0] not in ['d','c']: start_line = 8
        coord_start = start_line + 1
        coord = np.array([[float(c) for c in line.split()[:3]] for line in lines[coord_start:sum(atomnums)+coord_start]])
        if 'direct' in lines[start_line].lower(): coord = np.dot(coord, lv)
    return lv, coord, atomtypes, atomnums

# #############################################################################
# MAIN STM IMAGE SIMULATOR CLASS (FULLY GPU VECTORIZED)
# #############################################################################

class STM_Image_Simulator:
    def __init__(self, filepath):
        self.filepath = filepath
        self.unit_cell_num = 4
        if not exists(filepath): raise FileNotFoundError(f"Directory not found: {filepath}")
        chdir(filepath)
        
        try:
            self.dev = cp.cuda.Device(0)
            print(f"--- INITIALIZING FULL GPU VECTORIZATION ---")
            print(f"Active GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
            print(f"Total VRAM: {self.dev.mem_info[1] / 1024**3:.2f} GB")
            print(f"--------------------------------------------\n")
        except Exception as e:
            print(f"\n[GPU ERROR] Failed to initialize CUDA: {e}")
            sys.exit()

    def _find_bulk_surface_z(self, bin_width=0.5125, filter_atom_type=None):
        z_coords_all = self.coord[:, 2]
        if filter_atom_type:
            all_atom_types = np.repeat(self.atomtypes, self.atomnums)
            type_mask = (all_atom_types == filter_atom_type)
            z_coords_to_scan = z_coords_all[type_mask] if np.any(type_mask) else z_coords_all
        else:
            z_coords_to_scan = z_coords_all
        z_cell_midpoint = self.lv[2, 2] / 2.0
        bins = np.arange(np.min(z_coords_to_scan), z_cell_midpoint + bin_width, bin_width)
        counts, bin_edges = np.histogram(z_coords_to_scan[z_coords_to_scan < z_cell_midpoint], bins=bins)
        surface_z = bin_edges[np.max(np.where(counts == np.max(counts))[0]) + 1] if len(counts) > 0 else np.max(z_coords_all)
        return surface_z

    def parse_vasp_outputs(self, locpot_path, surface_mode='bulk', bulk_atom_type=None):
        poscar_path = './CONTCAR' if exists('./CONTCAR') and getsize('./CONTCAR') > 0 else './POSCAR'
        try:
            self.lv, self.coord, self.atomtypes, self.atomnums = parse_poscar(poscar_path)
            self.dos, self.energies, self.ef, self.orbitals = parse_doscar('./DOSCAR')
            self.z_highest_atom = np.max(self.coord[:, 2])
            self.z_bulk_surface = self._find_bulk_surface_z(filter_atom_type=bulk_atom_type) if surface_mode == 'bulk' else self.z_highest_atom

            locpot_npy_path = join(self.filepath, "LOCPOT.npy")
            self.locpot_data = np.load(locpot_npy_path) if exists(locpot_npy_path) else _parse_vasp_locpot(locpot_path)
            if not exists(locpot_npy_path): np.save(locpot_npy_path, self.locpot_data)
            
            # Move constant data to GPU once
            self.locpot_gpu = cp.array(self.locpot_data)
            self.inv_lv_gpu = cp.array(inv(self.lv))
            self.locpot_dims_gpu = cp.array(self.locpot_data.shape)
            
            # Pre-calculate DOS on GPU
            all_atom_dos = [np.sum([self.dos[a+1][o] for o in range(len(self.orbitals))], axis=0) for a in range(sum(self.atomnums))]
            self.dos_gpu = cp.array(all_atom_dos)

            print("Data structures loaded to GPU VRAM.")
        except Exception as e:
            print(f"Error reading VASP files: {e}"); sys.exit()

    def _calculate_integrated_ldos_gpu(self, tip_positions, emin, emax):
        """FULLY VECTORIZED GPU Integrated LDOS (No Python Loops)"""
        estart, eend = np.searchsorted(self.energies, [emin, emax])
        calc_energies_gpu = cp.array(self.energies[estart:eend])
        tip_pos_gpu = cp.array(tip_positions) 
        
        # 1. Vectorized Phi Interpolation directly on GPU
        frac_coords = cp.dot(tip_pos_gpu, self.inv_lv_gpu)
        grid_indices = (frac_coords % 1.0).T * self.locpot_dims_gpu[:, None]
        phi_local = cp_ndimage.map_coordinates(self.locpot_gpu, grid_indices, order=1, mode='wrap')
        kappa = 0.512 * cp.sqrt(cp.maximum(0.1, phi_local - self.ef))
        
        # 2. Periodic Coordinates on GPU
        if not hasattr(self, 'periodic_coord_gpu'):
            coords, idx_list = [], []
            base_idx = np.arange(len(self.coord))
            for i in range(-self.unit_cell_num, self.unit_cell_num + 1):
                for j in range(-self.unit_cell_num, self.unit_cell_num + 1):
                    coords.append(self.coord + self.lv[0] * i + self.lv[1] * j)
                    idx_list.append(base_idx)
            self.periodic_coord_gpu = cp.array(np.concatenate(coords))
            self.atom_indices_periodic_gpu = cp.array(np.concatenate(idx_list))

        # 3. Vectorized Distance and SF Calculation (Broadcasting)
        diffs = self.periodic_coord_gpu[:, None, :] - tip_pos_gpu[None, :, :]
        dists = cp.sqrt(cp.sum(diffs**2, axis=2))
        sf_batch = cp.exp(-2.0 * kappa[None, :] * dists)
        
        # 4. Summing SF contributions per atom and weighting DOS
        num_atoms = sum(self.atomnums)
        weight_per_atom = cp.zeros((num_atoms, tip_positions.shape[0]))
        for atom_idx in range(num_atoms):
            weight_per_atom[atom_idx] = cp.sum(sf_batch[self.atom_indices_periodic_gpu == atom_idx], axis=0)
            
        total_ldos = cp.dot(weight_per_atom.T, self.dos_gpu[:, estart:eend])
        
        # 5. Integrate on GPU
        return cp.asnumpy(cp.trapz(total_ldos, x=calc_energies_gpu, axis=1))

    def run_stm_image(self, grid_res, topo_bias, const_height, plot_cells=1, cmap_topo=None):
        print(f"--- Running FULL GPU Vectorized Map Scan ({grid_res}x{grid_res}) ---")
        start_time = time()
        u = np.linspace(0, 1, grid_res); v = np.linspace(0, 1, grid_res)
        uu, vv = np.meshgrid(u, v)
        grid_xy = (uu.ravel()[:, None] * self.lv[0, :2]) + (vv.ravel()[:, None] * self.lv[1, :2])
        
        print(f"\n[STEP 1] Initial Scan for Setpoint (Bias = {topo_bias:.2f}V, Height = {const_height} Å)...")
        current_z_profile = np.full(grid_xy.shape[0], self.z_highest_atom + const_height)
        e_min, e_max = sorted([0.0, topo_bias])
        
        current_ldos_int = self._calculate_integrated_ldos_gpu(np.hstack([grid_xy, current_z_profile[:, None]]), e_min, e_max)
        target_setpoint = np.max(current_ldos_int)
        print(f"Setpoint: {target_setpoint:.4e}")
        
        print("\n[STEP 2] Iterative Height Convergence...")
        for iteration in range(30):
            currents = self._calculate_integrated_ldos_gpu(np.hstack([grid_xy, current_z_profile[:, None]]), e_min, e_max)
            ratios = np.maximum(currents, 1e-20) / target_setpoint
            max_error = np.max(np.abs(ratios - 1.0))
            
            # --- HIGH SPECIFICITY ITERATION STATS ---
            z_min, z_max = np.min(current_z_profile), np.max(current_z_profile)
            z_range, mean_z = z_max - z_min, np.mean(current_z_profile)
            print(f"   Iter {iteration+1:02d}: Error = {max_error*100:6.2f}% | Mean Z = {mean_z:6.3f} Å | Range = {z_range:8.4f} Å | (Min: {z_min:8.4f} Å, Max: {z_max:8.4f} Å)")
            
            if max_error < 0.01:
                print("   Convergence reached."); break
            current_z_profile += 0.5 * np.log(ratios)

        print(f"\nTotal Simulation Time: {time() - start_time:.2f} seconds.")
        self.last_scan = {'x': grid_xy[:, 0], 'y': grid_xy[:, 1], 'z': current_z_profile}
        self.plot_result(grid_xy, current_z_profile, plot_cells, cmap_topo, topo_bias, const_height)

    def save_topography_data(self, filename="STM_topo_data.npy"):
        if not hasattr(self, 'last_scan'): return
        save_path = join(self.filepath, filename)
        np.save(save_path, self.last_scan)
        print(f"--- Data saved to: {save_path} ---")

    def plot_result(self, grid_xy, z_profile, plot_cells, cmap, bias, height):
        fig, ax = plt.subplots(figsize=(10, 10))
        all_x, all_y, all_z = [], [], []
        for i in range(plot_cells):
            for j in range(plot_cells):
                offset = i * self.lv[0, :2] + j * self.lv[1, :2]
                all_x.append(grid_xy[:, 0] + offset[0]); all_y.append(grid_xy[:, 1] + offset[1]); all_z.append(z_profile)
        img = ax.tricontourf(np.concatenate(all_x), np.concatenate(all_y), np.concatenate(all_z), levels=100, cmap=cmap)
        plt.colorbar(img, ax=ax, label='Tip Height (Å)')
        ax.set_aspect('equal')
        ax.set(title=f"Full GPU Vectorized STM Topography\nBias: {bias}V", xlabel="X (Å)", ylabel="Y (Å)")
        plt.show()

if __name__ == "__main__":
    custom_topo = LinearSegmentedColormap.from_list("custom_topo", ["black", "firebrick", "yellow"])
    vasp_dir = r'dir'
    sim = STM_Image_Simulator(vasp_dir)
    sim.parse_vasp_outputs(locpot_path="LOCPOT", bulk_atom_type='Au')
    sim.run_stm_image(grid_res=64, topo_bias=0.2, const_height=0.75, plot_cells=2, cmap_topo=custom_topo)

    sim.save_topography_data("Converged_STM_FullCell_GPU_Vectorized.npy")
