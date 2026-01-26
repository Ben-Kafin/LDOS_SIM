import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import cupy as cp 
import cupyx.scipy.ndimage as cp_ndimage 
from os.path import exists, getsize, join
from os import chdir
from numpy.linalg import norm, inv
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec

# --- DLL SEARCH ---
try:
    import nvidia
    nvidia_path = os.path.dirname(nvidia.__file__)
    for root, dirs, files in os.walk(nvidia_path):
        for f in files:
            if f.startswith('nvrtc64_') and f.endswith('.dll'):
                os.add_dll_directory(root)
                break
except Exception:
    pass

# #############################################################################
# VECTORIZED SIMPSON'S RULE (GPU)
# #############################################################################
def gpu_simpson(y, x):
    """Vectorized Simpson's Rule for GPU parity."""
    n = y.shape[1]
    if n % 2 == 0:
        return cp.trapz(y, x=x, axis=1)
    dx = (x[-1] - x[0]) / (n - 1)
    weights = cp.ones(n)
    weights[1:-1:2] = 4
    weights[2:-2:2] = 2
    return (dx / 3.0) * cp.sum(weights * y, axis=1)

# #############################################################################
# MAIN STM_Simulator CLASS
# #############################################################################

class STM_Simulator:
    def __init__(self, filepath):
        self.filepath = filepath
        self.unit_cell_num = 4
        chdir(filepath)
        print(f"--- INITIALIZING GPU LDOS SIMULATOR (RTX 4080) ---")

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
        return bin_edges[np.max(np.where(counts == np.max(counts))[0]) + 1] if len(counts) > 0 else np.max(z_coords_all)

    def _parse_poscar(self, ifile):
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

    def _parse_doscar(self, filepath):
        with open(filepath, 'r') as f:
            atomnum = int(f.readline().split()[0])
            [f.readline() for _ in range(4)]
            line = f.readline().split()
            nedos, ef = int(line[2]), float(line[3])
            dos, energies = [], []
            for i in range(atomnum + 1):
                if i != 0: f.readline()
                for j in range(nedos):
                    l = f.readline().split()
                    if i == 0: energies.append(float(l[0]))
                    if j == 0: t_dos = [[] for _ in range(len(l) - 1)]
                    for k in range(len(l) - 1): t_dos[k].append(float(l[k+1]))
                dos.append(t_dos)
        return dos, np.array(energies) - ef, ef, None

    def parse_vasp_outputs(self, locpot_path, surface_mode='bulk', bulk_atom_type=None):
        poscar_path = './CONTCAR' if exists('./CONTCAR') and getsize('./CONTCAR') > 0 else './POSCAR'
        self.lv, self.coord, self.atomtypes, self.atomnums = self._parse_poscar(poscar_path)
        self.dos, self.energies, self.ef, _ = self._parse_doscar('./DOSCAR')
        self.z_highest_atom = np.max(self.coord[:, 2])
        self.z_bulk_surface = self._find_bulk_surface_z(filter_atom_type=bulk_atom_type) if surface_mode == 'bulk' else self.z_highest_atom
        
        self.locpot_gpu = cp.array(np.load(join(self.filepath, "LOCPOT.npy")))
        self.inv_lv_gpu = cp.array(inv(self.lv))
        self.locpot_dims_gpu = cp.array(self.locpot_gpu.shape)
        self.dos_gpu = cp.array([np.sum(self.dos[i+1], axis=0) for i in range(sum(self.atomnums))])
        
        coords, idx_list = [], []
        base_idx = np.arange(len(self.coord))
        for i in range(-self.unit_cell_num, self.unit_cell_num + 1):
            for j in range(-self.unit_cell_num, self.unit_cell_num + 1):
                coords.append(self.coord + self.lv[0] * i + self.lv[1] * j)
                idx_list.append(base_idx)
        self.periodic_coord_gpu = cp.array(np.concatenate(coords))
        self.atom_indices_periodic_gpu = cp.array(np.concatenate(idx_list))

    def _calculate_ldos_at_points_gpu(self, tip_positions, emin, emax, target_energies=None):
        if target_energies is not None:
            energy_indices = cp.array([np.argmin(np.abs(self.energies - E)) for E in target_energies])
            calc_energies_gpu = cp.array(target_energies)
        else:
            estart = np.searchsorted(self.energies, emin)
            eend = np.searchsorted(self.energies, emax, side='right')
            energy_indices = cp.arange(estart, eend)
            calc_energies_gpu = cp.array(self.energies[estart:eend])
        
        tip_pos_gpu = cp.array(tip_positions)
        frac_coords = cp.dot(tip_pos_gpu, self.inv_lv_gpu)
        grid_indices = (frac_coords % 1.0).T * self.locpot_dims_gpu[:, None]
        phi_local = cp_ndimage.map_coordinates(self.locpot_gpu, grid_indices, order=1, mode='wrap')
        kappa = 0.512 * cp.sqrt(cp.maximum(0.1, phi_local - self.ef))
        
        diffs = self.periodic_coord_gpu[:, None, :] - tip_pos_gpu[None, :, :]
        dists = cp.sqrt(cp.sum(diffs**2, axis=2))
        sf_batch = cp.exp(-2.0 * kappa[None, :] * dists)
        
        weight_per_atom = cp.zeros((sum(self.atomnums), tip_positions.shape[0]))
        for atom_idx in range(sum(self.atomnums)):
            weight_per_atom[atom_idx] = cp.sum(sf_batch[self.atom_indices_periodic_gpu == atom_idx], axis=0)
            
        total_ldos = cp.dot(weight_per_atom.T, self.dos_gpu[:, energy_indices])
        return total_ldos, calc_energies_gpu

    def run_line_scan(self, path_coords, npts, erange, topo_bias, const_height, locpot_path, surface_mode, bulk_atom_type, cmap_ldos, cmap_topo, normalize_spectra, highlight_positions=None):
        print("--- Running Constant-Current (Iterative) Line Scan ---")
        self.parse_vasp_outputs(locpot_path, surface_mode, bulk_atom_type)
        path_xy = np.array([np.linspace(path_coords[0], path_coords[2], npts), np.linspace(path_coords[1], path_coords[3], npts)]).T
        path_dist = np.linspace(0, norm(np.array(path_coords[2:]) - np.array(path_coords[:2])), npts)
        
        print(f"\n[STEP 1] Initial Scan for Setpoint (Bias = {topo_bias:.2f}V, Height = {const_height} A)...")
        current_z = np.full(npts, self.z_highest_atom + const_height)
        t_emin, t_emax = sorted([0.0, topo_bias])
        
        def get_current(z_vals):
            pts = np.hstack([path_xy, z_vals[:, None]])
            ldos, engs = self._calculate_ldos_at_points_gpu(pts, t_emin, t_emax)
            return cp.asnumpy(gpu_simpson(ldos, x=engs))
            
        current_ldos_int = get_current(current_z)
        target_setpoint = np.max(current_ldos_int)
        
        print("\n[STEP 2] Feedback Loop Iterations...")
        for iteration in range(25):
            ratios = np.maximum(current_ldos_int, 1e-20) / target_setpoint
            max_error = np.max(np.abs(ratios - 1.0))
            
            # --- HIGH SPECIFICITY ITERATION STATS ---
            z_min, z_max = np.min(current_z), np.max(current_z)
            z_range = z_max - z_min
            mean_z = np.mean(current_z)
            
            print(f"   Iter {iteration+1:02d}: Error ={max_error*100:6.2f}% | Mean Z ={mean_z:6.3f} Å | Range ={z_range:8.4f} Å | (Min:{z_min:8.4f} Å, Max:{z_max:8.4f} Å)")
            
            if max_error < 0.01: break
            current_z += 0.5 * np.log(ratios)
            current_ldos_int = get_current(current_z)

        print(f"\n[STEP 3] Final Spectroscopic Scan...")
        f_ldos_gpu, f_engs_gpu = self._calculate_ldos_at_points_gpu(np.hstack([path_xy, current_z[:, None]]), erange[0], erange[1])
        
        if normalize_spectra:
            areas = gpu_simpson(f_ldos_gpu, x=f_engs_gpu)
            f_ldos_gpu = f_ldos_gpu / areas[:, cp.newaxis]

        f_ldos, f_engs = cp.asnumpy(f_ldos_gpu), cp.asnumpy(f_engs_gpu)
        self.scan_data = {'spectra': f_ldos, 'energies': f_engs, 'distances': path_dist}
        
        self._plot_integrated_results(path_coords, path_dist, current_z, f_ldos, f_engs, cmap_ldos, cmap_topo, highlight_positions, const_height, topo_bias, erange, normalize_spectra)

    def _plot_integrated_results(self, path_coords, path_distance, variable_z, f_ldos, f_engs, cmap_ldos, cmap_topo, highlights, const_height, topo_bias, erange, normalize_spectra):
        fig = plt.figure(figsize=(15, 10))
        outer_gs = gridspec.GridSpec(2, 1, height_ratios=[2.5, 1], hspace=0.3)
        top_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[0], width_ratios=[0.15, 2, 1], wspace=0.2)
        map_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=top_gs[:2], width_ratios=[0.15, 2], wspace=0)
        
        ax_stripe = fig.add_subplot(map_gs[0])
        ax_ldos = fig.add_subplot(map_gs[1], sharey=ax_stripe)
        ax_slices = fig.add_subplot(top_gs[2])
        ax_line = fig.add_subplot(outer_gs[1])
        
        # 1. Topo Stripe
        points = np.array([np.zeros_like(path_distance), path_distance]).T
        segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
        lc = LineCollection(segments, cmap=cmap_topo, norm=plt.Normalize(variable_z.min(), variable_z.max()), linewidth=40)
        lc.set_array((variable_z[:-1] + variable_z[1:]) / 2.0)
        ax_stripe.add_collection(lc)
        ax_stripe.set(xlim=(-0.1, 0.1), ylim=(path_distance.min(), path_distance.max()), xticks=[], ylabel="Position (Å)", title="Topo")
        
        # 2. LDOS Color Map & Synchronized Indicators
        e_mesh, d_mesh = np.meshgrid(f_engs, path_distance)
        ldos_map = ax_ldos.pcolormesh(e_mesh, d_mesh, f_ldos, cmap=cmap_ldos, shading='gouraud', rasterized=True)
        
        if highlights:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            for i, p in enumerate(highlights):
                color = colors[i % len(colors)]
                idx = np.argmin(np.abs(path_distance - p))
                
                # Dashed line on Colored Topo Stripe
                ax_stripe.axhline(y=p, color=color, ls='--', alpha=0.8)
                
                # Dashed line on Spec Map
                ax_ldos.axhline(y=p, color=color, ls='--', alpha=0.8)
                
                # Dashed line on Topo Plot
                ax_line.axvline(x=p, color=color, ls='--', alpha=0.6)
                
                # Corresponding Curve in Point Spectra
                ax_slices.plot(f_engs, f_ldos[idx, :], color=color, label=f'{path_distance[idx]:.2f} Å')
            
            ax_slices.legend(); ax_slices.grid(True)
            ax_slices.set(xlabel="Energy (eV)", ylabel="LDOS Intensity", title="Point Spectra")
            ax_slices.tick_params(left=False, labelleft=False)

        ax_ldos.set(xlabel="Energy (eV)", title="LDOS Spectroscopic Map")
        ax_ldos.tick_params(left=False, labelleft=False)
        
        # 3. Topo Line Plot
        ax_line.plot(path_distance, variable_z, color='black', lw=1.5)
        ax_line.set(ylabel=r'Tip Z-Height ($\AA$)', xlabel=r'Distance Along Path ($\AA$)', 
                     title='Converged Constant-Current Topography', ylim=(variable_z.min()*0.95, variable_z.max()*1.05))
        ax_line.grid(True)
        
        # Metadata
        norm_status = "ON" if normalize_spectra else "OFF"
        title_str = (f"Path: ({path_coords[0]:.2f}, {path_coords[1]:.2f}) -> ({path_coords[2]:.2f}, {path_coords[3]:.2f}) | "
                     f"Setpoint Defined at Height: {const_height:.1f} Å | Normalization: {norm_status}\n"
                     f"Topo Bias: {topo_bias:.2f} V | Spec Range: {erange}")
        fig.suptitle(title_str, y=0.98, fontsize=12)
        plt.show()

if __name__ == "__main__":
    v_dir = r'dir'
    sim = STM_Simulator(v_dir)
    #my_slices = [29.5, 37.6]
    my_slices = [17.64, 24.79]
    
    sim.run_line_scan(
        #path_coords=[34.87265, 30.65221, -6.56552, -7.22864],
        path_coords = [13.27028,  -8.77865, 4.00869,  23.77771],
        #npts=72, erange=[-2.125, 1.0], topo_bias=1.0, const_height=1.0,
        npts=72, erange=[-2.525, -1.3], topo_bias=-2.525, const_height=1.25,
        locpot_path="LOCPOT", surface_mode='bulk', bulk_atom_type='Au',
        cmap_ldos='jet', cmap_topo=LinearSegmentedColormap.from_list("t", ["black", "firebrick", "yellow"]),
        normalize_spectra=False, highlight_positions=my_slices

    )
