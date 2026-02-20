# -*- coding: utf-8 -*-
"""
Interactive STM Simulator: Multi-Tiered GPU Optimization
Flush Layout with Clean Vertical Topography Stripe
Developed for Benjamin Carlos Kafin
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp  
import cupyx.scipy.ndimage as cp_ndimage 
from os.path import exists, getsize, join
from os import chdir
from numpy.linalg import norm, inv
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider, CheckButtons, Button
import matplotlib.gridspec as gridspec
from time import time

# --- ROBUST DLL SEARCH ---
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

# --- CORE UTILITIES ---
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

def gpu_chen_tunneling_factor(V, E, phi):
    """Vectorized Julian Chen barrier model for GPU."""
    me, h, q = 9.1093837e-31, 6.62607015e-34, 1.60217663e-19
    V_eff = cp.where(cp.abs(V) < 1e-6, 1e-6, V)
    V_j, E_j, phi_j = cp.abs(V_eff) * q, E * q, phi * q
    prefactor = (8.0 / (3.0 * V_j)) * cp.pi * cp.sqrt(2.0 * me) / h
    term1 = cp.power(cp.maximum(0.1*q, phi_j - E_j + V_j), 1.5)
    term2 = cp.power(cp.maximum(0.1*q, phi_j - E_j), 1.5)
    return prefactor * (term1 - term2)

class Unified_STM_Simulator:
    def __init__(self, filepath):
        self.filepath = filepath
        self.unit_cell_num = 4
        if not exists(filepath): raise FileNotFoundError(f"Directory not found: {filepath}")
        chdir(filepath)
        try:
            self.dev = cp.cuda.Device(0)
            print(f"--- INITIALIZING GPU TENSOR SIMULATOR (RTX 4080) ---")
        except Exception as e:
            print(f"\n[GPU ERROR] Failed to initialize CUDA: {e}"); sys.exit()

    def _print_iteration_status(self, phase, iteration, t0, z_vals, ratios):
        cp.cuda.Stream.null.synchronize()
        elapsed = time() - t0
        max_error = np.max(np.abs(ratios - 1.0))
        z_min, z_max = np.min(z_vals), np.max(z_vals)
        mean_z = np.mean(z_vals)
        print(f"   {phase} Iter {iteration+1:02d}: Error ={max_error*100:6.2f}% | Mean Z ={mean_z:6.3f} Å | Range ={z_max-z_min:8.4f} Å | Time ={elapsed:6.3f}s")
        return max_error

    def _parse_poscar(self, ifile):
        with open(ifile, 'r') as f:
            lines = f.readlines()
            sf = float(lines[1])
            lv = np.array([float(c) for c in ' '.join(lines[2:5]).split()]).reshape(3,3) * sf
            atomtypes = lines[5].split()
            atomnums = [int(i) for i in lines[6].split()]
            start_line = 7 if lines[7].strip().lower()[0] in ['d', 'c'] else 8
            coord = np.array([[float(c) for c in line.split()[:3]] for line in lines[start_line+1:sum(atomnums)+start_line+1]])
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

    def _parse_locpot(self, ifile):
        with open(ifile, 'r') as f:
            [f.readline() for _ in range(5)]; f.readline()
            atom_counts = [int(i) for i in f.readline().split()]; f.readline()
            [f.readline() for _ in range(sum(atom_counts))]; f.readline()
            dim = [int(i) for i in f.readline().split()]
            data = np.fromfile(f, sep=' ', count=np.prod(dim)).reshape(dim, order='F')
        return data

    def parse_vasp_outputs(self, locpot_path):
        poscar_path = './CONTCAR' if exists('./CONTCAR') and getsize('./CONTCAR') > 0 else './POSCAR'
        self.lv, self.coord, self.atomtypes, self.atomnums = self._parse_poscar(poscar_path)
        self.dos, self.energies, self.ef, _ = self._parse_doscar('./DOSCAR')
        self.z_highest_atom = np.max(self.coord[:, 2])
        locpot_npy_path = join(self.filepath, "LOCPOT.npy")
        locpot_data = np.load(locpot_npy_path) if exists(locpot_npy_path) else self._parse_locpot(locpot_path)
        self.locpot_gpu = cp.array(locpot_data, dtype=cp.float32)
        self.inv_lv_gpu = cp.array(inv(self.lv), dtype=cp.float32)
        self.locpot_dims_gpu = cp.array(self.locpot_gpu.shape, dtype=cp.float32)
        self.num_total_atoms = sum(self.atomnums)
        self.dos_gpu = cp.array([np.sum(self.dos[i+1], axis=0) for i in range(self.num_total_atoms)], dtype=cp.float32)
        coords, idx_list = [], []
        base_idx = np.arange(len(self.coord))
        for i in range(-self.unit_cell_num, self.unit_cell_num + 1):
            for j in range(-self.unit_cell_num, self.unit_cell_num + 1):
                coords.append(self.coord + self.lv[0] * i + self.lv[1] * j)
                idx_list.append(base_idx)
        self.periodic_coord_gpu = cp.array(np.concatenate(coords), dtype=cp.float32)
        self.atom_indices_periodic_gpu = cp.array(np.concatenate(idx_list))

    def _calculate_ldos_at_points_gpu(self, tip_positions, emin, emax, use_energy_decay=False):
        estart, eend = np.searchsorted(self.energies, emin), np.searchsorted(self.energies, emax, side='right')
        energy_indices = cp.arange(estart, eend)
        calc_energies_gpu = cp.array(self.energies[estart:eend], dtype=cp.float32)
        tip_pos_gpu = cp.array(tip_positions, dtype=cp.float32)
        frac_coords = cp.dot(tip_pos_gpu, self.inv_lv_gpu)
        grid_indices = (frac_coords % 1.0).T * self.locpot_dims_gpu[:, None]
        phi_local = cp_ndimage.map_coordinates(self.locpot_gpu, grid_indices, order=1, mode='wrap') - self.ef
        diffs = self.periodic_coord_gpu[:, None, :] - tip_pos_gpu[None, :, :]
        dists = cp.sqrt(cp.sum(diffs**2, axis=2))

        if use_energy_decay:
            bias_v = cp.array(emax - emin, dtype=cp.float32)
            K = gpu_chen_tunneling_factor(bias_v, calc_energies_gpu[:, None], phi_local[None, :])
            sf_batch = cp.exp(-1.0 * dists[None, :, :] * K[:, None, :] * 1e-10)
            weight_per_atom = cp.zeros((len(calc_energies_gpu), self.num_total_atoms, tip_positions.shape[0]), dtype=cp.float32)
            for atom_idx in range(self.num_total_atoms):
                weight_per_atom[:, atom_idx, :] = cp.sum(sf_batch[:, self.atom_indices_periodic_gpu == atom_idx, :], axis=1)
            total_ldos = cp.sum(weight_per_atom * self.dos_gpu[:, energy_indices, None].transpose(1, 0, 2), axis=1).T
        else:
            kappa = 0.512 * cp.sqrt(cp.maximum(0.1, phi_local))
            sf_batch = cp.exp(-2.0 * kappa[None, :] * dists)
            weight_per_atom = cp.zeros((self.num_total_atoms, tip_positions.shape[0]), dtype=cp.float32)
            for atom_idx in range(self.num_total_atoms):
                weight_per_atom[atom_idx] = cp.sum(sf_batch[self.atom_indices_periodic_gpu == atom_idx], axis=0)
            total_ldos = cp.dot(weight_per_atom.T, self.dos_gpu[:, energy_indices])
        return total_ldos, calc_energies_gpu

class Interactive_STM_Simulator(Unified_STM_Simulator):
    def __init__(self, filepath, path_coords, erange, highlights, ldos_height, cmap_topo):
        super().__init__(filepath)
        self.p1, self.p2 = np.array(path_coords[:2]), np.array(path_coords[2:])
        self.erange = list(erange)
        self.ldos_height = ldos_height
        self.cmap_topo = cmap_topo
        self.npts = 72
        self.is_running, self.use_decay, self.normalize = False, True, False
        self.display_cells = 1
        
        # Cache Tier Logic
        self.cached_p1, self.cached_p2 = None, None
        self.cached_emin, self.cached_emax = None, None
        self.cached_decay = None
        self.current_z_line = None
        self.cached_f_ldos = None
        self.cached_f_engs = None

        dist = norm(self.p2 - self.p1)
        self.marker_ratios = [np.clip(h/dist, 0, 1) for h in highlights] if dist > 0 else [0.2, 0.8]
        self.active_obj = None

    def run_interactive(self, grid_res=64, topo_bias=0.2, topo_height=2.5, ldos_bias_sign='neg'):
        self.parse_vasp_outputs("LOCPOT")
        self.ldos_bias_sign = ldos_bias_sign
        self.current_z_line = np.full(self.npts, self.z_highest_atom + self.ldos_height)

        print(f"\n--- Phase 1: Global Topography ---")
        u, v = np.linspace(0, 1, grid_res), np.linspace(0, 1, grid_res)
        uu, vv = np.meshgrid(u, v)
        self.grid_xy = (uu.ravel()[:, None] * self.lv[0, :2]) + (vv.ravel()[:, None] * self.lv[1, :2])
        self.current_z_map = np.full(self.grid_xy.shape[0], self.z_highest_atom + topo_height)
        t_emin, t_emax = sorted([0.0, topo_bias])
        
        def get_grid_current(z_vals):
            pts = np.hstack([self.grid_xy, z_vals[:, None]])
            ldos, engs = self._calculate_ldos_at_points_gpu(pts, t_emin, t_emax, use_energy_decay=False)
            return cp.asnumpy(gpu_simpson(ldos, x=engs))

        target_setp = np.max(get_grid_current(self.current_z_map))
        for iteration in range(50):
            t0 = time(); ratios = np.maximum(get_grid_current(self.current_z_map), 1e-20) / target_setp
            self.current_z_map += 0.5 * np.log(ratios)
            if self._print_iteration_status("Global-Topo", iteration, t0, self.current_z_map, ratios) < 0.01: break

        self.fig = plt.figure(figsize=(20, 14))
        gs = gridspec.GridSpec(3, 2, height_ratios=[2.5, 1, 0.25], hspace=0.35, wspace=0.25)
        self.ax_map = self.fig.add_subplot(gs[0, 0])
        
        # Flush Layout: wspace=0.0
        lgs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], width_ratios=[0.08, 2], wspace=0.0)
        self.ax_stripe, self.ax_ldos = self.fig.add_subplot(lgs[0]), self.fig.add_subplot(lgs[1])
        self.ax_prof, self.ax_spec = self.fig.add_subplot(gs[1, 0]), self.fig.add_subplot(gs[1, 1])

        self.line_art, = self.ax_map.plot([], [], 'r--', lw=2.5, zorder=5)
        self.ends = self.ax_map.scatter([], [], c='white', edgecolors='red', s=100, zorder=10, picker=5)
        self.marks = self.ax_map.scatter([], [], s=150, edgecolors='black', zorder=15, picker=5)
        self.m_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        ax_run = plt.axes([0.02, 0.02, 0.08, 0.06]); self.btn_run = Button(ax_run, 'RUN', color='lightgray', hovercolor='lime')
        ax_chk = plt.axes([0.11, 0.02, 0.12, 0.06]); self.chk = CheckButtons(ax_chk, ['Decay', 'Norm'], [self.use_decay, self.normalize])
        ax_cell = plt.axes([0.25, 0.02, 0.15, 0.03]); self.s_cell = Slider(ax_cell, 'Cells', 0, 4, valinit=self.display_cells, valstep=1)
        ax_emin = plt.axes([0.45, 0.05, 0.25, 0.02]); self.s_emin = Slider(ax_emin, 'E Min', -5.0, 5.0, valinit=self.erange[0])
        ax_emax = plt.axes([0.45, 0.02, 0.25, 0.02]); self.s_emax = Slider(ax_emax, 'E Max', -5.0, 5.0, valinit=self.erange[1])

        self.fig.canvas.mpl_connect('pick_event', self._on_pick)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self._on_rel)
        self.btn_run.on_clicked(self._toggle_run); self.chk.on_clicked(self._on_ui_change)
        for s in [self.s_cell, self.s_emin, self.s_emax]: s.on_changed(self._on_ui_change)

        self._update_all(full_refresh=True)
        plt.show()

    def _toggle_run(self, event):
        self.is_running = not self.is_running
        self.btn_run.label.set_text('STOP' if self.is_running else 'RUN')
        self.btn_run.color = 'salmon' if self.is_running else 'lightgray'
        if self.is_running: self._update_all()

    def _update_all(self, full_refresh=False):
        if full_refresh:
            self.ax_map.clear(); n = int(self.s_cell.val)
            for i in range(-n, n + 1):
                for j in range(-n, n + 1):
                    off = i * self.lv[0, :2] + j * self.lv[1, :2]
                    self.ax_map.tricontourf(self.grid_xy[:, 0] + off[0], self.grid_xy[:, 1] + off[1], self.current_z_map, levels=60, cmap=self.cmap_topo, zorder=1)
                    tr = np.repeat(self.atomtypes, self.atomnums)
                    for t_idx, t_name in enumerate(self.atomtypes):
                        m = (tr == t_name); self.ax_map.scatter(self.coord[m, 0] + off[0], self.coord[m, 1] + off[1], s=10, color=plt.cm.tab10(t_idx/10), alpha=0.3, zorder=2)
            self.ax_map.set_aspect('equal'); self.ax_map.add_line(self.line_art); self.ax_map.add_collection(self.ends); self.ax_map.add_collection(self.marks)

        v = self.p2 - self.p1; p_len = norm(v); p_dist = np.linspace(0, p_len, self.npts); p_xy = np.array([self.p1 + r * v for r in np.linspace(0, 1, self.npts)])
        self.marks.set_offsets(np.array([self.p1 + r * v for r in self.marker_ratios])); self.marks.set_facecolors(self.m_colors[:len(self.marker_ratios)])
        self.line_art.set_data([self.p1[0], self.p2[0]], [self.p1[1], self.p2[1]]); self.ends.set_offsets([self.p1, self.p2])

        if not self.is_running: self.fig.canvas.draw_idle(); return

        # --- TIERED LOGIC ---
        bias_energy = self.s_emin.val if str(self.ldos_bias_sign).lower() in ['neg', '-', 'negative'] else self.s_emax.val
        needs_topo_refresh = (self.cached_p1 is None or not np.array_equal(self.p1, self.cached_p1) or 
                              not np.array_equal(self.p2, self.cached_p2) or self.cached_bias_energy != bias_energy or
                              self.cached_decay != self.use_decay)
        needs_heatmap_refresh = (needs_topo_refresh or self.cached_emin != self.s_emin.val or self.cached_emax != self.s_emax.val)

        if needs_topo_refresh:
            print(f"\n[RUN] Recalculating Path-Topo at {bias_energy:.3f}V...")
            self.current_z_line = np.full(self.npts, self.z_highest_atom + self.ldos_height)
            l_emin, l_emax = sorted([0.0, bias_energy])
            def get_line_cur(z_vals):
                ldos, engs = self._calculate_ldos_at_points_gpu(np.hstack([p_xy, z_vals[:, None]]), l_emin, l_emax, use_energy_decay=self.use_decay)
                return cp.asnumpy(gpu_simpson(ldos, x=engs))
            target_setp = np.max(get_line_cur(self.current_z_line))
            for iteration in range(25):
                t0 = time(); ratios = np.maximum(get_line_cur(self.current_z_line), 1e-20) / target_setp
                self.current_z_line += 0.5 * np.log(ratios)
                if self._print_iteration_status("Path-Topo", iteration, t0, self.current_z_line, ratios) < 0.01: break
            self.cached_p1, self.cached_p2, self.cached_bias_energy, self.cached_decay = self.p1.copy(), self.p2.copy(), bias_energy, self.use_decay

        if needs_heatmap_refresh:
            print("[RUN] Refreshing LDOS Heatmap...")
            lg, eg = self._calculate_ldos_at_points_gpu(np.hstack([p_xy, self.current_z_line[:, None]]), self.s_emin.val, self.s_emax.val, use_energy_decay=self.use_decay)
            self.cached_f_ldos, self.cached_f_engs = cp.asnumpy(lg), cp.asnumpy(eg)
            self.cached_emin, self.cached_emax = self.s_emin.val, self.s_emax.val

        f_ldos = self.cached_f_ldos.copy()
        if self.normalize: f_ldos /= (np.trapz(f_ldos, x=self.cached_f_engs, axis=1)[:, None] + 1e-15)

        self.ax_ldos.clear(); self.ax_spec.clear(); self.ax_prof.clear(); self.ax_stripe.clear()
        
        # ax_stripe: KEEP Y-labels/ticks, REMOVE X-labels/ticks/values
        lc = LineCollection(np.array([np.array([np.zeros_like(p_dist), p_dist]).T[:-1], np.array([np.zeros_like(p_dist), p_dist]).T[1:]]).transpose(1, 0, 2), 
                             cmap=self.cmap_topo, norm=plt.Normalize(self.current_z_line.min(), self.current_z_line.max()), linewidth=40)
        lc.set_array(self.current_z_line[:-1]); self.ax_stripe.add_collection(lc)
        self.ax_stripe.set(xlim=(-0.1, 0.1), ylim=(0, p_len), ylabel="Path Dist (Å)")
        self.ax_stripe.set_xticks([]); self.ax_stripe.set_xlabel("")
        self.ax_stripe.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        # ax_ldos: REMOVE Y-labels/ticks/values
        self.ax_ldos.pcolormesh(self.cached_f_engs, p_dist, f_ldos, cmap='jet', shading='auto')
        self.ax_ldos.set(xlabel="Energy (eV)")
        self.ax_ldos.tick_params(left=False, labelleft=False) 
        
        self.ax_prof.plot(p_dist, self.current_z_line, 'k-', lw=1.5); self.ax_prof.set_ylabel("Height (Å)")
        
        for i, r in enumerate(self.marker_ratios):
            idx = int(r * (self.npts - 1))
            self.ax_spec.plot(self.cached_f_engs, f_ldos[idx], color=self.m_colors[i], label=f'{p_dist[idx]:.2f} Å')
            self.ax_prof.axvline(x=p_dist[idx], color=self.m_colors[i], ls='--', lw=2, alpha=0.7, picker=5, label=f'marker_{i}')
            for ax in [self.ax_ldos, self.ax_stripe]: ax.axhline(y=p_dist[idx], color=self.m_colors[i], ls='--')
            
        self.ax_spec.legend(fontsize='x-small'); self.fig.canvas.draw_idle()

    def _on_pick(self, event):
        if event.artist == self.ends: self.active_obj = ('end', event.ind[0])
        elif event.artist == self.marks: self.active_obj = ('mark_map', event.ind[0])
        elif isinstance(event.artist, plt.Line2D) and 'marker_' in event.artist.get_label():
            self.active_obj = ('mark_prof', int(event.artist.get_label().split('_')[1]))

    def _on_motion(self, event):
        if self.active_obj is None or event.xdata is None: return
        t_obj, idx = self.active_obj
        if t_obj == 'end':
            if idx == 0: self.p1 = np.array([event.xdata, event.ydata])
            else: self.p2 = np.array([event.xdata, event.ydata])
        elif t_obj == 'mark_map':
            v = self.p2 - self.p1; v_sq = np.dot(v, v)
            if v_sq > 1e-9: self.marker_ratios[idx] = np.clip(np.dot(np.array([event.xdata, event.ydata]) - self.p1, v) / v_sq, 0, 1)
        elif t_obj == 'mark_prof':
            p_len = norm(self.p2 - self.p1)
            if p_len > 1e-9: self.marker_ratios[idx] = np.clip(event.xdata / p_len, 0, 1)
        self._update_all()

    def _on_ui_change(self, val):
        self.use_decay, self.normalize = self.chk.get_status()
        ref = (int(self.s_cell.val) != self.display_cells); self.display_cells = int(self.s_cell.val)
        self._update_all(full_refresh=ref)

    def _on_rel(self, event): self.active_obj = None

if __name__ == "__main__":
    v_dir = r'C:/Users/Benjamin Kafin/Documents/VASP/SAM/zigzag/kpoints551/dpl_corr/kp551'
    my_cmap = LinearSegmentedColormap.from_list("t", ["black", "firebrick", "yellow"])
    sim = Interactive_STM_Simulator(v_dir, [15, -4, 3, 20], [-2.525, -1.3], [17, 24.5], 1.3, my_cmap)
    sim.run_interactive(grid_res=64, topo_bias=0.2, topo_height=2.5, ldos_bias_sign='neg')