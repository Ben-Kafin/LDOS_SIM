# -*- coding: utf-8 -*-
"""
Interactive STM Simulator: Multi-Tiered GPU Optimization
Restored 2D Physics Kernel: Aligned with Reference units
Developed by Benjamin Kafin
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

# --- SURGICAL IMPORT INTEGRATION ---
from DOSCAR_spin import SpinAwareDosParser
from LOCPOT import LocpotManager

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
    """Vectorized Julian Chen barrier model character-for-character."""
    me, h, q = 9.1093837e-31, 6.62607015e-34, 1.60217663e-19
    V_eff = cp.where(cp.abs(V) < 1e-6, 1e-6, V)
    V_j, E_j, phi_j = cp.abs(V_eff) * q, E * q, phi * q
    prefactor = (8.0 / (3.0 * V_j)) * cp.pi * cp.sqrt(2.0 * me) / h
    term1 = cp.power(phi_j - E_j + V_j, 1.5)
    term2 = cp.power(phi_j - E_j, 1.5)
    return prefactor * (term1 - term2)

class Unified_STM_Simulator:
    def __init__(self, filepath):
        self.filepath = filepath
        self.unit_cell_num = 4
        chdir(filepath)
        self.dev = cp.cuda.Device(0)
        print(f"--- INITIALIZING GPU TENSOR SIMULATOR (RTX 4080) ---")

    def _converge_tip_height(self, z_map_gpu, grid_xy_gpu, emin, emax, target_ldos, 
                             target_threshold=0.01, topo_gain=0.5, max_iter=1000, use_decay=True):
        """Exhaustive Point-Wise Convergence Engine."""
        t_start = time()
        print(f"--- INITIALIZING TIP CONVERGENCE ENGINE ---")
        for i in range(max_iter):
            t0 = time()
            ld_up, ld_dn, a_grid = self._calculate_ldos_at_points_gpu(cp.hstack([grid_xy_gpu, z_map_gpu[:, None]]), emin, emax, use_energy_decay=use_decay)
            cur_ldos = gpu_simpson(ld_up + ld_dn if ld_dn is not None else ld_up, a_grid)
            rat = cp.maximum(cur_ldos, 1e-20) / target_ldos
            
            active_mask = cp.abs(rat - 1.0) > target_threshold
            active_count = int(cp.sum(active_mask))
            max_error = float(cp.max(cp.abs(rat - 1.0)))
            z_min, z_max = float(cp.min(z_map_gpu)), float(cp.max(z_map_gpu))
            print(f"   Iter {i+1:02d}: Active Pts ={active_count:4d} | Max Error ={max_error*100:6.2f}% | Z_min ={z_min:6.3f} Å | Z_max ={z_max:6.3f} Å | Range ={z_max-z_min:8.4f} Å | Time ={time()-t0:6.3f}s")
            if active_count == 0: break
            
            pts_active = cp.hstack([grid_xy_gpu, z_map_gpu[:, None]])[active_mask]
            idx_g = (cp.dot(pts_active, self.inv_lv_gpu) % 1.0).T * self.locpot_dims_gpu[(-3 if self.locpot_gpu.ndim==4 else 0):, None]
            pot_g = self.locpot_gpu[0] if self.locpot_gpu.ndim == 4 else self.locpot_gpu
            phi_l = cp_ndimage.map_coordinates(pot_g, idx_g, order=1, mode='wrap') - self.ef
            kappa = 0.512 * cp.sqrt(cp.maximum(0.1, phi_l))
            z_map_gpu[active_mask] += (topo_gain / (2.0 * kappa)) * cp.log(rat[active_mask])
        print(f"--- CONVERGENCE COMPLETE: Total Time ={time() - t_start:6.3f}s ---")
        return z_map_gpu

    def _parse_poscar(self, ifile):
        with open(ifile, 'r') as f:
            lines = f.readlines(); sf = float(lines[1]); lv = np.array([float(c) for c in ' '.join(lines[2:5]).split()]).reshape(3,3) * sf
            atomtypes = lines[5].split(); atomnums = [int(i) for i in lines[6].split()]
            start_line = 7 if lines[7].strip().lower()[0] in ['d', 'c'] else 8
            coord = np.array([[float(c) for c in line.split()[:3]] for line in lines[start_line+1:sum(atomnums)+start_line+1]])
            if 'direct' in lines[start_line].lower(): coord = np.dot(coord, lv)
        return lv, coord, atomtypes, atomnums

    def parse_vasp_outputs(self, locpot_path):
            poscar_path = './CONTCAR' if exists('./CONTCAR') and getsize('./CONTCAR') > 0 else './POSCAR'
            self.lv, self.coord, self.atomtypes, self.atomnums = self._parse_poscar(poscar_path)
            dos_parser = SpinAwareDosParser(join(self.filepath, 'DOSCAR'))
            self.energies, self.ef = dos_parser.energies, dos_parser.ef
            self.is_polarized = dos_parser.is_polarized
            lpt_mgr = LocpotManager(self.filepath, ispin=2 if self.is_polarized else 1)
            locpot_raw = cp.array(lpt_mgr.get_data(), dtype=cp.float32)
            
            if self.is_polarized:
                v_total, v_diff = locpot_raw[0], locpot_raw[1]
                self.locpot_gpu = cp.stack([v_total + 0.5 * v_diff, v_total - 0.5 * v_diff])
            else:
                self.locpot_gpu = locpot_raw
                
            self.inv_lv_gpu = cp.array(inv(self.lv), dtype=cp.float32)
            self.locpot_dims_gpu = cp.array(self.locpot_gpu.shape[-3:], dtype=cp.float32)
            self.num_total_atoms = sum(self.atomnums)
            self.dos_up_gpu = cp.array(dos_parser.get_dos_for_simulator(spin='up'), dtype=cp.float32)
            self.dos_dn_gpu = cp.array(dos_parser.get_dos_for_simulator(spin='down'), dtype=cp.float32) if self.is_polarized else None
            
            # Filter: Only consider atoms where fractional z < 0.9
            inv_lv = inv(self.lv)
            frac_coords = np.dot(self.coord, inv_lv)
            z_filter_mask = frac_coords[:, 2] < 0.9
            self.z_highest_atom = np.max(self.coord[z_filter_mask, 2])
            
            coords, idx_list = [], []; base_idx = np.arange(len(self.coord))
            for i in range(-self.unit_cell_num, self.unit_cell_num + 1):
                for j in range(-self.unit_cell_num, self.unit_cell_num + 1):
                    coords.append(self.coord + self.lv[0] * i + self.lv[1] * j); idx_list.append(base_idx)
            self.periodic_coord_gpu = cp.array(np.concatenate(coords), dtype=cp.float32)
            self.atom_indices_periodic_gpu = cp.array(np.concatenate(idx_list))
            self.map_mat_gpu = cp.zeros((self.num_total_atoms, len(self.atom_indices_periodic_gpu)), dtype=cp.float32)
            self.map_mat_gpu[self.atom_indices_periodic_gpu, cp.arange(len(self.atom_indices_periodic_gpu))] = 1.0

    def _calculate_ldos_at_points_gpu(self, tip_positions, emin, emax, use_energy_decay=False):
        estart, eend = np.searchsorted(self.energies, emin), np.searchsorted(self.energies, emax, side='right')
        energy_indices = cp.arange(estart, eend); calc_energies_gpu = cp.array(self.energies[estart:eend], dtype=cp.float32)
        num_pts, num_e = tip_positions.shape[0], len(calc_energies_gpu)
        tip_pos_gpu = cp.array(tip_positions, dtype=cp.float32); frac_coords = cp.dot(tip_pos_gpu, self.inv_lv_gpu)
        grid_indices = (frac_coords % 1.0).T * self.locpot_dims_gpu[:, None]
        dists = cp.sqrt(cp.sum((self.periodic_coord_gpu[:, None, :] - tip_pos_gpu[None, :, :])**2, axis=2))

        def _compute_channel(pot, dos_gpu):
            phi_local = cp_ndimage.map_coordinates(pot, grid_indices, order=1, mode='wrap') - self.ef
            output_ldos = cp.zeros((num_pts, num_e), dtype=cp.float32)
            if use_energy_decay:
                bias_v = cp.array(emax - emin, dtype=cp.float32)
                for e_idx in range(num_e):
                    K = gpu_chen_tunneling_factor(bias_v, calc_energies_gpu[e_idx], phi_local)
                    sf = cp.exp(-1.0 * dists * K[None, :] * 1e-10)
                    w_atom = cp.dot(self.map_mat_gpu, sf)
                    output_ldos[:, e_idx] = cp.dot(w_atom.T, dos_gpu[:, energy_indices[e_idx]])
            else:
                kappa = 0.512 * cp.sqrt(cp.maximum(0.1, phi_local))
                sf = cp.exp(-2.0 * kappa[None, :] * dists)
                w_atom = cp.dot(self.map_mat_gpu, sf)
                output_ldos = cp.dot(w_atom.T, dos_gpu[:, energy_indices])
            return output_ldos

        if self.is_polarized:
            return _compute_channel(self.locpot_gpu[0], self.dos_up_gpu), _compute_channel(self.locpot_gpu[1], self.dos_dn_gpu), calc_energies_gpu
        return _compute_channel(self.locpot_gpu, self.dos_up_gpu), None, calc_energies_gpu

class Interactive_STM_Simulator(Unified_STM_Simulator):
    def __init__(self, filepath, path_coords, erange, highlights, ldos_height, cmap_topo):
        super().__init__(filepath)
        self.p1, self.p2 = np.array(path_coords[:2]), np.array(path_coords[2:])
        self.erange, self.ldos_height, self.cmap_topo = list(erange), ldos_height, cmap_topo
        self.npts = 72; self.is_running, self.normalize, self.show_mag = False, False, False
        self.show_atoms = True
        self.use_decay_topo, self.use_decay_ldos = True, True; self.display_cells = 1
        
        self.cached_p1, self.cached_p2 = None, None
        self.cached_emin, self.cached_emax = None, None
        self.cached_d_topo, self.cached_d_ldos = None, None
        self.cached_bias_energy = None
        self.cached_ld_up, self.cached_ld_dn = None, None
        self.cached_eg = None

        dist = norm(self.p2 - self.p1); self.marker_ratios = [np.clip(h/dist, 0, 1) for h in highlights] if dist > 0 else [0.2, 0.8]
        self.active_obj = None

    def run_interactive(self, grid_res=64, topo_bias=0.2, topo_height=2.5, ldos_bias_sign='neg', use_decay_topo=True, use_decay_ldos=True):
        self.parse_vasp_outputs("LOCPOT"); self.ldos_bias_sign, self.use_decay_topo, self.use_decay_ldos = ldos_bias_sign, use_decay_topo, use_decay_ldos
        print(f"\n--- Phase 1: Global Topography Pre-Calculation ---")
        grid_xy = (np.meshgrid(np.linspace(0,1,grid_res), np.linspace(0,1,grid_res))[0].ravel()[:, None] * self.lv[0, :2]) + (np.meshgrid(np.linspace(0,1,grid_res), np.linspace(0,1,grid_res))[1].ravel()[:, None] * self.lv[1, :2])
        grid_xy_gpu = cp.array(grid_xy, dtype=cp.float32); z_fixed = cp.full(grid_xy_gpu.shape[0], self.z_highest_atom + topo_height, dtype=cp.float32)
        t_emin, t_emax = sorted([0.0, topo_bias])
        ld_up, ld_dn, init_engs = self._calculate_ldos_at_points_gpu(cp.hstack([grid_xy_gpu, z_fixed[:, None]]), t_emin, t_emax, use_energy_decay=self.use_decay_topo)
        target_setp = cp.max(gpu_simpson(ld_up + ld_dn if ld_dn is not None else ld_up, init_engs))
        print(f"[*] Global Maximum Setpoint Current (Target LDOS): {float(target_setp):.6e}")
        z_map_gpu = self._converge_tip_height(z_fixed, grid_xy_gpu, t_emin, t_emax, target_setp, use_decay=self.use_decay_topo)
        self.current_z_map, self.grid_xy = cp.asnumpy(z_map_gpu), grid_xy

        print(f"\n--- Initializing Graphical User Interface ---")
        self.fig = plt.figure(figsize=(20, 14)); gs = gridspec.GridSpec(3, 2, height_ratios=[2.5, 1, 0.25], hspace=0.35, wspace=0.25)
        self.ax_map, self.ax_prof, self.ax_spec = self.fig.add_subplot(gs[0, 0]), self.fig.add_subplot(gs[1, 0]), self.fig.add_subplot(gs[1, 1])
        
        # 4-column layout to allow empty spacer at index 2
        lgs = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0, 1], width_ratios=[0.08, 2, 0.03, 0.05], wspace=0.0)
        self.ax_stripe, self.ax_ldos, self.cax = self.fig.add_subplot(lgs[0]), self.fig.add_subplot(lgs[1]), self.fig.add_subplot(lgs[3])
        
        self.line_art, = self.ax_map.plot([], [], 'r--', lw=2.5, zorder=5); self.m_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.ends = self.ax_map.scatter([], [], c='white', edgecolors='red', s=100, zorder=10, picker=5); self.marks = self.ax_map.scatter([], [], s=150, edgecolors='black', zorder=15, picker=5)
        self.btn_run = Button(plt.axes([0.02, 0.02, 0.08, 0.06]), 'RUN', color='lightgray', hovercolor='lime')
        self.chk = CheckButtons(plt.axes([0.11, 0.02, 0.12, 0.06]), ['Atoms', 'Decay', 'Norm', 'Mag'], [self.show_atoms, self.use_decay_ldos, self.normalize, self.show_mag])
        self.s_cell = Slider(plt.axes([0.25, 0.02, 0.15, 0.03]), 'Cells', 0, 4, valinit=self.display_cells, valstep=1)
        self.s_emin = Slider(plt.axes([0.45, 0.05, 0.25, 0.02]), 'E Min', -5.0, 5.0, valinit=self.erange[0]); self.s_emax = Slider(plt.axes([0.45, 0.02, 0.25, 0.02]), 'E Max', -5.0, 5.0, valinit=self.erange[1])
        
        self.fig.canvas.mpl_connect('pick_event', self._on_pick); self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion); self.fig.canvas.mpl_connect('button_release_event', self._on_rel)
        self.btn_run.on_clicked(self._toggle_run); self.chk.on_clicked(self._on_ui_change)
        for s in [self.s_cell, self.s_emin, self.s_emax]: s.on_changed(self._on_ui_change)
        self._update_all(full_refresh=True); plt.show()

    def _toggle_run(self, event):
        self.is_running = not self.is_running
        self.btn_run.label.set_text('STOP' if self.is_running else 'RUN')
        if self.is_running: self._update_all()

    def _update_all(self, full_refresh=False):
        if full_refresh:
            self.ax_map.clear(); n = int(self.s_cell.val)
            for i in range(-n, n + 1):
                for j in range(-n, n + 1):
                    off = i * self.lv[0, :2] + j * self.lv[1, :2]
                    self.ax_map.tricontourf(self.grid_xy[:, 0] + off[0], self.grid_xy[:, 1] + off[1], self.current_z_map, levels=60, cmap=self.cmap_topo, zorder=1)
                    if self.show_atoms:
                        tr = np.repeat(self.atomtypes, self.atomnums)
                        for t_idx, t_name in enumerate(self.atomtypes):
                            m = (tr == t_name); self.ax_map.scatter(self.coord[m, 0] + off[0], self.coord[m, 1] + off[1], s=10, color=plt.cm.tab10(t_idx/10), alpha=0.3, zorder=2)
            self.ax_map.set_aspect('equal'); self.ax_map.add_line(self.line_art); self.ax_map.add_collection(self.ends); self.ax_map.add_collection(self.marks)
            self.ax_map.set_title("Topo"); self.ax_map.set_xlabel("Distance (Å)"); self.ax_map.set_ylabel("Distance (Å)")

        v = self.p2 - self.p1; p_len = norm(v); p_dist = np.linspace(0, p_len, self.npts); p_xy = np.array([self.p1 + r * v for r in np.linspace(0, 1, self.npts)])
        self.line_art.set_data([self.p1[0], self.p2[0]], [self.p1[1], self.p2[1]]); self.ends.set_offsets([self.p1, self.p2])
        self.marks.set_offsets(np.array([self.p1 + r * v for r in self.marker_ratios])); self.marks.set_facecolors(self.m_colors[:len(self.marker_ratios)])
        if not self.is_running: self.fig.canvas.draw_idle(); return

        bias_e = self.s_emin.val if str(self.ldos_bias_sign).lower() in ['neg', '-', 'negative'] else self.s_emax.val
        needs_topo = (self.cached_p1 is None or not np.array_equal(self.p1, self.cached_p1) or not np.array_equal(self.p2, self.cached_p2) or self.cached_bias_energy != bias_e or self.cached_d_topo != self.use_decay_topo)
        needs_ldos = (needs_topo or self.cached_emin != self.s_emin.val or self.cached_emax != self.s_emax.val or self.cached_d_ldos != self.use_decay_ldos)

        if needs_topo:
            print(f"\n[RUN] Recalculating Path-Topo at {bias_e:.3f}V...")
            l_emin, l_emax = sorted([0.0, bias_e]); p_xy_gpu = cp.array(p_xy, dtype=cp.float32)
            ld_up, ld_dn, l_engs = self._calculate_ldos_at_points_gpu(cp.hstack([p_xy_gpu, cp.full((self.npts,1), self.z_highest_atom + self.ldos_height, dtype=cp.float32)]), l_emin, l_emax, use_energy_decay=self.use_decay_topo)
            target = cp.max(gpu_simpson(ld_up + ld_dn if ld_dn is not None else ld_up, l_engs))
            z_line = self._converge_tip_height(cp.full(self.npts, self.z_highest_atom + self.ldos_height, dtype=cp.float32), p_xy_gpu, l_emin, l_emax, target, use_decay=self.use_decay_topo)
            self.current_z_line = cp.asnumpy(z_line)
            self.cached_p1, self.cached_p2, self.cached_bias_energy, self.cached_d_topo = self.p1.copy(), self.p2.copy(), bias_e, self.use_decay_topo

        if needs_ldos:
            print("[RUN] Refreshing Spin-LDOS Heatmap...")
            ld_up, ld_dn, eg = self._calculate_ldos_at_points_gpu(np.hstack([p_xy, self.current_z_line[:, None]]), self.s_emin.val, self.s_emax.val, use_energy_decay=self.use_decay_ldos)
            self.cached_ld_up, self.cached_ld_dn, self.cached_eg = cp.asnumpy(ld_up), (cp.asnumpy(ld_dn) if ld_dn is not None else None), cp.asnumpy(eg)
            self.cached_emin, self.cached_emax, self.cached_d_ldos = self.s_emin.val, self.s_emax.val, self.use_decay_ldos

        f_up, f_dn = self.cached_ld_up.copy(), (self.cached_ld_dn.copy() if self.cached_ld_dn is not None else None)
        f_ldos = (f_up - f_dn) if (self.show_mag and f_dn is not None) else (f_up + f_dn if f_dn is not None else f_up)
        if self.normalize: f_ldos /= (np.trapezoid(f_ldos, x=self.cached_eg, axis=1)[:, None] + 1e-15)

        self.ax_ldos.clear(); self.ax_spec.clear(); self.ax_prof.clear(); self.ax_stripe.clear(); self.cax.clear()
        
        lc = LineCollection(np.array([np.array([np.zeros_like(p_dist), p_dist]).T[:-1], np.array([np.zeros_like(p_dist), p_dist]).T[1:]]).transpose(1, 0, 2), cmap=self.cmap_topo, norm=plt.Normalize(self.current_z_line.min(), self.current_z_line.max()), linewidth=40)
        lc.set_array(self.current_z_line[:-1]); self.ax_stripe.add_collection(lc)
        self.ax_stripe.set(xlim=(-0.1, 0.1), ylim=(0, p_len), ylabel="Path Dist (Å)"); self.ax_stripe.set_xticks([]); self.ax_stripe.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        
        if self.show_mag and f_dn is not None:
            v_max = np.max(np.abs(f_ldos))
            mesh = self.ax_ldos.pcolormesh(self.cached_eg, p_dist, f_ldos, cmap='bwr', shading='auto', vmin=-v_max, vmax=v_max)
        else:
            mesh = self.ax_ldos.pcolormesh(self.cached_eg, p_dist, f_ldos, cmap='jet', shading='auto')
        
        self.fig.colorbar(mesh, cax=self.cax)
        self.ax_ldos.set_title("LDOS"); self.ax_ldos.set_xlabel("Energy (eV)"); self.ax_ldos.tick_params(left=False, labelleft=False)
        
        self.ax_prof.plot(p_dist, self.current_z_line, 'k-', lw=1.5); self.ax_prof.set_ylabel("Height (Å)")
        self.ax_prof.set_title("Tip Height Along Path"); self.ax_prof.set_xlabel("Distance (Å)")
        
        for i, r in enumerate(self.marker_ratios):
            idx = int(r * (self.npts - 1)); self.ax_spec.plot(self.cached_eg, f_ldos[idx], color=self.m_colors[i], label=f'{p_dist[idx]:.2f} Å')
            self.ax_prof.axvline(x=p_dist[idx], color=self.m_colors[i], ls='--', lw=2, alpha=0.7, picker=5, label=f'marker_{i}')
            for ax in [self.ax_ldos, self.ax_stripe]: ax.axhline(y=p_dist[idx], color=self.m_colors[i], ls='--')
        
        self.ax_spec.set_title("Single Point LDOS"); self.ax_spec.set_xlabel("Energy (eV)")
        self.ax_spec.legend(fontsize='x-small'); self.fig.canvas.draw_idle()

    def _on_pick(self, event):
        if event.artist == self.ends: self.active_obj = ('end', event.ind[0])
        elif event.artist == self.marks: self.active_obj = ('mark_map', event.ind[0])
        elif isinstance(event.artist, plt.Line2D) and 'marker_' in event.artist.get_label(): self.active_obj = ('mark_prof', int(event.artist.get_label().split('_')[1]))
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
        self.show_atoms, self.use_decay_ldos, self.normalize, self.show_mag = self.chk.get_status()
        self._update_all(full_refresh=True)
        self.display_cells = int(self.s_cell.val)
    def _on_rel(self, event): self.active_obj = None

if __name__ == "__main__":
    v_dir = r'C:/dir'
    sim = Interactive_STM_Simulator(v_dir, [15, -4, 3, 20], [-2.1, 1.25], [17, 24.5], 1.2, LinearSegmentedColormap.from_list("t", ["black", "firebrick", "yellow"]))
    # Topo decay behavior is set here via use_decay_topo=True
    sim.run_interactive(grid_res=64, topo_bias=1.5, topo_height=2.5, ldos_bias_sign='pos', use_decay_topo=True)
