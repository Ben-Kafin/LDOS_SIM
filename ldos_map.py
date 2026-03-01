# -*- coding: utf-8 -*-
"""
Interactive STM Simulator: Multi-Tiered GPU Optimization (2D Map Mode Extension)
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
        print("--- INITIALIZING GPU TENSOR SIMULATOR ---")

    def _converge_tip_height(self, z_map_gpu, grid_xy_gpu, emin, emax, target_ldos, 
                             target_threshold=0.01, topo_gain=0.5, max_iter=1000, use_decay=True):
        """Exhaustive Point-Wise Convergence Engine."""
        t_start = time()
        print("--- INITIALIZING TIP CONVERGENCE ENGINE ---")
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

class MapMode_STM_Simulator(Unified_STM_Simulator):
    def __init__(self, filepath, erange, num_e_pts, ldos_height, cmap_topo):
        super().__init__(filepath)
        self.erange, self.num_e_pts, self.ldos_height, self.cmap_topo = list(erange), num_e_pts, ldos_height, cmap_topo
        self.is_running, self.normalize, self.show_mag = False, False, False
        self.show_atoms, self.show_unit_cell = True, False
        self.use_decay_topo, self.use_decay_ldos = True, True; self.display_cells = 1
        
        self.m_colors = ['#1f77b4', '#2ca02c', '#9467bd', '#00ced1', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f', '#8c564b', '#d62728']
        self.active_obj = None
        self.marker_coords = None

        self.cached_emin, self.cached_emax = None, None
        self.cached_nepts = None
        self.cached_d_topo, self.cached_d_ldos = None, None
        self.cached_bias_energy = None
        self.cached_ld_up, self.cached_ld_dn = None, None
        self.cached_eg = None
        
        self.cached_marker_coords = None
        self.cached_spec_ldos = None

    def run_map_mode(self, grid_res=64, topo_height=2.5, ldos_bias_sign='neg', use_decay_topo=True, use_decay_ldos=True):
        self.parse_vasp_outputs("LOCPOT"); self.ldos_bias_sign, self.use_decay_topo, self.use_decay_ldos = ldos_bias_sign, use_decay_topo, use_decay_ldos
        
        self.marker_coords = [[self.lv[0,0]*0.2, self.lv[1,1]*0.2], [self.lv[0,0]*0.8, self.lv[1,1]*0.8]]
        
        print("\n--- Phase 1: Global Topography Pre-Calculation ---")
        grid_xy = (np.meshgrid(np.linspace(0,1,grid_res), np.linspace(0,1,grid_res))[0].ravel()[:, None] * self.lv[0, :2]) + (np.meshgrid(np.linspace(0,1,grid_res), np.linspace(0,1,grid_res))[1].ravel()[:, None] * self.lv[1, :2])
        self.grid_xy_gpu = cp.array(grid_xy, dtype=cp.float32); self.grid_xy = grid_xy
        z_fixed = cp.full(self.grid_xy_gpu.shape[0], self.z_highest_atom + topo_height, dtype=cp.float32)
        
        init_bias_e = self.erange[0] if str(self.ldos_bias_sign).lower() in ['neg', '-', 'negative'] else self.erange[1]
        t_emin, t_emax = sorted([0.0, init_bias_e])
        ld_up, ld_dn, init_engs = self._calculate_ldos_at_points_gpu(cp.hstack([self.grid_xy_gpu, z_fixed[:, None]]), t_emin, t_emax, use_energy_decay=self.use_decay_topo)
        target_setp = cp.max(gpu_simpson(ld_up + ld_dn if ld_dn is not None else ld_up, init_engs))
        print(f"[*] Global Maximum Setpoint Current (Target LDOS): {float(target_setp):.6e}")
        z_map_gpu = self._converge_tip_height(z_fixed, self.grid_xy_gpu, t_emin, t_emax, target_setp, use_decay=self.use_decay_topo)
        self.current_z_map = cp.asnumpy(z_map_gpu)
        self.cached_bias_energy, self.cached_d_topo = init_bias_e, self.use_decay_topo

        print("\n--- Initializing Graphical User Interface ---")
        self.fig = plt.figure(figsize=(20, 14))
        self.gs = gridspec.GridSpec(2, 2, height_ratios=[2.5, 1], width_ratios=[3, 1], hspace=0.35, wspace=0.2)
        self.ax_map = self.fig.add_subplot(self.gs[0, 0])
        self.ax_spec = self.fig.add_subplot(self.gs[0, 1])
        self.map_axes = []
        
        self.marks = self.ax_map.scatter([], [], s=150, edgecolors='black', zorder=15, picker=5)
        
        self.btn_run = Button(plt.axes([0.02, 0.02, 0.05, 0.06]), 'RUN', color='lightgray', hovercolor='lime')
        self.chk = CheckButtons(plt.axes([0.08, 0.02, 0.12, 0.06]), ['Atoms', 'Decay', 'Norm', 'Mag', 'Cell'], [self.show_atoms, self.use_decay_ldos, self.normalize, self.show_mag, self.show_unit_cell])
        self.s_cell = Slider(plt.axes([0.22, 0.02, 0.1, 0.03]), 'Cells', 0, 4, valinit=self.display_cells, valstep=1)
        self.s_num_marks = Slider(plt.axes([0.35, 0.02, 0.1, 0.03]), 'Points', 1, 10, valinit=len(self.marker_coords), valstep=1)
        self.s_emin = Slider(plt.axes([0.50, 0.05, 0.20, 0.02]), 'E Min', -5.0, 5.0, valinit=self.erange[0])
        self.s_emax = Slider(plt.axes([0.50, 0.02, 0.20, 0.02]), 'E Max', -5.0, 5.0, valinit=self.erange[1])
        self.s_nepts = Slider(plt.axes([0.75, 0.035, 0.15, 0.02]), 'E Pts', 1, 20, valinit=self.num_e_pts, valstep=1)
        
        self.fig.canvas.mpl_connect('pick_event', self._on_pick)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self._on_rel)
        self.btn_run.on_clicked(self._toggle_run); self.chk.on_clicked(self._on_ui_change)
        for s in [self.s_cell, self.s_num_marks, self.s_emin, self.s_emax, self.s_nepts]: s.on_changed(self._on_ui_change)
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
            
            if self.show_unit_cell:
                v0, v1, v2, v3 = np.array([0,0]), self.lv[0, :2], self.lv[0, :2] + self.lv[1, :2], self.lv[1, :2]
                cell_pts = np.array([v0, v1, v2, v3, v0])
                self.ax_map.plot(cell_pts[:, 0], cell_pts[:, 1], color='cyan', lw=2.0, ls='-', zorder=4, label='Unit Cell')

            self.ax_map.set_aspect('equal')
            self.ax_map.set_title("Global Topography"); self.ax_map.set_xlabel("Distance (Å)"); self.ax_map.set_ylabel("Distance (Å)")
            self.ax_map.add_collection(self.marks)

        self.marks.set_offsets(self.marker_coords)
        self.marks.set_facecolors(self.m_colors[:len(self.marker_coords)])
        if not self.is_running: self.fig.canvas.draw_idle(); return

        bias_e = self.s_emin.val if str(self.ldos_bias_sign).lower() in ['neg', '-', 'negative'] else self.s_emax.val
        nepts = int(self.s_nepts.val)
        needs_topo = (self.cached_bias_energy != bias_e or self.cached_d_topo != self.use_decay_topo)
        needs_ldos = (needs_topo or self.cached_emin != self.s_emin.val or self.cached_emax != self.s_emax.val or self.cached_d_ldos != self.use_decay_ldos or self.cached_nepts != nepts)
        needs_spec = (needs_ldos or self.cached_marker_coords is None or not np.array_equal(self.marker_coords, self.cached_marker_coords))

        if needs_topo:
            print(f"\n[RUN] Recalculating Map-Topo at {bias_e:.3f}V...")
            l_emin, l_emax = sorted([0.0, bias_e])
            z_fixed = cp.full(self.grid_xy_gpu.shape[0], self.z_highest_atom + self.ldos_height, dtype=cp.float32)
            ld_up, ld_dn, l_engs = self._calculate_ldos_at_points_gpu(cp.hstack([self.grid_xy_gpu, z_fixed[:, None]]), l_emin, l_emax, use_energy_decay=self.use_decay_topo)
            target = cp.max(gpu_simpson(ld_up + ld_dn if ld_dn is not None else ld_up, l_engs))
            z_map_gpu = self._converge_tip_height(z_fixed, self.grid_xy_gpu, l_emin, l_emax, target, use_decay=self.use_decay_topo)
            self.current_z_map = cp.asnumpy(z_map_gpu)
            self.cached_bias_energy, self.cached_d_topo = bias_e, self.use_decay_topo
            self._update_all(full_refresh=True)
            return

        if needs_ldos:
            print("[RUN] Refreshing Spin-LDOS Heatmap Tensor...")
            ld_up, ld_dn, eg = self._calculate_ldos_at_points_gpu(cp.hstack([self.grid_xy_gpu, cp.array(self.current_z_map)[:, None]]), self.s_emin.val, self.s_emax.val, use_energy_decay=self.use_decay_ldos)
            self.cached_ld_up, self.cached_ld_dn, self.cached_eg = cp.asnumpy(ld_up), (cp.asnumpy(ld_dn) if ld_dn is not None else None), cp.asnumpy(eg)
            self.cached_emin, self.cached_emax, self.cached_d_ldos, self.cached_nepts = self.s_emin.val, self.s_emax.val, self.use_decay_ldos, nepts

        if needs_spec:
            m_coords = np.array(self.marker_coords)
            z_marks = []
            for pt in m_coords:
                dist_sq = (self.grid_xy[:, 0] - pt[0])**2 + (self.grid_xy[:, 1] - pt[1])**2
                z_marks.append(self.current_z_map[np.argmin(dist_sq)])
            z_marks = np.array(z_marks)
            
            pt_gpu = cp.array(np.hstack([m_coords, z_marks[:, None]]), dtype=cp.float32)
            s_up, s_dn, _ = self._calculate_ldos_at_points_gpu(pt_gpu, self.s_emin.val, self.s_emax.val, use_energy_decay=self.use_decay_ldos)
            s_up_np, s_dn_np = cp.asnumpy(s_up), (cp.asnumpy(s_dn) if s_dn is not None else None)
            
            spec_ldos = (s_up_np - s_dn_np) if (self.show_mag and s_dn_np is not None) else (s_up_np + s_dn_np if s_dn_np is not None else s_up_np)
            if self.normalize:
                spec_ldos /= (np.trapezoid(spec_ldos, x=self.cached_eg, axis=1)[:, None] + 1e-15)
                
            self.cached_spec_ldos = spec_ldos
            self.cached_marker_coords = np.array(self.marker_coords).copy()

        self.ax_spec.clear()
        for i, pt in enumerate(self.marker_coords):
            color = self.m_colors[i % len(self.m_colors)]
            self.ax_spec.plot(self.cached_eg, self.cached_spec_ldos[i], color=color, label=f'[{pt[0]:.1f}, {pt[1]:.1f}]')
        self.ax_spec.set(title="Single Point LDOS", xlabel="Energy (eV)")
        self.ax_spec.legend(fontsize='x-small')

        f_up, f_dn = self.cached_ld_up.copy(), (self.cached_ld_dn.copy() if self.cached_ld_dn is not None else None)
        f_ldos = (f_up - f_dn) if (self.show_mag and f_dn is not None) else (f_up + f_dn if f_dn is not None else f_up)
        if self.normalize: f_ldos /= (np.trapezoid(f_ldos, x=self.cached_eg, axis=1)[:, None] + 1e-15)

        if len(self.map_axes) != nepts or full_refresh:
            for ax in self.map_axes: ax.remove()
            self.map_axes.clear()
            sub_gs = gridspec.GridSpecFromSubplotSpec(1, nepts, subplot_spec=self.gs[1, :], wspace=0.1)
            for i in range(nepts): self.map_axes.append(self.fig.add_subplot(sub_gs[0, i]))

        e_targets = np.linspace(self.s_emin.val, self.s_emax.val, nepts)
        v_max = np.max(np.abs(f_ldos)) if self.show_mag else None
        m_coords_np = np.array(self.marker_coords)

        for i, target_e in enumerate(e_targets):
            ax = self.map_axes[i]
            ax.clear()
            e_idx = np.abs(self.cached_eg - target_e).argmin()
            slice_data = f_ldos[:, e_idx]

            if self.show_mag and f_dn is not None:
                ax.tricontourf(self.grid_xy[:,0], self.grid_xy[:,1], slice_data, levels=40, cmap='bwr', vmin=-v_max, vmax=v_max)
            else:
                ax.tricontourf(self.grid_xy[:,0], self.grid_xy[:,1], slice_data, levels=40, cmap='jet')
            
            ax.scatter(m_coords_np[:, 0], m_coords_np[:, 1], color=self.m_colors[:len(m_coords_np)], s=30, edgecolors='white', zorder=5)
            ax.set_title(f"E = {self.cached_eg[e_idx]:.3f} eV", fontsize=10)
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])

        self.fig.canvas.draw_idle()

    def _on_pick(self, event):
        if event.artist == self.marks: 
            self.active_obj = ('mark_map', event.ind[0])

    def _on_motion(self, event):
        if self.active_obj is None or event.xdata is None: return
        t_obj, idx = self.active_obj
        if t_obj == 'mark_map' and event.inaxes == self.ax_map:
            self.marker_coords[idx] = [event.xdata, event.ydata]
            self._update_all()

    def _on_ui_change(self, val):
        self.show_atoms, self.use_decay_ldos, self.normalize, self.show_mag, self.show_unit_cell = self.chk.get_status()
        self.display_cells = int(self.s_cell.val)
        
        new_count = int(self.s_num_marks.val)
        if new_count != len(self.marker_coords):
            if new_count > len(self.marker_coords):
                for _ in range(new_count - len(self.marker_coords)):
                    self.marker_coords.append([self.lv[0,0]*0.5 + np.random.rand(), self.lv[1,1]*0.5 + np.random.rand()])
            else:
                self.marker_coords = self.marker_coords[:new_count]
                
        self._update_all(full_refresh=True)

    def _on_rel(self, event): self.active_obj = None

if __name__ == "__main__":
    v_dir = r'C:/dir'
    sim = MapMode_STM_Simulator(v_dir, [-2.0, -0.8], 5, 1.5, LinearSegmentedColormap.from_list("t", ["black", "firebrick", "yellow"]))

    sim.run_map_mode(grid_res=48, ldos_bias_sign='pos')
