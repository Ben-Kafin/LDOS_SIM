# -*- coding: utf-8 -*-
"""
Phase-Sensitive Interactive STM Simulator: Atomic PDOS Architecture
Developed BY: Benjamin Carlos Kafin
Unified Version: Fully Parallelized GPU Architecture (VASP 6.5.1 LORBIT=14)
Surgical Update: Corrected Pymatgen Imports, Robust LOCPOT Loader, Simpson Setpoints
Anchors: Tersoff-Hamann, Julian Chen, Constant-Current Topo, Localized Workfunction
Immutable Anchors: Global Topography and Path Topography are INDEPENDENT.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp  
import cupyx.scipy.ndimage as cp_ndimage 
from cupyx import scatter_add as cp_scatter_add 
from os.path import exists, getsize, join
from os import chdir
from numpy.linalg import norm, inv
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider, CheckButtons, Button
import matplotlib.gridspec as gridspec
from time import time

try:
    from procar_spin import GPUComplexProcarParser
except ImportError:
    print("[!] Error: procar_spin.py not found."); sys.exit()

# --- CORE UTILITIES ---
def gpu_simpson(y, x):
    """Vectorized Simpson's Rule for GPU parity: Immutable Anchor."""
    n = y.shape[1]
    if n % 2 == 0:
        return cp.trapz(y, x=x, axis=1)
    dx = (x[-1] - x[0]) / (n - 1)
    weights = cp.ones(n)
    weights[1:-1:2] = 4
    weights[2:-2:2] = 2
    return (dx / 3.0) * cp.sum(weights * y, axis=1)

# --- GPU-ACCELERATED PHYSICS & BASIS KERNELS ---
def gpu_chen_tunneling_factor(V, E, phi):
    """Julian Chen Barrier Model: Immutable Anchor."""
    me, h, q = 9.1093837e-31, 6.62607015e-34, 1.60217663e-19
    V_eff = cp.where(cp.abs(V) < 1e-6, 1e-6, V)
    V_j, E_j, phi_j = cp.abs(V_eff) * q, E * q, phi * q
    prefactor = (8.0 / (3.0 * V_j)) * cp.pi * cp.sqrt(2.0 * me) / h
    term1 = cp.power(cp.maximum(0.1*q, phi_j - E_j + V_j), 1.5)
    term2 = cp.power(cp.maximum(0.1*q, phi_j - E_j), 1.5)
    return prefactor * (term1 - term2)

def sph_r_gpu(coords, l):
    n_pts = coords.shape[0]
    dist = cp.linalg.norm(coords, axis=1)
    dist_safe = cp.where(dist < 1e-12, 1e-12, dist)
    x, y, z = coords[:, 0]/dist_safe, coords[:, 1]/dist_safe, coords[:, 2]/dist_safe
    if l == 0: return cp.ones((1, n_pts), dtype=cp.float32) * 0.28209479
    if l == 1: return cp.stack([y, z, x]) * 0.48860251
    if l == 2: return cp.stack([x*y, y*z, (2.0*z**2-x**2-y**2)/1.73205, x*z, (x**2-y**2)/2.0]) * 0.63576474
    return cp.zeros((2*l+1, n_pts), dtype=cp.float32)

class Unified_STM_Simulator:
    def __init__(self, filepath):
        self.filepath = filepath; self.unit_cell_num = 4
        if not exists(filepath): raise FileNotFoundError(f"Path not found: {filepath}")
        chdir(filepath); self.radial_data = {}; self.rgrid_data = {}; self.header_map = []
        self.magnetization_mode = False 
        self.global_setpoint = 1.0; self.tile_size = 1024
        try:
            self.dev = cp.cuda.Device(0)
            print(f"--- INITIALIZING PHASE-SENSITIVE SIMULATOR (BENJAMIN KAFIN) ---")
        except Exception as e: print(f"\n[GPU ERROR]: {e}"); sys.exit()

    def _prepare_radial_basis(self):
        npy_path = join(self.filepath, 'basis_multi_PLO.npy')
        raw = np.load(npy_path, allow_pickle=True).item()
        for elem, data in raw.items():
            self.rgrid_data[elem] = cp.array(data['rgrid'], dtype=cp.float32)
            self.radial_data[elem] = {int(k[-1]): cp.array(v, dtype=cp.float32) for k, v in data.items() if k.startswith('r_opt_l')}

    def _map_headers(self, headers):
        v_map = {'s':(0,0), 'py':(1,0), 'pz':(1,1), 'px':(1,2), 'dxy':(2,0), 'dyz':(2,1), 'dz2':(2,2), 'dxz':(2,3), 'dx2-y2':(2,4)}
        self.header_map = [v_map.get(h.lower(), (0, 0)) for h in headers]

    def _print_iteration_status(self, phase, iteration, t0, z_vals, ratios, active_count, min_ldos=None):
        cp.cuda.Stream.null.synchronize(); elapsed = time() - t0
        max_error = np.max(np.abs(ratios - 1.0)); z_min, z_max = np.min(z_vals), np.max(z_vals)
        ldos_str = f" | Min Current ={min_ldos:8.4e}" if min_ldos is not None else ""
        print(f"   {phase} Iter {iteration+1:02d}: Active Pts ={active_count:4d} | Error ={max_error*100:6.2f}% | Z_min ={z_min:6.3f} Å | Z_max ={z_max:6.3f} Å{ldos_str} | Range ={z_max-z_min:8.4f} Å | Time ={elapsed:6.3f}s")
        return max_error

    def _parse_poscar(self, ifile):
        with open(ifile, 'r') as f:
            lines = f.readlines(); sf = float(lines[1])
            lv = np.array([float(c) for c in ' '.join(lines[2:5]).split()]).reshape(3,3) * sf
            types = lines[5].split(); nums = [int(i) for i in lines[6].split()]
            offset = 9 if lines[7].lower().startswith('s') else 8
            coord = np.array([[float(c) for c in l.split()[:3]] for l in lines[offset:sum(nums)+offset]])
            if 'direct' in lines[offset-1].lower(): coord = np.dot(coord, lv)
        return lv, coord, types, nums

    def parse_vasp_outputs(self, locpot_path, my_sigmas, force_rebuild_cache=False):
        poscar = './CONTCAR' if exists('./CONTCAR') and getsize('./CONTCAR') > 0 else './POSCAR'
        self.lv, self.coord, self.atomtypes, self.atomnums = self._parse_poscar(poscar)
        self._prepare_radial_basis(); self.z_highest_atom = np.max(self.coord[:, 2])
        cache_path = join(self.filepath, "Broadened_Orbital_Curves.npy")
        
        if force_rebuild_cache or not exists(cache_path):
            parser = GPUComplexProcarParser(self.filepath); parser.run_gpu_workflow(sigmas=my_sigmas); self.ispin = parser.ispin
        else:
            raw_c = np.load(cache_path, allow_pickle=True).item(); self.ispin = 2 if raw_c['broadened_coeffs'].ndim == 5 else 1
            
        c = np.load(cache_path, allow_pickle=True).item()
        self.energy_grid, self.broadened_coeffs, self.ef = cp.array(c['energy_grid']), cp.array(c['broadened_coeffs']), c['ef']
        self._map_headers(c['headers'])
        
        locpot_npy_path = join(self.filepath, "LOCPOT.npy")
        rebuild_locpot = force_rebuild_cache or not exists(locpot_npy_path)
        
        if not rebuild_locpot:
            try:
                locpot_data = np.load(locpot_npy_path)
                if (self.ispin == 2 and locpot_data.ndim == 3) or (self.ispin == 1 and locpot_data.ndim == 4):
                    print(f"[*] LOCPOT cache dimension mismatch detected (ISPIN={self.ispin}, NDIM={locpot_data.ndim}). Forcing rebuild.")
                    rebuild_locpot = True
            except Exception:
                rebuild_locpot = True
                
        if rebuild_locpot:
            from pymatgen.io.vasp import Locpot
            try:
                from pymatgen.electronic_structure.core import Spin
            except ImportError:
                from pymatgen.core.spin import Spin
            print(f"[*] Parsing raw LOCPOT via pymatgen (resolving sections 1 & 2)...")
            lpt = Locpot.from_file(locpot_path)
            
            # Key-Blind Sequential Section Extraction (Total Potential then Magnetization)
            vol_sections = list(lpt.data.values())
            if self.ispin == 2 and len(vol_sections) >= 2:
                v_tot, v_mag = vol_sections[0], vol_sections[1]
                locpot_data = np.stack([(v_tot + v_mag)/2.0, (v_tot - v_mag)/2.0])
            else:
                locpot_data = vol_sections[0]
            np.save(locpot_npy_path, locpot_data)
            print(f"[*] Saved unified LOCPOT cache to {locpot_npy_path} with shape {locpot_data.shape}")

        self.locpot_gpu = cp.array(locpot_data, dtype=cp.float32)
        self.inv_lv_gpu = cp.array(inv(self.lv), dtype=cp.float32)
        spatial_shape = self.locpot_gpu.shape[1:] if self.locpot_gpu.ndim == 4 else self.locpot_gpu.shape
        self.locpot_dims_gpu = cp.array(spatial_shape, dtype=cp.float32)
        self.periodic_images = [cp.array(self.lv[0]*i + self.lv[1]*j, dtype=cp.float32) for i in range(-self.unit_cell_num, self.unit_cell_num + 1) for j in range(-self.unit_cell_num, self.unit_cell_num + 1)]

    def _get_angular_tensor(self, dx, dy, dz, dist, num_orbs):
        angs = cp.zeros((num_orbs, dist.shape[0], dist.shape[1]), dtype=cp.float32)
        for t_name, m_idx in self.element_indices_gpu.items():
            dx_e, dy_e, dz_e, dist_e = dx[m_idx], dy[m_idx], dz[m_idx], dist[m_idx]
            dist_bohr = dist_e / 0.529177
            ylm = {l: sph_r_gpu(cp.stack([dx_e.ravel(), dy_e.ravel(), dz_e.ravel()], axis=1), l).reshape(2*l+1, len(m_idx), -1) for l in [0, 1, 2]}
            r_opt = {l: cp.interp(dist_bohr.ravel(), self.rgrid_data[t_name], self.radial_data[t_name][l]).reshape(len(m_idx), -1) for l in self.radial_data[t_name]}
            for o_idx, (l, m) in enumerate(self.header_map):
                if l in r_opt: angs[o_idx, m_idx] = r_opt[l] * ylm[l][m]
        return angs

    def _calculate_ldos_at_points_gpu(self, tip_positions_gpu, emin, emax, use_energy_decay=False):
        """Coherent Intra-Atomic Summation & Incoherent Inter-K-Point Accumulation Loop."""
        mask = (self.energy_grid >= emin) & (self.energy_grid <= emax)
        if cp.sum(mask) == 0: return cp.zeros((tip_positions_gpu.shape[0], 2)), self.energy_grid[:2]
        a_grid, a_curves = self.energy_grid[mask], self.broadened_coeffs[..., mask]
        num_pts, num_engs = tip_positions_gpu.shape[0], len(a_grid)
        final_ldos = cp.zeros((num_pts, num_engs), dtype=cp.float32)
        coord_gpu = cp.array(self.coord, dtype=cp.float32)
        self.element_indices_gpu = {t: cp.where(cp.array(np.repeat(self.atomtypes, self.atomnums) == t))[0] for t in self.radial_data}
        
        for start in range(0, num_pts, self.tile_size):
            end = min(start + self.tile_size, num_pts); tile_pts = tip_positions_gpu[start:end]
            idx_g = (cp.dot(tile_pts, self.inv_lv_gpu) % 1.0).T * self.locpot_dims_gpu[:, None]
            if self.ispin == 2:
                pot_up = self.locpot_gpu[0] if self.locpot_gpu.ndim == 4 else self.locpot_gpu
                pot_dn = self.locpot_gpu[1] if self.locpot_gpu.ndim == 4 else self.locpot_gpu
                phi_up = cp_ndimage.map_coordinates(pot_up, idx_g, order=1, mode='wrap') - self.ef
                phi_dn = cp_ndimage.map_coordinates(pot_dn, idx_g, order=1, mode='wrap') - self.ef
            else:
                pot_l = self.locpot_gpu[0] if self.locpot_gpu.ndim == 4 else self.locpot_gpu
                phi_l = cp_ndimage.map_coordinates(pot_l, idx_g, order=1, mode='wrap') - self.ef
            total_tile_ldos = cp.zeros((tile_pts.shape[0], num_engs), dtype=cp.float32)
            
            for img_off in self.periodic_images:
                current_coords = coord_gpu + img_off; diffs = tile_pts[None, :, :] - current_coords[:, None, :] 
                dists = cp.sqrt(cp.sum(diffs**2, axis=2)); angs = self._get_angular_tensor(diffs[...,0], diffs[...,1], diffs[...,2], dists, a_curves.shape[-2])
                if self.ispin == 2:
                    N_kpts = a_curves.shape[1]
                    if not use_energy_decay:
                        k_up, k_dn = (0.512 * cp.sqrt(cp.maximum(0.1, phi_up)))[None, None, :], (0.512 * cp.sqrt(cp.maximum(0.1, phi_dn)))[None, None, :]
                        decay_up, decay_dn = cp.exp(-1.0 * k_up * dists[None, :, :]), cp.exp(-1.0 * k_dn * dists[None, :, :])
                    else:
                        b_v = cp.array(emax - emin)
                        K_u, K_d = gpu_chen_tunneling_factor(b_v, a_grid[:, None], phi_up[None, :]), gpu_chen_tunneling_factor(b_v, a_grid[:, None], phi_dn[None, :])
                        decay_up, decay_dn = cp.exp(-0.5 * dists[None, :, :] * K_u[:, None, :] * 1e-10), cp.exp(-0.5 * dists[None, :, :] * K_d[:, None, :] * 1e-10)
                    for k_idx in range(N_kpts):
                        a_k = a_curves[:, k_idx, ...]
                        psi_up = cp.einsum('aoe,oap,eap->pe' if use_energy_decay else 'aoe,oap->pe', a_k[0], angs, decay_up)
                        psi_dn = cp.einsum('aoe,oap,eap->pe' if use_energy_decay else 'aoe,oap->pe', a_k[1], angs, decay_dn)
                        total_tile_ldos += (cp.abs(psi_up)**2 - cp.abs(psi_dn)**2) if self.magnetization_mode else (cp.abs(psi_up)**2 + cp.abs(psi_dn)**2)
                else:
                    N_kpts = a_curves.shape[0]
                    K_vals = gpu_chen_tunneling_factor(cp.array(emax - emin), a_grid[:, None], phi_l[None, :]) if use_energy_decay else (0.512 * cp.sqrt(cp.maximum(0.1, phi_l)))[None, None, :]
                    decay = cp.exp(-0.5 * dists[None, :, :] * K_vals[:, None, :] * 1e-10) if use_energy_decay else cp.exp(-1.0 * K_vals * dists[None, :, :])
                    for k_idx in range(N_kpts):
                        psi = cp.einsum('aoe,oap,eap->pe' if use_energy_decay else 'aoe,oap->pe', a_curves[k_idx, ...], angs, decay)
                        total_tile_ldos += cp.abs(psi)**2
            final_ldos[start:end] = total_tile_ldos
        return final_ldos, a_grid

class Interactive_STM_Simulator(Unified_STM_Simulator):
    def __init__(self, filepath, path_coords, erange, highlights, ldos_height, cmap_topo, npts=72, threshold=0.01):
        super().__init__(filepath); self.p1, self.p2 = np.array(path_coords[:2]), np.array(path_coords[2:])
        self.erange, self.ldos_height, self.cmap_topo, self.npts = list(erange), ldos_height, cmap_topo, npts
        self.target_threshold = threshold; self.topo_gain = 0.4; self.is_running, self.use_decay, self.normalize, self.display_cells = False, True, False, 1
        d = norm(self.p2 - self.p1); self.marker_ratios = [np.clip(h/d, 0, 1) for h in highlights] if d > 0 else [0.2, 0.8]
        self.cached_p1, self.cached_p2, self.cached_emin, self.cached_emax = None, None, None, None; self.cached_bias_energy, self.cached_decay, self.current_z_line = None, None, None
        self.cached_topo_h, self.cached_path_h, self.cached_mag = None, None, None; self.cached_f_ldos, self.cached_f_engs, self.active_obj = None, None, None

    def run_interactive(self, my_sigmas, grid_res=64, topo_bias=0.2, topo_height=2.5, ldos_bias_sign='neg', topo_use_decay=False, topo_gain=0.125, tile_size=1024, force_rebuild_cache=False):
        self.topo_gain = topo_gain; self.tile_size = tile_size; self.parse_vasp_outputs("LOCPOT", my_sigmas, force_rebuild_cache=force_rebuild_cache); self.ldos_bias_sign = ldos_bias_sign
        u, v = np.linspace(0, 1, grid_res), np.linspace(0, 1, grid_res); uu, vv = np.meshgrid(u, v)
        grid_xy = (uu.ravel()[:, None] * self.lv[0, :2]) + (vv.ravel()[:, None] * self.lv[1, :2])
        self.grid_xy_gpu = cp.array(grid_xy, dtype=cp.float32); self.current_z_map_gpu = cp.full(grid_xy.shape[0], self.z_highest_atom + topo_height, dtype=cp.float32)
        t_emin, t_emax = sorted([0.0, topo_bias])
        init_res, a_grid = self._calculate_ldos_at_points_gpu(cp.hstack([self.grid_xy_gpu, self.current_z_map_gpu[:, None]]), t_emin, t_emax, use_energy_decay=topo_use_decay)
        self.anchor_max = float(cp.max(gpu_simpson(init_res, a_grid)))
        print(f"[*] Reference Anchor Established: {self.anchor_max:.6e}")
        for i in range(500):
            t0 = time(); cp.cuda.Stream.null.synchronize()
            ld_res, _ = self._calculate_ldos_at_points_gpu(cp.hstack([self.grid_xy_gpu, self.current_z_map_gpu[:, None]]), t_emin, t_emax, use_energy_decay=topo_use_decay)
            cur_ldos = gpu_simpson(ld_res, a_grid); rat = cp.maximum(cur_ldos, 1e-20) / self.anchor_max
            active_mask = cp.abs(rat - 1.0) > self.target_threshold; active_count = int(cp.sum(active_mask))
            if active_count == 0: break
            pts_active = cp.hstack([self.grid_xy_gpu[active_mask], self.current_z_map_gpu[active_mask][:, None]])
            idx_g = (cp.dot(pts_active, self.inv_lv_gpu) % 1.0).T * self.locpot_dims_gpu[:, None]
            pot_g = self.locpot_gpu[0] if self.locpot_gpu.ndim == 4 else self.locpot_gpu
            phi_l = cp_ndimage.map_coordinates(pot_g, idx_g, order=1, mode='wrap') - self.ef
            kappa = 0.512 * cp.sqrt(cp.maximum(0.1, phi_l))
            self.current_z_map_gpu[active_mask] += (self.topo_gain / (2.0 * kappa)) * cp.log(rat[active_mask])
            self._print_iteration_status("Global-Topo", i, t0, cp.asnumpy(self.current_z_map_gpu), cp.asnumpy(rat), active_count, float(cp.min(cur_ldos)))
        self.current_z_map = cp.asnumpy(self.current_z_map_gpu); self.grid_xy = grid_xy; self.current_z_line = np.full(self.npts, self.z_highest_atom + self.ldos_height)
        self.fig = plt.figure(figsize=(20, 14)); gs = gridspec.GridSpec(3, 2, height_ratios=[2.5, 1, 0.25], hspace=0.35, wspace=0.25)
        self.ax_map, self.ax_prof, self.ax_spec = self.fig.add_subplot(gs[0, 0]), self.fig.add_subplot(gs[1, 0]), self.fig.add_subplot(gs[1, 1])
        lgs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], width_ratios=[0.08, 2], wspace=0.0); self.ax_stripe, self.ax_ldos = self.fig.add_subplot(lgs[0]), self.fig.add_subplot(lgs[1])
        self.line_art, = self.ax_map.plot([], [], 'r--', lw=2.5, zorder=5); self.ends = self.ax_map.scatter([], [], c='white', edgecolors='red', s=100, zorder=10, picker=5); self.marks = self.ax_map.scatter([], [], s=150, edgecolors='black', zorder=15, picker=5); self.m_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        ax_run, ax_chk = plt.axes([0.02, 0.02, 0.08, 0.06]), plt.axes([0.11, 0.02, 0.12, 0.06]); self.btn_run, self.chk = Button(ax_run, 'RUN', color='lightgray', hovercolor='lime'), CheckButtons(ax_chk, ['EnergyDecay', 'Norm', 'Mag'], [self.use_decay, self.normalize, self.magnetization_mode])
        ax_cell, ax_emin, ax_emax = plt.axes([0.25, 0.02, 0.15, 0.03]), plt.axes([0.45, 0.05, 0.25, 0.02]), plt.axes([0.45, 0.02, 0.25, 0.02]); ax_topo_h, ax_path_h = plt.axes([0.75, 0.05, 0.20, 0.02]), plt.axes([0.75, 0.02, 0.20, 0.02])
        self.s_cell, self.s_emin, self.s_emax = Slider(ax_cell, 'Cells', 0, 4, valinit=self.display_cells, valstep=1), Slider(ax_emin, 'E Min', -5.0, 5.0, valinit=self.erange[0]), Slider(ax_emax, 'E Max', -5.0, 5.0, valinit=self.erange[1]); self.s_topo_h = Slider(ax_topo_h, 'Topo H (Å)', 1.0, 10.0, valinit=topo_height); self.s_path_h = Slider(ax_path_h, 'Path H (Å)', 1.0, 10.0, valinit=self.ldos_height)
        self.fig.canvas.mpl_connect('pick_event', self._on_pick); self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion); self.fig.canvas.mpl_connect('button_release_event', self._on_rel)
        self.btn_run.on_clicked(self._toggle_run); self.chk.on_clicked(self._on_ui_change); [s.on_changed(self._on_ui_change) for s in [self.s_cell, self.s_emin, self.s_emax, self.s_topo_h, self.s_path_h]]; self._update_all(full_refresh=True); plt.show()

    def _toggle_run(self, event):
        self.is_running = not self.is_running; self.btn_run.label.set_text('STOP' if self.is_running else 'RUN'); self.btn_run.color = 'salmon' if self.is_running else 'lightgray'
        if self.is_running: self._update_all()

    def _update_all(self, full_refresh=False):
        if full_refresh:
            self.ax_map.clear(); n = int(self.s_cell.val)
            for i in range(-n, n + 1):
                for j in range(-n, n + 1):
                    off = i * self.lv[0, :2] + j * self.lv[1, :2]; self.ax_map.tricontourf(self.grid_xy[:, 0] + off[0], self.grid_xy[:, 1] + off[1], self.current_z_map, levels=60, cmap=self.cmap_topo, zorder=1)
                    tr = np.repeat(self.atomtypes, self.atomnums)
                    for t_idx, t_name in enumerate(self.atomtypes): self.ax_map.scatter(self.coord[tr==t_name, 0] + off[0], self.coord[tr==t_name, 1] + off[1], s=10, color=plt.cm.tab10(t_idx/10), alpha=0.3, zorder=2)
            self.ax_map.set_aspect('equal'); self.ax_map.add_line(self.line_art); self.ax_map.add_collection(self.ends); self.ax_map.add_collection(self.marks)
        v = self.p2 - self.p1; p_len = norm(v); p_dist = np.linspace(0, p_len, self.npts); p_xy = np.array([self.p1 + r * v for r in np.linspace(0, 1, self.npts)])
        self.marks.set_offsets(np.array([self.p1 + r * v for r in self.marker_ratios])); self.marks.set_facecolors(self.m_colors[:len(self.marker_ratios)]); self.line_art.set_data([self.p1[0], self.p2[0]], [self.p1[1], self.p2[1]]); self.ends.set_offsets([self.p1, self.p2])
        if not self.is_running: self.fig.canvas.draw_idle(); return
        bias = self.s_emin.val if str(self.ldos_bias_sign).lower() in ['neg', '-', 'negative'] else self.s_emax.val
        needs_topo_refresh = (self.cached_p1 is None or not np.array_equal(self.p1, self.cached_p1) or not np.array_equal(self.p2, self.cached_p2) or self.cached_bias_energy != bias or self.cached_decay != self.use_decay or self.cached_path_h != self.s_path_h.val)
        needs_heatmap_refresh = (needs_topo_refresh or self.cached_emin != self.s_emin.val or self.cached_emax != self.s_emax.val or self.cached_mag != self.magnetization_mode)
        if needs_topo_refresh:
            z_line_gpu = cp.full(self.npts, self.z_highest_atom + self.s_path_h.val, dtype=cp.float32); p_xy_gpu = cp.array(p_xy, dtype=cp.float32); l_e, l_x = sorted([0.0, bias])
            init_p, a_p_grid = self._calculate_ldos_at_points_gpu(cp.hstack([p_xy_gpu, z_line_gpu[:, None]]), l_e, l_x, use_energy_decay=self.use_decay)
            self.anchor_p_max = float(cp.max(gpu_simpson(init_p, a_p_grid)))
            for i in range(500):
                t0 = time(); cp.cuda.Stream.null.synchronize(); lp_res, _ = self._calculate_ldos_at_points_gpu(cp.hstack([p_xy_gpu, z_line_gpu[:, None]]), l_e, l_x, use_energy_decay=self.use_decay)
                ci_gpu = gpu_simpson(lp_res, a_p_grid); rat_p = cp.maximum(ci_gpu, 1e-20) / self.anchor_p_max
                p_active = cp.abs(rat_p - 1.0) > self.target_threshold; p_act_count = int(cp.sum(p_active))
                if p_act_count == 0: break
                p_idx = (cp.dot(cp.hstack([p_xy_gpu[p_active], z_line_gpu[p_active][:, None]]), self.inv_lv_gpu) % 1.0).T * self.locpot_dims_gpu[:, None]
                pot_p = self.locpot_gpu[0] if self.locpot_gpu.ndim == 4 else self.locpot_gpu
                phi_p = cp_ndimage.map_coordinates(pot_p, p_idx, order=1, mode='wrap') - self.ef
                kappa_p = 0.512 * cp.sqrt(cp.maximum(0.1, phi_p))
                z_line_gpu[p_active] += (self.topo_gain / (2.0 * kappa_p)) * cp.log(rat_p[p_active])
                self._print_iteration_status("Path-Topo", i, t0, cp.asnumpy(z_line_gpu), cp.asnumpy(rat_p), p_act_count, float(cp.min(ci_gpu)))
            self.current_z_line = cp.asnumpy(z_line_gpu); self.cached_p1, self.cached_p2, self.cached_bias_energy, self.cached_decay, self.cached_path_h = self.p1.copy(), self.p2.copy(), bias, self.use_decay, self.s_path_h.val
        if needs_heatmap_refresh:
            lg, eg = self._calculate_ldos_at_points_gpu(cp.array(np.hstack([p_xy, self.current_z_line[:, None]]), dtype=cp.float32), self.s_emin.val, self.s_emax.val, use_energy_decay=self.use_decay)
            self.cached_f_ldos, self.cached_f_engs = cp.asnumpy(lg), cp.asnumpy(eg); self.cached_emin, self.cached_emax, self.cached_mag = self.s_emin.val, self.s_emax.val, self.magnetization_mode
        f_l = self.cached_f_ldos.copy()
        if self.normalize: f_l /= (np.trapz(f_l, x=self.cached_f_engs, axis=1)[:, None] + 1e-15)
        m_cmap = 'seismic' if self.magnetization_mode else 'jet'
        [ax.clear() for ax in [self.ax_ldos, self.ax_spec, self.ax_prof, self.ax_stripe]]; lc = LineCollection(np.array([np.array([np.zeros_like(p_dist), p_dist]).T[:-1], np.array([np.zeros_like(p_dist), p_dist]).T[1:]]).transpose(1, 0, 2), cmap=self.cmap_topo, norm=plt.Normalize(self.current_z_line.min(), self.current_z_line.max()), linewidth=40); lc.set_array(self.current_z_line[:-1]); self.ax_stripe.add_collection(lc); self.ax_stripe.set(xlim=(-0.1, 0.1), ylim=(0, p_len), ylabel="Path Dist (Å)"); self.ax_stripe.set_xticks([]); self.ax_stripe.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        self.ax_ldos.pcolormesh(self.cached_f_engs, p_dist, f_l, cmap=m_cmap, shading='auto'); self.ax_ldos.set(xlabel="Energy (eV)"); self.ax_ldos.tick_params(left=False, labelleft=False); self.ax_prof.plot(p_dist, self.current_z_line, 'k-', lw=1.5); self.ax_prof.set_ylabel("Height (Å)")
        for i, r in enumerate(self.marker_ratios):
            idx = int(r * (self.npts - 1)); self.ax_spec.plot(self.cached_f_engs, f_l[idx], color=self.m_colors[i], label=f'{p_dist[idx]:.2f} Å'); self.ax_prof.axvline(x=p_dist[idx], color=self.m_colors[i], ls='--', lw=2, alpha=0.7, picker=5, label=f'marker_{i}'); [ax.axhline(y=p_dist[idx], color=self.m_colors[i], ls='--') for ax in [self.ax_ldos, self.ax_stripe]]
        if self.magnetization_mode: self.ax_spec.axhline(0, color='black', lw=0.5, ls='-')
        self.ax_spec.legend(fontsize='x-small'); self.fig.canvas.draw_idle()

    def _on_pick(self, event):
        if event.artist == self.ends: self.active_obj = ('end', event.ind[0])
        elif event.artist == self.marks: self.active_obj = ('mark_map', event.ind[0])
        elif isinstance(event.artist, plt.Line2D) and 'marker_' in event.artist.get_label(): self.active_obj = ('mark_prof', int(event.artist.get_label().split('_')[1]))

    def _on_motion(self, event):
        if self.active_obj is None or event.xdata is None: return
        t, idx = self.active_obj
        if t == 'end':
            if idx == 0: self.p1 = np.array([event.xdata, event.ydata])
            else: self.p2 = np.array([event.xdata, event.ydata])
        elif t == 'mark_map':
            v = self.p2 - self.p1; v_s = np.dot(v, v); self.marker_ratios[idx] = np.clip(np.dot(np.array([event.xdata, event.ydata]) - self.p1, v) / v_s, 0, 1) if v_s > 1e-9 else 0
        elif t == 'mark_prof':
            p = norm(self.p2 - self.p1); self.marker_ratios[idx] = np.clip(event.xdata / p, 0, 1) if p > 1e-9 else 0
        self._update_all()

    def _on_ui_change(self, val):
        self.use_decay, self.normalize, self.magnetization_mode = self.chk.get_status()
        ref = (int(self.s_cell.val) != self.display_cells); self.display_cells = int(self.s_cell.val)
        if self.is_running:
            if self.cached_topo_h != self.s_topo_h.val:
                self.current_z_map_gpu = cp.full(self.grid_xy.shape[0], self.z_highest_atom + self.s_topo_h.val, dtype=cp.float32)
                t_emin, t_emax = sorted([0.0, 0.2]); init_res_g, a_grid_g = self._calculate_ldos_at_points_gpu(cp.hstack([self.grid_xy_gpu, self.current_z_map_gpu[:, None]]), t_emin, t_emax, use_energy_decay=True)
                for i in range(150): 
                    cp.cuda.Stream.null.synchronize(); lg_res, _ = self._calculate_ldos_at_points_gpu(cp.hstack([self.grid_xy_gpu, self.current_z_map_gpu[:, None]]), t_emin, t_emax, use_energy_decay=True)
                    cur_g = gpu_simpson(lg_res, a_grid_g); rat_g = cp.maximum(cur_g, 1e-20) / self.anchor_max
                    g_active = cp.abs(rat_g - 1.0) > self.target_threshold
                    if not cp.any(g_active): break
                    idx_g = (cp.dot(cp.hstack([self.grid_xy_gpu[g_active], self.current_z_map_gpu[g_active][:, None]]), self.inv_lv_gpu) % 1.0).T * self.locpot_dims_gpu[:, None]
                    pot_g = self.locpot_gpu[0] if self.locpot_gpu.ndim == 4 else self.locpot_gpu
                    phi_g = cp_ndimage.map_coordinates(pot_g, idx_g, order=1, mode='wrap') - self.ef
                    kappa_g = 0.512 * cp.sqrt(cp.maximum(0.1, phi_g))
                    self.current_z_map_gpu[g_active] += (self.topo_gain / (2.0 * kappa_g)) * cp.log(rat_g[g_active])
                self.current_z_map = cp.asnumpy(self.current_z_map_gpu); self.cached_topo_h = self.s_topo_h.val; ref = True
        self._update_all(full_refresh=ref)

    def _on_rel(self, event): self.active_obj = None

if __name__ == "__main__":
    v_dir = r'C:/Users/Benjamin Kafin/Documents/VASP/SAM/zigzag/kpoints551/dpl_corr/kp551'
    grid_resolution = 64; path_npts = 72; target_error = 0.01; gpu_tile_size = 8192
    topo_bias_V = 0.2; initial_topo_height = 4.0; ldos_path_height = 1.5; force_cache_rewrite = True 
    my_cmap = LinearSegmentedColormap.from_list("t", ["black", "firebrick", "yellow"])
    sigmas = {'C': 0.1, 'H': 0.1, 'N': 0.1, 'Au': 0.2}
    sim = Interactive_STM_Simulator(v_dir, [15, -4, 3, 20], [-2.5, -1.3], [17, 24.5], ldos_path_height, my_cmap, npts=path_npts, threshold=target_error)
    sim.run_interactive(sigmas, grid_res=grid_resolution, topo_bias=topo_bias_V, topo_height=initial_topo_height, ldos_bias_sign='neg', topo_use_decay=True, topo_gain=0.5, tile_size=gpu_tile_size, force_rebuild_cache=force_cache_rewrite)