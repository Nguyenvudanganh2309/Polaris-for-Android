"""
PCB Layout Viewer – Fabmaster (.txt) format  v3
================================================
* Board outline (SIP shape) với lines + arcs
* Test points (TP / TPA / PPA …) vẽ bằng Circle patch data-coords
  → tự động scale khi zoom in/out
* Không hiện label mặc định – chỉ hiện khi:
    - Click chuột trái vào test point
    - Search linh kiện và test point được highlight
* Khi search: bỏ qua các test point chỉ kết nối qua GND
* Click vào vùng trống để bỏ label

Usage
-----
    python pcb_viewer.py
    python pcb_viewer.py path/to/fabmaster.txt
"""

import math
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.widgets import TextBox, Button
import numpy as np


# ─── Fabmaster parser ─────────────────────────────────────────────────────────

def parse_fabmaster(filepath: str) -> dict:
    filepath = Path(filepath)
    lines    = filepath.read_text(encoding="utf-8", errors="replace").splitlines()

    COMP_HEADER = ("A!REFDES!COMP_CLASS!COMP_PART_NUMBER!COMP_HEIGHT!"
                   "COMP_DEVICE_LABEL!COMP_INSERTION_CODE!SYM_TYPE!SYM_NAME!"
                   "SYM_MIRROR!SYM_ROTATE!SYM_X!SYM_Y!COMP_VALUE")
    NET_HEADER  = "A!NET_NAME!REFDES!PIN_NUMBER!PIN_NAME!PIN_GROUND!PIN_POWER!"
    PAD_HEADER  = "A!PAD_NAME!REC_NUMBER!LAYER!FIXFLAG!VIAFLAG!PADSHAPE1!PADWIDTH!PADHGHT"
    PIN_HEADER  = "A!SYM_NAME!PIN_NAME!PIN_NUMBER!PIN_X!PIN_Y!PAD_STACK_NAME!REFDES"

    components: dict = {}
    netlist          = defaultdict(list)
    comp_nets        = defaultdict(list)
    outline: list    = []
    pad_sizes: dict  = {}   # pad_stack_name -> diameter (mm)
    tp_pad_map: dict = {}   # refdes -> pad_stack_name
    mode = None

    for line in lines:
        if not line or line[0] not in ('A', 'S'):
            continue
        if line.startswith('A!'):
            if line.startswith(COMP_HEADER):   mode = 'comp'
            elif line.startswith(NET_HEADER):  mode = 'net'
            elif line.startswith(PAD_HEADER):  mode = 'pad'
            elif line.startswith(PIN_HEADER):  mode = 'pin'
            else:                              mode = None
            continue
        if not line.startswith('S!'):
            continue
        parts = line.split('!')

        if mode == 'comp' and len(parts) >= 13:
            try:
                components[parts[1]] = dict(
                    refdes=parts[1], comp_class=parts[2], sym_name=parts[8],
                    mirror=(parts[9] == 'YES'),
                    rotate=float(parts[10]),
                    x=float(parts[11]), y=float(parts[12]),
                    value=parts[13] if len(parts) > 13 else '',
                )
            except ValueError:
                pass

        elif mode == 'net' and len(parts) >= 5:
            net = parts[1]; refdes = parts[2]
            netlist[net].append((refdes, parts[3], parts[4]))
            comp_nets[refdes].append((net, parts[3], parts[4]))

        elif mode == 'pad' and len(parts) >= 9:
            # Only grab TOP layer rows with actual shape
            pad_name = parts[1]; layer = parts[3]; shape = parts[6]
            if layer == 'TOP' and shape and shape != '0.0000':
                try:
                    w = float(parts[7]); h = float(parts[8])
                    d = (w + h) / 2          # average for non-circle
                    if pad_name not in pad_sizes or d > pad_sizes[pad_name]:
                        pad_sizes[pad_name] = d
                except ValueError:
                    pass

        elif mode == 'pin' and len(parts) >= 8:
            # A!SYM_NAME!PIN_NAME!PIN_NUMBER!PIN_X!PIN_Y!PAD_STACK_NAME!REFDES
            pad_stack = parts[6]; refdes = parts[7]
            if refdes and pad_stack:
                tp_pad_map[refdes] = pad_stack

        elif (len(parts) > 2 and parts[1] == 'BOARD GEOMETRY'
              and parts[2] == 'DESIGN_OUTLINE'):
            seg = parts[3] if len(parts) > 3 else ''
            try:
                if seg == 'LINE' and len(parts) >= 10:
                    outline.append(dict(type='line',
                        x1=float(parts[6]), y1=float(parts[7]),
                        x2=float(parts[8]), y2=float(parts[9])))
                elif seg == 'ARC' and len(parts) >= 14:
                    outline.append(dict(type='arc',
                        x1=float(parts[6]),  y1=float(parts[7]),
                        x2=float(parts[8]),  y2=float(parts[9]),
                        cx=float(parts[10]), cy=float(parts[11]),
                        r =float(parts[12]),
                        dir=parts[14] if len(parts) > 14 else 'COUNTERCLOCKWISE'))
            except ValueError:
                pass

    return dict(components=components, netlist=netlist,
                comp_nets=comp_nets, outline=outline,
                pad_sizes=pad_sizes, tp_pad_map=tp_pad_map)


# ─── Geometry ─────────────────────────────────────────────────────────────────

def _arc_pts(seg: dict):
    cx, cy, r = seg['cx'], seg['cy'], seg['r']
    a1 = math.atan2(seg['y1'] - cy, seg['x1'] - cx)
    a2 = math.atan2(seg['y2'] - cy, seg['x2'] - cx)
    if seg.get('dir', 'COUNTERCLOCKWISE') == 'COUNTERCLOCKWISE':
        if a2 <= a1: a2 += 2 * math.pi
    else:
        if a1 <= a2: a1 += 2 * math.pi
        a1, a2 = a2, a1
    n = max(6, int(abs(a2 - a1) / math.radians(4)))
    t = np.linspace(a1, a2, n)
    return cx + r * np.cos(t), cy + r * np.sin(t)


# ─── Test-point helpers ───────────────────────────────────────────────────────

def is_testpoint(comp: dict) -> bool:
    s = comp.get('sym_name', '').upper()
    return s.startswith('TP-') or s.startswith('PP-')


def _tp_radius(comp: dict, pad_sizes: dict = None, tp_pad_map: dict = None) -> float:
    """Return pad radius in mm using actual pad stack data from the file.

    Falls back to sym_name heuristic if pad data unavailable.
    Physical diameter / 2, capped to [0.10, 0.32] for clean rendering.
    """
    MIN_R, MAX_R = 0.10, 0.32
    if pad_sizes and tp_pad_map:
        refdes    = comp.get('refdes', '')
        pad_stack = tp_pad_map.get(refdes)
        if pad_stack and pad_stack in pad_sizes:
            return min(max(pad_sizes[pad_stack] / 2, MIN_R), MAX_R)
    # Fallback: parse sym_name (TP-P4 → 0.4mm → r=0.20)
    import re
    m = re.search(r'P(\d+)', comp.get('sym_name', ''))
    if m:
        val = int(m.group(1))
        r   = val * 0.005 if val >= 10 else val * 0.05
        return min(max(r, MIN_R), MAX_R)
    return 0.15


_TP_COLORS = {
    'TPA':  '#00CFFF',
    'TPAA': '#00B8E0',
    'TP':   '#50FF90',
    'PPA':  '#FFB347',
    'PP':   '#FFA020',
}

def _tp_color(refdes: str) -> str:
    r = refdes.upper()
    for pfx, col in _TP_COLORS.items():
        if r.startswith(pfx):
            return col
    return '#FF6EFF'


_GND_EXACT = frozenset(['GND', 'AGND', 'DGND', 'AVSS', 'VSS', 'AUX_AVSS',
                         'GND_SMPS', 'GND_VRCP', 'GND_RF'])

def is_gnd_net(net: str) -> bool:
    n = net.upper()
    return n in _GND_EXACT or n.startswith('GND') or n.startswith('VSS_')


# ─── PCB Viewer ───────────────────────────────────────────────────────────────

class PCBViewer:

    # TP circle size for highlight ring (relative multiplier)
    _RING_MULT = 1.8

    def __init__(self, data: dict, title: str = "PCB Layout Viewer"):
        self.data  = data
        self.title = title

        self._testpoints: dict = {}
        self._tp_net:     dict = {}
        self._comp_to_tps      = defaultdict(lambda: defaultdict(list))
        self._tp_order: list   = []     # [(refdes, comp), ...]
        self._pad_sizes        = data.get('pad_sizes', {})
        self._tp_pad_map       = data.get('tp_pad_map', {})
        self._build_indices()

        # ── figure ───────────────────────────────────────────────────────
        self.fig = plt.figure(figsize=(17, 10), facecolor='#111118')
        self.fig.canvas.manager.set_window_title(title)

        self.ax = self.fig.add_axes([0.005, 0.11, 0.625, 0.875],
                                    facecolor='#0A1018')
        self.ax_info = self.fig.add_axes([0.645, 0.11, 0.35, 0.875],
                                         facecolor='#0A1018')
        self.ax_info.axis('off')

        # ── widgets ──────────────────────────────────────────────────────
        ax_tb  = self.fig.add_axes([0.005, 0.020, 0.40, 0.066])
        ax_btn = self.fig.add_axes([0.412, 0.020, 0.10, 0.066])
        ax_clr = self.fig.add_axes([0.518, 0.020, 0.08, 0.066])

        self.textbox = TextBox(ax_tb, 'Search: ', initial='',
                               color='#181828', hovercolor='#22223A',
                               label_pad=0.04)
        self.textbox.label.set_color('#8899BB')
        self.textbox.text_disp.set_color('#FFDD57')
        self.textbox.text_disp.set_fontsize(10)

        self.btn_find  = Button(ax_btn, 'Find',  color='#0F3460', hovercolor='#1E5C9E')
        self.btn_clear = Button(ax_clr, 'Clear', color='#2A2A50', hovercolor='#44447A')
        for b in (self.btn_find, self.btn_clear):
            b.label.set_color('white'); b.label.set_fontsize(9)

        self.btn_find.on_clicked(self._on_search)
        self.btn_clear.on_clicked(self._on_clear)
        self.textbox.on_submit(self._on_search)

        # ── state ────────────────────────────────────────────────────────
        self._highlights: list  = []   # artists to remove on clear
        self._click_label       = None # floating annotation from click
        self._ring_collection   = None # highlight ring PatchCollection

        # ── draw ─────────────────────────────────────────────────────────
        self._draw_board()
        self._draw_components()
        self._draw_testpoints()          # base circles (no labels)
        self._auto_zoom()
        self._draw_legend()
        self._show_idle_info()

        # ── events ───────────────────────────────────────────────────────
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        plt.show()

    # ─── index building ───────────────────────────────────────────────────

    def _build_indices(self):
        comps     = self.data['components']
        comp_nets = self.data['comp_nets']

        self._testpoints = {r: c for r, c in comps.items() if is_testpoint(c)}
        self._tp_order   = list(self._testpoints.items())

        for r in self._testpoints:
            nets = comp_nets.get(r, [])
            if nets:
                self._tp_net[r] = nets[0][0]

        tp_by_net: dict = defaultdict(list)
        for r in self._testpoints:
            for net, _, _ in comp_nets.get(r, []):
                tp_by_net[net].append(r)

        for refdes, comp in comps.items():
            if is_testpoint(comp):
                continue
            for net, pin_n, pin_nm in comp_nets.get(refdes, []):
                for tp in tp_by_net.get(net, []):
                    self._comp_to_tps[refdes][tp].append((net, pin_n, pin_nm))

    # ─── drawing ──────────────────────────────────────────────────────────

    def _draw_board(self):
        ax = self.ax
        for seg in self.data['outline']:
            if seg['type'] == 'line':
                ax.plot([seg['x1'], seg['x2']], [seg['y1'], seg['y2']],
                        color='#C8D8EA', lw=1.5, solid_capstyle='round', zorder=3)
            else:
                xs, ys = _arc_pts(seg)
                ax.plot(xs, ys, color='#C8D8EA', lw=1.5, zorder=3)

        ax.set_title(self.title, color='#B0C4D8', fontsize=10, pad=5)
        ax.tick_params(colors='#445060', labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor('#202838')
        ax.set_aspect('equal')
        ax.grid(True, color='#101820', lw=0.35, linestyle='--', zorder=0)

    def _draw_components(self):
        xs = [c['x'] for c in self.data['components'].values() if not is_testpoint(c)]
        ys = [c['y'] for c in self.data['components'].values() if not is_testpoint(c)]
        if xs:
            self.ax.scatter(xs, ys, s=5, color='#2E3E50', marker='s',
                            alpha=0.6, zorder=2, linewidths=0)

    def _draw_testpoints(self):
        """Draw all TPs as Circle patches in data coords (scales with zoom)."""
        patches, fc, ec = [], [], []
        for refdes, c in self._tp_order:
            patches.append(Circle((c['x'], c['y']), _tp_radius(c, self._pad_sizes, self._tp_pad_map)))
            fc.append(_tp_color(refdes))
            ec.append('#111111')

        self._tp_collection = PatchCollection(
            patches, facecolors=fc, edgecolors=ec,
            linewidths=0.7, zorder=5, alpha=0.90,
        )
        self.ax.add_collection(self._tp_collection)

    def _auto_zoom(self):
        xs = [c['x'] for c in self.data['components'].values()]
        ys = [c['y'] for c in self.data['components'].values()]
        if not xs: return
        pad = max(max(xs)-min(xs), max(ys)-min(ys)) * 0.05 + 1.0
        self.ax.set_xlim(min(xs)-pad, max(xs)+pad)
        self.ax.set_ylim(min(ys)-pad, max(ys)+pad)

    def _draw_legend(self):
        handles = [
            mpatches.Patch(color='#00CFFF', label='TPA – test access'),
            mpatches.Patch(color='#50FF90', label='TP  – test point'),
            mpatches.Patch(color='#FFB347', label='PPA/PP – power probe'),
            mpatches.Patch(color='#2E3E50', label='Component'),
        ]
        self.ax.legend(handles=handles, loc='upper right',
                       fontsize=7, framealpha=0.35,
                       labelcolor='white', facecolor='#0A1018',
                       edgecolor='#202838')

    # ─── click handler ────────────────────────────────────────────────────

    def _on_click(self, event):
        """Left-click on a TP → show floating label + update info panel."""
        if event.inaxes != self.ax or event.button != 1:
            return
        # Ignore clicks used by zoom/pan toolbar
        if self.fig.canvas.toolbar and self.fig.canvas.toolbar.mode:
            return

        # Find nearest TP within tolerance
        TOL = max(_tp_radius(c, self._pad_sizes, self._tp_pad_map) * 2.5 for _, c in self._tp_order) if self._tp_order else 1.0
        best_dist, best_refdes = TOL, None
        for refdes, c in self._tp_order:
            d = math.hypot(c['x'] - event.xdata, c['y'] - event.ydata)
            if d < best_dist:
                best_dist, best_refdes = d, refdes

        if best_refdes:
            self._show_clicked_tp(best_refdes)
        else:
            self._remove_click_label()
            self.fig.canvas.draw_idle()

    def _remove_click_label(self):
        if self._click_label:
            try: self._click_label.remove()
            except Exception: pass
            self._click_label = None

    def _show_clicked_tp(self, refdes: str):
        self._remove_click_label()
        c   = self._testpoints[refdes]
        net = self._tp_net.get(refdes, '?')
        r   = _tp_radius(c, self._pad_sizes, self._tp_pad_map)

        label = f" {refdes} \n {net} "
        self._click_label = self.ax.annotate(
            label,
            xy=(c['x'], c['y'] + r),
            xytext=(c['x'], c['y'] + r + 0.55),
            color='#FFEE88', fontsize=6.5, ha='center', va='bottom',
            fontweight='bold', zorder=20,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#0A1018',
                      edgecolor='#FFDD57', linewidth=1.2, alpha=0.92),
            arrowprops=dict(arrowstyle='->', color='#FFDD57',
                            lw=0.8, connectionstyle='arc3,rad=0'),
        )
        self._show_info_tp(refdes, c, net)
        self.fig.canvas.draw_idle()

    # ─── search / highlight ───────────────────────────────────────────────

    def _clear_highlights(self):
        for a in self._highlights:
            try: a.remove()
            except Exception: pass
        self._highlights.clear()
        if self._ring_collection:
            try: self._ring_collection.remove()
            except Exception: pass
            self._ring_collection = None
        self._remove_click_label()

    def _on_clear(self, _=None):
        self.textbox.set_val('')
        self._clear_highlights()
        self._show_idle_info()
        self.fig.canvas.draw_idle()

    def _on_search(self, text=None):
        query = (text if isinstance(text, str) else self.textbox.text).strip().upper()
        self._clear_highlights()
        if not query:
            self._show_idle_info()
            self.fig.canvas.draw_idle()
            return

        comps = self.data['components']
        matches = ([r for r in comps if r.upper() == query]
                   or [r for r in comps if r.upper().startswith(query)]
                   or [r for r in comps if query in r.upper()])

        if not matches:
            self._show_info_none(query)
            self.fig.canvas.draw_idle()
            return

        refdes = matches[0]
        comp   = comps[refdes]
        raw_map = self._comp_to_tps.get(refdes, {})

        # Filter: keep only TPs that have at least one non-GND signal
        tp_map = {}
        for tp_r, net_list in raw_map.items():
            sig_nets = [(n, pn, pnm) for n, pn, pnm in net_list if not is_gnd_net(n)]
            if sig_nets:
                tp_map[tp_r] = sig_nets

        # ── star marker on component ──────────────────────────────────────
        a = self.ax.scatter([comp['x']], [comp['y']], s=240,
                            color='#FFDD57', marker='*', zorder=12,
                            linewidths=0.5, edgecolors='#AA8800')
        self._highlights.append(a)
        a2 = self.ax.text(comp['x'], comp['y'] + 0.75, refdes,
                          color='#FFDD57', fontsize=8.5, ha='center',
                          fontweight='bold', zorder=13,
                          bbox=dict(boxstyle='round,pad=0.2',
                                    facecolor='#0A1018', edgecolor='#FFDD57',
                                    alpha=0.85))
        self._highlights.append(a2)

        if not tp_map:
            self._show_info_comp(refdes, comp, {})
            self.fig.canvas.draw_idle()
            return

        # ── highlight rings for TPs (PatchCollection) ─────────────────────
        ring_patches, ring_colors = [], []
        for tp_r in tp_map:
            tp = self._testpoints.get(tp_r)
            if not tp: continue
            r = _tp_radius(tp, self._pad_sizes, self._tp_pad_map) * self._RING_MULT
            ring_patches.append(Circle((tp['x'], tp['y']), r))
            ring_colors.append(_tp_color(tp_r))

        self._ring_collection = PatchCollection(
            ring_patches, facecolors='none',
            edgecolors='#FFDD57', linewidths=1.8,
            zorder=8, alpha=0.9,
        )
        self.ax.add_collection(self._ring_collection)

        # ── net label bubble near each highlighted TP ──────────────────────
        for tp_r, sig_nets in tp_map.items():
            tp = self._testpoints.get(tp_r)
            if not tp: continue
            r       = _tp_radius(tp, self._pad_sizes, self._tp_pad_map)
            nets_str = '\n'.join(sorted(set(n for n, _, _ in sig_nets)))

            a3 = self.ax.text(
                tp['x'], tp['y'] + r + 0.30, f"{tp_r}\n{nets_str}",
                color='#FFEE88', fontsize=5.2, ha='center', va='bottom',
                zorder=14, clip_on=True,
                bbox=dict(boxstyle='round,pad=0.18', facecolor='#0A1018',
                          edgecolor='#FFDD57', linewidth=0.9, alpha=0.88),
            )
            self._highlights.append(a3)

            # dashed line to component
            a4, = self.ax.plot([comp['x'], tp['x']], [comp['y'], tp['y']],
                               color='#FFDD57', lw=0.45, alpha=0.22,
                               linestyle='--', zorder=7)
            self._highlights.append(a4)

        self._show_info_comp(refdes, comp, tp_map)
        self.fig.canvas.draw_idle()

    # ─── info panel ───────────────────────────────────────────────────────

    def _clear_info(self):
        self.ax_info.cla()
        self.ax_info.axis('off')
        self.ax_info.set_facecolor('#0A1018')

    # ── Click: single TP info ─────────────────────────────────────────────
    def _show_info_tp(self, refdes: str, comp: dict, net: str):
        self._clear_info()
        ax = self.ax_info
        FM = ax.transAxes
        X0, y = 0.04, 0.96
        dy = 0.032

        def row(txt, col='#C0D0E0', sz=8, bold=False, indent=0):
            nonlocal y
            ax.text(X0 + indent, y, txt, transform=FM,
                    color=col, fontsize=sz, va='top',
                    fontweight='bold' if bold else 'normal',
                    fontfamily='monospace', clip_on=True)
            y -= dy

        row(refdes, col='#FFDD57', sz=12, bold=True); y -= 0.008
        row(f"Net  : {net}", col='#68D391', sz=9)
        row(f"Pos  : ({comp['x']:.3f}, {comp['y']:.3f}) mm", col='#8899AA', sz=8)
        row(f"Pad  : {comp['sym_name']}", col='#8899AA', sz=8)
        y -= 0.016

        ax.plot([0, 1], [y, y], color='#1E2A3A', lw=0.6, transform=FM)
        y -= 0.020

        # Which components share this TP's nets
        comp_nets_data = self.data['comp_nets']
        netlist = self.data['netlist']
        comps_on_net = [
            (r, pn, pnm)
            for r, pn, pnm in netlist.get(net, [])
            if r != refdes and not is_testpoint(self.data['components'].get(r, {}))
        ]
        row(f"Components on {net}:", col='#90CDF4', sz=8, bold=True)
        for r, pn, pnm in sorted(comps_on_net)[:18]:
            if y < 0.06: row("  … more …", col='#445566', sz=7); break
            row(f"  {r:<12} pin {pn}({pnm})", col='#9AE6B4', sz=7)

    # ── Search: component info ────────────────────────────────────────────
    def _show_info_comp(self, refdes: str, comp: dict, tp_map: dict):
        self._clear_info()
        ax = self.ax_info
        FM = ax.transAxes
        X0, y = 0.03, 0.97

        def row(txt, col='#C0D0E0', sz=8, bold=False, dy=0.027):
            nonlocal y
            ax.text(X0, y, txt, transform=FM, color=col, fontsize=sz,
                    va='top', fontweight='bold' if bold else 'normal',
                    fontfamily='monospace', clip_on=True)
            y -= dy

        def sep(dy=0.012):
            nonlocal y
            ax.plot([0, 1], [y + dy/2, y + dy/2], color='#1E2A3A', lw=0.6, transform=FM)
            y -= dy

        row(refdes, col='#FFDD57', sz=11, bold=True, dy=0.036)
        row(f"Class : {comp['comp_class']}", col='#6A7E94', sz=7.5, dy=0.022)
        row(f"Value : {comp['value']}", col='#6A7E94', sz=7.5, dy=0.022)
        row(f"Pos   : ({comp['x']:.3f}, {comp['y']:.3f}) mm",
            col='#6A7E94', sz=7.5, dy=0.025)
        sep()

        if not tp_map:
            row("Không có test point tín hiệu kết nối.", col='#FC8181', sz=8.5)
            return

        # Signal nets reachable via TPs
        tp_nets = set(n for nlist in tp_map.values() for n, _, _ in nlist)

        row(f"{len(tp_map)} test point(s) có tín hiệu:",
            col='#68D391', sz=9, bold=True, dy=0.030)
        sep()

        # Pins list
        row("Pins (xanh = đo được qua TP):", col='#90CDF4', sz=8, bold=True, dy=0.026)
        shown = 0
        all_pins = sorted(self.data['comp_nets'].get(refdes, []), key=lambda t: t[0])
        for net, pin_n, pin_nm in all_pins:
            if y < 0.44:
                row(f"  … {len(all_pins)-shown} pin nữa …", col='#445566', sz=7, dy=0.020)
                break
            col = '#9AE6B4' if net in tp_nets else '#3A4A5A'
            row(f"  p{pin_n:<5} {pin_nm:<16} {net}", col=col, sz=7, dy=0.019)
            shown += 1

        sep()
        row("Test points:", col='#90CDF4', sz=8, bold=True, dy=0.028)

        tp_list = sorted(tp_map.items())
        for i, (tp_r, sig_nets) in enumerate(tp_list):
            if y < 0.10:
                row(f"  … và {len(tp_list)-i} TP nữa", col='#445566', sz=7, dy=0.022)
                break
            tp  = self._testpoints.get(tp_r)
            pos = f"({tp['x']:.2f},{tp['y']:.2f})" if tp else "?"
            row(f"  {tp_r:<11} {pos}", col=_tp_color(tp_r), sz=7.5, bold=True, dy=0.024)
            for net, _, _ in sorted(set((n, pn, pnm) for n, pn, pnm in sig_nets),
                                    key=lambda t: t[0]):
                if y < 0.08: break
                row(f"    ↳ {net}", col='#B8C8D8', sz=7, dy=0.019)

    def _show_idle_info(self):
        self._clear_info()
        ax = self.ax_info
        ax.text(0.5, 0.60, "PCB Layout Viewer",
                transform=ax.transAxes, color='#3D5A80',
                fontsize=13, ha='center', fontweight='bold')
        ax.text(0.5, 0.53, "Nhập refdes → Find  hoặc  Click vào TP",
                transform=ax.transAxes, color='#2A3A50',
                fontsize=8.5, ha='center')
        ax.text(0.5, 0.47, "(vd: U7500  U3100  TPA033  PPA301)",
                transform=ax.transAxes, color='#22303E',
                fontsize=8, ha='center', style='italic')
        n_tp   = len(self._testpoints)
        n_comp = len(self.data['components']) - n_tp
        n_net  = len(self.data['netlist'])
        for i, t in enumerate([f"{n_comp} linh kiện",
                                f"{n_tp} test points",
                                f"{n_net} nets"]):
            ax.text(0.5, 0.33 - i*0.07, t, transform=ax.transAxes,
                    color='#3D5A80', fontsize=9, ha='center')

    def _show_info_none(self, query: str):
        self._clear_info()
        self.ax_info.text(0.5, 0.55, f'Không tìm thấy "{query}"',
                          transform=self.ax_info.transAxes,
                          color='#FC8181', fontsize=10, ha='center')


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    filepath = (sys.argv[1] if len(sys.argv) > 1 else
                str(Path(__file__).parent / "820-03733_N244S_EVT_fabmaster.txt"))
    print(f"Loading {filepath} …")
    data  = parse_fabmaster(filepath)
    n_c   = len(data['components'])
    n_tp  = sum(1 for c in data['components'].values() if is_testpoint(c))
    n_net = len(data['netlist'])
    print(f"  Components : {n_c}")
    print(f"  Test points: {n_tp}")
    print(f"  Nets       : {n_net}")
    print(f"  Outline    : {len(data['outline'])} segments")
    title = (f"PCB Viewer — {Path(filepath).stem}"
             f"   ({n_c} comps · {n_tp} TPs · {n_net} nets)")
    PCBViewer(data, title=title)


if __name__ == "__main__":
    main()
