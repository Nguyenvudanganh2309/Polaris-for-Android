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


# TPA / TP / TPAA → same cyan; PPA / PP → orange
_TP_COLORS = {
    'TPA':  '#00CFFF',
    'TPAA': '#00CFFF',
    'TP':   '#00CFFF',
    'PPA':  '#FF9F43',
    'PP':   '#FF9F43',
}

def _tp_color(refdes: str) -> str:
    r = refdes.upper()
    for pfx, col in _TP_COLORS.items():
        if r.startswith(pfx):
            return col
    return '#00CFFF'


_GND_EXACT = frozenset(['GND', 'AGND', 'DGND', 'AVSS', 'VSS', 'AUX_AVSS',
                         'GND_SMPS', 'GND_VRCP', 'GND_RF'])

def is_gnd_net(net: str) -> bool:
    n = net.upper()
    return n in _GND_EXACT or n.startswith('GND') or n.startswith('VSS_')


# ─── Format detection ────────────────────────────────────────────────────────

def detect_and_parse(filepath: str) -> dict:
    """Detect PCB file format and call the appropriate parser.

    Currently supported
    -------------------
    Fabmaster (.txt / .fab / .asc)
        Identified by header starting with "A!REFDES!COMP_CLASS!"

    Returns the same dict structure as parse_fabmaster().
    Raises ValueError if format is not recognised.
    """
    filepath = str(filepath)
    p = Path(filepath)
    if not p.exists():
        raise ValueError(f"File not found: {filepath}")

    # Read first non-empty line to fingerprint format
    with open(filepath, encoding='utf-8', errors='replace') as f:
        first = ''
        for line in f:
            line = line.strip()
            if line:
                first = line
                break

    if first.startswith('A!REFDES!COMP_CLASS!'):
        data = parse_fabmaster(filepath)
        data['_filepath'] = filepath
        return data

    # Future formats can be added here:
    # if first.startswith('...'):  return parse_altium(filepath)

    raise ValueError(
        f"Unrecognised PCB format.\n"
        f"First line: {first[:80]!r}\n\n"
        f"Supported: Fabmaster (.txt / .fab / .asc)"
    )


# ─── PCB Viewer ───────────────────────────────────────────────────────────────

class PCBViewer:

    _RING_MULT = 1.15  # ring hugs the pad: just slightly outside

    def __init__(self, data: dict, title: str = "PCB Layout Viewer"):
        self.data  = data
        self.title = title

        self._testpoints: dict = {}
        self._tp_net:     dict = {}          # refdes -> primary net
        self._tp_all_nets: dict = {}         # refdes -> [net, ...]
        self._comp_to_tps      = defaultdict(lambda: defaultdict(list))
        self._tp_order: list   = []
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
        ax_tb   = self.fig.add_axes([0.005, 0.020, 0.36, 0.066])
        ax_btn  = self.fig.add_axes([0.372, 0.020, 0.09, 0.066])
        ax_clr  = self.fig.add_axes([0.468, 0.020, 0.07, 0.066])
        ax_open = self.fig.add_axes([0.545, 0.020, 0.10, 0.066])

        self.textbox = TextBox(ax_tb, 'Search: ', initial='',
                               color='#181828', hovercolor='#22223A',
                               label_pad=0.04)
        self.textbox.label.set_color('#8899BB')
        self.textbox.text_disp.set_color('#FFDD57')
        self.textbox.text_disp.set_fontsize(10)

        self.btn_find  = Button(ax_btn,  'Find',      color='#0F3460', hovercolor='#1E5C9E')
        self.btn_clear = Button(ax_clr,  'Clear',     color='#2A2A50', hovercolor='#44447A')
        self.btn_open  = Button(ax_open, 'Open File', color='#1A3A1A', hovercolor='#2A5A2A')
        for b in (self.btn_find, self.btn_clear, self.btn_open):
            b.label.set_color('white'); b.label.set_fontsize(9)

        self.btn_find.on_clicked(self._on_search)
        self.btn_clear.on_clicked(self._on_clear)
        self.btn_open.on_clicked(self._on_open_file)
        self.textbox.on_submit(self._on_search)

        # ── panel filter textbox (right side, below info panel) ──────────
        ax_filter = self.fig.add_axes([0.645, 0.020, 0.305, 0.066])
        self.filter_box = TextBox(ax_filter, 'Filter: ', initial='',
                                  color='#0E1A14', hovercolor='#162010',
                                  label_pad=0.04)
        self.filter_box.label.set_color('#6ABB8A')
        self.filter_box.text_disp.set_color('#AAFFCC')
        self.filter_box.text_disp.set_fontsize(9)
        self.filter_box.on_submit(self._on_filter)

        # ── state ────────────────────────────────────────────────────────
        self._highlights: list   = []
        self._ring_collection    = None
        self._board_annot        = None
        self._board_ring         = None
        self._info_rows: list    = []
        self._all_info_rows: list = []   # full unfiltered rows
        self._info_scroll        = 0
        self._info_visible       = 30
        self._current_comp       = None
        self._panel_filter       = ''    # current filter string

        # ── draw ─────────────────────────────────────────────────────────
        self._draw_board()
        self._draw_components()
        self._draw_testpoints()
        self._auto_zoom()
        self._draw_legend()
        self._show_idle_info()

        # ── events ───────────────────────────────────────────────────────
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('scroll_event',       self._on_scroll)

        # Drag & drop (TkAgg only)
        try:
            self.fig.canvas.get_tk_widget().drop_target_register('DND_Files')
            self.fig.canvas.get_tk_widget().dnd_bind('<<Drop>>', self._on_drop)
        except Exception:
            pass   # drag-drop not available on this backend

        plt.show()

    # ─── file loading ─────────────────────────────────────────────────────

    def _on_open_file(self, _=None):
        """Open a file-picker dialog and load the selected PCB file."""
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
        path = filedialog.askopenfilename(
            title="Open PCB Layout File",
            filetypes=[
                ("Fabmaster / text", "*.txt *.fab *.asc"),
                ("All files",        "*.*"),
            ],
            initialdir=str(Path(self.data.get('_filepath',
                                __file__)).parent),
        )
        root.destroy()
        if path:
            self._load_file(path)

    def _on_drop(self, event):
        """Handle drag-and-drop of a file onto the window."""
        raw = event.data.strip()
        # Tk gives path in curly braces when it has spaces
        path = raw.strip('{}').strip('"').strip("'")
        if path:
            self._load_file(path)

    def _load_file(self, filepath: str):
        """Parse filepath and reload the entire board view."""
        filepath = str(filepath)
        try:
            data = detect_and_parse(filepath)
        except Exception as e:
            self._show_info_error(str(e))
            self.fig.canvas.draw_idle()
            return

        self.data         = data
        self._pad_sizes   = data.get('pad_sizes', {})
        self._tp_pad_map  = data.get('tp_pad_map', {})
        self._testpoints  = {}
        self._tp_net      = {}
        self._tp_all_nets = {}
        self._comp_to_tps = defaultdict(lambda: defaultdict(list))
        self._tp_order    = []
        self._build_indices()

        # Reset state
        self._clear_highlights()
        self._info_comp_refdes = None
        self._info_comp        = None

        # Redraw board
        self.ax.cla()
        self.ax.set_facecolor('#0A1018')
        self._draw_board()
        self._draw_components()
        self._draw_testpoints()
        self._auto_zoom()
        self._draw_legend()

        # Update title
        n_c  = len(data['components'])
        n_tp = len(self._testpoints)
        n_n  = len(data['netlist'])
        title = (f"PCB Viewer — {Path(filepath).stem}"
                 f"   ({n_c} comps · {n_tp} TPs · {n_n} nets)")
        self.title = title
        self.fig.canvas.manager.set_window_title(title)
        self.ax.set_title(title, color='#B0C4D8', fontsize=10, pad=5)

        self._show_idle_info()
        self.fig.canvas.draw_idle()
        print(f"Loaded: {filepath}")
        print(f"  Components: {n_c}  TPs: {n_tp}  Nets: {n_n}")

    def _show_info_error(self, msg: str):
        self._clear_info()
        ax = self.ax_info
        ax.text(0.5, 0.6, "Load Error", transform=ax.transAxes,
                color='#FC8181', fontsize=12, ha='center', fontweight='bold')
        ax.text(0.5, 0.5, msg, transform=ax.transAxes,
                color='#FC8181', fontsize=7.5, ha='center',
                wrap=True)

    # ─── index building ───────────────────────────────────────────────────

    def _build_indices(self):
        comps     = self.data['components']
        comp_nets = self.data['comp_nets']

        self._testpoints = {r: c for r, c in comps.items() if is_testpoint(c)}
        self._tp_order   = list(self._testpoints.items())

        for r in self._testpoints:
            nets = [n for n, _, _ in comp_nets.get(r, [])]
            if nets:
                self._tp_net[r]      = nets[0]
                self._tp_all_nets[r] = nets

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

    def _r(self, comp):
        return _tp_radius(comp, self._pad_sizes, self._tp_pad_map)

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
        patches, fc, ec = [], [], []
        for refdes, c in self._tp_order:
            patches.append(Circle((c['x'], c['y']), self._r(c)))
            fc.append(_tp_color(refdes))
            ec.append('#111111')
        self._tp_collection = PatchCollection(
            patches, facecolors=fc, edgecolors=ec,
            linewidths=0.7, zorder=5, alpha=0.90)
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
            mpatches.Patch(color='#00CFFF', label='TP / TPA – test point'),
            mpatches.Patch(color='#FF9F43', label='PPA / PP – power probe'),
            mpatches.Patch(color='#2E3E50', label='Component'),
        ]
        self.ax.legend(handles=handles, loc='upper right',
                       fontsize=7, framealpha=0.35,
                       labelcolor='white', facecolor='#0A1018',
                       edgecolor='#202838')

    # ─── scroll handler ───────────────────────────────────────────────────

    def _on_filter(self, text=None):
        """Filter signal rows in the info panel by signal or TP name."""
        self._panel_filter = (text if isinstance(text, str)
                              else self.filter_box.text).strip().upper()
        self._info_scroll = 0
        self._redraw_info_table()
        self.fig.canvas.draw_idle()

    def _filtered_rows(self):
        """Return self._all_info_rows filtered by self._panel_filter."""
        q = self._panel_filter
        if not q:
            return self._all_info_rows
        return [
            row for row in self._all_info_rows
            if q in row[0].upper()                          # net name
            or (row[3] and q in row[3].upper())            # TP refdes
        ]

    def _on_scroll(self, event):
        if event.inaxes == self.ax:
            # ── zoom board centered on cursor ─────────────────────────────
            factor = 0.82 if event.button == 'up' else 1.0 / 0.82
            xdata, ydata = event.xdata, event.ydata
            if xdata is None or ydata is None: return
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            self.ax.set_xlim([xdata + (x - xdata) * factor for x in xlim])
            self.ax.set_ylim([ydata + (y - ydata) * factor for y in ylim])
            self.fig.canvas.draw_idle()

        elif event.inaxes == self.ax_info:
            # ── scroll info table ─────────────────────────────────────────
            if not self._all_info_rows: return
            delta   = -2 if event.button == 'up' else 2
            total   = len(self._filtered_rows())
            max_off = max(0, total - self._info_visible)
            self._info_scroll = max(0, min(max_off, self._info_scroll + delta))
            self._redraw_info_table()
            self.fig.canvas.draw_idle()

    # ─── board annotation helpers ──────────────────────────────────────────

    def _remove_board_annot(self):
        for artist in (self._board_annot, self._board_ring):
            if artist:
                try: artist.remove()
                except Exception: pass
        self._board_annot = None
        self._board_ring  = None

    def _place_board_annot(self, tp_refdes: str):
        """Put a floating label + glow ring on the PCB for a single TP."""
        self._remove_board_annot()
        tp  = self._testpoints.get(tp_refdes)
        if not tp: return
        net = self._tp_net.get(tp_refdes, '?')
        r   = self._r(tp)

        # Glow ring – tight around pad, bright red
        self._board_ring = Circle(
            (tp['x'], tp['y']), r * self._RING_MULT,
            fill=False, edgecolor='#FF2222', linewidth=2.2,
            zorder=15, alpha=1.0)
        self.ax.add_patch(self._board_ring)

        # Compact annotation just above the TP
        self._board_annot = self.ax.annotate(
            f" {tp_refdes} \n {net} ",
            xy=(tp['x'], tp['y'] + r),
            xytext=(tp['x'], tp['y'] + r + 0.18),
            color='#FFEE88', fontsize=6.5, ha='center', va='bottom',
            fontweight='bold', zorder=20,
            bbox=dict(boxstyle='round,pad=0.22', facecolor='#0A1018',
                      edgecolor='#FFDD57', linewidth=1.1, alpha=0.93),
            arrowprops=dict(arrowstyle='-', color='#FFDD57',
                            lw=0.6, connectionstyle='arc3,rad=0'),
        )
        self.fig.canvas.draw_idle()

    # ─── unified click dispatcher ─────────────────────────────────────────

    def _on_click(self, event):
        if event.button != 1: return
        if self.fig.canvas.toolbar and self.fig.canvas.toolbar.mode: return

        if event.inaxes == self.ax:
            self._handle_board_click(event)
        elif event.inaxes == self.ax_info:
            self._handle_info_click(event)

    # ── click on PCB board ────────────────────────────────────────────────

    def _handle_board_click(self, event):
        TOL = max(self._r(c) * 3.0 for _, c in self._tp_order) if self._tp_order else 1.0
        best_dist, best_r = TOL, None
        for refdes, c in self._tp_order:
            d = math.hypot(c['x'] - event.xdata, c['y'] - event.ydata)
            if d < best_dist:
                best_dist, best_r = d, refdes

        if best_r:
            self._place_board_annot(best_r)
            self._show_info_tp(best_r)
        else:
            self._remove_board_annot()
            self.fig.canvas.draw_idle()

    # ── click on info panel row ───────────────────────────────────────────

    def _handle_info_click(self, event):
        if not self._info_rows: return
        ax_y = self.ax_info.transAxes.inverted().transform(
            (event.x, event.y))[1]
        best_dist, best_tp = 0.018, None
        for (y_ctr, tp_r, _net) in self._info_rows:
            if tp_r and abs(ax_y - y_ctr) < best_dist:
                best_dist = abs(ax_y - y_ctr)
                best_tp   = tp_r
        if best_tp:
            self._place_board_annot(best_tp)
            self._pan_to_tp(best_tp)

    def _pan_to_tp(self, tp_refdes: str):
        """Pan board to center on tp_refdes, keeping current zoom level."""
        tp = self._testpoints.get(tp_refdes)
        if not tp: return
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        half_w = (xlim[1] - xlim[0]) / 2
        half_h = (ylim[1] - ylim[0]) / 2
        self.ax.set_xlim(tp['x'] - half_w, tp['x'] + half_w)
        self.ax.set_ylim(tp['y'] - half_h, tp['y'] + half_h)
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
        self._remove_board_annot()
        self._info_rows.clear()
        self._all_info_rows.clear()
        self._info_scroll  = 0
        self._current_comp = None

    def _on_clear(self, _=None):
        self.textbox.set_val('')
        self.filter_box.set_val('')
        self._panel_filter = ''
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

        comps     = self.data['components']
        comp_nets = self.data['comp_nets']
        netlist   = self.data['netlist']

        # 1. Exact refdes match
        matches = [r for r in comps if r.upper() == query]

        # 2. Refdes prefix / substring
        if not matches:
            matches = [r for r in comps if r.upper().startswith(query)]
        if not matches:
            matches = [r for r in comps if query in r.upper()]

        # 3. TP refdes — find components connected to this TP
        if not matches:
            tp_match = next((r for r in self._testpoints if r.upper() == query), None)
            if tp_match is None:
                tp_match = next((r for r in self._testpoints
                                 if r.upper().startswith(query)), None)
            if tp_match is None:
                tp_match = next((r for r in self._testpoints
                                 if query in r.upper()), None)
            if tp_match:
                # Collect non-TP comps that share a net with this TP
                nets_of_tp = [n for n, _, _ in comp_nets.get(tp_match, [])]
                seen = set()
                for net in nets_of_tp:
                    for r, _, _ in netlist.get(net, []):
                        if r not in seen and r in comps and not is_testpoint(comps[r]):
                            seen.add(r)
                            matches.append(r)

        # 4. Net / signal name — find components on that net
        if not matches:
            found_nets = [n for n in netlist if query in n.upper()]
            seen = set()
            for net in found_nets:
                for r, _, _ in netlist.get(net, []):
                    if r not in seen and r in comps and not is_testpoint(comps[r]):
                        seen.add(r)
                        matches.append(r)

        if not matches:
            self._show_info_none(query)
            self.fig.canvas.draw_idle()
            return

        refdes = matches[0]
        comp   = comps[refdes]
        self._current_comp = refdes

        raw_map = self._comp_to_tps.get(refdes, {})
        # Filter GND-only TPs
        tp_map = {
            tp_r: [(n, pn, pnm) for n, pn, pnm in nl if not is_gnd_net(n)]
            for tp_r, nl in raw_map.items()
        }
        tp_map = {k: v for k, v in tp_map.items() if v}

        # ── star marker on component ──────────────────────────────────────
        a = self.ax.scatter([comp['x']], [comp['y']], s=240,
                            color='#FFDD57', marker='*', zorder=12,
                            linewidths=0.5, edgecolors='#AA8800')
        self._highlights.append(a)
        a2 = self.ax.text(comp['x'], comp['y'] + 0.65, refdes,
                          color='#FFDD57', fontsize=8, ha='center',
                          fontweight='bold', zorder=13,
                          bbox=dict(boxstyle='round,pad=0.18',
                                    facecolor='#0A1018', edgecolor='#FFDD57',
                                    alpha=0.85))
        self._highlights.append(a2)

        # ── highlight rings only (no labels) ─────────────────────────────
        if tp_map:
            ring_patches = []
            for tp_r in tp_map:
                tp = self._testpoints.get(tp_r)
                if not tp: continue
                ring_patches.append(
                    Circle((tp['x'], tp['y']),
                           self._r(tp) * self._RING_MULT))

            self._ring_collection = PatchCollection(
                ring_patches, facecolors='none',
                edgecolors='#FF2222', linewidths=2.2,
                zorder=8, alpha=1.0)
            self.ax.add_collection(self._ring_collection)

            # Dashed lines from component to each TP
            for tp_r in tp_map:
                tp = self._testpoints.get(tp_r)
                if not tp: continue
                a3, = self.ax.plot(
                    [comp['x'], tp['x']], [comp['y'], tp['y']],
                    color='#FFDD57', lw=0.4, alpha=0.18,
                    linestyle='--', zorder=7)
                self._highlights.append(a3)

        self._show_info_comp(refdes, comp, tp_map)
        self.fig.canvas.draw_idle()

    # ─── info panel ───────────────────────────────────────────────────────

    def _clear_info(self):
        self.ax_info.cla()
        self.ax_info.axis('off')
        self.ax_info.set_facecolor('#0A1018')
        self._info_rows.clear()
        self._all_info_rows.clear()
        self._info_scroll = 0

    # ── Click on TP: show TP info in panel ───────────────────────────────
    def _show_info_tp(self, refdes: str):
        self._clear_info()
        tp   = self._testpoints[refdes]
        nets = self._tp_all_nets.get(refdes, [])
        ax   = self.ax_info
        FM   = ax.transAxes
        y    = 0.96

        def t(text, col='#C0D0E0', sz=8, bold=False, dy=0.030):
            nonlocal y
            ax.text(0.04, y, text, transform=FM, color=col, fontsize=sz,
                    va='top', fontweight='bold' if bold else 'normal',
                    fontfamily='monospace', clip_on=True)
            y -= dy

        t(refdes, col='#FFDD57', sz=12, bold=True, dy=0.038)
        t(f"Pos : ({tp['x']:.3f}, {tp['y']:.3f}) mm", col='#8899AA', sz=7.5, dy=0.022)
        t(f"Pad : {tp['sym_name']}", col='#8899AA', sz=7.5, dy=0.024)
        ax.plot([0,1],[y,y], color='#1E2A3A', lw=0.6, transform=FM); y -= 0.016

        t("Signals:", col='#90CDF4', sz=8, bold=True, dy=0.026)
        for net in nets:
            if y < 0.06: break
            col = '#68D391' if not is_gnd_net(net) else '#3A5048'
            t(f"  {net}", col=col, sz=7.5, dy=0.022)

        ax.plot([0,1],[y,y], color='#1E2A3A', lw=0.6, transform=FM); y -= 0.016
        t("Components on these nets:", col='#90CDF4', sz=8, bold=True, dy=0.026)
        netlist = self.data['netlist']
        seen = set()
        for net in nets:
            if is_gnd_net(net): continue
            for r, pn, pnm in netlist.get(net, []):
                if r == refdes or r in seen: continue
                if is_testpoint(self.data['components'].get(r, {})): continue
                seen.add(r)
                if y < 0.05: t("  …", col='#445566', sz=7); break
                t(f"  {r:<12} {net}", col='#9AE6B4', sz=7, dy=0.020)

    # ── Search: two-column scrollable table (signal → test point) ────────
    def _show_info_comp(self, refdes: str, comp: dict, tp_map: dict):
        self._clear_info()

        # Build net → first matching TP
        net_to_tp: dict = {}
        for tp_r, nl in tp_map.items():
            for net, _, _ in nl:
                if net not in net_to_tp:
                    net_to_tp[net] = tp_r

        # Build filtered, deduplicated list:
        # - skip GND nets
        # - skip NC pins (pin_nm == 'NC' or net == 'NC' / 'NO_CONNECT')
        # - one row per unique net (first pin occurrence kept)
        # - all non-GND/NC nets shown; TP column = TP refdes or None
        all_pins = sorted(self.data['comp_nets'].get(refdes, []),
                          key=lambda t: t[0])

        def is_nc(net, pin_nm):
            n = net.strip().upper(); p = pin_nm.strip().upper()
            return n in ('NC', 'NO_CONNECT', 'NOCONNECT', '') or p == 'NC'

        seen_nets = set()
        filtered  = []
        for net, pin_n, pin_nm in all_pins:
            if is_gnd_net(net) or is_nc(net, pin_nm):
                continue
            if net in seen_nets:
                continue
            seen_nets.add(net)
            filtered.append((net, pin_n, pin_nm, net_to_tp.get(net)))

        # Sort: TP first → PP next → no TP last (alphabetical within each group)
        def _row_sort_key(row):
            tp_r = row[3]
            if tp_r is None:
                return (2, row[0])                        # no TP
            r = tp_r.upper()
            if r.startswith('TPA') or r.startswith('TP'):
                return (0, row[0])                        # TP / TPA
            return (1, row[0])                            # PP / PPA

        filtered.sort(key=_row_sort_key)
        self._all_info_rows = filtered   # full list for scrolling
        self._info_scroll    = 0

        # Store comp info for redraw
        self._info_comp_refdes = refdes
        self._info_comp        = comp
        self._info_total       = len(filtered)

        self._redraw_info_table()

    def _redraw_info_table(self):
        """Render the currently visible slice of the info table."""
        ax = self.ax_info
        ax.cla(); ax.axis('off'); ax.set_facecolor('#0A1018')
        self._info_rows.clear()

        refdes = getattr(self, '_info_comp_refdes', None)
        comp   = getattr(self, '_info_comp', None)
        if refdes is None: return

        FM = ax.transAxes

        # ── fixed header (always visible) ─────────────────────────────────
        HEADER_H = 0.175    # fraction of axes height used by header
        y = 0.985
        ax.text(0.03, y, refdes, transform=FM, color='#FFDD57',
                fontsize=11, va='top', fontweight='bold',
                fontfamily='monospace')
        y -= 0.032
        ax.text(0.03, y,
                f"Class: {comp['comp_class']}   Value: {comp['value']}",
                transform=FM, color='#6A7E94', fontsize=7, va='top',
                fontfamily='monospace')
        y -= 0.020
        ax.text(0.03, y,
                f"Pos: ({comp['x']:.3f}, {comp['y']:.3f}) mm",
                transform=FM, color='#6A7E94', fontsize=7, va='top',
                fontfamily='monospace')
        y -= 0.015
        ax.plot([0,1],[y,y], color='#1E2A3A', lw=0.7, transform=FM)
        y -= 0.010

        total_all = len(self._all_info_rows)
        n_with_tp = sum(1 for _, _, _, t in self._all_info_rows if t)

        # Apply panel filter
        visible_rows = self._filtered_rows()
        total        = len(visible_rows)

        q = self._panel_filter
        if q:
            ax.text(0.03, y,
                    f"Filter '{q}': {total}/{total_all}  (TP:{n_with_tp})",
                    transform=FM, color='#AAFFCC', fontsize=6.5, va='top',
                    fontfamily='monospace')
        else:
            ax.text(0.03, y,
                    f"Signals: {total_all}  (TP: {n_with_tp}  no TP: {total_all-n_with_tp})",
                    transform=FM, color='#4A6880', fontsize=6.5, va='top',
                    fontfamily='monospace')
        y -= 0.018
        ax.plot([0,1],[y,y], color='#1E2A3A', lw=0.5, transform=FM)
        y -= 0.004

        # Column header
        C1, C2 = 0.02, 0.58
        DY     = 0.0235
        ax.text(C1, y, "PIN   SIGNAL", transform=FM, color='#90CDF4',
                fontsize=6.8, va='top', fontweight='bold',
                fontfamily='monospace')
        ax.text(C2, y, "TEST POINT", transform=FM, color='#90CDF4',
                fontsize=6.8, va='top', fontweight='bold',
                fontfamily='monospace')
        y -= DY * 0.8
        ax.plot([0,1],[y,y], color='#253545', lw=0.5, transform=FM)
        y -= DY * 0.3

        table_top = y
        avail_h   = table_top - 0.03
        max_rows  = max(1, int(avail_h / DY))
        self._info_visible = max_rows

        # ── visible slice from filtered rows ──────────────────────────────
        start  = min(self._info_scroll, max(0, total - max_rows))
        self._info_scroll = start
        end    = min(start + max_rows, total)
        slice_ = visible_rows[start:end]

        def _group(tp_r):
            if tp_r is None: return 2
            r = tp_r.upper()
            return 0 if (r.startswith('TPA') or r.startswith('TP')) else 1

        labels = {0: 'TP / TPA', 1: 'PP / PPA', 2: 'No test point'}
        prev_group = None

        # Label for first group visible in this slice
        if slice_:
            first_g = _group(slice_[0][3])
            ax.text(C1, y, labels[first_g],
                    transform=FM, color='#3A5060', fontsize=5.5,
                    va='top', fontfamily='monospace', style='italic')
            y -= DY * 0.65

        for net, pin_n, pin_nm, tp_r in slice_:
            cur_group = _group(tp_r)

            # Draw separator between groups (only when group changes)
            if prev_group is not None and cur_group != prev_group:
                ax.plot([0.01, 0.96], [y + DY*0.55, y + DY*0.55],
                        color='#1E3040', lw=0.5, transform=FM)
                # Group label
                labels = {0: 'TP / TPA', 1: 'PP / PPA', 2: 'No test point'}
                ax.text(C1, y + DY*0.52, labels[cur_group],
                        transform=FM, color='#3A5060', fontsize=5.5,
                        va='bottom', fontfamily='monospace', style='italic')
            prev_group = cur_group

            has_tp  = tp_r is not None
            sig_col = '#9AE6B4' if has_tp else '#5A7A8A'
            net_str = net if len(net) <= 26 else net[:25] + '\u2026'

            ax.text(C1, y, f"p{pin_n:<4} {net_str}",
                    transform=FM, color=sig_col, fontsize=6.2, va='top',
                    fontfamily='monospace', clip_on=True)

            if has_tp:
                tp_col = _tp_color(tp_r)
                ax.text(C2, y, tp_r,
                        transform=FM, color=tp_col, fontsize=6.2, va='top',
                        fontweight='bold', fontfamily='monospace', clip_on=True,
                        bbox=dict(boxstyle='round,pad=0.10', facecolor='#0F1A28',
                                  edgecolor=tp_col, alpha=0.6, linewidth=0.5))
            else:
                ax.text(C2, y, '\u2014',
                        transform=FM, color='#2A3A48', fontsize=6.2, va='top',
                        fontfamily='monospace', clip_on=True)

            self._info_rows.append((y - DY * 0.4, tp_r, net))
            y -= DY

        # ── scroll indicator ──────────────────────────────────────────────
        if total > max_rows:
            sb_x  = 0.975
            sb_h  = table_top - 0.03
            # track
            ax.plot([sb_x, sb_x], [0.03, table_top],
                    color='#1A2A38', lw=3, transform=FM,
                    solid_capstyle='round')
            # thumb
            frac   = max_rows / total
            top_f  = 1.0 - start / total
            bot_f  = top_f - frac
            ax.plot([sb_x, sb_x],
                    [0.03 + bot_f * sb_h, 0.03 + top_f * sb_h],
                    color='#4A7090', lw=3, transform=FM,
                    solid_capstyle='round')
            # page info
            ax.text(0.50, 0.015,
                    f"rows {start+1}–{end} of {total}",
                    transform=FM, color='#3A5060', fontsize=6,
                    ha='center', va='bottom', fontfamily='monospace')

    def _show_idle_info(self):
        self._clear_info()
        ax = self.ax_info
        ax.text(0.5, 0.60, "PCB Layout Viewer",
                transform=ax.transAxes, color='#3D5A80',
                fontsize=13, ha='center', fontweight='bold')
        ax.text(0.5, 0.53, "Nhap refdes → Find  hoac  Click vao TP",
                transform=ax.transAxes, color='#2A3A50',
                fontsize=8.5, ha='center')
        ax.text(0.5, 0.47, "(vd: U7500  U3100  TPA033  PPA301)",
                transform=ax.transAxes, color='#22303E',
                fontsize=8, ha='center', style='italic')
        ax.text(0.5, 0.38, "Click dong trong bang → highlight TP",
                transform=ax.transAxes, color='#1E3040',
                fontsize=7.5, ha='center')
        n_tp   = len(self._testpoints)
        n_comp = len(self.data['components']) - n_tp
        n_net  = len(self.data['netlist'])
        for i, t in enumerate([f"{n_comp} linh kien",
                                f"{n_tp} test points",
                                f"{n_net} nets"]):
            ax.text(0.5, 0.28 - i*0.07, t, transform=ax.transAxes,
                    color='#3D5A80', fontsize=9, ha='center')

    def _show_info_none(self, query: str):
        self._clear_info()
        self.ax_info.text(0.5, 0.55, f'Khong tim thay "{query}"',
                          transform=self.ax_info.transAxes,
                          color='#FC8181', fontsize=10, ha='center')


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    # ── determine initial file ────────────────────────────────────────────
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        default = Path(__file__).parent / "820-03733_N244S_EVT_fabmaster.txt"
        if default.exists():
            filepath = str(default)
        else:
            # No default → open file picker immediately
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
            filepath = filedialog.askopenfilename(
                title="Open PCB Layout File",
                filetypes=[("Fabmaster / text", "*.txt *.fab *.asc"),
                           ("All files", "*.*")],
            )
            root.destroy()
            if not filepath:
                print("No file selected. Exiting.")
                return

    print(f"Loading {filepath} …")
    try:
        data = detect_and_parse(filepath)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

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
