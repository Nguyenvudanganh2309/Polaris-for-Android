"""
PCB Layout Viewer – Fabmaster (.txt) format  v3
================================================
* Board outline with lines + arcs
* Test points (TP / TPA / PPA …) drawn as Circle patches in data-coords
  → scale automatically on zoom
* Labels hidden by default – shown only when:
    - Left-clicking a test point
    - Searching a component / test point (highlighted)
* Search: ignores test points connected only through GND nets
* Click on empty board area to clear label

Usage
-----
    python pcb_viewer.py
    python pcb_viewer.py path/to/fabmaster.txt
"""

import math
import sys
import time
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection, PolyCollection
from matplotlib.widgets import TextBox, Button
import numpy as np


# ─── Fabmaster parser ─────────────────────────────────────────────────────────

def parse_fabmaster(filepath: str) -> dict:
    filepath = Path(filepath)
    lines    = filepath.read_text(encoding="utf-8", errors="replace").splitlines()

    COMP_HEADER    = ("A!REFDES!COMP_CLASS!COMP_PART_NUMBER!COMP_HEIGHT!"
                      "COMP_DEVICE_LABEL!COMP_INSERTION_CODE!SYM_TYPE!SYM_NAME!"
                      "SYM_MIRROR!SYM_ROTATE!SYM_X!SYM_Y!COMP_VALUE")
    NET_HEADER     = "A!NET_NAME!REFDES!PIN_NUMBER!PIN_NAME!PIN_GROUND!PIN_POWER!"
    PAD_HEADER     = "A!PAD_NAME!REC_NUMBER!LAYER!FIXFLAG!VIAFLAG!PADSHAPE1!PADWIDTH!PADHGHT"
    PIN_HEADER     = "A!SYM_NAME!PIN_NAME!PIN_NUMBER!PIN_X!PIN_Y!PAD_STACK_NAME!REFDES"
    GRAPHIC_HEADER = "A!GRAPHIC_DATA_NAME!GRAPHIC_DATA_NUMBER!RECORD_TAG!"

    # GRAPHIC_DATA subclasses that define component body outlines
    _ASSY_TOP = frozenset(['ASSEMBLY_TOP', 'DFA_BOUND_TOP'])
    _ASSY_BOT = frozenset(['ASSEMBLY_BOTTOM', 'PLACE_BOUND_BOTTOM'])
    _ASSY_ALL = _ASSY_TOP | _ASSY_BOT

    components: dict = {}
    netlist          = defaultdict(list)
    comp_nets        = defaultdict(list)
    outline: list    = []
    pad_sizes: dict  = {}   # pad_stack_name -> diameter (mm)
    tp_pad_map: dict = {}   # refdes -> pad_stack_name
    # raw outline coords collected from GRAPHIC_DATA assembly lines
    _assy_xs: dict   = defaultdict(list)   # refdes -> [x, ...]
    _assy_ys: dict   = defaultdict(list)   # refdes -> [y, ...]
    comp_pins: dict  = defaultdict(list)   # refdes -> [(pin_x, pin_y, pin_name, pad_stack)]
    mode = None

    for line in lines:
        if not line or line[0] not in ('A', 'S'):
            continue
        if line.startswith('A!'):
            if   line.startswith(COMP_HEADER):    mode = 'comp'
            elif line.startswith(NET_HEADER):     mode = 'net'
            elif line.startswith(PAD_HEADER):     mode = 'pad'
            elif line.startswith(PIN_HEADER):     mode = 'pin'
            elif line.startswith(GRAPHIC_HEADER): mode = 'graphic'
            else:                                 mode = None
            continue
        if not line.startswith('S!'):
            continue
        parts = line.split('!')

        if mode == 'comp' and len(parts) >= 13:
            try:
                mirror = (parts[9] == 'YES')
                components[parts[1]] = dict(
                    refdes=parts[1], comp_class=parts[2], sym_name=parts[8],
                    mirror=mirror,
                    side='BOT' if mirror else 'TOP',
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
            pad_name = parts[1]; layer = parts[3]; shape = parts[6]
            if layer == 'TOP' and shape and shape != '0.0000':
                try:
                    w = float(parts[7]); h = float(parts[8])
                    d = (w + h) / 2
                    if pad_name not in pad_sizes or d > pad_sizes[pad_name]:
                        pad_sizes[pad_name] = d
                except ValueError:
                    pass

        elif mode == 'pin' and len(parts) >= 8:
            # A!SYM_NAME!PIN_NAME!PIN_NUMBER!PIN_X!PIN_Y!PAD_STACK_NAME!REFDES
            pad_stack = parts[6]; refdes = parts[7]
            if refdes and pad_stack:
                tp_pad_map[refdes] = pad_stack
            try:
                comp_pins[refdes].append((
                    float(parts[4]), float(parts[5]),   # absolute x, y
                    parts[2],                            # pin name
                    parts[3],                            # pin number
                    pad_stack,
                ))
            except (ValueError, IndexError):
                pass

        elif mode == 'graphic' and len(parts) >= 16:
            # A!GRAPHIC_DATA_NAME!...!SUBCLASS!SYM_NAME!REFDES!
            # parts[1]=type  parts[4-7]=x1,y1,x2,y2  parts[13]=subclass  parts[15]=refdes
            subclass = parts[13]
            if subclass in _ASSY_ALL and parts[1] in ('LINE', 'RECTANGLE'):
                refdes = parts[15].strip()
                if refdes:
                    try:
                        _assy_xs[refdes].extend([float(parts[4]), float(parts[6])])
                        _assy_ys[refdes].extend([float(parts[5]), float(parts[7])])
                    except (ValueError, IndexError):
                        pass

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

    # Build accurate bounding boxes from ASSEMBLY geometry outlines
    comp_bounds: dict = {}
    for refdes, xs in _assy_xs.items():
        ys = _assy_ys[refdes]
        if xs and ys:
            comp_bounds[refdes] = dict(
                xmin=min(xs), xmax=max(xs),
                ymin=min(ys), ymax=max(ys),
            )

    return dict(components=components, netlist=netlist,
                comp_nets=comp_nets, outline=outline,
                pad_sizes=pad_sizes, tp_pad_map=tp_pad_map,
                comp_bounds=comp_bounds,
                comp_pins=dict(comp_pins))


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


# ─── Component size heuristic ────────────────────────────────────────────────

def _comp_size(comp: dict) -> tuple:
    """Return (width_mm, height_mm) for a component rectangle.

    Tries to extract size from sym_name (e.g. '0402', '0805', 'SOIC8'),
    then falls back to comp_class / refdes prefix.
    """
    sym = comp.get('sym_name', '').upper()
    cls = comp.get('comp_class', '').upper()
    ref = comp.get('refdes', '').upper()

    # Imperial passive codes  w    h
    for code, wh in [('2010', (5.0, 2.5)),
                     ('1812', (4.5, 3.2)),
                     ('1210', (3.2, 2.5)),
                     ('1206', (3.2, 1.6)),
                     ('0805', (2.0, 1.2)),
                     ('0603', (1.6, 0.8)),
                     ('0402', (1.0, 0.5)),
                     ('0201', (0.6, 0.3))]:
        if code in sym:
            return wh

    # SOT / SOD / QFN / BGA size hints
    for kw, wh in [('BGA',  (12.0, 12.0)),
                   ('QFN',  ( 7.0,  7.0)),
                   ('QFP',  (14.0, 14.0)),
                   ('SOIC', ( 5.0,  4.0)),
                   ('SOP',  ( 5.0,  4.0)),
                   ('SOT',  ( 3.0,  2.0)),
                   ('SOD',  ( 2.7,  1.4)),
                   ('DIP',  ( 8.0,  2.5))]:
        if kw in sym:
            return wh

    # Class / refdes prefix defaults
    if cls == 'IC'         or ref[:1] == 'U': return (8.0, 8.0)
    if cls == 'CONNECTOR'  or ref[:1] in ('J', 'P'): return (6.0, 3.5)
    if cls == 'INDUCTOR'   or ref[:1] == 'L': return (3.0, 2.5)
    if cls == 'TRANSISTOR' or ref[:1] == 'Q': return (2.5, 2.0)
    if cls == 'DIODE'      or ref[:1] == 'D': return (2.5, 1.4)
    if cls in ('RESISTOR', 'CAPACITOR') or ref[:1] in ('R', 'C'): return (1.6, 0.8)
    return (3.0, 2.0)


def _rotated_box(cx, cy, w, h, angle_deg):
    """Return list of 4 (x, y) corner points for a rotated rectangle."""
    a   = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    hw, hh = w / 2, h / 2
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    return [(cx + x * cos_a - y * sin_a,
             cy + x * sin_a + y * cos_a) for x, y in corners]


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

        self.ax = self.fig.add_axes([0.005, 0.11, 0.625, 0.835],
                                    facecolor='#0A1018')
        self.ax_info = self.fig.add_axes([0.645, 0.11, 0.35, 0.875],
                                         facecolor='#0A1018')
        self.ax_info.axis('off')

        # ── widgets ──────────────────────────────────────────────────────
        ax_open   = self.fig.add_axes([0.005, 0.952, 0.13,  0.038])
        ax_top    = self.fig.add_axes([0.145, 0.952, 0.075, 0.038])
        ax_bottom = self.fig.add_axes([0.230, 0.952, 0.085, 0.038])
        ax_link   = self.fig.add_axes([0.330, 0.952, 0.075, 0.038])
        ax_tb   = self.fig.add_axes([0.005, 0.020, 0.42, 0.066])
        ax_btn  = self.fig.add_axes([0.430, 0.020, 0.09, 0.066])
        ax_clr  = self.fig.add_axes([0.526, 0.020, 0.07, 0.066])

        self.textbox = TextBox(ax_tb, 'Search: ', initial='',
                               color="#E5E5F1", hovercolor="#E8E8F3",
                               label_pad=0.04)
        self.textbox.label.set_color("#2F3646")
        self.textbox.text_disp.set_color("#050402")
        self.textbox.text_disp.set_fontsize(10)

        self.btn_find   = Button(ax_btn,    'Find',      color='#0F3460', hovercolor='#1E5C9E')
        self.btn_clear  = Button(ax_clr,    'Clear',     color='#2A2A50', hovercolor='#44447A')
        self.btn_open   = Button(ax_open,   'Open File', color='#1A3A1A', hovercolor='#2A5A2A')
        self.btn_top    = Button(ax_top,    'TOP',       color='#1A2A3A', hovercolor='#2A4A6A')
        self.btn_bottom = Button(ax_bottom, 'BOTTOM',    color='#2A1A3A', hovercolor='#4A2A6A')
        self.btn_link   = Button(ax_link,   'LINK',      color='#1A2A1A', hovercolor='#2A4A2A')
        for b in (self.btn_find, self.btn_clear, self.btn_open,
                  self.btn_top, self.btn_bottom, self.btn_link):
            b.label.set_color('white'); b.label.set_fontsize(9)

        self.btn_find.on_clicked(self._on_search)
        self.btn_clear.on_clicked(self._on_clear)
        self.btn_open.on_clicked(self._on_open_file)
        self.btn_top.on_clicked(self._on_top_view)
        self.btn_bottom.on_clicked(self._on_bottom_view)
        self.btn_link.on_clicked(self._on_toggle_link)
        self.textbox.on_submit(self._on_search)

        # ── panel filter textbox (right side, below info panel) ──────────
        self._ax_filter = self.fig.add_axes([0.645, 0.020, 0.305, 0.066])
        self.filter_box = TextBox(self._ax_filter, 'Filter: ', initial='',
                                  color="#F4F7F5", hovercolor="#F7F9F6",
                                  label_pad=0.04)
        self.filter_box.label.set_color("#2AD3AB")
        self.filter_box.text_disp.set_color("#000000")
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
        self._info_mode          = 'idle'  # 'idle' | 'comp' | 'net' | 'tp'
        self._info_net_query     = ''
        self._search_results: list = []    # ranked search candidates
        self._selected_result_idx = 0
        self._view_mode          = 'top'   # 'top' | 'bottom'
        self._project_data       = data    # the one loaded file — used for both views
        self._project_dir        = str(Path(data.get('_filepath', __file__)).parent)
        self._tp_collection      = None
        self._pin_collection     = None
        self._bottom_comps       = []
        self._comp_labels        = []
        self._bot_pin_data       = []
        self._link_mode          = False   # True = board click shows all net connections
        self._net_conn_current   = ''      # net name being shown in net-connection sidebar
        self._net_conn_points: list = []   # cached all_points from last _show_net_connections
        self._last_table_redraw  = 0.0     # perf_counter timestamp of last _redraw_info_table

        # Sidebar resize state
        self._sidebar_left      = 0.645   # figure fraction where info panel starts
        self._dragging_sidebar  = False
        self._divider_line      = None
        self._no_tp_pin_rows: list = []   # (y_ctr, refdes, pin_num, px, py)
        self._TP_COL_FIG_W      = 0.42 * 0.350  # fixed TP column width in figure fractions

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

        # Draw resizable divider between board and sidebar
        from matplotlib.lines import Line2D as _Line2D
        self._divider_line = _Line2D(
            [self._sidebar_left - 0.008, self._sidebar_left - 0.008], [0.09, 1.0],
            transform=self.fig.transFigure, figure=self.fig,
            color='#2A4060', linewidth=2.5, zorder=50, solid_capstyle='round')
        self.fig.add_artist(self._divider_line)

        self.fig.canvas.mpl_connect('motion_notify_event',  self._on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)

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
            initialdir=self._project_dir,
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

        self._project_data = data
        self._project_dir  = str(Path(filepath).parent)
        self._view_mode    = 'top'
        self._show_idle_info()
        self.fig.canvas.draw_idle()
        print(f"Loaded: {filepath}")
        print(f"  Components: {n_c}  TPs: {n_tp}  Nets: {n_n}")

    # ─── TOP / BOTTOM view toggle ──────────────────────────────────────────

    def _on_top_view(self, _=None):
        """Switch to TOP view."""
        if self._view_mode == 'top':
            return
        self._view_mode = 'top'
        self._switch_data(self._project_data)

    def _on_bottom_view(self, _=None):
        """Switch to BOTTOM view (same file, different side filter)."""
        if self._view_mode == 'bottom':
            return
        self._view_mode = 'bottom'
        self._switch_data(self._project_data)

    def _on_toggle_link(self, _=None):
        """Toggle LINK mode: board click shows all net connections."""
        self._link_mode = not self._link_mode
        if self._link_mode:
            self.btn_link.ax.set_facecolor('#1A5A2A')
            self.btn_link.label.set_color('#88FF88')
            self.btn_link.label.set_fontsize(11)
        else:
            self.btn_link.ax.set_facecolor('#1A2A1A')
            self.btn_link.label.set_color('white')
            self.btn_link.label.set_fontsize(9)
        self.fig.canvas.draw_idle()

    def _switch_data(self, data: dict):
        """Rebuild indices and redraw the board with the given data dict."""
        self.data         = data
        self._pad_sizes   = data.get('pad_sizes', {})
        self._tp_pad_map  = data.get('tp_pad_map', {})
        self._testpoints  = {}
        self._tp_net      = {}
        self._tp_all_nets = {}
        self._comp_to_tps = defaultdict(lambda: defaultdict(list))
        self._tp_order    = []
        self._build_indices()
        self._comp_label_map = {}   # cla() removes artists; reset reference
        self._clear_highlights()
        self.ax.cla()
        self._tp_collection  = None   # cla() removed it from axes; reset reference
        self._pin_collection = None
        self.ax.set_facecolor('#0A1018')
        self._draw_board()
        if self._view_mode == 'bottom':
            self._draw_components_bottom()
        else:
            self._draw_components()
            self._draw_testpoints()
        self._auto_zoom()
        self._draw_legend()
        self._show_idle_info()
        self.fig.canvas.draw_idle()

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

        # Cache click tolerance (max TP pad radius × 3) so _handle_board_click
        # doesn't recompute it on every click.
        self._tp_click_tol = (
            max(self._r(c) * 3.0 for _, c in self._tp_order)
            if self._tp_order else 1.0
        )

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
        xs = [c['x'] for c in self.data['components'].values()
              if not is_testpoint(c) and c.get('side', 'TOP') == 'TOP']
        ys = [c['y'] for c in self.data['components'].values()
              if not is_testpoint(c) and c.get('side', 'TOP') == 'TOP']
        if xs:
            self.ax.scatter(xs, ys, s=5, color='#2E3E50', marker='s',
                            alpha=0.6, zorder=2, linewidths=0)

    _COMP_COLORS = {
        'IC':         "#976421",
        'RESISTOR':   '#4FC3F7',
        'CAPACITOR':  '#81C784',
        'CONNECTOR':  '#FF8A65',
        'INDUCTOR':   '#CE93D8',
        'TRANSISTOR': '#F48FB1',
        'DIODE':      '#80DEEA',
    }
    _COMP_COLOR_DEFAULT = '#B0BEC5'

    def _draw_components_bottom(self):
        """Draw BOT components as rectangles + all their pins as one PatchCollection."""
        comps = [(r, c) for r, c in self.data['components'].items()
                 if not is_testpoint(c) and c.get('side') == 'BOT']
        if not comps:
            return

        self._bottom_comps = comps
        bounds    = self.data.get('comp_bounds', {})
        comp_pins = self.data.get('comp_pins', {})
        pad_sizes = self._pad_sizes

        # ── component rectangles ─────────────────────────────────────────────
        verts, rect_colors = [], []
        for refdes, c in comps:
            b = bounds.get(refdes)
            if b and (b['xmax'] - b['xmin']) > 0.05 and (b['ymax'] - b['ymin']) > 0.05:
                verts.append([(b['xmin'], b['ymin']), (b['xmax'], b['ymin']),
                               (b['xmax'], b['ymax']), (b['xmin'], b['ymax'])])
            else:
                w, h = _comp_size(c)
                verts.append(_rotated_box(c['x'], c['y'], w, h, c.get('rotate', 0)))
            rect_colors.append(self._COMP_COLORS.get(
                c.get('comp_class', '').upper(), self._COMP_COLOR_DEFAULT))

        col = PolyCollection(verts, facecolors=rect_colors, edgecolors='#444444',
                             linewidths=0.5, alpha=0.80, zorder=4, clip_on=True)
        self.ax.add_collection(col)

        # ── pin circles (data-coords → auto-scales with zoom) ───────────────
        DEFAULT_R   = 0.045   # mm fallback
        pin_patches = []
        pin_colors  = []
        self._bot_pin_data = []  # (refdes, pin_x, pin_y, pin_name, pin_number, pad_stack)

        bot_refdes_set = {r for r, _ in comps}
        for refdes, pins in comp_pins.items():
            if refdes not in bot_refdes_set:
                continue
            for pin_x, pin_y, pin_name, pin_number, pad_stack in pins:
                r = pad_sizes.get(pad_stack, 0) / 2
                if r <= 0:
                    r = DEFAULT_R
                pin_patches.append(Circle((pin_x, pin_y), r))
                pin_colors.append('#22AAFF')
                self._bot_pin_data.append(
                    (refdes, pin_x, pin_y, pin_name, pin_number, pad_stack))

        if pin_patches:
            self._pin_collection = PatchCollection(
                pin_patches, facecolors=pin_colors, edgecolors='#004488',
                linewidths=0.2, alpha=0.88, zorder=7, clip_on=True)
            self.ax.add_collection(self._pin_collection)

        # Pre-create all label artists (hidden); toggled by _refresh_bottom_labels
        self._comp_label_map: dict = {}
        for refdes, c in comps:
            t = self.ax.text(
                c['x'], c['y'], refdes,
                fontsize=7, color='white', ha='center', va='center',
                fontweight='bold', zorder=6, clip_on=True, visible=False,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#000000',
                          alpha=0.55, edgecolor='none'))
            self._comp_label_map[refdes] = t

    def _refresh_bottom_labels(self):
        """Toggle label visibility for components in the current viewport.
        No artists are created or destroyed — only set_visible() is called."""
        label_map = getattr(self, '_comp_label_map', {})
        if not label_map:
            return
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        comps = getattr(self, '_bottom_comps', [])
        in_view = [
            (r, c) for r, c in comps
            if xlim[0] <= c['x'] <= xlim[1] and ylim[0] <= c['y'] <= ylim[1]
        ]
        show_labels = len(in_view) <= 80
        for refdes, c in comps:
            artist = label_map.get(refdes)
            if artist is None:
                continue
            in_viewport = xlim[0] <= c['x'] <= xlim[1] and ylim[0] <= c['y'] <= ylim[1]
            artist.set_visible(show_labels and in_viewport)

    def _refresh_bottom_pins(self):
        """Draw pad circles (data-coords) for BOT components visible in viewport.
        Called on every scroll/zoom so pins scale naturally with the board."""
        if self._pin_collection is not None:
            try: self._pin_collection.remove()
            except Exception: pass
            self._pin_collection = None

        if not self._bottom_comps or self._view_mode != 'bottom':
            return

        comp_pins = self.data.get('comp_pins', {})
        pad_sizes = self._pad_sizes
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # Only show pins when few enough BOT components are in view
        visible = [(r, c) for r, c in self._bottom_comps
                   if xlim[0] <= c['x'] <= xlim[1] and ylim[0] <= c['y'] <= ylim[1]]
        if len(visible) > 30:
            return

        DEFAULT_R = 0.045   # mm fallback radius
        patches, colors = [], []
        for refdes, _ in visible:
            for pin_x, pin_y, _, pad_stack in comp_pins.get(refdes, []):
                r = pad_sizes.get(pad_stack, 0) / 2
                if r <= 0:
                    r = DEFAULT_R
                patches.append(Circle((pin_x, pin_y), r))
                colors.append('#22AAFF')

        if patches:
            self._pin_collection = PatchCollection(
                patches, facecolors=colors, edgecolors='#004488',
                linewidths=0.2, alpha=0.88, zorder=7, clip_on=True)
            self.ax.add_collection(self._pin_collection)

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
        """Fit the view to the board outline (SIP edges) so both views share the same frame."""
        xs, ys = [], []
        for seg in self.data.get('outline', []):
            xs += [seg['x1'], seg['x2']]
            ys += [seg['y1'], seg['y2']]
        if not xs:                          # fallback: use component positions
            xs = [c['x'] for c in self.data['components'].values()]
            ys = [c['y'] for c in self.data['components'].values()]
        if not xs: return
        w   = max(xs) - min(xs)
        h   = max(ys) - min(ys)
        pad = max(w, h) * 0.04 + 1.0
        self.ax.set_xlim(min(xs) - pad, max(xs) + pad)
        self.ax.set_ylim(min(ys) - pad, max(ys) + pad)

    def _draw_legend(self):
        if self._view_mode == 'bottom':
            handles = [
                mpatches.Patch(color='#FFD700', label='IC'),
                mpatches.Patch(color='#4FC3F7', label='Resistor'),
                mpatches.Patch(color='#81C784', label='Capacitor'),
                mpatches.Patch(color='#FF8A65', label='Connector'),
                mpatches.Patch(color='#CE93D8', label='Inductor'),
                mpatches.Patch(color='#F48FB1', label='Transistor'),
                mpatches.Patch(color='#80DEEA', label='Diode'),
                mpatches.Patch(color='#B0BEC5', label='Other'),
            ]
        else:
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
            if self._view_mode == 'bottom':
                self._refresh_bottom_labels()
            self.fig.canvas.draw_idle()

        elif event.inaxes == self.ax_info:
            # ── scroll info table — rate-limited to 30 fps ────────────────
            if not self._all_info_rows: return
            delta   = -2 if event.button == 'up' else 2
            total   = len(self._filtered_rows())
            max_off = max(0, total - self._info_visible)
            self._info_scroll = max(0, min(max_off, self._info_scroll + delta))
            now = time.perf_counter()
            if now - self._last_table_redraw >= 1 / 30:
                self._last_table_redraw = now
                self._redraw_info_table()
                self.fig.canvas.draw_idle()

    # ─── sidebar resize ───────────────────────────────────────────────────────

    def _is_near_divider(self, event) -> bool:
        if event.x is None:
            return False
        bbox = self.fig.bbox
        fig_x = (event.x - bbox.x0) / bbox.width
        return abs(fig_x - (self._sidebar_left - 0.008)) < 0.012

    def _resize_layout(self):
        """Reposition axes and widgets after sidebar_left changes."""
        sl  = self._sidebar_left
        gap = 0.010
        # Board axes
        self.ax.set_position([0.005, 0.11, sl - gap - 0.005, 0.835])
        # Info axes
        iw = max(0.08, 1.0 - sl - 0.005)
        self.ax_info.set_position([sl, 0.11, iw, 0.875])
        # Filter textbox
        self._ax_filter.set_position([sl, 0.020, max(0.04, iw - 0.005), 0.066])
        # Divider line
        dx = sl - gap / 2
        self._divider_line.set_xdata([dx, dx])

    def _get_c2(self) -> float:
        """C2 axes-fraction so TP column keeps same physical width as sidebar grows."""
        sidebar_w = max(0.08, 1.0 - self._sidebar_left - 0.005)
        tp_col_axes_frac = min(0.50, self._TP_COL_FIG_W / sidebar_w)
        return max(0.40, 1.0 - tp_col_axes_frac - 0.02)

    def _max_net_chars(self) -> int:
        """Max chars for net name in table based on current sidebar width."""
        sidebar_w = max(0.08, 1.0 - self._sidebar_left - 0.005)
        return max(8, int(18 * sidebar_w / 0.350))

    def _on_motion(self, event):
        if not self._dragging_sidebar:
            return
        if event.x is None:
            return
        bbox = self.fig.bbox
        new_left = (event.x - bbox.x0) / bbox.width + 0.008
        self._sidebar_left = max(0.25, min(0.90, new_left))
        self._resize_layout()

        # Refresh table/sidebar content while dragging so truncation updates live.
        now = time.perf_counter()
        if now - self._last_table_redraw >= 1 / 30:
            self._last_table_redraw = now
            if self._info_mode == 'no_tp_pins' and self._net_conn_current:
                current_side = 'TOP' if self._view_mode == 'top' else 'BOT'
                self._draw_net_conn_sidebar(
                    self._net_conn_current, self._net_conn_points, current_side)
            elif self._info_mode in ('comp', 'net'):
                self._redraw_info_table()
        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        if self._dragging_sidebar:
            self._dragging_sidebar = False
            if self._info_mode == 'no_tp_pins' and self._net_conn_current:
                # Redraw net-connection sidebar with updated column widths/truncation;
                # do NOT call _clear_highlights so board highlights are preserved.
                current_side = 'TOP' if self._view_mode == 'top' else 'BOT'
                self._draw_net_conn_sidebar(
                    self._net_conn_current, self._net_conn_points, current_side)
            elif self._info_mode not in ('idle',):
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

        if self._is_near_divider(event):
            self._dragging_sidebar = True
            return

        if event.inaxes == self.ax:
            self._handle_board_click(event)
        elif event.inaxes == self.ax_info:
            self._handle_info_click(event)

    # ── click on PCB board ────────────────────────────────────────────────

    def _handle_board_click(self, event):
        if self._view_mode == 'bottom':
            self._handle_bottom_click(event)
            return

        TOL = self._tp_click_tol
        best_dist, best_r = TOL, None
        for refdes, c in self._tp_order:
            d = math.hypot(c['x'] - event.xdata, c['y'] - event.ydata)
            if d < best_dist:
                best_dist, best_r = d, refdes

        if best_r:
            if self._link_mode:
                net = self._tp_net.get(best_r, '')
                if net:
                    self._show_net_connections(net)
                    return
            self._clear_highlights()
            self._place_board_annot(best_r)
            self._show_info_tp(best_r)
        else:
            self._remove_board_annot()
            self.fig.canvas.draw_idle()

    def _handle_bottom_click(self, event):
        """Click handler for BOTTOM view.
        Priority: 1) nearest pin  2) smallest enclosing rect  3) nearest centre."""
        mx, my = event.xdata, event.ydata
        xlim   = self.ax.get_xlim()
        # tolerance scales with zoom: 1.5 % of viewport width
        tol = (xlim[1] - xlim[0]) * 0.015

        # ── 1. nearest pin ────────────────────────────────────────────────
        best_pin_d, best_pin = tol, None
        for entry in getattr(self, '_bot_pin_data', []):
            refdes, px, py, pin_name, pin_number, pad_stack = entry
            d = math.hypot(px - mx, py - my)
            if d < best_pin_d:
                best_pin_d, best_pin = d, entry

        if best_pin:
            refdes, px, py, pin_name, pin_number, pad_stack = best_pin
            if self._link_mode:
                comp_nets_list = self.data['comp_nets'].get(refdes, [])
                pin_nets = [net for net, pn, pnm in comp_nets_list
                            if str(pn) == str(pin_number)
                            or (pnm and pnm.upper() == pin_name.upper())]
                if pin_nets:
                    self._show_net_connections(pin_nets[0])
                    self.fig.canvas.draw_idle()
                    return
            self._highlight_bottom_pin(px, py, pad_stack)
            self._show_info_pin(refdes, pin_name, pin_number, px, py)
            self.fig.canvas.draw_idle()
            return

        # ── 2. smallest bounding-box enclosing the click ──────────────────
        bounds = self.data.get('comp_bounds', {})
        comps  = {r: c for r, c in self.data['components'].items()
                  if not is_testpoint(c) and c.get('side') == 'BOT'}

        best, best_area = None, float('inf')
        for refdes, b in bounds.items():
            if refdes not in comps: continue
            if b['xmin'] <= mx <= b['xmax'] and b['ymin'] <= my <= b['ymax']:
                area = (b['xmax'] - b['xmin']) * (b['ymax'] - b['ymin'])
                if area < best_area:
                    best_area, best = area, refdes

        # ── 3. nearest component centre within tolerance ───────────────────
        if best is None:
            best_d = tol
            for refdes, c in comps.items():
                d = math.hypot(c['x'] - mx, c['y'] - my)
                if d < best_d:
                    best_d, best = d, refdes

        if best:
            self._highlight_bottom_comp(best)
            self._show_info_comp(best, comps[best], {})
            self.fig.canvas.draw_idle()
        else:
            self._clear_highlights()
            self._show_idle_info()
            self.fig.canvas.draw_idle()

    def _highlight_bottom_comp(self, refdes: str):
        """Draw a red selection border + label over the clicked component."""
        self._clear_highlights()
        bounds = self.data.get('comp_bounds', {})
        comp   = self.data['components'].get(refdes)
        if not comp: return

        b = bounds.get(refdes)
        if b and (b['xmax'] - b['xmin']) > 0.05:
            x0, y0 = b['xmin'], b['ymin']
            w,  h  = b['xmax'] - b['xmin'], b['ymax'] - b['ymin']
            label_x = (b['xmin'] + b['xmax']) / 2
            label_y = b['ymax']
        else:
            w, h   = _comp_size(comp)
            x0, y0 = comp['x'] - w / 2, comp['y'] - h / 2
            label_x, label_y = comp['x'], comp['y'] + h / 2

        border = mpatches.Rectangle(
            (x0, y0), w, h,
            fill=False, edgecolor='#FF3333', linewidth=2.2,
            zorder=10, linestyle='--')
        self.ax.add_patch(border)
        self._highlights.append(border)

        lbl = self.ax.text(
            label_x, label_y + 0.25, refdes,
            color='#FFEE88', fontsize=8, ha='center', va='bottom',
            fontweight='bold', zorder=11, clip_on=True,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#0A1018',
                      edgecolor='#FFDD57', linewidth=1.0, alpha=0.92))
        self._highlights.append(lbl)

    def _highlight_bottom_pin(self, pin_x: float, pin_y: float, pad_stack: str):
        """Draw a bright ring around a clicked pin."""
        self._clear_highlights()
        r = self._pad_sizes.get(pad_stack, 0) / 2
        if r <= 0:
            r = 0.045
        ring = Circle((pin_x, pin_y), r * 2.5,
                      fill=False, edgecolor='#FF3333', linewidth=1.8,
                      zorder=12, linestyle='-')
        self.ax.add_patch(ring)
        self._highlights.append(ring)

    def _show_info_pin(self, refdes: str, pin_name: str, pin_number: str,
                       pin_x: float, pin_y: float):
        """Show pin details in the right-hand info panel."""
        self._clear_info()
        comp     = self.data['components'].get(refdes, {})
        comp_nets_list = self.data['comp_nets'].get(refdes, [])
        ax  = self.ax_info
        FM  = ax.transAxes
        y   = 0.96

        def t(text, col='#C0D0E0', sz=8, bold=False, dy=0.030):
            nonlocal y
            ax.text(0.04, y, text, transform=FM, color=col, fontsize=sz,
                    va='top', fontweight='bold' if bold else 'normal',
                    fontfamily='monospace', clip_on=True)
            y -= dy

        # Header
        t(f"{refdes}", col='#FFDD57', sz=15, bold=True, dy=0.052)
        t(f"Pin  : {pin_number}  ({pin_name})", col='#4FC3F7', sz=13, bold=True, dy=0.040)
        t(f"Pos  : ({pin_x:.3f}, {pin_y:.3f}) mm", col='#8899AA', sz=11, dy=0.033)
        t(f"Class: {comp.get('comp_class', '?')}   Value: {comp.get('value', '?')}",
          col='#8899AA', sz=11, dy=0.033)
        ax.plot([0, 1], [y, y], color='#1E2A3A', lw=0.6, transform=FM)
        y -= 0.016

        # Find net(s) for this pin (match by pin_name or pin_number)
        pin_name_u = pin_name.upper()
        pin_num_u  = pin_number.upper()
        pin_nets   = [net for net, pn, pnm in comp_nets_list
                      if pn.upper() == pin_num_u or pnm.upper() == pin_name_u]

        t("Net:", col='#90CDF4', sz=12, bold=True, dy=0.038)
        if pin_nets:
            for net in pin_nets[:6]:
                col = '#68D391' if not is_gnd_net(net) else '#3A5048'
                t(f"  {net}", col=col, sz=12, dy=0.036)
        else:
            t("  (no connection)", col='#445566', sz=11, dy=0.033)

    # ── click on info panel row ───────────────────────────────────────────

    def _handle_info_click(self, event):
        if not self._info_rows and not self._no_tp_pin_rows: return
        ax_y = self.ax_info.transAxes.inverted().transform(
            (event.x, event.y))[1]

        if self._info_mode == 'search_results' and self._search_results:
            best_dist, best_idx = 0.028, None
            for y_ctr, idx, _ in self._info_rows:
                dist = abs(ax_y - y_ctr)
                if dist < best_dist:
                    best_dist, best_idx = dist, idx
            if best_idx is not None and 0 <= best_idx < len(self._search_results):
                _score, kind, key = self._search_results[best_idx]
                if kind == 'net':
                    self._show_info_nets(str(key), [str(key)])
                else:
                    self.textbox.set_val(str(key))
                    self._on_search(str(key))
            return

        # ── no_tp_pins mode: click selects a specific point ───────────────
        if self._info_mode == 'no_tp_pins' and self._no_tp_pin_rows:
            best_dist, best_row = 0.028, None
            for row in self._no_tp_pin_rows:
                if abs(ax_y - row[0]) < best_dist:
                    best_dist = abs(ax_y - row[0])
                    best_row  = row
            if best_row:
                _, refdes, pin_num, px, py, side, is_tp, pad_stack = best_row
                # Auto-switch side if needed
                saved_net = self._net_conn_current
                if side == 'TOP' and self._view_mode != 'top':
                    self._on_top_view()          # redraws board & clears state
                    self._show_net_connections(saved_net)   # re-populate
                elif side == 'BOT' and self._view_mode != 'bottom':
                    self._on_bottom_view()
                    self._show_net_connections(saved_net)
                # Clear all orange highlights, draw single red selected patch
                for a in list(self._highlights):
                    try: a.remove()
                    except Exception: pass
                self._highlights.clear()
                patch = self._make_pad_highlight_patch(px, py, pad_stack, is_tp, '#FF2222')
                self.ax.add_patch(patch)
                self._highlights.append(patch)
                self._pan_to_point(px, py)
            return

        # ── normal comp/net table mode ────────────────────────────────────
        best_dist, best_tp = 0.028, None
        best_net           = None
        best_no_tp_net     = None
        for (y_ctr, tp_r, net) in self._info_rows:
            dist = abs(ax_y - y_ctr)
            if dist < best_dist:
                best_dist = dist
                if tp_r:
                    best_tp        = tp_r
                    best_net       = net
                    best_no_tp_net = None
                else:
                    best_no_tp_net = net
                    best_net       = net
                    best_tp        = None

        if self._info_mode == 'net' and (best_tp or best_no_tp_net):
            self._show_net_connections(best_net or best_no_tp_net)
        elif best_tp:
            self._clear_board_highlights()
            self._place_board_annot(best_tp)
            self._pan_to_tp(best_tp)
        elif best_no_tp_net:
            self._handle_no_tp_click(best_no_tp_net)

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

    def _pan_to_point(self, px: float, py: float):
        """Pan board to centre on (px, py) keeping current zoom."""
        xlim   = self.ax.get_xlim()
        ylim   = self.ax.get_ylim()
        half_w = (xlim[1] - xlim[0]) / 2
        half_h = (ylim[1] - ylim[0]) / 2
        self.ax.set_xlim(px - half_w, px + half_w)
        self.ax.set_ylim(py - half_h, py + half_h)
        self.fig.canvas.draw_idle()

    def _highlight_single_net_pin(self, px: float, py: float):
        """Keep existing net-pin scatter but add a bright ring on the selected pin."""
        # Remove any previous single-pin ring
        for a in list(self._highlights):
            if getattr(a, '_is_single_pin_ring', False):
                try: a.remove()
                except Exception: pass
                self._highlights.remove(a)
        ring = Circle((px, py), 0.28, fill=False, edgecolor='#FF3333',
                      linewidth=2.5, zorder=20)
        ring._is_single_pin_ring = True
        self.ax.add_patch(ring)
        self._highlights.append(ring)
        self.fig.canvas.draw_idle()

    def _handle_no_tp_click(self, net: str):
        """User clicked a no-TP signal row. Show all net connections."""
        self._show_net_connections(net)

    def _show_net_connections(self, net: str):
        """Unified: collect ALL points on net, highlight current-side on board, list all in sidebar."""
        self._net_conn_current = net
        netlist    = self.data.get('netlist', {})
        comp_pins  = self.data.get('comp_pins', {})
        components = self.data['components']
        current_side = 'TOP' if self._view_mode == 'top' else 'BOT'

        # Collect ALL connected points from BOTH sides
        # each entry: (label, refdes, pin_num_or_none, px, py, side, is_tp, pad_stack)
        all_points = []
        for refdes, pn, pnm in netlist.get(net, []):
            comp = components.get(refdes)
            if not comp:
                continue
            comp_side = comp.get('side', 'TOP')
            is_tp = is_testpoint(comp)

            if is_tp:
                tp = self._testpoints.get(refdes)
                if not tp:
                    continue
                pad_stack = self._tp_pad_map.get(refdes, '')
                all_points.append((refdes, refdes, None, tp['x'], tp['y'],
                                   comp_side, True, pad_stack))
            else:
                for px, py, pin_name, pin_number, pad_stack in comp_pins.get(refdes, []):
                    if str(pin_number) == str(pn) or (pnm and pin_name.upper() == pnm.upper()):
                        label = f'{refdes}.{pn}'
                        all_points.append((label, refdes, pn, px, py,
                                           comp_side, False, pad_stack))
                        break

        self._net_conn_points = all_points   # cache for sidebar-resize redraw

        # Highlight only current-side points on board (shape-aware)
        self._clear_highlights()
        for _lbl, refdes, pin_num, px, py, side, is_tp, pad_stack in all_points:
            if side != current_side:
                continue
            patch = self._make_pad_highlight_patch(px, py, pad_stack, is_tp, '#FF8800')
            self.ax.add_patch(patch)
            self._highlights.append(patch)
        self.fig.canvas.draw_idle()

        # Draw sidebar with complete list
        self._draw_net_conn_sidebar(net, all_points, current_side)

    def _make_pad_highlight_patch(self, px, py, pad_stack, is_tp, color):
        """Return a highlight patch shaped to the pad type."""
        raw  = self._pad_sizes.get(pad_stack, 0.0)
        size = max(0.12, min(0.40, raw * 0.5)) if raw else 0.15
        if is_tp:
            return Circle((px, py), size * 2.2,
                          fill=False, edgecolor=color, linewidth=2.2, zorder=15)
        # Component pin — use rectangle (SMD default)
        hw = size * 1.8 * 1.2   # half-width (slightly wider)
        hh = size * 1.8 * 0.8   # half-height
        ps_up = (pad_stack or '').upper()
        if any(s in ps_up for s in ('SQ', 'SQUARE')):
            hw = hh = size * 1.8
        return mpatches.Rectangle((px - hw, py - hh), hw * 2, hh * 2,
                                   fill=False, edgecolor=color,
                                   linewidth=2.2, zorder=15)

    def _draw_net_conn_sidebar(self, net: str, all_points: list, current_side: str):
        """Sidebar: list ALL connected points for net (no coordinates), grouped by side."""
        ax = self.ax_info
        ax.cla(); ax.axis('off'); ax.set_facecolor('#0A1018')
        self._info_rows.clear()
        self._no_tp_pin_rows = []
        self._info_mode      = 'no_tp_pins'
        FM  = ax.transAxes
        y   = 0.985
        DY  = 0.038

        # Header
        ax.text(0.03, y, f'Net: {net}', transform=FM, color='#FFAA44',
                fontsize=13, va='top', fontweight='bold',
                fontfamily='monospace', clip_on=True)
        y -= 0.052

        n_total = len(all_points)
        n_this  = sum(1 for p in all_points if p[5] == current_side)
        side_name = 'TOP' if current_side == 'TOP' else 'BOTTOM'
        ax.text(0.03, y,
                f'{n_total} points  •  highlighted: {n_this} on {side_name}',
                transform=FM, color='#6A8EA0', fontsize=9,
                va='top', fontfamily='monospace')
        y -= 0.030
        ax.plot([0, 1], [y, y], color='#1E2A3A', lw=0.7, transform=FM)
        y -= 0.014

        if not all_points:
            ax.text(0.03, y, 'No connection points found',
                    transform=FM, color='#FC8181', fontsize=10,
                    va='top', fontfamily='monospace')
            self.fig.canvas.draw_idle()
            return

        # Sort: current-side first → TPs before pins → alphabetical
        def _sort_key(p):
            same = 0 if p[5] == current_side else 1
            is_tp = 0 if p[6] else 1
            return (same, is_tp, p[0])
        sorted_pts = sorted(all_points, key=_sort_key)

        prev_side = None
        for label, refdes, pin_num, px, py, side, is_tp, pad_stack in sorted_pts:
            if y < 0.04:
                ax.text(0.03, y, '  …', transform=FM, color='#445566',
                        fontsize=9, va='top', fontfamily='monospace')
                break

            # Group header when side changes
            if side != prev_side:
                grp_name = 'TOP' if side == 'TOP' else 'BOTTOM'
                marker   = '●' if side == current_side else '○'
                grp_col  = '#4A8840' if side == current_side else '#3A4A5A'
                ax.text(0.03, y, f'{marker} {grp_name}',
                        transform=FM, color=grp_col, fontsize=8,
                        va='top', fontfamily='monospace', style='italic')
                y -= DY * 0.55
                ax.plot([0.01, 0.96], [y + DY * 0.15, y + DY * 0.15],
                        color='#1A2A38', lw=0.5, transform=FM)
                y -= DY * 0.15
                prev_side = side

            col = '#00CFFF' if is_tp else '#FFA07A'
            ax.text(0.06, y, label, transform=FM, color=col,
                    fontsize=9, va='top', fontfamily='monospace', clip_on=True)

            self._no_tp_pin_rows.append(
                (y - DY * 0.40, refdes, pin_num, px, py, side, is_tp, pad_stack))
            y -= DY

        self.fig.canvas.draw_idle()

    # ─── search / highlight ───────────────────────────────────────────────

    def _reset_info_state(self):
        self._info_rows.clear()
        self._all_info_rows.clear()
        self._info_scroll    = 0
        self._current_comp   = None
        self._info_mode      = 'idle'
        self._info_net_query = ''
        self._no_tp_pin_rows = []
        self._net_conn_current   = ''
        self._net_conn_points    = []
        self._search_results     = []
        self._selected_result_idx = 0

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
        self._reset_info_state()

    def _clear_board_highlights(self):
        """Remove board highlight patches/rings/annotation; preserve info-panel state."""
        for a in self._highlights:
            try: a.remove()
            except Exception: pass
        self._highlights.clear()
        if self._ring_collection:
            try: self._ring_collection.remove()
            except Exception: pass
            self._ring_collection = None
        self._remove_board_annot()

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
        candidates = []

        for refdes, comp in comps.items():
            ru = refdes.upper()
            side_bonus = 5 if comp.get('side', 'TOP') == ('TOP' if self._view_mode == 'top' else 'BOT') else 0
            if ru == query:
                candidates.append((100 + side_bonus, 'comp', refdes))
            elif ru.startswith(query):
                candidates.append((80 + side_bonus, 'comp', refdes))
            elif query in ru:
                candidates.append((70 + side_bonus, 'comp', refdes))

        for net in netlist:
            nu = net.upper()
            if nu == query:
                candidates.append((90, 'net', net))
            elif query in nu:
                candidates.append((50, 'net', net))

        # TP-driven component candidates
        tp_match = next((r for r in self._testpoints if r.upper() == query), None)
        if tp_match is None:
            tp_match = next((r for r in self._testpoints if r.upper().startswith(query)), None)
        if tp_match is None:
            tp_match = next((r for r in self._testpoints if query in r.upper()), None)
        if tp_match:
            nets_of_tp = [n for n, _, _ in comp_nets.get(tp_match, [])]
            seen = set()
            for net in nets_of_tp:
                for r, _, _ in netlist.get(net, []):
                    if r not in seen and r in comps and not is_testpoint(comps[r]):
                        seen.add(r)
                        candidates.append((60, 'comp', r))

        if not candidates:
            self._show_info_none(query)
            self.fig.canvas.draw_idle()
            return

        # Stable ranking: score desc, shorter id first, alphabetical
        candidates.sort(key=lambda t: (-t[0], len(str(t[2])), str(t[2])))
        top_score = candidates[0][0]
        top_band = [c for c in candidates if top_score - c[0] <= 5]
        if len(top_band) > 1:
            self._show_info_search_results(query, candidates[:100])
            self.fig.canvas.draw_idle()
            return

        kind, key = candidates[0][1], candidates[0][2]
        if kind == 'net':
            self._show_info_nets(query, [key])
            self.fig.canvas.draw_idle()
            return

        refdes = key
        comp   = comps.get(refdes)
        if not comp:
            self._show_info_none(query)
            self.fig.canvas.draw_idle()
            return
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
        self._search_results = []
        self._selected_result_idx = 0

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

        t(refdes, col='#FFDD57', sz=18, bold=True, dy=0.056)
        t(f"Pos : ({tp['x']:.3f}, {tp['y']:.3f}) mm", col='#8899AA', sz=11, dy=0.033)
        t(f"Pad : {tp['sym_name']}", col='#8899AA', sz=11, dy=0.036)
        ax.plot([0,1],[y,y], color='#1E2A3A', lw=0.6, transform=FM); y -= 0.016

        t("Signals:", col='#90CDF4', sz=12, bold=True, dy=0.038)
        for net in nets:
            if y < 0.08: break
            col = '#68D391' if not is_gnd_net(net) else '#3A5048'
            t(f"  {net}", col=col, sz=11, dy=0.033)

        ax.plot([0,1],[y,y], color='#1E2A3A', lw=0.6, transform=FM); y -= 0.016
        t("Components on these nets:", col='#90CDF4', sz=12, bold=True, dy=0.038)
        netlist = self.data['netlist']
        seen = set()
        for net in nets:
            if is_gnd_net(net): continue
            for r, pn, pnm in netlist.get(net, []):
                if r == refdes or r in seen: continue
                if is_testpoint(self.data['components'].get(r, {})): continue
                seen.add(r)
                if y < 0.07: t("  …", col='#445566', sz=10); break
                t(f"  {r:<12} {net}", col='#9AE6B4', sz=10, dy=0.030)

    # ── Search: two-column scrollable table (signal → test point) ────────
    def _show_info_comp(self, refdes: str, comp: dict, tp_map: dict):
        self._clear_info()
        self._info_mode = 'comp'

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
        HEADER_H = 0.26    # fraction of axes height used by header
        y = 0.985
        info_mode = getattr(self, '_info_mode', 'comp')
        if info_mode == 'net':
            ax.text(0.03, y, f'Signal: "{self._info_net_query}"',
                    transform=FM, color='#AAFFCC', fontsize=13, va='top',
                    fontweight='bold', fontfamily='monospace')
            y -= 0.060
        else:
            ax.text(0.03, y, refdes, transform=FM, color='#FFDD57',
                    fontsize=15, va='top', fontweight='bold',
                    fontfamily='monospace')
            y -= 0.048
            ax.text(0.03, y,
                    f"Class: {comp['comp_class']}   Value: {comp['value']}",
                    transform=FM, color='#6A7E94', fontsize=10, va='top',
                    fontfamily='monospace')
            y -= 0.028
            ax.text(0.03, y,
                    f"Pos: ({comp['x']:.3f}, {comp['y']:.3f}) mm",
                    transform=FM, color='#6A7E94', fontsize=10, va='top',
                    fontfamily='monospace')
            y -= 0.022
        ax.plot([0,1],[y,y], color='#1E2A3A', lw=0.7, transform=FM)
        y -= 0.012

        total_all = len(self._all_info_rows)
        n_with_tp = sum(1 for _, _, _, t in self._all_info_rows if t)

        # Apply panel filter
        visible_rows = self._filtered_rows()
        total        = len(visible_rows)

        q = self._panel_filter
        if q:
            ax.text(0.03, y,
                    f"Filter '{q}': {total}/{total_all}  (TP:{n_with_tp})",
                    transform=FM, color='#AAFFCC', fontsize=9, va='top',
                    fontfamily='monospace')
        else:
            ax.text(0.03, y,
                    f"Signals: {total_all}  (TP: {n_with_tp}  no TP: {total_all-n_with_tp})",
                    transform=FM, color='#4A6880', fontsize=9, va='top',
                    fontfamily='monospace')
        y -= 0.027
        ax.plot([0,1],[y,y], color='#1E2A3A', lw=0.5, transform=FM)
        y -= 0.004

        # Column header
        C1 = 0.02
        C2 = self._get_c2()
        DY = 0.035
        col_hdr = "SIGNAL" if info_mode == 'net' else "PIN   SIGNAL"
        ax.text(C1, y, col_hdr, transform=FM, color='#90CDF4',
                fontsize=10, va='top', fontweight='bold',
                fontfamily='monospace')
        if self._view_mode != 'bottom':
            ax.text(C2, y, "TEST POINT", transform=FM, color='#90CDF4',
                    fontsize=10, va='top', fontweight='bold',
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
                    transform=FM, color='#3A5060', fontsize=8,
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
                        transform=FM, color='#3A5060', fontsize=8,
                        va='bottom', fontfamily='monospace', style='italic')
            prev_group = cur_group

            has_tp  = tp_r is not None
            sig_col = '#9AE6B4' if has_tp else '#5A7A8A'
            _mc     = self._max_net_chars()
            net_str = net if len(net) <= _mc else net[:_mc - 1] + '\u2026'

            row_label = net_str if info_mode == 'net' else f"p{pin_n:<4} {net_str}"
            ax.text(C1, y, row_label,
                    transform=FM, color=sig_col, fontsize=9, va='top',
                    fontfamily='monospace', clip_on=True)

            if has_tp:
                tp_col = _tp_color(tp_r)
                ax.text(C2, y, tp_r,
                        transform=FM, color=tp_col, fontsize=9, va='top',
                        fontweight='bold', fontfamily='monospace', clip_on=True,
                        bbox=dict(boxstyle='round,pad=0.10', facecolor='#0F1A28',
                                  edgecolor=tp_col, alpha=0.6, linewidth=0.5))
            else:
                ax.text(C2, y, '\u2014',
                        transform=FM, color='#2A3A48', fontsize=9, va='top',
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
                    transform=FM, color='#3A5060', fontsize=9,
                    ha='center', va='bottom', fontfamily='monospace')

    def _show_info_nets(self, query: str, nets: list):
        """Signal-centric panel: list all nets matching query with their TPs."""
        self._clear_highlights()
        self._clear_info()
        self._info_mode      = 'net'
        self._info_net_query = query

        comp_nets = self.data['comp_nets']
        tp_by_net: dict = defaultdict(list)
        for r in self._testpoints:
            for net, _, _ in comp_nets.get(r, []):
                tp_by_net[net].append(r)

        rows = []
        all_tp_refdes = set()
        for net in nets:
            if is_gnd_net(net):
                continue
            tps = tp_by_net.get(net, [])
            if tps:
                for tp_r in sorted(set(tps)):
                    all_tp_refdes.add(tp_r)
                    rows.append((net, '', '', tp_r))
            else:
                rows.append((net, '', '', None))

        def _sort_key(row):
            tp_r = row[3]
            if tp_r is None: return (2, row[0])
            r = tp_r.upper()
            if r.startswith('TPA') or r.startswith('TP'): return (0, row[0])
            return (1, row[0])
        rows.sort(key=_sort_key)

        self._all_info_rows    = rows
        self._info_scroll      = 0
        self._info_comp_refdes = f'NET:{query}'
        self._info_comp        = {'comp_class': '', 'value': '', 'x': 0, 'y': 0}

        # Highlight all matching TPs on board
        if all_tp_refdes:
            ring_patches = []
            for tp_r in all_tp_refdes:
                tp = self._testpoints.get(tp_r)
                if not tp: continue
                ring_patches.append(
                    Circle((tp['x'], tp['y']), self._r(tp) * self._RING_MULT))
            self._ring_collection = PatchCollection(
                ring_patches, facecolors='none',
                edgecolors='#FF2222', linewidths=2.2,
                zorder=8, alpha=1.0)
            self.ax.add_collection(self._ring_collection)

        self._redraw_info_table()

    def _show_info_search_results(self, query: str, results: list):
        """Show ranked mixed search results (components + nets)."""
        self._clear_info()
        self._info_mode = 'search_results'
        self._search_results = results
        self._selected_result_idx = 0
        ax = self.ax_info
        FM = ax.transAxes
        y = 0.985
        DY = 0.034
        self._info_rows.clear()

        ax.text(0.03, y, f'Search: "{query}"', transform=FM, color='#AAFFCC',
                fontsize=13, va='top', fontweight='bold', fontfamily='monospace')
        y -= 0.048
        ax.text(0.03, y, f'{len(results)} close matches, click a row to focus',
                transform=FM, color='#6A8EA0', fontsize=9, va='top',
                fontfamily='monospace')
        y -= 0.028
        ax.plot([0, 1], [y, y], color='#1E2A3A', lw=0.7, transform=FM)
        y -= 0.012

        for idx, (score, kind, key) in enumerate(results[:28]):
            if y < 0.04:
                ax.text(0.03, y, '  …', transform=FM, color='#445566',
                        fontsize=9, va='top', fontfamily='monospace')
                break
            tag = 'COMP' if kind == 'comp' else 'NET'
            col = '#9AE6B4' if kind == 'comp' else '#90CDF4'
            text = f'[{tag}] {key:<20}  score={score}'
            ax.text(0.04, y, text, transform=FM, color=col,
                    fontsize=9, va='top', fontfamily='monospace', clip_on=True)
            self._info_rows.append((y - DY * 0.4, idx, None))
            y -= DY

    def _show_idle_info(self):
        self._clear_info()
        ax = self.ax_info
        ax.text(0.5, 0.60, "PCB Layout Viewer",
                transform=ax.transAxes, color='#3D5A80',
                fontsize=18, ha='center', fontweight='bold')
        ax.text(0.5, 0.53, "Enter refdes → Find   or   Click on a TP",
                transform=ax.transAxes, color='#2A3A50',
                fontsize=12, ha='center')
        ax.text(0.5, 0.47, "(e.g. U7500  U3100  TPA033  PPA301)",
                transform=ax.transAxes, color='#22303E',
                fontsize=11, ha='center', style='italic')
        ax.text(0.5, 0.38, "Click a row in the table → highlight TP",
                transform=ax.transAxes, color='#1E3040',
                fontsize=11, ha='center')
        n_tp   = len(self._testpoints)
        n_comp = len(self.data['components']) - n_tp
        n_net  = len(self.data['netlist'])
        for i, t in enumerate([f"{n_comp} components",
                                f"{n_tp} test points",
                                f"{n_net} nets"]):
            ax.text(0.5, 0.28 - i*0.07, t, transform=ax.transAxes,
                    color='#3D5A80', fontsize=13, ha='center')

    def _show_info_none(self, query: str):
        self._clear_info()
        self.ax_info.text(0.5, 0.55, f'Not found: "{query}"',
                          transform=self.ax_info.transAxes,
                          color='#FC8181', fontsize=14, ha='center')


# ─── Entry point ─────────────────────────────────────────────────────────────

def _pick_file(title: str, initialdir: str = None):
    """Open a Tk file-picker and return the chosen path, or None if cancelled."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
    path = filedialog.askopenfilename(
        title=title,
        filetypes=[("Fabmaster / text", "*.txt *.fab *.asc"),
                   ("All files", "*.*")],
        initialdir=initialdir,
    )
    root.destroy()
    return path or None


def main():
    # ── Pick the project file (used for both TOP and BOTTOM views) ────────
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = _pick_file("Open PCB Layout File (fabmaster .txt)")
        if not filepath:
            print("No file selected. Exiting.")
            return

    print(f"Loading {filepath} …")
    try:
        data = detect_and_parse(filepath)
    except ValueError as e:
        print(f"Error: {e}"); sys.exit(1)

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
