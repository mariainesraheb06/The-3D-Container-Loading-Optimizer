"""
3D Container Loading Optimizer — Desktop App v2
New features:
  • Full zoom + rotate with toolbar & mouse scroll
  • Click placed box → flip orientation or reposition (non-fragile only)
  • Edit & validate predefined container sizes inline
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random, math, time, threading
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from copy import deepcopy

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class Box:
    id: int
    length: float
    width: float
    height: float
    weight_kg: float
    fragile: bool = False

    @property
    def volume(self):
        return self.length * self.width * self.height

    def get_orientations(self):
        l, w, h = self.length, self.width, self.height
        if self.fragile:
            return [(l, w, h), (w, l, h)]
        return [
            (l, w, h), (l, h, w),
            (w, l, h), (w, h, l),
            (h, l, w), (h, w, l)
        ]

    def orientation_labels(self):
        """Human-readable labels for each orientation."""
        l, w, h = self.length, self.width, self.height
        oris = self.get_orientations()
        return [f"{o[0]}×{o[1]}×{o[2]} cm" for o in oris]


@dataclass
class Container:
    name: str
    length: float
    width: float
    height: float

    @property
    def volume(self):
        return self.length * self.width * self.height

    def __str__(self):
        return f"{self.name} ({self.length}×{self.width}×{self.height} cm)"


# ─────────────────────────────────────────────
#  PRESET CONTAINERS (mutable list — editable)
# ─────────────────────────────────────────────

PRESET_CONTAINERS = [
    Container("ISO 20ft Container",  589.0, 235.0, 239.0),
    Container("ISO 40ft Container", 1203.0, 235.0, 239.0),
    Container("Standard Truck",      600.0, 240.0, 250.0),
    Container("Delivery Van",        250.0, 160.0, 160.0),
    Container("Pallet Box",          120.0,  80.0, 100.0),
]


# ─────────────────────────────────────────────
#  PACKING ENGINE
# ─────────────────────────────────────────────

def check_overlap(p1, d1, p2, d2):
    x1,y1,z1=p1; l1,w1,h1=d1
    x2,y2,z2=p2; l2,w2,h2=d2
    if (x1+l1<=x2 or x2+l2<=x1 or
        y1+w1<=y2 or y2+w2<=y1 or
        z1+h1<=z2 or z2+h2<=z1):
        return False
    return True

def is_supported(x, y, z, l, w, placed):
    if z == 0:
        return True
    for pb in placed:
        px,py,pz=pb['pos']; pl,pw,ph=pb['dim']
        if abs((pz+ph)-z) < 0.01:
            ox = min(x+l, px+pl) - max(x, px)
            oy = min(y+w, py+pw) - max(y, py)
            if ox > 0 and oy > 0:
                return True
    return False

def pack_sequence(sequence, container):
    placed = []
    candidates = [(0.0, 0.0, 0.0)]
    vol_packed = 0.0
    for box in sequence:
        best_pos=None; best_ori=None; best_score=float('inf')
        for ori in box.get_orientations():
            bl,bw,bh=ori
            for (cx,cy,cz) in candidates:
                if cx+bl>container.length or cy+bw>container.width or cz+bh>container.height:
                    continue
                if not is_supported(cx,cy,cz,bl,bw,placed):
                    continue
                if any(check_overlap((cx,cy,cz),(bl,bw,bh),pb['pos'],pb['dim']) for pb in placed):
                    continue
                score = cx+cy+cz + cz*box.weight_kg
                if score < best_score:
                    best_score=score; best_pos=(cx,cy,cz); best_ori=ori
        if best_pos:
            bx,by,bz=best_pos; bl,bw,bh=best_ori
            placed.append({'id':box.id,'pos':best_pos,'dim':best_ori,
                           'weight':box.weight_kg,'fragile':box.fragile,
                           'original_box': box})
            vol_packed += bl*bw*bh
            candidates += [(bx+bl,by,bz),(bx,by+bw,bz),(bx,by,bz+bh)]
    return placed, (vol_packed/container.volume)*100.0

def pack_sequence_with_forced(sequence, container, forced_orientations):
    """
    Like pack_sequence but some boxes have a forced orientation.
    forced_orientations: dict {box_id: (l, w, h)}
    """
    placed = []
    candidates = [(0.0, 0.0, 0.0)]
    vol_packed = 0.0
    for box in sequence:
        if box.id in forced_orientations:
            oris = [forced_orientations[box.id]]
        else:
            oris = box.get_orientations()
        best_pos=None; best_ori=None; best_score=float('inf')
        for ori in oris:
            bl,bw,bh=ori
            for (cx,cy,cz) in candidates:
                if cx+bl>container.length or cy+bw>container.width or cz+bh>container.height:
                    continue
                if not is_supported(cx,cy,cz,bl,bw,placed):
                    continue
                if any(check_overlap((cx,cy,cz),(bl,bw,bh),pb['pos'],pb['dim']) for pb in placed):
                    continue
                score = cx+cy+cz + cz*box.weight_kg
                if score < best_score:
                    best_score=score; best_pos=(cx,cy,cz); best_ori=ori
        if best_pos:
            bx,by,bz=best_pos; bl,bw,bh=best_ori
            placed.append({'id':box.id,'pos':best_pos,'dim':best_ori,
                           'weight':box.weight_kg,'fragile':box.fragile,
                           'original_box': box})
            vol_packed += bl*bw*bh
            candidates += [(bx+bl,by,bz),(bx,by+bw,bz),(bx,by,bz+bh)]
    return placed, (vol_packed/container.volume)*100.0


# ─────────────────────────────────────────────
#  ALGORITHMS
# ─────────────────────────────────────────────

def greedy_pack(boxes, container):
    seq = sorted(boxes, key=lambda b: b.volume, reverse=True)
    return pack_sequence(seq, container)

def genetic_algorithm(boxes, container, pop_size=30, generations=50, progress_cb=None):
    n = len(boxes)
    def rand_ind():
        s=boxes[:]; random.shuffle(s); return s
    def fitness(ind):
        _,u=pack_sequence(ind,container); return u
    def crossover(p1,p2):
        cut=random.randint(1,n-1)
        c=p1[:cut]; c+=[b for b in p2 if b not in c]; return c
    def mutate(ind,rate=0.1):
        i=ind[:]
        for k in range(n):
            if random.random()<rate:
                j=random.randint(0,n-1); i[k],i[j]=i[j],i[k]
        return i
    pop=[rand_ind() for _ in range(pop_size)]
    best_ind=None; best_fit=0.0
    for gen in range(generations):
        sc=[(fitness(i),i) for i in pop]
        sc.sort(key=lambda x:x[0],reverse=True)
        if sc[0][0]>best_fit: best_fit=sc[0][0]; best_ind=sc[0][1]
        if progress_cb: progress_cb(gen,generations,best_fit)
        elites=[i for _,i in sc[:pop_size//4]]
        new_pop=elites[:]
        while len(new_pop)<pop_size:
            p1,p2=random.sample(elites,2)
            new_pop.append(mutate(crossover(p1,p2)))
        pop=new_pop
    return pack_sequence(best_ind, container)

def simulated_annealing(boxes, container, initial_sequence=None,
                         T_start=500, T_end=0.1, cooling=0.995,
                         iters=30, progress_cb=None):
    cur = initial_sequence[:] if initial_sequence else sorted(boxes, key=lambda b:b.volume, reverse=True)
    _,cur_util = pack_sequence(cur, container)
    best_seq=cur[:]; best_placed,best_util=pack_sequence(best_seq,container)
    T=T_start; step=0
    while T>T_end:
        for _ in range(iters):
            new=cur[:]
            i,j=random.sample(range(len(new)),2)
            new[i],new[j]=new[j],new[i]
            _,new_util=pack_sequence(new,container)
            delta=new_util-cur_util
            if delta>0 or random.random()<math.exp(delta/T):
                cur=new; cur_util=new_util
            if cur_util>best_util:
                best_util=cur_util; best_seq=cur[:]
                best_placed,best_util=pack_sequence(best_seq,container)
        T*=cooling; step+=1
        if progress_cb and step%10==0: progress_cb(T,T_start,best_util)
    return best_placed, best_util


# ─────────────────────────────────────────────
#  3D DRAWING
# ─────────────────────────────────────────────

def draw_box_3d(ax, pos, dim, color, alpha=0.55, highlight=False):
    x,y,z=pos; l,w,h=dim
    xx=[x,x+l,x+l,x,x,x+l,x+l,x]
    yy=[y,y,y+w,y+w,y,y,y+w,y+w]
    zz=[z,z,z,z,z+h,z+h,z+h,z+h]
    faces=[[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,3,7,4],[1,2,6,5]]
    ec='#FFD700' if highlight else '#333333'
    lw=2.0 if highlight else 0.3
    poly=Poly3DCollection(
        [[[xx[i],yy[i],zz[i]] for i in f] for f in faces],
        alpha=alpha, facecolor=color, edgecolor=ec, linewidth=lw
    )
    ax.add_collection3d(poly)


# ─────────────────────────────────────────────
#  EDIT CONTAINER DIALOG
# ─────────────────────────────────────────────

class EditContainerDialog(tk.Toplevel):
    """
    A popup window to edit a preset container's dimensions.
    Shows current values, validates input, and saves changes.
    """
    def __init__(self, parent, container: Container, on_save):
        super().__init__(parent)
        self.title(f"Edit — {container.name}")
        self.geometry("340x280")
        self.resizable(False, False)
        self.configure(bg="#f0f4f8")
        self.grab_set()  # Modal — blocks the main window

        self.container = container
        self.on_save = on_save

        tk.Label(self, text=f"Edit Container: {container.name}",
                 font=("Helvetica",11,"bold"), bg="#1a237e", fg="white",
                 pady=8).pack(fill="x")

        form = tk.Frame(self, bg="#f0f4f8", padx=20, pady=10)
        form.pack(fill="both", expand=True)

        fields = [
            ("Name",   container.name,   "Name shown in the list"),
            ("Length", container.length, "Internal length in cm  (e.g. 589)"),
            ("Width",  container.width,  "Internal width in cm  (e.g. 235)"),
            ("Height", container.height, "Internal height in cm  (e.g. 239)"),
        ]

        self._entries = {}
        for i,(lbl,default,hint) in enumerate(fields):
            tk.Label(form, text=lbl+":", bg="#f0f4f8",
                     font=("Helvetica",9,"bold"), width=8,
                     anchor="w").grid(row=i, column=0, pady=5, sticky="w")
            e = tk.Entry(form, font=("Helvetica",10), width=18,
                         relief="solid", bd=1)
            e.insert(0, str(default))
            e.grid(row=i, column=1, padx=8, pady=5)
            tk.Label(form, text=hint, bg="#f0f4f8",
                     font=("Helvetica",7), fg="#888").grid(
                         row=i, column=2, sticky="w", padx=4)
            self._entries[lbl] = e

        # Validation message
        self._msg = tk.Label(self, text="", fg="#c62828",
                             bg="#f0f4f8", font=("Helvetica",9))
        self._msg.pack()

        btn_row = tk.Frame(self, bg="#f0f4f8", pady=8)
        btn_row.pack()
        tk.Button(btn_row, text="✔  Save Changes",
                  font=("Helvetica",10,"bold"),
                  bg="#1565c0", fg="white", relief="flat",
                  padx=12, pady=5, cursor="hand2",
                  command=self._save).pack(side="left", padx=8)
        tk.Button(btn_row, text="✖  Cancel",
                  font=("Helvetica",10),
                  bg="#616161", fg="white", relief="flat",
                  padx=12, pady=5, cursor="hand2",
                  command=self.destroy).pack(side="left")

    def _save(self):
        name = self._entries["Name"].get().strip()
        errors = []
        if not name:
            errors.append("Name cannot be empty.")
        try:
            l = float(self._entries["Length"].get())
            if l <= 0: errors.append("Length must be > 0.")
        except ValueError:
            errors.append("Length must be a number."); l = 0

        try:
            w = float(self._entries["Width"].get())
            if w <= 0: errors.append("Width must be > 0.")
        except ValueError:
            errors.append("Width must be a number."); w = 0

        try:
            h = float(self._entries["Height"].get())
            if h <= 0: errors.append("Height must be > 0.")
        except ValueError:
            errors.append("Height must be a number."); h = 0

        if errors:
            self._msg.config(text="\n".join(errors))
            return

        # Apply changes
        self.container.name   = name
        self.container.length = l
        self.container.width  = w
        self.container.height = h

        self.on_save()
        self.destroy()


# ─────────────────────────────────────────────
#  FLIP/REPOSITION BOX DIALOG
# ─────────────────────────────────────────────

class FlipBoxDialog(tk.Toplevel):
    """
    Popup to flip orientation or manually set position of a non-fragile placed box.
    """
    def __init__(self, parent, placed_box: dict, box_obj: Box, container: Container, on_apply):
        super().__init__(parent)
        self.title(f"Edit Box #{placed_box['id']}")
        self.geometry("380x340")
        self.resizable(False, False)
        self.configure(bg="#f0f4f8")
        self.grab_set()

        self.pb = placed_box
        self.box_obj = box_obj
        self.container = container
        self.on_apply = on_apply

        tk.Label(self, text=f"Box #{placed_box['id']}  —  Non-Fragile",
                 font=("Helvetica",11,"bold"), bg="#00695c", fg="white",
                 pady=8).pack(fill="x")

        # Info
        info = tk.Frame(self, bg="#e8f5e9", pady=6, padx=12)
        info.pack(fill="x")
        cur_l,cur_w,cur_h = placed_box['dim']
        cur_x,cur_y,cur_z = placed_box['pos']
        tk.Label(info, text=f"Current orientation:  {cur_l}×{cur_w}×{cur_h} cm",
                 bg="#e8f5e9", font=("Helvetica",9)).pack(anchor="w")
        tk.Label(info, text=f"Current position:  x={cur_x}, y={cur_y}, z={cur_z}",
                 bg="#e8f5e9", font=("Helvetica",9)).pack(anchor="w")
        tk.Label(info, text=f"Weight: {placed_box['weight']} kg",
                 bg="#e8f5e9", font=("Helvetica",9)).pack(anchor="w")

        # Orientation selector
        ori_sec = tk.LabelFrame(self, text="Choose orientation",
                                bg="#f0f4f8", font=("Helvetica",9,"bold"),
                                fg="#1a237e", padx=10, pady=6)
        ori_sec.pack(fill="x", padx=12, pady=(10,0))

        self._ori_var = tk.StringVar()
        oris = box_obj.get_orientations()
        labels = box_obj.orientation_labels()
        current_ori_str = f"{cur_l}×{cur_w}×{cur_h} cm"
        for lbl, ori in zip(labels, oris):
            marker = "  ← current" if lbl == current_ori_str else ""
            rb = tk.Radiobutton(ori_sec, text=lbl+marker,
                                variable=self._ori_var, value=lbl,
                                bg="#f0f4f8", font=("Helvetica",9))
            rb.pack(anchor="w")
        self._ori_var.set(current_ori_str)
        self._oris = dict(zip(labels, oris))

        # Manual position override
        pos_sec = tk.LabelFrame(self, text="Override position (optional, cm)",
                                bg="#f0f4f8", font=("Helvetica",9,"bold"),
                                fg="#1a237e", padx=10, pady=6)
        pos_sec.pack(fill="x", padx=12, pady=(8,0))

        row = tk.Frame(pos_sec, bg="#f0f4f8")
        row.pack(fill="x")
        self._pos_entries = {}
        for axis, val in [("X", cur_x), ("Y", cur_y), ("Z", cur_z)]:
            tk.Label(row, text=axis+":", bg="#f0f4f8",
                     font=("Helvetica",9)).pack(side="left", padx=(4,0))
            e = tk.Entry(row, width=7, font=("Helvetica",9))
            e.insert(0, str(val))
            e.pack(side="left", padx=(2,6))
            self._pos_entries[axis] = e

        tk.Label(pos_sec,
                 text="Note: position will be validated against container bounds.\n"
                      "If it causes overlap, the change will be rejected.",
                 bg="#f0f4f8", fg="#666", font=("Helvetica",8),
                 wraplength=320, justify="left").pack(anchor="w", pady=2)

        self._msg = tk.Label(self, text="", fg="#c62828",
                             bg="#f0f4f8", font=("Helvetica",9))
        self._msg.pack()

        btn_row = tk.Frame(self, bg="#f0f4f8", pady=6)
        btn_row.pack()
        tk.Button(btn_row, text="✔  Apply",
                  font=("Helvetica",10,"bold"),
                  bg="#1565c0", fg="white", relief="flat",
                  padx=12, pady=5, cursor="hand2",
                  command=self._apply).pack(side="left", padx=8)
        tk.Button(btn_row, text="✖  Cancel",
                  font=("Helvetica",10),
                  bg="#616161", fg="white", relief="flat",
                  padx=12, pady=5, cursor="hand2",
                  command=self.destroy).pack(side="left")

    def _apply(self):
        # Get chosen orientation
        chosen_label = self._ori_var.get()
        new_ori = self._oris.get(chosen_label)
        if not new_ori:
            self._msg.config(text="Please select an orientation."); return

        # Get position
        try:
            nx = float(self._pos_entries["X"].get())
            ny = float(self._pos_entries["Y"].get())
            nz = float(self._pos_entries["Z"].get())
        except ValueError:
            self._msg.config(text="Position values must be numbers."); return

        # Validate bounds
        bl,bw,bh = new_ori
        c = self.container
        if nx<0 or ny<0 or nz<0 or nx+bl>c.length or ny+bw>c.width or nz+bh>c.height:
            self._msg.config(
                text=f"Box doesn't fit at that position/orientation in the container.\n"
                     f"Container: {c.length}×{c.width}×{c.height} cm")
            return

        self.on_apply(new_ori, (nx, ny, nz))
        self.destroy()


# ─────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("3D Container Loading Optimizer  v2")
        self.geometry("1280x860")
        self.configure(bg="#f0f4f8")
        self.resizable(True, True)

        self.boxes: List[Box] = []
        self.container: Container = PRESET_CONTAINERS[0]
        self.last_result  = None
        self.last_util    = 0.0
        self.last_algo    = ""
        self.selected_box_id = None   # highlighted box in 3D view
        self._forced_orientations = {}  # {box_id: (l,w,h)} — user overrides

        self._build_ui()

    # ── UI CONSTRUCTION ──────────────────────

    def _build_ui(self):
        top = tk.Frame(self, bg="#1a237e", pady=10)
        top.pack(fill="x")
        tk.Label(top, text="3D Container Loading Optimizer",
                 font=("Helvetica",17,"bold"), fg="white", bg="#1a237e").pack()
        tk.Label(top, text="Greedy  •  Genetic Algorithm  •  Simulated Annealing  •  Interactive 3D",
                 font=("Helvetica",9), fg="#90caf9", bg="#1a237e").pack()

        main = tk.Frame(self, bg="#f0f4f8")
        main.pack(fill="both", expand=True, padx=10, pady=10)

        left = tk.Frame(main, bg="#f0f4f8", width=430)
        left.pack(side="left", fill="y", padx=(0,8))
        left.pack_propagate(False)

        right = tk.Frame(main, bg="white",
                         highlightbackground="#dde3ed", highlightthickness=1)
        right.pack(side="left", fill="both", expand=True)

        self._build_left(left)
        self._build_right(right)

    def _build_left(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True)
        t1=tk.Frame(nb,bg="#f0f4f8"); nb.add(t1, text="  Container  ")
        t2=tk.Frame(nb,bg="#f0f4f8"); nb.add(t2, text="  Boxes  ")
        t3=tk.Frame(nb,bg="#f0f4f8"); nb.add(t3, text="  Run  ")
        t4=tk.Frame(nb,bg="#f0f4f8"); nb.add(t4, text="  Edit Boxes  ")
        self._build_container_tab(t1)
        self._build_boxes_tab(t2)
        self._build_run_tab(t3)
        self._build_edit_tab(t4)

    def _section(self, parent, title):
        f = tk.LabelFrame(parent, text=title, font=("Helvetica",9,"bold"),
                          bg="#f0f4f8", fg="#1a237e", padx=8, pady=6,
                          relief="groove", bd=1)
        f.pack(fill="x", padx=8, pady=(8,0))
        return f

    # ── TAB 1: Container ──────────────────────

    def _build_container_tab(self, parent):
        sec = self._section(parent, "Preset containers  (click ✏ to edit dimensions)")
        self._container_var = tk.StringVar(value=PRESET_CONTAINERS[0].name)
        self._preset_rows = []  # keep refs to update labels

        for c in PRESET_CONTAINERS:
            row = tk.Frame(sec, bg="#f0f4f8")
            row.pack(fill="x", pady=1)

            rb = tk.Radiobutton(row, textvariable=tk.StringVar(value=str(c)),
                                variable=self._container_var, value=c.name,
                                bg="#f0f4f8", font=("Helvetica",9),
                                command=self._on_preset_select)
            # Use a label that we can update after edits
            rb_label = tk.StringVar(value=str(c))
            rb = tk.Radiobutton(row, textvariable=rb_label,
                                variable=self._container_var, value=c.name,
                                bg="#f0f4f8", font=("Helvetica",9),
                                command=self._on_preset_select)
            rb.pack(side="left")

            edit_btn = tk.Button(row, text="✏",
                                 font=("Helvetica",8), bg="#e3f2fd",
                                 fg="#1565c0", relief="flat",
                                 cursor="hand2", padx=4,
                                 command=lambda cont=c, lv=rb_label: self._edit_preset(cont, lv))
            edit_btn.pack(side="right", padx=2)
            self._preset_rows.append((c, rb_label))

        sec2 = self._section(parent, "Or enter custom dimensions (cm)")
        grid = tk.Frame(sec2, bg="#f0f4f8"); grid.pack(fill="x")
        self._custom_entries = {}
        for i,(lbl,default) in enumerate([("Name","My Container"),
                                           ("Length","400"),("Width","200"),("Height","200")]):
            tk.Label(grid, text=lbl+":", bg="#f0f4f8",
                     font=("Helvetica",9), width=8, anchor="w").grid(row=i,column=0,pady=2,sticky="w")
            e = tk.Entry(grid, font=("Helvetica",9), width=16)
            e.insert(0,default)
            e.grid(row=i,column=1,padx=6,pady=2)
            self._custom_entries[lbl] = e
        tk.Button(sec2, text="✔  Use Custom Container",
                  font=("Helvetica",9,"bold"), bg="#1565c0", fg="white",
                  relief="flat", padx=8, pady=4, cursor="hand2",
                  command=self._apply_custom_container).pack(pady=6)

        self._container_info = tk.Label(parent, text="", font=("Helvetica",9),
                                        bg="#e8eaf6", fg="#283593",
                                        relief="flat", pady=6, wraplength=390)
        self._container_info.pack(fill="x", padx=8, pady=8)
        self._update_container_info()

    def _edit_preset(self, container, label_var):
        """Open the edit dialog for a preset container."""
        def on_save():
            # Update the radiobutton label to reflect new dimensions
            label_var.set(str(container))
            # If this container is currently selected, update info
            if self.container is container:
                self._update_container_info()
        EditContainerDialog(self, container, on_save)

    def _on_preset_select(self):
        name = self._container_var.get()
        for c in PRESET_CONTAINERS:
            if c.name == name:
                self.container = c; break
        self._update_container_info()

    def _apply_custom_container(self):
        errors = []
        name = self._custom_entries["Name"].get().strip() or "Custom"
        dims = {}
        for d in ["Length","Width","Height"]:
            try:
                v = float(self._custom_entries[d].get())
                if v <= 0: errors.append(f"{d} must be > 0.")
                dims[d] = v
            except ValueError:
                errors.append(f"{d} must be a number.")
                dims[d] = 0
        if errors:
            messagebox.showerror("Validation Error", "\n".join(errors)); return
        self.container = Container(name, dims["Length"], dims["Width"], dims["Height"])
        self._container_var.set("__custom__")
        self._update_container_info()
        messagebox.showinfo("Container Set", f"✔ Custom container set:\n{self.container}")

    def _update_container_info(self):
        c = self.container
        self._container_info.config(
            text=f"Selected: {c.name}\n"
                 f"Dimensions: {c.length} × {c.width} × {c.height} cm\n"
                 f"Volume: {c.volume/1e6:.3f} m³  ({c.volume:,.0f} cm³)"
        )

    # ── TAB 2: Boxes ──────────────────────────

    def _build_boxes_tab(self, parent):
        sec = self._section(parent, "Load from CSV")
        tk.Button(sec, text="📂  Load CSV file",
                  font=("Helvetica",9), bg="#2e7d32", fg="white",
                  relief="flat", padx=8, pady=4, cursor="hand2",
                  command=self._load_csv).pack(fill="x", pady=2)

        sec2 = self._section(parent, "Add a box manually")
        self._box_entries = {}
        row_f = tk.Frame(sec2, bg="#f0f4f8"); row_f.pack(fill="x")
        for i,lbl in enumerate(["Length","Width","Height","Weight(kg)"]):
            tk.Label(row_f, text=lbl, bg="#f0f4f8",
                     font=("Helvetica",8)).grid(row=0,column=i*2,padx=(4,0))
            e = tk.Entry(row_f, width=7, font=("Helvetica",9))
            e.grid(row=0,column=i*2+1,padx=(2,4))
            self._box_entries[lbl] = e
        row2 = tk.Frame(sec2, bg="#f0f4f8"); row2.pack(fill="x", pady=4)
        self._fragile_var = tk.BooleanVar()
        tk.Checkbutton(row2, text="Fragile", variable=self._fragile_var,
                       bg="#f0f4f8", font=("Helvetica",9)).pack(side="left")
        tk.Button(row2, text="➕ Add Box", font=("Helvetica",9,"bold"),
                  bg="#1565c0", fg="white", relief="flat",
                  padx=8, pady=3, cursor="hand2",
                  command=self._add_box_manual).pack(side="right")

        sec3 = self._section(parent, "Box list")
        cols=("id","L","W","H","kg","fragile")
        self._box_tree = ttk.Treeview(sec3, columns=cols, show="headings", height=8)
        for c in cols:
            self._box_tree.heading(c, text=c)
            self._box_tree.column(c, width=55, anchor="center")
        sc=ttk.Scrollbar(sec3, orient="vertical", command=self._box_tree.yview)
        self._box_tree.configure(yscrollcommand=sc.set)
        self._box_tree.pack(side="left", fill="both", expand=True)
        sc.pack(side="right", fill="y")

        row3=tk.Frame(parent,bg="#f0f4f8"); row3.pack(fill="x",padx=8,pady=4)
        tk.Button(row3,text="🗑 Remove",font=("Helvetica",9),bg="#b71c1c",
                  fg="white",relief="flat",padx=6,pady=3,cursor="hand2",
                  command=self._remove_box).pack(side="left")
        tk.Button(row3,text="Clear All",font=("Helvetica",9),bg="#616161",
                  fg="white",relief="flat",padx=6,pady=3,cursor="hand2",
                  command=self._clear_boxes).pack(side="left",padx=6)
        self._boxes_label = tk.Label(parent, text="No boxes loaded.",
                                     font=("Helvetica",9,"italic"),
                                     bg="#f0f4f8", fg="#555")
        self._boxes_label.pack(pady=2)

    def _load_csv(self):
        path = filedialog.askopenfilename(
            title="Open boxes CSV",
            filetypes=[("CSV files","*.csv"),("All files","*.*")])
        if not path: return
        try:
            df = pd.read_csv(path)
            df['fragile'] = df['fragile'].astype(str).str.lower().map(
                {'true':True,'false':False,'1':True,'0':False}).fillna(False)
            self.boxes = [
                Box(id=int(r['id']), length=float(r['length']),
                    width=float(r['width']), height=float(r['height']),
                    weight_kg=float(r['weight_kg']), fragile=bool(r['fragile']))
                for _,r in df.iterrows()]
            self._refresh_box_tree()
            messagebox.showinfo("Loaded", f"✔ Loaded {len(self.boxes)} boxes.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load CSV:\n{e}")

    def _add_box_manual(self):
        try:
            l=float(self._box_entries["Length"].get())
            w=float(self._box_entries["Width"].get())
            h=float(self._box_entries["Height"].get())
            kg=float(self._box_entries["Weight(kg)"].get())
            if l<=0 or w<=0 or h<=0 or kg<0: raise ValueError
            new_id=max((b.id for b in self.boxes),default=0)+1
            self.boxes.append(Box(id=new_id,length=l,width=w,height=h,
                                  weight_kg=kg,fragile=self._fragile_var.get()))
            self._refresh_box_tree()
            for e in self._box_entries.values(): e.delete(0,'end')
        except ValueError:
            messagebox.showerror("Invalid Input","Enter valid positive numbers.")

    def _remove_box(self):
        for item in self._box_tree.selection():
            vid = self._box_tree.item(item)['values'][0]
            self.boxes = [b for b in self.boxes if b.id != vid]
        self._refresh_box_tree()

    def _clear_boxes(self):
        if messagebox.askyesno("Confirm","Clear all boxes?"):
            self.boxes=[]; self._refresh_box_tree()

    def _refresh_box_tree(self):
        self._box_tree.delete(*self._box_tree.get_children())
        for b in self.boxes:
            self._box_tree.insert("","end",values=(
                b.id,b.length,b.width,b.height,b.weight_kg,"Yes" if b.fragile else "No"))
        n=len(self.boxes)
        self._boxes_label.config(
            text=f"{n} box{'es' if n!=1 else ''} loaded  |  "
                 f"{sum(1 for b in self.boxes if b.fragile)} fragile")

    # ── TAB 3: Run ────────────────────────────

    def _build_run_tab(self, parent):
        sec = self._section(parent, "Step 1 — Choose base algorithm")
        tk.Label(sec, text="Run Greedy or Genetic Algorithm first:",
                 bg="#f0f4f8", font=("Helvetica",9)).pack(anchor="w")
        bf=tk.Frame(sec,bg="#f0f4f8"); bf.pack(fill="x",pady=6)
        self._btn_greedy=tk.Button(bf,text="▶  Run Greedy",
                  font=("Helvetica",10,"bold"),bg="#00695c",fg="white",
                  relief="flat",padx=10,pady=8,cursor="hand2",width=15,
                  command=lambda:self._run_algo("greedy"))
        self._btn_greedy.pack(side="left",padx=(0,8))
        self._btn_ga=tk.Button(bf,text="▶  Run Genetic (GA)",
                  font=("Helvetica",10,"bold"),bg="#6a1b9a",fg="white",
                  relief="flat",padx=10,pady=8,cursor="hand2",
                  command=lambda:self._run_algo("ga"))
        self._btn_ga.pack(side="left")
        self._progress=ttk.Progressbar(sec,mode="indeterminate",length=380)
        self._progress.pack(fill="x",pady=4)
        self._result_label=tk.Label(sec,text="No results yet.",
                                    font=("Helvetica",10),bg="#e8f5e9",
                                    fg="#1b5e20",relief="flat",pady=8,wraplength=380)
        self._result_label.pack(fill="x",pady=4)

        sec2 = self._section(parent, "Step 2 — Improve with Simulated Annealing")
        tk.Label(sec2, text="Apply SA to improve the baseline result:",
                 bg="#f0f4f8", font=("Helvetica",9)).pack(anchor="w")
        self._btn_sa=tk.Button(sec2,text="🌡  Improve with SA",
                  font=("Helvetica",10,"bold"),bg="#e65100",fg="white",
                  relief="flat",padx=10,pady=8,cursor="hand2",
                  state="disabled",command=self._run_sa)
        self._btn_sa.pack(fill="x",pady=6)
        self._sa_label=tk.Label(sec2,text="",font=("Helvetica",10),
                                bg="#fff3e0",fg="#e65100",
                                relief="flat",pady=6,wraplength=380)
        self._sa_label.pack(fill="x")

        sec3 = self._section(parent, "SA Parameters")
        self._sa_params={}
        for lbl,default in [("Start Temp","1000"),("End Temp","0.1"),
                             ("Cooling Rate","0.995"),("Iters/Step","30")]:
            row=tk.Frame(sec3,bg="#f0f4f8"); row.pack(fill="x",pady=1)
            tk.Label(row,text=lbl+":",bg="#f0f4f8",
                     font=("Helvetica",9),width=14,anchor="w").pack(side="left")
            e=tk.Entry(row,width=10,font=("Helvetica",9))
            e.insert(0,default); e.pack(side="left",padx=4)
            self._sa_params[lbl]=e

    # ── TAB 4: Edit Placed Boxes ─────────────

    def _build_edit_tab(self, parent):
        tk.Label(parent,
                 text="After running an algorithm, select a placed box\n"
                      "to flip its orientation or change its position.\n"
                      "Only non-fragile boxes can be modified.",
                 bg="#f0f4f8", font=("Helvetica",9), fg="#555",
                 justify="left", pady=6).pack(padx=12, anchor="w")

        sec = self._section(parent, "Placed boxes")
        cols=("id","x","y","z","L","W","H","kg","fragile")
        self._placed_tree = ttk.Treeview(sec, columns=cols, show="headings", height=10)
        widths={"id":40,"x":60,"y":60,"z":60,"L":55,"W":55,"H":55,"kg":55,"fragile":55}
        for c in cols:
            self._placed_tree.heading(c,text=c)
            self._placed_tree.column(c,width=widths.get(c,55),anchor="center")
        sc2=ttk.Scrollbar(sec,orient="vertical",command=self._placed_tree.yview)
        self._placed_tree.configure(yscrollcommand=sc2.set)
        self._placed_tree.pack(side="left",fill="both",expand=True)
        sc2.pack(side="right",fill="y")
        self._placed_tree.bind("<<TreeviewSelect>>", self._on_placed_select)

        row=tk.Frame(parent,bg="#f0f4f8"); row.pack(fill="x",padx=8,pady=6)
        self._btn_flip=tk.Button(row,text="✏  Flip / Reposition Selected Box",
                  font=("Helvetica",9,"bold"),bg="#1565c0",fg="white",
                  relief="flat",padx=10,pady=6,cursor="hand2",
                  state="disabled",command=self._flip_box)
        self._btn_flip.pack(fill="x")

        self._edit_msg=tk.Label(parent,text="",font=("Helvetica",9),
                                bg="#f0f4f8",fg="#555",wraplength=380)
        self._edit_msg.pack(padx=12,anchor="w")

    def _on_placed_select(self, event):
        sel = self._placed_tree.selection()
        if not sel:
            self._btn_flip.config(state="disabled"); return
        vals = self._placed_tree.item(sel[0])['values']
        box_id = vals[0]
        is_fragile = vals[8] == "Yes"
        self.selected_box_id = box_id
        # Re-render with highlight
        if self.last_result:
            self._render_3d(self.last_result,
                            f"{self.last_algo} — Box #{box_id} selected",
                            highlight_id=box_id)
        if is_fragile:
            self._btn_flip.config(state="disabled")
            self._edit_msg.config(text="Fragile boxes cannot be modified.")
        else:
            self._btn_flip.config(state="normal")
            self._edit_msg.config(text=f"Box #{box_id} selected. Click Flip/Reposition to edit.")

    def _flip_box(self):
        if not self.selected_box_id or not self.last_result: return
        pb = next((p for p in self.last_result if p['id']==self.selected_box_id), None)
        box_obj = next((b for b in self.boxes if b.id==self.selected_box_id), None)
        if not pb or not box_obj: return

        def on_apply(new_ori, new_pos):
            # Apply forced orientation and position
            self._forced_orientations[self.selected_box_id] = new_ori
            # Repack using forced orientations
            seq = [b for b in self.boxes if b.id in [p['id'] for p in self.last_result]]
            placed, util = pack_sequence_with_forced(seq, self.container, self._forced_orientations)
            self.last_result = placed
            self.last_util   = util
            self._refresh_placed_tree()
            self._render_3d(placed, f"{self.last_algo} (edited) — {util:.1f}%")
            self._edit_msg.config(text=f"✔ Box #{self.selected_box_id} updated! Utilization: {util:.2f}%")

        FlipBoxDialog(self, pb, box_obj, self.container, on_apply)

    def _refresh_placed_tree(self):
        self._placed_tree.delete(*self._placed_tree.get_children())
        if not self.last_result: return
        for p in self.last_result:
            x,y,z=p['pos']; l,w,h=p['dim']
            self._placed_tree.insert("","end",values=(
                p['id'],round(x,1),round(y,1),round(z,1),
                l,w,h,p['weight'],"Yes" if p['fragile'] else "No"))

    # ── RIGHT PANEL: 3D Visualization ────────

    def _build_right(self, parent):
        # Title + controls row
        top_row = tk.Frame(parent, bg="white")
        top_row.pack(fill="x", padx=8, pady=(8,0))
        tk.Label(top_row, text="3D Packing Visualization",
                 font=("Helvetica",12,"bold"), bg="white",
                 fg="#1a237e").pack(side="left")

        # View angle preset buttons
        angle_frame = tk.Frame(top_row, bg="white")
        angle_frame.pack(side="right")
        for label, elev, azim in [("Front",0,0),("Side",0,90),("Top",90,0),("ISO",25,45)]:
            tk.Button(angle_frame, text=label, font=("Helvetica",8),
                      bg="#e3f2fd", fg="#1565c0", relief="flat",
                      padx=6, pady=2, cursor="hand2",
                      command=lambda e=elev,a=azim: self._set_view_angle(e,a)
                      ).pack(side="left", padx=2)

        # Matplotlib figure
        self._fig = plt.Figure(figsize=(7.5, 6), dpi=90)
        self._ax  = self._fig.add_subplot(111, projection='3d')
        self._ax.set_title("Run an algorithm to see the result",
                           fontsize=10, color="#888")

        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=8)

        # Navigation toolbar — gives zoom, pan, save buttons
        toolbar_frame = tk.Frame(parent, bg="white")
        toolbar_frame.pack(fill="x", padx=8)
        self._toolbar = NavigationToolbar2Tk(self._canvas, toolbar_frame)
        self._toolbar.update()

        # Mouse scroll to zoom
        self._canvas.mpl_connect('scroll_event', self._on_scroll)

        # Stats bar
        self._stats_bar = tk.Label(parent, text="",
                                   font=("Helvetica",10,"bold"),
                                   bg="#e8eaf6", fg="#283593",
                                   relief="flat", pady=6)
        self._stats_bar.pack(fill="x", padx=8, pady=(2,8))

        # Zoom hint
        tk.Label(parent,
                 text="🖱 Left-drag: rotate  |  Right-drag / scroll: zoom  |  Use toolbar for more controls",
                 font=("Helvetica",8), fg="#888", bg="white").pack(pady=(0,4))

    def _set_view_angle(self, elev, azim):
        """Snap to a preset view angle."""
        self._ax.view_init(elev=elev, azim=azim)
        self._canvas.draw()

    def _on_scroll(self, event):
        """Zoom in/out with mouse scroll wheel."""
        ax = self._ax
        factor = 0.9 if event.button == 'up' else 1.1
        # Scale the axis limits around the center
        for get_lim, set_lim in [
            (ax.get_xlim, ax.set_xlim),
            (ax.get_ylim, ax.set_ylim),
            (ax.get_zlim, ax.set_zlim)
        ]:
            lo, hi = get_lim()
            mid = (lo + hi) / 2
            half = (hi - lo) / 2 * factor
            set_lim(mid - half, mid + half)
        self._canvas.draw_idle()

    # ── ALGORITHMS ───────────────────────────

    def _validate_ready(self):
        if not self.boxes:
            messagebox.showwarning("No Boxes","Please load or add boxes first."); return False
        if not self.container:
            messagebox.showwarning("No Container","Please select a container first."); return False
        return True

    def _run_algo(self, algo):
        if not self._validate_ready(): return
        self._progress.start()
        self._btn_greedy.config(state="disabled")
        self._btn_ga.config(state="disabled")
        self._btn_sa.config(state="disabled")
        self._result_label.config(text="Running…")
        self._forced_orientations = {}  # reset manual edits on new run

        def task():
            t0=time.time()
            if algo=="greedy":
                placed,util=greedy_pack(self.boxes,self.container)
                name="Greedy"
            else:
                def prog(gen,total,best):
                    self._result_label.config(text=f"GA: gen {gen}/{total}  best={best:.1f}%")
                placed,util=genetic_algorithm(self.boxes,self.container,
                                              pop_size=30,generations=50,progress_cb=prog)
                name="Genetic Algorithm"
            rt=time.time()-t0
            self.last_result=placed; self.last_util=util; self.last_algo=name
            self.after(0,lambda:self._on_algo_done(placed,util,name,rt))

        threading.Thread(target=task,daemon=True).start()

    def _on_algo_done(self, placed, util, name, rt):
        self._progress.stop()
        self._btn_greedy.config(state="normal")
        self._btn_ga.config(state="normal")
        self._btn_sa.config(state="normal")
        self._result_label.config(
            text=f"✅ {name} done!\n"
                 f"Placed: {len(placed)}/{len(self.boxes)}  |  "
                 f"Utilization: {util:.2f}%  |  Time: {rt:.1f}s")
        self._render_3d(placed, f"{name} — {util:.1f}% utilization")
        self._refresh_placed_tree()

    def _run_sa(self):
        if not self.last_result:
            messagebox.showwarning("No Baseline","Run Greedy or GA first."); return
        try:
            T_start = float(self._sa_params["Start Temp"].get())
            T_end   = float(self._sa_params["End Temp"].get())
            cooling = float(self._sa_params["Cooling Rate"].get())
            iters   = int(self._sa_params["Iters/Step"].get())
        except ValueError:
            messagebox.showerror("Invalid Params","Check SA parameter values."); return
        self._progress.start()
        self._btn_sa.config(state="disabled")
        self._sa_label.config(text="SA running…")
        last_ids=[p['id'] for p in self.last_result]
        init_seq=sorted(self.boxes, key=lambda b:last_ids.index(b.id) if b.id in last_ids else 999)

        def task():
            t0=time.time()
            def prog(T,T0,best):
                self._sa_label.config(text=f"SA: T={T:.2f}  best={best:.2f}%")
            placed,util=simulated_annealing(
                self.boxes,self.container,initial_sequence=init_seq,
                T_start=T_start,T_end=T_end,cooling=cooling,
                iters=iters,progress_cb=prog)
            rt=time.time()-t0
            self.after(0,lambda:self._on_sa_done(placed,util,rt))

        threading.Thread(target=task,daemon=True).start()

    def _on_sa_done(self, placed, util, rt):
        self._progress.stop()
        self._btn_sa.config(state="normal")
        imp=util-self.last_util
        self._sa_label.config(
            text=f"✅ SA done!  {util:.2f}%  (+{imp:.2f}% improvement)  {rt:.1f}s")
        self.last_result=placed; self.last_util=util
        self._render_3d(placed, f"SA Improved — {util:.1f}% utilization")
        self._refresh_placed_tree()

    # ── 3D RENDER ────────────────────────────

    def _render_3d(self, placed, title, highlight_id=None):
        # Save current view angle so rotation is preserved after refresh
        elev = self._ax.elev
        azim = self._ax.azim
        self._ax.cla()
        c = self.container

        # Container wireframe
        verts=[(0,0,0),(c.length,0,0),(c.length,c.width,0),(0,c.width,0),
               (0,0,c.height),(c.length,0,c.height),(c.length,c.width,c.height),(0,c.width,c.height)]
        edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for e in edges:
            p1,p2=verts[e[0]],verts[e[1]]
            self._ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],
                          'k--',alpha=0.25,linewidth=0.7)

        normal_cols=['#4fc3f7','#81c784','#ffb74d','#ce93d8',
                     '#80cbc4','#fff176','#a5d6a7','#ef9a9a']
        fragile_cols=['#ef5350','#ff7043','#ec407a']
        ni=fi=0

        for pb in placed:
            is_highlight = (highlight_id is not None and pb['id']==highlight_id)
            if pb['fragile']:
                color=fragile_cols[fi%len(fragile_cols)]; fi+=1
            else:
                color=normal_cols[ni%len(normal_cols)]; ni+=1
            draw_box_3d(self._ax, pb['pos'], pb['dim'], color,
                        alpha=0.8 if is_highlight else 0.55,
                        highlight=is_highlight)

        # Restore the same view angle
        self._ax.view_init(elev=elev, azim=azim)
        self._ax.set_xlim(0,c.length)
        self._ax.set_ylim(0,c.width)
        self._ax.set_zlim(0,c.height)
        self._ax.set_xlabel('Length (cm)',fontsize=8)
        self._ax.set_ylabel('Width (cm)',fontsize=8)
        self._ax.set_zlabel('Height (cm)',fontsize=8)
        self._ax.set_title(title,fontsize=10,fontweight='bold')

        handles=[mpatches.Patch(color='#4fc3f7',label='Normal'),
                 mpatches.Patch(color='#ef5350',label='Fragile')]
        if highlight_id:
            handles.append(mpatches.Patch(color='gold',label=f'Selected #{highlight_id}'))
        self._ax.legend(handles=handles,loc='upper left',fontsize=8)

        self._canvas.draw()

        vol_used=sum(p['dim'][0]*p['dim'][1]*p['dim'][2] for p in placed)
        util_pct=vol_used/c.volume*100
        self._stats_bar.config(
            text=f"Container: {c.name}   |   "
                 f"Boxes: {len(placed)}/{len(self.boxes)}   |   "
                 f"{vol_used/1e6:.3f} m³ / {c.volume/1e6:.3f} m³   |   "
                 f"Utilization: {util_pct:.2f}%")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()

