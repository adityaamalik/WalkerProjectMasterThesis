"""
Walker2D Thesis Viewer — Tkinter GUI for exploring trained agents.

Lets the user select a curriculum strategy and seed, view per-seed statistics,
launch the MuJoCo viewer to watch the trained agent, and inspect learning
curves and gravity-robustness profiles. The best-performing seed for each
strategy is highlighted with a star marker.

Run:
    python viewer_app.py

Requirements: data must be present under experiments/thesis_morph/<strategy>/seed_<NN>/.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "thesis_morph"
N_SEEDS = 10

# Strategy display order (best → worst by mean final score, from thesis results)
STRATEGY_ORDER = [
    "staged_evolution",
    "gradual_transition",
    "archive_based",
    "multi_environment",
    "fixed_gravity",
    "random_variable_gravity",
    "adaptive_progression",
]

STRATEGY_DISPLAY = {
    "staged_evolution":        "Staged Evolution",
    "gradual_transition":      "Gradual Transition",
    "archive_based":           "Archive-Based",
    "multi_environment":       "Multi-Environment",
    "fixed_gravity":           "Fixed Gravity (Baseline)",
    "random_variable_gravity": "Random Variable (Control)",
    "adaptive_progression":    "Adaptive Progression",
}

STRATEGY_ROLE = {
    "staged_evolution":        "curriculum",
    "gradual_transition":      "curriculum",
    "archive_based":           "curriculum",
    "multi_environment":       "curriculum",
    "adaptive_progression":    "curriculum",
    "fixed_gravity":           "baseline",
    "random_variable_gravity": "control",
}

GRAVITY_OPTIONS = [-6.0, -7.5, -9.81, -11.0, -12.0]

# Color palette
COLOR_BG = "#f4f4f7"
COLOR_PANEL = "#ffffff"
COLOR_ACCENT = "#1f4e79"
COLOR_BEST = "#d97706"
COLOR_TEXT = "#1f2937"
COLOR_MUTED = "#6b7280"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_summary(strategy: str, seed: int) -> dict | None:
    path = RESULTS_DIR / strategy / f"seed_{seed:02d}" / "summary.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_csv(strategy: str, seed: int, filename: str) -> list[dict]:
    path = RESULTS_DIR / strategy / f"seed_{seed:02d}" / filename
    if not path.exists():
        return []
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def load_all_summaries() -> dict[str, dict[int, dict]]:
    """Load every available summary.json, indexed by [strategy][seed]."""
    out: dict[str, dict[int, dict]] = {}
    for strategy in STRATEGY_ORDER:
        out[strategy] = {}
        for seed in range(N_SEEDS):
            summary = load_summary(strategy, seed)
            if summary is not None:
                out[strategy][seed] = summary
    return out


def best_seed_for(summaries: dict[int, dict]) -> int | None:
    """Return the seed with the highest final Earth score."""
    if not summaries:
        return None
    return max(
        summaries.keys(),
        key=lambda s: summaries[s].get("final_earth", {}).get("score", -1e9),
    )


def get_checkpoint_path(strategy: str, seed: int) -> Path | None:
    """Return the best checkpoint path for a run, or None if none exists.

    Curriculum arms produce both ``best_earth_params.npy`` (best Earth-probe
    score) and ``best_params.npy`` (best training-gravity score). The
    fixed-gravity baseline trains on Earth throughout, so it only saves
    ``best_params.npy`` — which is already the best Earth solution.
    """
    base = RESULTS_DIR / strategy / f"seed_{seed:02d}" / "checkpoints"
    for name in ("best_earth_params.npy", "best_params.npy"):
        path = base / name
        if path.exists():
            return path
    return None


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class ViewerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Walker2D — Curriculum Learning Viewer")
        self.root.geometry("960x640")
        self.root.configure(bg=COLOR_BG)

        # Load all data once at startup
        self.summaries = load_all_summaries()

        # Tk variables
        self.var_strategy = tk.StringVar(value=STRATEGY_ORDER[0])
        self.var_seed = tk.IntVar(value=0)
        self.var_gravity = tk.DoubleVar(value=-9.81)
        self.var_episodes = tk.IntVar(value=3)

        # Subprocess handles for any running renders (single or multi-window)
        self.render_procs: list[subprocess.Popen] = []

        self._configure_styles()
        self._build_layout()
        self._refresh_seed_list()
        self._refresh_stats()
        self._update_render_buttons()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ----- styling -------------------------------------------------------

    def _configure_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=COLOR_BG)
        style.configure("Panel.TFrame", background=COLOR_PANEL)
        style.configure("TLabel", background=COLOR_BG, foreground=COLOR_TEXT, font=("Helvetica", 11))
        style.configure("Panel.TLabel", background=COLOR_PANEL, foreground=COLOR_TEXT, font=("Helvetica", 11))
        style.configure("Title.TLabel", background=COLOR_BG, foreground=COLOR_ACCENT,
                        font=("Helvetica", 18, "bold"))
        style.configure("Heading.TLabel", background=COLOR_PANEL, foreground=COLOR_ACCENT,
                        font=("Helvetica", 12, "bold"))
        style.configure("Muted.TLabel", background=COLOR_PANEL, foreground=COLOR_MUTED,
                        font=("Helvetica", 10))
        style.configure("Stat.TLabel", background=COLOR_PANEL, foreground=COLOR_TEXT,
                        font=("Helvetica", 11))
        style.configure("StatValue.TLabel", background=COLOR_PANEL, foreground=COLOR_ACCENT,
                        font=("Helvetica", 11, "bold"))
        style.configure("Accent.TButton", font=("Helvetica", 11, "bold"), padding=8)
        style.configure("TButton", font=("Helvetica", 10), padding=6)
        style.configure("TCombobox", padding=4)

    # ----- layout --------------------------------------------------------

    def _build_layout(self):
        # ===== Top header =====
        header = ttk.Frame(self.root, style="TFrame", padding=(20, 16, 20, 8))
        header.pack(fill="x")
        ttk.Label(header, text="Walker2D — Curriculum Learning Viewer",
                  style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Select a strategy and seed to inspect, render, or compare results.",
            style="TLabel",
        ).pack(anchor="w", pady=(2, 0))

        # ===== Body: two columns =====
        body = ttk.Frame(self.root, style="TFrame", padding=(20, 8, 20, 12))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1, minsize=380)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        # ----- LEFT: selectors -----
        left = ttk.Frame(body, style="Panel.TFrame", padding=14)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        ttk.Label(left, text="1.  Curriculum strategy", style="Heading.TLabel").pack(anchor="w")
        ttk.Label(
            left,
            text="Sorted by mean final score across 10 seeds.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(0, 6))

        for strategy in STRATEGY_ORDER:
            seeds = self.summaries.get(strategy, {})
            mean_score = (
                sum(s.get("final_earth", {}).get("score", 0) for s in seeds.values()) / len(seeds)
                if seeds else 0
            )
            label = (
                f"{STRATEGY_DISPLAY[strategy]}  —  "
                f"{mean_score:>6.0f}  ({STRATEGY_ROLE[strategy]})"
            )
            ttk.Radiobutton(
                left, text=label,
                variable=self.var_strategy, value=strategy,
                command=self._on_strategy_change,
            ).pack(anchor="w", pady=1)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)

        ttk.Label(left, text="2.  Seed", style="Heading.TLabel").pack(anchor="w")
        ttk.Label(
            left,
            text="⭐ marks the best-performing seed for this strategy.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(0, 6))

        seed_frame = ttk.Frame(left, style="Panel.TFrame")
        seed_frame.pack(fill="both", expand=True)
        self.seed_listbox = tk.Listbox(
            seed_frame,
            height=10,
            font=("Menlo", 11),
            selectmode="single",
            activestyle="dotbox",
            background="#fafafa",
            foreground=COLOR_TEXT,
            selectbackground=COLOR_ACCENT,
            selectforeground="white",
            highlightthickness=0,
            relief="flat",
        )
        self.seed_listbox.pack(side="left", fill="both", expand=True)
        self.seed_listbox.bind("<<ListboxSelect>>", lambda _e: self._on_seed_change())

        sb = ttk.Scrollbar(seed_frame, orient="vertical", command=self.seed_listbox.yview)
        sb.pack(side="right", fill="y")
        self.seed_listbox.configure(yscrollcommand=sb.set)

        # ----- RIGHT: stats + actions -----
        right = ttk.Frame(body, style="Panel.TFrame", padding=14)
        right.grid(row=0, column=1, sticky="nsew")

        ttk.Label(right, text="Selected run", style="Heading.TLabel").pack(anchor="w")
        self.lbl_run_title = ttk.Label(right, text="", style="Stat.TLabel")
        self.lbl_run_title.pack(anchor="w", pady=(0, 8))

        # Stats grid
        self.stats_frame = ttk.Frame(right, style="Panel.TFrame")
        self.stats_frame.pack(fill="x", pady=(0, 12))
        self.stat_widgets: dict[str, ttk.Label] = {}
        for i, (key, label) in enumerate([
            ("best_earth_probe", "Best Earth probe score"),
            ("final_score",      "Final Earth score (20 ep)"),
            ("net_progress",     "Net forward progress"),
            ("fall_rate",        "Fall rate"),
            ("tte",              "Time to effect (gen)"),
            ("robustness",       "Mean robustness (5 gravities)"),
        ]):
            row = i // 2
            col = (i % 2) * 2
            ttk.Label(self.stats_frame, text=label, style="Stat.TLabel").grid(
                row=row, column=col, sticky="w", padx=(0, 8), pady=2
            )
            value_label = ttk.Label(self.stats_frame, text="—", style="StatValue.TLabel")
            value_label.grid(row=row, column=col + 1, sticky="w", padx=(0, 24), pady=2)
            self.stat_widgets[key] = value_label

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=4)

        # Render settings
        ttk.Label(right, text="Render settings", style="Heading.TLabel").pack(anchor="w", pady=(6, 4))
        settings = ttk.Frame(right, style="Panel.TFrame")
        settings.pack(fill="x", pady=(0, 10))

        ttk.Label(settings, text="Gravity (m/s²):", style="Stat.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 6))
        gravity_combo = ttk.Combobox(
            settings, textvariable=self.var_gravity,
            values=[str(g) for g in GRAVITY_OPTIONS],
            state="readonly", width=8,
        )
        gravity_combo.grid(row=0, column=1, sticky="w", padx=(0, 16))
        gravity_combo.set("-9.81")

        ttk.Label(settings, text="Episodes:", style="Stat.TLabel").grid(
            row=0, column=2, sticky="w", padx=(0, 6))
        ttk.Spinbox(settings, from_=1, to=10, width=4, textvariable=self.var_episodes).grid(
            row=0, column=3, sticky="w")

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=4)

        # Action buttons
        ttk.Label(right, text="Actions", style="Heading.TLabel").pack(anchor="w", pady=(6, 6))
        btns = ttk.Frame(right, style="Panel.TFrame")
        btns.pack(fill="x")
        btns.columnconfigure((0, 1), weight=1)

        self.btn_render = ttk.Button(
            btns, text="▶  Render agent in MuJoCo",
            style="Accent.TButton", command=self._action_render,
        )
        self.btn_render.grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=(0, 6))

        self.btn_stop = ttk.Button(
            btns, text="■  Stop",
            command=self._action_stop, state="disabled",
        )
        self.btn_stop.grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=(0, 6))

        self.btn_run_all = ttk.Button(
            btns, text="🪟  Run all best simulations (tiled)",
            style="Accent.TButton", command=self._action_run_all_best,
        )
        self.btn_run_all.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 6))

        ttk.Button(btns, text="📈  Learning curve",
                   command=self._action_learning_curve).grid(
            row=2, column=0, sticky="ew", padx=(0, 4), pady=2)
        ttk.Button(btns, text="🌍  Gravity robustness",
                   command=self._action_gravity_sweep).grid(
            row=2, column=1, sticky="ew", padx=(4, 0), pady=2)
        ttk.Button(btns, text="📊  Compare all strategies",
                   command=self._action_compare).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=2)

        # ===== Status bar =====
        self.status_var = tk.StringVar(value="Ready.")
        status = ttk.Label(self.root, textvariable=self.status_var,
                           background=COLOR_ACCENT, foreground="white",
                           padding=(14, 6), font=("Helvetica", 10))
        status.pack(fill="x", side="bottom")

    # ----- callbacks -----------------------------------------------------

    def _on_strategy_change(self):
        self._refresh_seed_list()
        self._refresh_stats()

    def _on_seed_change(self):
        sel = self.seed_listbox.curselection()
        if not sel:
            return
        # Listbox values are formatted; the seed index equals the list index
        self.var_seed.set(int(sel[0]))
        self._refresh_stats()

    def _refresh_seed_list(self):
        strategy = self.var_strategy.get()
        seeds = self.summaries.get(strategy, {})
        best = best_seed_for(seeds)

        self.seed_listbox.delete(0, "end")
        for seed in range(N_SEEDS):
            summary = seeds.get(seed)
            if summary is None:
                line = f"  seed {seed:02d}  —  (missing)"
            else:
                score = summary.get("final_earth", {}).get("score", 0)
                marker = "⭐" if seed == best else "  "
                line = f"{marker} seed {seed:02d}  —  score {score:>7.1f}"
            self.seed_listbox.insert("end", line)
            if seed == best:
                self.seed_listbox.itemconfig(seed, foreground=COLOR_BEST)

        # Default-select the best seed
        if best is not None:
            self.seed_listbox.selection_clear(0, "end")
            self.seed_listbox.selection_set(best)
            self.seed_listbox.see(best)
            self.var_seed.set(best)

    def _refresh_stats(self):
        strategy = self.var_strategy.get()
        seed = self.var_seed.get()
        summary = self.summaries.get(strategy, {}).get(seed)

        title = f"{STRATEGY_DISPLAY[strategy]}, seed {seed:02d}"
        if summary is None:
            self.lbl_run_title.configure(text=f"{title}  —  (no data)")
            for w in self.stat_widgets.values():
                w.configure(text="—")
            return

        self.lbl_run_title.configure(text=title)
        final_earth = summary.get("final_earth", {})
        robustness = summary.get("robustness", {})

        self.stat_widgets["best_earth_probe"].configure(
            text=f"{summary.get('best_earth_probe_score', 0):.1f}"
        )
        self.stat_widgets["final_score"].configure(
            text=f"{final_earth.get('score', 0):.1f}"
        )
        self.stat_widgets["net_progress"].configure(
            text=f"{final_earth.get('net_progress_m', 0):.2f} m"
        )
        self.stat_widgets["fall_rate"].configure(
            text=f"{final_earth.get('fell', 0):.0%}"
        )
        tte_gen = summary.get("tte_generation")
        tte_text = f"{tte_gen}" if tte_gen and not summary.get("tte_censored") else "censored"
        self.stat_widgets["tte"].configure(text=tte_text)
        self.stat_widgets["robustness"].configure(
            text=f"{robustness.get('mean_net_progress_m', 0):.2f} m"
        )

    # ----- actions -------------------------------------------------------

    def _action_render(self):
        if self._any_process_running():
            messagebox.showinfo(
                "Already running",
                "A simulation is already running. Stop it first or wait for it to finish."
            )
            return

        strategy = self.var_strategy.get()
        seed = self.var_seed.get()
        ckpt = get_checkpoint_path(strategy, seed)
        if ckpt is None:
            base = RESULTS_DIR / strategy / f"seed_{seed:02d}" / "checkpoints"
            messagebox.showerror(
                "Checkpoint not found",
                f"No checkpoint found in:\n{base}\n\nRun training for this seed first."
            )
            return

        gravity = float(self.var_gravity.get())
        episodes = int(self.var_episodes.get())
        cmd = [
            sys.executable, str(PROJECT_ROOT / "render_agent.py"),
            str(ckpt),
            "--episodes", str(episodes),
            "--gravity", str(gravity),
            "--fullscreen",
            "--window-title", f"{STRATEGY_DISPLAY[strategy]} (seed {seed:02d})",
        ]
        self.status_var.set(
            f"Launching MuJoCo viewer for {STRATEGY_DISPLAY[strategy]} seed {seed:02d} "
            f"at g={gravity:.2f} ({episodes} ep)..."
        )
        try:
            proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
        except OSError as e:
            messagebox.showerror("Failed to launch renderer", str(e))
            self.status_var.set("Ready.")
            return

        self.render_procs.append(proc)
        self._update_render_buttons()
        self._poll_render_procs()

    def _action_run_all_best(self):
        if self._any_process_running():
            messagebox.showinfo(
                "Already running",
                "A simulation is already running. Stop it first before launching the tiled view."
            )
            return

        # Compute screen geometry. macOS reserves ~25 px at the top for the menu bar.
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        menu_bar = 28 if sys.platform == "darwin" else 0
        usable_h = max(200, screen_h - menu_bar)

        # 4+3 grid layout: top row 4 windows, bottom row 3 centered windows
        cell_w = screen_w // 4
        cell_h = usable_h // 2

        positions: list[tuple[int, int]] = []
        # Top row: indices 0-3 at x = 0, W/4, 2W/4, 3W/4
        for i in range(4):
            positions.append((i * cell_w, menu_bar))
        # Bottom row: indices 4-6 centered (W/8 padding on each side)
        bottom_offset = cell_w // 2
        for i in range(3):
            positions.append((bottom_offset + i * cell_w, menu_bar + cell_h))

        gravity = -9.81
        episodes = int(self.var_episodes.get())
        launched = 0
        skipped: list[str] = []

        for strategy, (x, y) in zip(STRATEGY_ORDER, positions):
            seeds = self.summaries.get(strategy, {})
            best = best_seed_for(seeds)
            if best is None:
                skipped.append(STRATEGY_DISPLAY[strategy])
                continue
            ckpt = get_checkpoint_path(strategy, best)
            if ckpt is None:
                skipped.append(STRATEGY_DISPLAY[strategy])
                continue

            cmd = [
                sys.executable, str(PROJECT_ROOT / "render_agent.py"),
                str(ckpt),
                "--episodes", str(episodes),
                "--gravity", str(gravity),
                "--no-fullscreen",
                "--no-hud",
                "--window-x", str(x),
                "--window-y", str(y),
                "--window-width", str(cell_w),
                "--window-height", str(cell_h),
                "--window-title", f"{STRATEGY_DISPLAY[strategy]} (best: seed {best:02d})",
            ]
            try:
                proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
                self.render_procs.append(proc)
                launched += 1
            except OSError:
                skipped.append(STRATEGY_DISPLAY[strategy])

        if launched == 0:
            messagebox.showerror(
                "Nothing to run",
                "No best-seed checkpoints found for any strategy."
            )
            return

        msg = f"Launched {launched} simulations in tiled layout."
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}."
        self.status_var.set(msg)
        self._update_render_buttons()
        self._poll_render_procs()

    def _action_stop(self):
        running = [p for p in self.render_procs if p.poll() is None]
        if not running:
            return
        self.status_var.set(f"Stopping {len(running)} simulation(s)...")
        for proc in running:
            try:
                proc.terminate()  # SIGTERM → clean exit in render_agent.py
            except OSError:
                pass
        # If any subprocess does not exit within 2 seconds, kill it
        self.root.after(2000, self._force_kill_alive)

    def _force_kill_alive(self):
        for proc in self.render_procs:
            if proc.poll() is None:
                try:
                    proc.kill()
                except OSError:
                    pass

    def _poll_render_procs(self):
        """Periodically check if any render subprocesses have finished."""
        # Drop finished processes from the tracked list
        self.render_procs = [p for p in self.render_procs if p.poll() is None]
        if self.render_procs:
            self.root.after(300, self._poll_render_procs)
            self._update_render_buttons()
            return
        # All finished
        self.status_var.set("Simulation finished. Ready.")
        self._update_render_buttons()

    def _any_process_running(self) -> bool:
        return any(p.poll() is None for p in self.render_procs)

    def _update_render_buttons(self):
        running = self._any_process_running()
        state_when_running = "disabled" if running else "normal"
        self.btn_render.configure(state=state_when_running)
        self.btn_run_all.configure(state=state_when_running)
        self.btn_stop.configure(state=("normal" if running else "disabled"))

    def _on_close(self):
        # Kill any orphan render subprocesses when the GUI closes
        for proc in self.render_procs:
            if proc.poll() is None:
                try:
                    proc.terminate()
                except OSError:
                    pass
        self.root.destroy()

    def _action_learning_curve(self):
        strategy = self.var_strategy.get()
        seed = self.var_seed.get()
        rows = load_csv(strategy, seed, "earth_probe.csv")
        if not rows:
            messagebox.showwarning("No data", "earth_probe.csv not found for this run.")
            return

        gens = [int(r["gen"]) for r in rows]
        scores = [float(r["score"]) for r in rows]
        progress = [float(r["net_progress_m"]) for r in rows]

        win = tk.Toplevel(self.root)
        win.title(f"Learning curve — {STRATEGY_DISPLAY[strategy]} seed {seed:02d}")
        win.geometry("780x540")

        fig = Figure(figsize=(7.5, 5), dpi=100)
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(gens, scores, color=COLOR_ACCENT, linewidth=2)
        ax1.set_ylabel("Earth probe score")
        ax1.grid(alpha=0.3)
        ax1.set_title(f"{STRATEGY_DISPLAY[strategy]} — seed {seed:02d}")

        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
        ax2.plot(gens, progress, color=COLOR_BEST, linewidth=2)
        ax2.axhline(4.0, color="green", linestyle="--", alpha=0.5,
                    label="TTE threshold (4 m)")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Net progress (m)")
        ax2.legend(loc="lower right")
        ax2.grid(alpha=0.3)

        fig.tight_layout()
        self._embed_plot(win, fig)

    def _action_gravity_sweep(self):
        strategy = self.var_strategy.get()
        seed = self.var_seed.get()
        rows = load_csv(strategy, seed, "gravity_sweep.csv")
        if not rows:
            messagebox.showwarning("No data", "gravity_sweep.csv not found for this run.")
            return

        gravities = [float(r["gravity"]) for r in rows]
        progress = [float(r["net_progress_m"]) for r in rows]
        fall_rates = [float(r["fell"]) for r in rows]

        win = tk.Toplevel(self.root)
        win.title(f"Gravity robustness — {STRATEGY_DISPLAY[strategy]} seed {seed:02d}")
        win.geometry("780x540")

        fig = Figure(figsize=(7.5, 5), dpi=100)
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(gravities, progress, "o-", color=COLOR_ACCENT, linewidth=2, markersize=8)
        ax1.axvline(-9.81, color="green", linestyle="--", alpha=0.5,
                    label="Earth gravity")
        ax1.set_ylabel("Net progress (m)")
        ax1.set_title(f"{STRATEGY_DISPLAY[strategy]} — seed {seed:02d}")
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
        ax2.bar([str(g) for g in gravities], fall_rates, color=COLOR_BEST, alpha=0.7)
        ax2.set_xlabel("Gravity (m/s²)")
        ax2.set_ylabel("Fall rate")
        ax2.set_ylim(0, 1)
        ax2.grid(alpha=0.3, axis="y")

        fig.tight_layout()
        self._embed_plot(win, fig)

    def _action_compare(self):
        # Box plot of final Earth scores across strategies
        data = []
        labels = []
        for strategy in STRATEGY_ORDER:
            seeds = self.summaries.get(strategy, {})
            scores = [s.get("final_earth", {}).get("score", 0) for s in seeds.values()]
            if scores:
                data.append(scores)
                labels.append(STRATEGY_DISPLAY[strategy])

        if not data:
            messagebox.showwarning("No data", "No summaries found to compare.")
            return

        win = tk.Toplevel(self.root)
        win.title("Strategy comparison — final Earth scores")
        win.geometry("900x560")

        fig = Figure(figsize=(8.6, 5.2), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)

        for i, patch in enumerate(bp["boxes"]):
            role = STRATEGY_ROLE[STRATEGY_ORDER[i]]
            color = {"curriculum": "#a7c7e7", "baseline": "#cccccc", "control": "#cccccc"}[role]
            patch.set_facecolor(color)
            patch.set_edgecolor(COLOR_ACCENT)

        # Per-seed scatter
        for i, scores in enumerate(data, start=1):
            ax.scatter([i] * len(scores), scores, color="black", s=18, alpha=0.6, zorder=3)

        ax.set_ylabel("Final Earth score")
        ax.set_title("Final Earth score across 10 seeds per strategy")
        ax.grid(alpha=0.3, axis="y")
        for label in ax.get_xticklabels():
            label.set_rotation(20)
            label.set_ha("right")
        fig.tight_layout()
        self._embed_plot(win, fig)

    # ----- helpers -------------------------------------------------------

    @staticmethod
    def _embed_plot(window: tk.Toplevel, fig: Figure):
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(canvas, window)
        toolbar.update()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not RESULTS_DIR.exists():
        print(f"Error: results directory not found at {RESULTS_DIR}", file=sys.stderr)
        print("Run the experiments first (scripts/run_thesis_batch.py).", file=sys.stderr)
        sys.exit(1)

    root = tk.Tk()
    try:
        # macOS dark-mode-aware default font
        root.option_add("*Font", "Helvetica 11")
    except tk.TclError:
        pass
    ViewerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
