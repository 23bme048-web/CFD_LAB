from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.35, "figure.constrained_layout.use": True})


def run(case: str, *args: object) -> Path:
    """Run sim.py with the given command-line arguments inside results/<case>/."""
    out = RESULTS / case
    out.mkdir(parents=True, exist_ok=True)
    print(f"running {case}: {' '.join(map(str, args))}")
    subprocess.run([sys.executable, str(HERE / "sim.py"), *map(str, args)], cwd=out, check=True, stdout=subprocess.DEVNULL)
    return out


def read_csv(path: Path) -> np.ndarray:
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8"))


def read_summary(path: Path) -> dict[str, float | str]:
    values: dict[str, float | str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                values[parts[0]] = float(parts[1])
            except ValueError:
                values[parts[0]] = parts[1]
    return values


def read_vtk(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Read the ASCII structured-grid VTK files written by common.py (scalars only)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    nx, ny = (int(v) for v in next(l for l in lines if l.startswith("DIMENSIONS")).split()[1:3])
    start = next(i for i, l in enumerate(lines) if l.startswith("POINTS")) + 1
    points = np.loadtxt(lines[start:start + nx * ny])
    scalars: dict[str, np.ndarray] = {}
    for i, line in enumerate(lines):
        if line.startswith("SCALARS"):
            scalars[line.split()[1]] = np.loadtxt(lines[i + 2:i + 2 + nx * ny]).reshape(ny, nx)
    return points[:, 0].reshape(ny, nx), points[:, 1].reshape(ny, nx), scalars


def save(name: str) -> None:
    plt.savefig(HERE / name)
    plt.close()
    print(f"saved {name}")


# Parameter set (sim.py argument order:
# nx ny_cold ny_wall ny_hot Re_hot Re_cold ks/kf counter_flow max_iterations Pr T_hot_in T_cold_in U_hot U_cold)
NX, NY_COLD, NY_WALL, NY_HOT = 200, 24, 8, 24
PR = 5.0
KS_BASE = 20.0
MAX_IT = 8000


def case(name, re=100.0, ks=KS_BASE, counter=1):
    return run(name, NX, NY_COLD, NY_WALL, NY_HOT, re, re, ks, counter, MAX_IT, PR, 1.0, 0.0, 1.0, 1.0)


def main() -> None:
    base = case("base")

    # Figure 1: base temperature field
    x, y, f = read_vtk(base / "double_pipe_cht.vtk")
    fig, ax = plt.subplots(figsize=(10, 3.6))
    cf = ax.contourf(x, y, f["temperature"], levels=40, cmap="inferno")
    fig.colorbar(cf, ax=ax, label="T*")
    for level in (0.5, 1.5):
        ax.contour(x, y, f["zone"], levels=[level], colors="w", linewidths=0.8, linestyles="--")
    ax.text(0.15, 0.25, "cold stream  <--", color="w")
    ax.text(0.15, 1.75, "hot stream  -->", color="k")
    ax.text(0.15, 1.0, "solid wall", color="w", va="center")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Counter-flow conjugate heat transfer: Re = 100, Pr = {PR:g}, ks/kf = {KS_BASE:g}")
    save("Figure_1.png")

    # Figure 2: Reynolds-number sweep
    res = [50.0, 100.0, 200.0]
    rows = []
    for re in res:
        s = read_summary(case(f"Re_{re:g}", re=re) / "summary.txt")
        rows.append((re, s["hot_outlet_temperature"], s["cold_outlet_temperature"], s["hot_stream_heat_loss"], s["effectiveness"], s["relative_energy_imbalance"], s["iterations_completed"]))
    r = np.array(rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(r[:, 0], r[:, 1], "o-", color="C3", label="Hot outlet")
    ax1.plot(r[:, 0], r[:, 2], "s-", color="C0", label="Cold outlet")
    ax1.set_xscale("log")
    ax1.set_xlabel("Re (both streams)")
    ax1.set_ylabel("Outlet temperature T*")
    ax1.set_title("(a) Outlet temperatures")
    ax1.legend()
    ax2.plot(r[:, 0], r[:, 4], "o-", label="Effectiveness")
    ax2.plot(r[:, 0], r[:, 3], "s--", label="Heat-transfer rate Q*")
    ax2.set_xscale("log")
    ax2.set_xlabel("Re (both streams)")
    ax2.set_title("(b) Effectiveness and heat duty")
    ax2.legend()
    save("Figure_2.png")

    # Figure 3: conductivity-ratio sweep, wall-normal profile at mid-length
    ks_rows = []
    plt.figure(figsize=(6, 4.5))
    for ks in (1.0, 20.0, 100.0):
        out = case(f"ks_{ks:g}", ks=ks)
        x, y, f = read_vtk(out / "double_pipe_cht.vtk")
        i = NX // 2
        plt.plot(f["temperature"][:, i], y[:, i], label=f"ks/kf = {ks:g}")
        s = read_summary(out / "summary.txt")
        ks_rows.append((ks, s["hot_outlet_temperature"], s["cold_outlet_temperature"], s["effectiveness"], s["relative_energy_imbalance"]))
    wall = y[NY_COLD:NY_COLD + NY_WALL, 0]
    plt.axhspan(wall[0], wall[-1], color="0.85", label="solid wall")
    plt.xlabel("T* at x = L/2")
    plt.ylabel("y")
    plt.title("Effect of wall conductivity ratio")
    plt.legend()
    save("Figure_3.png")

    # Figure 4: counter vs parallel flow and convergence
    par = case("parallel", counter=0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    mid_wall = NY_COLD + NY_WALL // 2
    for out, label in ((base, "counter-flow"), (par, "parallel-flow")):
        x, y, f = read_vtk(out / "double_pipe_cht.vtk")
        ax1.plot(x[0], f["temperature"][mid_wall], label=f"wall mid-plane, {label}")
        ax1.plot(x[0], f["temperature"][NY_COLD + NY_WALL], "--", label=f"hot-side interface, {label}")
        h = read_csv(out / "history.csv")
        ax2.semilogy(h["iteration"], h["residual_linf"], label=label)
    ax1.set_xlabel("x")
    ax1.set_ylabel("T*")
    ax1.set_title("(a) Wall temperature along the exchanger")
    ax1.legend(fontsize=7)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("L-infinity residual")
    ax2.set_title("(b) Convergence history")
    ax2.legend()
    save("Figure_4.png")

    sp = read_summary(par / "summary.txt")
    print("Re sweep (Re, Th_out, Tc_out, Q, eff, imbalance, iterations)")
    for row in rows:
        print("  ", tuple(float(f"{v:.5g}") for v in row))
    print("ks sweep (ks, Th_out, Tc_out, eff, imbalance)")
    for row in ks_rows:
        print("  ", tuple(float(f"{v:.5g}") for v in row))
    print("parallel:", sp["hot_outlet_temperature"], sp["cold_outlet_temperature"], sp["effectiveness"])


if __name__ == "__main__":
    main()
