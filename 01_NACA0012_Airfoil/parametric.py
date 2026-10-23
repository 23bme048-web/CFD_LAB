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


# Parameter set (sim.py argument order: nx ny Re alpha_deg iterations poisson_iterations)
NX, NY = 150, 75
RE = 100.0
ITERATIONS, POISSON = 1500, 100
ALPHAS = (-5.0, 0.0, 5.0, 10.0)
BASE_ALPHA = 5.0


def outline(ax, x, y, mask):
    ax.contourf(x, y, mask, levels=[0.5, 1.5], colors="0.25")
    ax.set_aspect("equal")
    ax.set_xlim(0.5, 4.5)
    ax.set_ylim(-1.0, 1.0)


def main() -> None:
    runs = {a: run(f"alpha_{a:+.0f}", NX, NY, RE, a, ITERATIONS, POISSON) for a in ALPHAS}
    fields = {a: read_vtk(out / "airfoil_final.vtk") for a, out in runs.items()}

    # Figure 1: base case speed field with streamlines
    x, y, f = fields[BASE_ALPHA]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    cf = ax.contourf(x, y, np.where(f["solid_mask"] > 0.5, np.nan, f["speed"]), levels=40, cmap="viridis")
    fig.colorbar(cf, ax=ax, label="|u| / U")
    _, _, vel = x, y, None
    outline(ax, x, y, f["solid_mask"])
    ax.set_title(f"NACA 0012, Re = {RE:g}, alpha = {BASE_ALPHA:g} deg: velocity magnitude")
    ax.set_xlabel("x / c")
    ax.set_ylabel("y / c")
    save("Figure_1.png")

    # Figure 2: speed fields for the angle-of-attack sweep
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.8), sharex=True, sharey=True)
    for ax, a in zip(axes.ravel(), ALPHAS):
        x, y, f = fields[a]
        cf = ax.contourf(x, y, np.where(f["solid_mask"] > 0.5, np.nan, f["speed"]), levels=np.linspace(0, 1.35, 28), cmap="viridis", extend="max")
        outline(ax, x, y, f["solid_mask"])
        ax.set_title(f"alpha = {a:g} deg")
    fig.colorbar(cf, ax=axes, label="|u| / U", shrink=0.9)
    fig.suptitle(f"Velocity magnitude, Re = {RE:g}")
    save("Figure_2.png")

    # Figure 3: pressure and vorticity at the highest angle of attack
    x, y, f = fields[ALPHAS[-1]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8))
    solid = f["solid_mask"] > 0.5
    p = np.where(solid, np.nan, f["pressure"])
    cf = ax1.contourf(x, y, p, levels=40, cmap="coolwarm")
    fig.colorbar(cf, ax=ax1, label="pressure (projection variable)")
    outline(ax1, x, y, f["solid_mask"])
    ax1.set_title(f"(a) Pressure, alpha = {ALPHAS[-1]:g} deg")
    w = np.where(solid, np.nan, f["vorticity"])
    lim = np.nanpercentile(np.abs(w), 98)
    cf = ax2.contourf(x, y, np.clip(w, -lim, lim), levels=np.linspace(-lim, lim, 41), cmap="RdBu_r")
    fig.colorbar(cf, ax=ax2, label="vorticity")
    outline(ax2, x, y, f["solid_mask"])
    ax2.set_title(f"(b) Vorticity, alpha = {ALPHAS[-1]:g} deg")
    save("Figure_3.png")

    # Figure 4: force trends and continuity residual
    cl, cd, res = [], [], []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for a, out in runs.items():
        h = read_csv(out / "history.csv")
        tail = slice(int(0.8 * h.size), None)
        cl.append(np.mean(h["Cl_penalty_qualitative"][tail]))
        cd.append(np.mean(h["Cd_penalty_qualitative"][tail]))
        res.append(h["continuity_l2"][-1])
        ax2.semilogy(h["iteration"], h["continuity_l2"], label=f"alpha = {a:g}")
    ax1.plot(ALPHAS, cl, "o-", label="C_L (qualitative)")
    ax1.plot(ALPHAS, cd, "s-", label="C_D (qualitative)")
    ax1.set_xlabel("Angle of attack [deg]")
    ax1.set_ylabel("Coefficient (mean of last 20 % of iterations)")
    ax1.set_title("(a) Penalty-force coefficient trends")
    ax1.legend()
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Continuity L2 residual")
    ax2.set_title("(b) Continuity residual")
    ax2.legend()
    save("Figure_4.png")

    print("alpha, CL, CD, final continuity residual")
    for row in zip(ALPHAS, cl, cd, res):
        print("  ", tuple(round(float(v), 4) for v in row))
    print(read_summary(runs[BASE_ALPHA] / "airfoil_summary.txt"))


if __name__ == "__main__":
    main()
