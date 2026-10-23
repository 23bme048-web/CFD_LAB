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


# Parameter set (sim.py argument order: nx ny L H T_left max_iterations tolerance omega screen_every)
L, H, T_LEFT = 1.5, 1.0, 150.0
TOL = 1e-8
MAX_IT = 40000
OMEGA = 1.6
GRIDS = ((31, 21), (61, 41), (121, 81))
BASE_GRID = (61, 41)
OMEGAS = (1.0, 1.2, 1.4, 1.6, 1.7, 1.8, 1.9)


def case(nx, ny, omega=OMEGA):
    return run(f"grid_{nx}x{ny}_w{omega:g}", nx, ny, L, H, T_LEFT, MAX_IT, TOL, omega, 10**6)


def main() -> None:
    grids = {g: case(*g) for g in GRIDS}
    base = grids[BASE_GRID]

    # Figure 1: temperature field and absolute error
    x, y, f = read_vtk(base / "plate_conduction_2d.vtk")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    cf = ax1.contourf(x, y, f["temperature_line_tdma"], levels=30, cmap="inferno")
    cs = ax1.contour(x, y, f["temperature_analytical"], levels=[5, 15, 30, 60, 100], colors="w", linewidths=0.8, linestyles="--")
    ax1.clabel(cs, fmt="%g", fontsize=7)
    fig.colorbar(cf, ax=ax1, label="T")
    ax1.set_title(f"(a) Line-TDMA/SOR solution {BASE_GRID[0]}x{BASE_GRID[1]} (dashed: analytical)")
    cf = ax2.contourf(x, y, f["line_tdma_absolute_error"], levels=30, cmap="magma")
    fig.colorbar(cf, ax=ax2, label="|T - T_exact|")
    ax2.set_title("(b) Absolute error, line-TDMA/SOR")
    for ax in (ax1, ax2):
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    save("Figure_1.png")

    # Figure 2: centreline profiles
    p = read_csv(base / "plate_conduction_profiles.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, label in ((ax1, "horizontal", "x / L at y = H/2"), (ax2, "vertical", "y / H at x = L/2")):
        ax.plot(p["coordinate"], p[f"{key}_jacobi"], label="Jacobi")
        ax.plot(p["coordinate"], p[f"{key}_line_tdma"], label="Line TDMA/SOR")
        ax.plot(p["coordinate"][::4], p[f"{key}_analytical"][::4], "ko", mfc="none", ms=4, label="Analytical")
        ax.set_xlabel(label)
        ax.set_ylabel("T")
        ax.legend()
    ax1.set_title("(a) Horizontal centreline")
    ax2.set_title("(b) Vertical centreline")
    save("Figure_2.png")

    # Figure 3: residual histories on all grids
    rows = []
    plt.figure(figsize=(7, 4.3))
    for k, (g, out) in enumerate(grids.items()):
        h = read_csv(out / "plate_conduction_history.csv")
        s = read_summary(out / "plate_conduction_summary.txt")
        rows.append((g[0], g[1], s["jacobi_iterations"], s["jacobi_relative_L2_error"], s["line_tdma_iterations"], s["line_tdma_relative_L2_error"]))
        for method, ls in (("jacobi", "-"), ("line_tdma", "--")):
            m = h["method"] == method
            plt.semilogy(h["iteration"][m], h["residual_linf"][m], ls, color=f"C{k}", label=f"{method}, {g[0]}x{g[1]}")
    plt.axhline(TOL, color="k", lw=0.7)
    plt.xscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("L-infinity residual")
    plt.title(f"Convergence (line TDMA uses omega = {OMEGA})")
    plt.legend(fontsize=7, ncol=2)
    save("Figure_3.png")

    # Figure 4: relaxation factor and grid convergence
    w_rows = []
    for w in OMEGAS:
        s = read_summary(case(*BASE_GRID, omega=w) / "plate_conduction_summary.txt")
        w_rows.append((w, s["line_tdma_iterations"]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    wr = np.array(w_rows)
    ax1.plot(wr[:, 0], wr[:, 1], "o-")
    ax1.set_xlabel("Relaxation factor omega")
    ax1.set_ylabel("Iterations to residual < 1e-8")
    ax1.set_title(f"(a) Line-TDMA/SOR, grid {BASE_GRID[0]}x{BASE_GRID[1]}")
    r = np.array(rows)
    dx = L / (r[:, 0] - 1)
    ax2.loglog(dx, r[:, 5], "o-", label="Line TDMA/SOR")
    ax2.loglog(dx, r[:, 3], "s--", label="Jacobi")
    ax2.loglog(dx, r[0, 5] * (dx / dx[0]) ** 2, "k:", label="slope 2")
    ax2.set_xlabel("dx")
    ax2.set_ylabel("Relative L2 error vs analytical")
    ax2.set_title("(b) Grid convergence")
    ax2.legend()
    save("Figure_4.png")

    print("nx, ny, jac_it, jac_err, tdma_it, tdma_err")
    for row in rows:
        print("  ", tuple(float(f"{v:.4g}") for v in row))
    print("omega, iterations:", w_rows)


if __name__ == "__main__":
    main()
