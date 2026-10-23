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


# Parameter set (sim.py argument order: nr ntheta rpm Re T_in T_wall iterations swirl wall_HTC loss_fraction)
NR, NT = 90, 180
RPM_BASE = 2000.0
RE = 600.0
T_IN, T_WALL = 300.0, 340.0
ITERATIONS = 2000
SWIRL, HTC, LOSS = 0.80, 150.0, 0.05
RPMS = (1000.0, 1500.0, 2000.0, 2500.0, 3000.0)


def case(rpm):
    return run(f"rpm_{rpm:g}", NR, NT, rpm, RE, T_IN, T_WALL, ITERATIONS, SWIRL, HTC, LOSS)


def closed(a):
    """Append the first theta column so polar contours close at theta = 2 pi."""
    return np.concatenate((a, a[:, :1]), axis=1)


def main() -> None:
    runs = {rpm: case(rpm) for rpm in RPMS}
    x, y, f = read_vtk(runs[RPM_BASE] / "pump_annulus_srf.vtk")
    # VTK arrays are stored (r, theta); close the annulus for plotting
    X, Y = closed(x), closed(y)

    # Figure 1: temperature and pressure in the annulus
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    cf = ax1.contourf(X, Y, closed(f["temperature"]), levels=40, cmap="inferno")
    fig.colorbar(cf, ax=ax1, label="T [K]")
    ax1.set_title(f"(a) Temperature, {RPM_BASE:g} rpm")
    cf = ax2.contourf(X, Y, closed(f["pressure"]) / 1e3, levels=40, cmap="viridis")
    fig.colorbar(cf, ax=ax2, label="p - p_inlet [kPa]")
    ax2.set_title(f"(b) Pressure rise, {RPM_BASE:g} rpm")
    for ax in (ax1, ax2):
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    save("Figure_1.png")

    # Figure 2: radial profiles at theta = 0
    r = np.hypot(x[:, 0], y[:, 0])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(r * 1e3, f["absolute_speed"][:, 0], label="|V| absolute frame")
    ax1.plot(r * 1e3, f["relative_speed"][:, 0], label="|W| rotating frame")
    ax1.set_xlabel("r [mm]")
    ax1.set_ylabel("speed [m/s]")
    ax1.set_title(f"(a) Velocity magnitudes, {RPM_BASE:g} rpm")
    ax1.set_yscale("log")
    ax1.legend()
    for rpm, out in runs.items():
        xx, yy, ff = read_vtk(out / "pump_annulus_srf.vtk")
        ax2.plot(r * 1e3, ff["temperature"].mean(axis=1), label=f"{rpm:g} rpm")
    ax2.set_xlabel("r [mm]")
    ax2.set_ylabel("circumferential-mean T [K]")
    ax2.set_title("(b) Temperature rise through the passage")
    ax2.legend(fontsize=8)
    save("Figure_2.png")

    # Figure 3: head scaling with rpm^2
    rows = []
    for rpm, out in runs.items():
        s = read_summary(out / "pump_summary.txt")
        rows.append((rpm, s["pump_head_m"], s["pressure_rise_Pa"], s["Tout_K"], s["fluid_heat_gain_W"], s["final_residual_linf"]))
    a = np.array(rows)
    slope, intercept = np.polyfit(a[:, 0] ** 2, a[:, 1], 1)
    fit = slope * a[:, 0] ** 2 + intercept
    r2 = 1.0 - np.sum((a[:, 1] - fit) ** 2) / np.sum((a[:, 1] - a[:, 1].mean()) ** 2)
    plt.figure(figsize=(6.5, 4.2))
    plt.plot(a[:, 0] ** 2 / 1e6, a[:, 1], "o", ms=7, label="Simulation")
    plt.plot(a[:, 0] ** 2 / 1e6, fit, "--", label=f"Linear fit, R^2 = {r2:.6f}")
    plt.xlabel("RPM^2  [x 10^6]")
    plt.ylabel("Pump head H [m]")
    plt.title("Pump head vs rotational speed squared")
    plt.legend()
    save("Figure_3.png")

    # Figure 4: thermal convergence
    plt.figure(figsize=(6.5, 4))
    for rpm, out in runs.items():
        h = read_csv(out / "pump_history.csv")
        plt.semilogy(h["iteration"], h["residual_linf"], label=f"{rpm:g} rpm")
    plt.xlabel("Iteration")
    plt.ylabel("L-infinity residual [K]")
    plt.title("Thermal convergence")
    plt.legend(fontsize=8)
    save("Figure_4.png")

    print("rpm, head_m, dp_Pa, Tout_K, Q_W, residual")
    for row in rows:
        print("  ", tuple(float(f"{v:.5g}") for v in row))
    print(f"fit H = {slope:.4e} rpm^2 + {intercept:.4e}, R2 = {r2:.8f}")
    print(read_summary(runs[RPM_BASE] / "pump_summary.txt"))


if __name__ == "__main__":
    main()
