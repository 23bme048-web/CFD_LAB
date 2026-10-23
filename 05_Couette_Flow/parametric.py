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


# Parameter set (sim.py argument order: ny H U nu tEnd Fo_explicit Fo_implicit screen_every)
NY, H, U, NU = 101, 1.0, 1.0, 0.02        # Re = U H / nu = 50
T_END = 1.5
FO_EXPLICIT, FO_IMPLICIT = 0.45, 4.0
T_STABILITY = 0.08                         # short physical time used for the Fourier-number sweep


def main() -> None:
    base = run("base", NY, H, U, NU, T_END, FO_EXPLICIT, FO_IMPLICIT, 100)

    # Figure 1: development of the velocity profile towards the linear steady state
    plt.figure(figsize=(6, 4.5))
    for k, t in enumerate((0.1, 0.5, T_END, 5.0, 25.0)):
        prof = read_csv(run(f"t_{t}", NY, H, U, NU, t, FO_EXPLICIT, FO_IMPLICIT, 10**6) / "couette_profiles.csv")
        color = f"C{k}"
        plt.plot(prof["explicit_velocity"], prof["y"], "-", color=color, label=f"t = {t:g}")
        plt.plot(prof["implicit_velocity"][::6], prof["y"][::6], "o", mfc="none", ms=4, color=color)
        plt.plot(prof["analytical_velocity"], prof["y"], "k--", lw=0.8)
    plt.plot([], [], "k-", label="Explicit FTCS")
    plt.plot([], [], "ko", mfc="none", label="Implicit BE")
    plt.plot([], [], "k--", lw=0.8, label="Analytical")
    plt.xlabel("u / U")
    plt.ylabel("y / H")
    plt.title("Transient Couette velocity profiles")
    plt.legend(fontsize=8)
    save("Figure_1.png")

    # Figure 2: error history for both schemes
    hist = read_csv(base / "couette_history.csv")
    plt.figure(figsize=(6.5, 4))
    for method in ("explicit", "implicit"):
        m = hist["method"] == method
        plt.semilogy(hist["time"][m], hist["relative_l2_error"][m], label=method)
    plt.xlabel("t")
    plt.ylabel("Relative L2 error vs analytical")
    plt.title("Error history")
    plt.legend()
    save("Figure_2.png")

    # Figure 3: explicit stability boundary, Fo = 0.25, 0.45, 0.55 at the same physical time
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    rows = []
    for fo in (0.25, 0.45, 0.55):
        out = run(f"Fo_{fo}", NY, H, U, NU, T_STABILITY, fo, FO_IMPLICIT, 100)
        p = read_csv(out / "couette_profiles.csv")
        s = read_summary(out / "couette_summary.txt")
        h = read_csv(out / "couette_history.csv")
        m = h["method"] == "explicit"
        rows.append((fo, int(s["explicit_steps"]), s["explicit_relative_L2_error"]))
        ax1.plot(p["explicit_velocity"], p["y"], ".-", ms=3, label=f"Fo = {fo}")
        ax2.semilogy(h["time"][m], h["relative_l2_error"][m], label=f"Fo = {fo}")
    ax1.plot(p["analytical_velocity"], p["y"], "k--", label="Analytical")
    ax1.set_ylim(0.75, 1.0)
    ax1.set_xlabel("u / U")
    ax1.set_ylabel("y / H")
    ax1.set_title(f"(a) Explicit profiles near the moving wall, t = {T_STABILITY}")
    ax1.legend()
    ax2.set_xlabel("t")
    ax2.set_ylabel("Relative L2 error")
    ax2.set_title("(b) Explicit error history")
    ax2.legend()
    save("Figure_3.png")

    # Figure 4: implicit cost vs accuracy (Fo sweep at t_end)
    fos = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    steps, errors = [], []
    for fo in fos:
        s = read_summary(run(f"implicit_Fo_{fo}", NY, H, U, NU, T_END, FO_EXPLICIT, fo, 1000) / "couette_summary.txt")
        steps.append(s["implicit_steps"])
        errors.append(s["implicit_relative_L2_error"])
    sb = read_summary(base / "couette_summary.txt")
    plt.figure(figsize=(6.5, 4))
    plt.loglog(steps, errors, "o-", label="Implicit BE (Fo sweep)")
    for fo, n, e in zip(fos, steps, errors):
        plt.annotate(f"Fo={fo:g}", (n, e), textcoords="offset points", xytext=(4, 4), fontsize=7)
    plt.loglog([sb["explicit_steps"]], [sb["explicit_relative_L2_error"]], "rs", label=f"Explicit FTCS (Fo={FO_EXPLICIT})")
    plt.xlabel("Time steps to reach t_end")
    plt.ylabel("Relative L2 error at t_end")
    plt.title("Accuracy vs computational effort")
    plt.legend()
    save("Figure_4.png")

    print("\nStability sweep (Fo, steps, explicit L2 error):")
    for row in rows:
        print("  ", row)
    print("Implicit sweep (Fo, steps, L2 error):")
    for row in zip(fos, steps, errors):
        print("  ", row)
    print("Base summary:", sb)


if __name__ == "__main__":
    main()
