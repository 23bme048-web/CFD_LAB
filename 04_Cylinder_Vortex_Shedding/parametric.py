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


# Parameter set (sim.py argument order: nx ny Re outer_steps numerical_iterations output_every)
NX, NY = 241, 97              # cylinder diameter = round(ny / 8) = 12 lattice cells
RE = 100.0
STEPS = 4500                  # 4 lattice sub-steps per outer step -> U t / D = 150
NUM_IT = 40                   # -> 4 sub-steps per outer step
FRAME_EVERY = 30              # VTK output every 1 convective time unit
RE_SWEEP = (60.0, 100.0, 150.0)


def williamson(re):
    """Empirical laminar-shedding fit St(Re) of Williamson (1998)."""
    return -3.3265 / re + 0.1816 + 1.6e-4 * re


def zero_crossing_strouhal(h, t_start):
    """St from the mean period between upward zero crossings of C_L after t_start."""
    t = h["dimensionless_time_Ut_over_D"]
    m = t > t_start
    t, cl = t[m], h["Cl_momentum_exchange"][m]
    cl = cl - cl.mean()
    k = np.where((cl[:-1] < 0) & (cl[1:] >= 0))[0]
    crossings = t[k] - cl[k] * (t[k + 1] - t[k]) / (cl[k + 1] - cl[k])
    return 1.0 / np.mean(np.diff(crossings)), len(crossings) - 1


def wake(ax, x, y, f, lim=3.0):
    w = np.where(f["solid_mask"] > 0.5, np.nan, f["vorticity_nondimensional"])
    cf = ax.contourf(x, y, np.clip(w, -lim, lim), levels=np.linspace(-lim, lim, 41), cmap="RdBu_r")
    ax.contourf(x, y, f["solid_mask"], levels=[0.5, 1.5], colors="0.2")
    ax.set_aspect("equal")
    return cf


def main() -> None:
    base = run(f"Re_{RE:g}", NX, NY, RE, STEPS, NUM_IT, FRAME_EVERY)
    s = read_summary(base / "cylinder_summary.txt")
    h = read_csv(base / "cylinder_history.csv")
    st_fft = s["estimated_Strouhal"]
    st, n_cycles = zero_crossing_strouhal(h, 90.0)

    # Figure 1: final vorticity field
    x, y, f = read_vtk(base / "cylinder_vortex_final.vtk")
    fig, ax = plt.subplots(figsize=(10, 4))
    cf = wake(ax, x, y, f)
    fig.colorbar(cf, ax=ax, label="omega D / U", shrink=0.8)
    ax.set_xlabel("x / D")
    ax.set_ylabel("y / D")
    ax.set_title(f"Karman vortex street, Re = {RE:g}, U t / D = {s['final_dimensionless_time_Ut_over_D']:.0f}")
    save("Figure_1.png")

    # Figure 2: vorticity over one shedding period (last cycle of the run)
    period = 1.0 / st                                  # in units of D / U, one frame per unit
    n_frames = 6
    last = STEPS // FRAME_EVERY * FRAME_EVERY
    picks = [last - k * FRAME_EVERY for k in range(n_frames)][::-1]
    fig, axes = plt.subplots(3, 2, figsize=(11, 7.5), sharex=True, sharey=True)
    t0 = picks[0] * 4 * 0.1 / 12
    for ax, step in zip(axes.ravel(), picks):
        x, y, f = read_vtk(base / f"cylinder_vortex_{step:06d}.vtk")
        cf = wake(ax, x, y, f)
        t = step * 4 * 0.1 / 12
        ax.set_title(f"t* = {t:.0f}  (t - t0) / T = {(t - t0) / period:.2f}", fontsize=9)
    fig.colorbar(cf, ax=axes, label="omega D / U", shrink=0.7)
    fig.suptitle(f"Vorticity over one shedding period T = 1/St = {period:.2f} D/U")
    save("Figure_2.png")

    # Figure 3: force histories
    t = h["dimensionless_time_Ut_over_D"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    ax1.plot(t, h["Cl_momentum_exchange"], lw=0.9)
    ax1.set_ylabel("C_L")
    ax1.set_title("Lift and drag coefficients (momentum exchange)")
    ax2.plot(t, h["Cd_momentum_exchange"], lw=0.9, color="C1")
    ax2.set_ylabel("C_D")
    ax2.set_ylim(1.4, 2.0)
    ax2.text(1, 1.95, "(impulsive-start spike at t* < 2 is off scale)", fontsize=8, va="top")
    ax2.set_xlabel("U t / D")
    save("Figure_3.png")

    # Figure 4: lift spectrum and Strouhal number vs Re
    rows = []
    for re in RE_SWEEP:
        out = base if re == RE else run(f"Re_{re:g}", NX, NY, re, STEPS, NUM_IT, STEPS)
        sr = read_summary(out / "cylinder_summary.txt")
        hr = read_csv(out / "cylinder_history.csv")
        half = int(hr.size * 0.6)
        rows.append((re, zero_crossing_strouhal(hr, 90.0)[0], sr["estimated_Strouhal"], williamson(re), np.mean(hr["Cd_momentum_exchange"][half:]),
                     0.5 * np.ptp(hr["Cl_momentum_exchange"][half:]), np.max(hr["maximum_density_deviation"])))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    lift = h["Cl_momentum_exchange"][int(h.size * 0.6):]
    lift = lift - lift.mean()
    spec = np.abs(np.fft.rfft(lift * np.hanning(lift.size))) ** 2
    freq = np.fft.rfftfreq(lift.size, d=t[1] - t[0])
    ax1.plot(freq, spec / spec.max())
    ax1.axvline(st, color="C3", ls="--", label=f"zero-crossing St = {st:.3f}")
    ax1.axvline(st_fft, color="C2", ls=":", label=f"sim.py FFT St = {st_fft:.3f}")
    ax1.set_xlim(0, 0.6)
    ax1.set_xlabel("f D / U")
    ax1.set_ylabel("Normalised power")
    ax1.set_title(f"(a) Lift spectrum, Re = {RE:g}")
    ax1.legend()
    r = np.array(rows)
    re_line = np.linspace(50, 170, 100)
    ax2.plot(re_line, williamson(re_line), "k--", label="Williamson (1998) fit")
    ax2.plot(r[:, 0], r[:, 1], "o", ms=8, label="LBM, zero-crossing St")
    ax2.plot(r[:, 0], r[:, 2], "x", ms=8, label="LBM, sim.py FFT St")
    ax2.set_xlabel("Re")
    ax2.set_ylabel("St")
    ax2.set_title("(b) Strouhal number vs Reynolds number")
    ax2.legend()
    save("Figure_4.png")

    print(f"zero-crossing St = {st:.4f} over {n_cycles} cycles, FFT St = {st_fft:.4f}")
    print("Re, St_zero_crossing, St_fft, St_Williamson, mean Cd, Cl amplitude, max density deviation")
    for row in rows:
        print("  ", tuple(float(f"{v:.4g}") for v in row))
    print(s)


if __name__ == "__main__":
    main()
