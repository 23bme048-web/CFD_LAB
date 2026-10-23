from __future__ import annotations

import csv
import math
import sys

import numpy as np

from common import relative_l2, tdma, write_multiline_vtk, write_structured_grid_vtk


def analytical_profile(y: np.ndarray, height: float, speed: float, nu: float, time: float, terms: int = 400) -> np.ndarray:
    eta = y / height
    if time <= 0.0:
        result = np.zeros_like(y)
        result[-1] = speed
        return result
    result = eta.copy()
    for n in range(1, terms + 1):
        result += 2.0 * ((-1.0) ** n) * np.sin(n * math.pi * eta) * math.exp(-n * n * math.pi * math.pi * nu * time / (height * height)) / (n * math.pi)
    result *= speed
    result[0] = 0.0
    result[-1] = speed
    return result


def implicit_step(field: np.ndarray, fourier: float, speed: float) -> np.ndarray:
    n = field.size - 2
    a = np.full(n, -fourier)
    b = np.full(n, 1.0 + 2.0 * fourier)
    c = np.full(n, -fourier)
    rhs = field[1:-1].copy()
    rhs[-1] += fourier * speed
    a[0] = 0.0
    c[-1] = 0.0
    result = field.copy()
    result[1:-1] = tdma(a, b, c, rhs)
    result[0] = 0.0
    result[-1] = speed
    return result


def main() -> int:
    ny = int(sys.argv[1]) if len(sys.argv) > 1 else 81
    height = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    speed = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    nu = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01
    end_time = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
    fo_explicit = float(sys.argv[6]) if len(sys.argv) > 6 else 0.45
    fo_implicit = float(sys.argv[7]) if len(sys.argv) > 7 else 5.0
    screen_every = int(sys.argv[8]) if len(sys.argv) > 8 else 50
    if ny < 5 or height <= 0.0 or speed <= 0.0 or nu <= 0.0 or end_time <= 0.0 or fo_explicit <= 0.0 or fo_implicit <= 0.0:
        raise ValueError("usage: python exp05_couette_explicit_implicit.py ny H U nu tEnd Fo_explicit Fo_implicit screen_every")
    y = np.linspace(0.0, height, ny)
    dy = y[1] - y[0]
    dt_explicit = fo_explicit * dy * dy / nu
    dt_implicit = fo_implicit * dy * dy / nu
    n_explicit = max(1, int(math.ceil(end_time / dt_explicit)))
    n_implicit = max(1, int(math.ceil(end_time / dt_implicit)))
    dt_explicit = end_time / n_explicit
    dt_implicit = end_time / n_implicit
    fo_explicit = nu * dt_explicit / (dy * dy)
    fo_implicit = nu * dt_implicit / (dy * dy)
    explicit = np.zeros(ny)
    implicit = np.zeros(ny)
    explicit[-1] = speed
    implicit[-1] = speed
    history: list[list[object]] = []
    time = 0.0
    for step in range(1, n_explicit + 1):
        old = explicit.copy()
        explicit[1:-1] = old[1:-1] + fo_explicit * (old[2:] - 2.0 * old[1:-1] + old[:-2])
        explicit[0] = 0.0
        explicit[-1] = speed
        time += dt_explicit
        reference = analytical_profile(y, height, speed, nu, time)
        l2_change = relative_l2(explicit, old)
        l2_error = relative_l2(explicit, reference)
        max_error = float(np.max(np.abs(explicit - reference)))
        history.append(["explicit", step, time, dt_explicit, fo_explicit, l2_change, l2_error, max_error])
        if step == 1 or step == n_explicit or step % max(1, screen_every) == 0:
            print(f"explicit_step={step} time={time:.6e} L2_error={l2_error:.6e}")
        if not np.all(np.isfinite(explicit)):
            break
    time = 0.0
    for step in range(1, n_implicit + 1):
        old = implicit.copy()
        implicit = implicit_step(implicit, fo_implicit, speed)
        time += dt_implicit
        reference = analytical_profile(y, height, speed, nu, time)
        l2_change = relative_l2(implicit, old)
        l2_error = relative_l2(implicit, reference)
        max_error = float(np.max(np.abs(implicit - reference)))
        history.append(["implicit", step, time, dt_implicit, fo_implicit, l2_change, l2_error, max_error])
        if step == 1 or step == n_implicit or step % max(1, screen_every) == 0:
            print(f"implicit_step={step} time={time:.6e} L2_error={l2_error:.6e}")
    reference = analytical_profile(y, height, speed, nu, end_time)
    explicit_error = relative_l2(explicit, reference) if np.all(np.isfinite(explicit)) else float("inf")
    implicit_error = relative_l2(implicit, reference)
    with open("couette_history.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "step", "time", "dt", "Fo", "relative_l2_change", "relative_l2_error", "maximum_error"])
        writer.writerows(history)
    with open("couette_profiles.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["y", "explicit_velocity", "implicit_velocity", "analytical_velocity"])
        for values in zip(y, explicit, implicit, reference):
            writer.writerow(values)
    x = np.array([0.0, 1.0])
    xx, yy = np.meshgrid(x, y)
    ue = np.repeat(explicit[:, None], 2, axis=1)
    ui = np.repeat(implicit[:, None], 2, axis=1)
    ua = np.repeat(reference[:, None], 2, axis=1)
    write_structured_grid_vtk("couette_field.vtk", xx, yy, {"explicit_velocity": ue, "implicit_velocity": ui, "analytical_velocity": ua}, {"explicit_vector": (ue, np.zeros_like(ue)), "implicit_vector": (ui, np.zeros_like(ui))}, "Transient Couette flow")
    write_multiline_vtk("couette_plates.vtk", [(np.array([0.0, 1.0]), np.array([0.0, 0.0])), (np.array([0.0, 1.0]), np.array([height, height]))], "Couette plates")
    with open("couette_summary.txt", "w", encoding="utf-8") as handle:
        handle.write("Couette flow explicit FTCS and implicit backward-Euler comparison\n")
        handle.write(f"ny {ny}\nH {height:.12e}\nU {speed:.12e}\nnu {nu:.12e}\nRe {speed * height / nu:.12e}\n")
        handle.write(f"tEnd {end_time:.12e}\nexplicit_dt {dt_explicit:.12e}\nexplicit_Fo {fo_explicit:.12e}\nexplicit_steps {n_explicit}\n")
        handle.write(f"explicit_relative_L2_error {explicit_error:.12e}\nexplicit_stable_by_Fo_criterion {str(fo_explicit <= 0.5).lower()}\n")
        handle.write(f"implicit_dt {dt_implicit:.12e}\nimplicit_Fo {fo_implicit:.12e}\nimplicit_steps {n_implicit}\nimplicit_relative_L2_error {implicit_error:.12e}\n")
    print("Wrote couette_field.vtk, couette_plates.vtk, couette_profiles.csv, couette_history.csv, couette_summary.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
