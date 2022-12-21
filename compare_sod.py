import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pl

from hll_solver import solve_MHD_Riemann_problem
from riemannSolver import RiemannSolver

left_state = {
    "rho": 1.0,
    "p": 1.0,
    "u": 0.0,
    "v": 0.0,
    "w": 0.0,
    "Bx": 0.0,
    "By": 0.0,
    "Bz": 0.0,
}
right_state = {
    "rho": 0.125,
    "p": 0.1,
    "u": 0.0,
    "v": 0.0,
    "w": 0.0,
    "Bx": 0.0,
    "By": 0.0,
    "Bz": 0.0,
}
gamma = 5.0 / 3.0
ncell = 800
xsize = 1.0
t = 0.2

rs_hydro = RiemannSolver(gamma)
xs = np.linspace(0.0, xsize, ncell)
xs -= xs.mean()
dxdt = xs / t
rhosol_hydro, usol_hydro, Psol_hydro, _ = rs_hydro.solve(
    left_state["rho"],
    left_state["u"],
    left_state["p"],
    right_state["rho"],
    right_state["u"],
    right_state["p"],
    dxdt,
)

riemann_problem = {
    "left_state": left_state,
    "right_state": right_state,
    "time": t,
    "dt": 4.0e-4,
    "xsize": xsize,
    "gamma": gamma,
    "ncell": ncell,
    "solver": "HLL",
}
xs_MHD, solution_MHD = solve_MHD_Riemann_problem(riemann_problem, False)

fig, ax = pl.subplots(3, 1, sharex=True, figsize=(4, 6))

ax[0].plot(xs, rhosol_hydro, label="exact")
ax[0].plot(xs_MHD, solution_MHD["rho"], label="HLL")
ax[0].set_ylabel("rho")

ax[1].plot(xs, usol_hydro, label="exact")
ax[1].plot(xs_MHD, solution_MHD["u"], label="HLL")
ax[1].set_ylabel("u")

ax[2].plot(xs, Psol_hydro, label="exact")
ax[2].plot(xs_MHD, solution_MHD["p"], label="HLL")
ax[2].set_ylabel("p")

ax[2].set_xlabel("x")

ax[0].legend(loc="best")

pl.tight_layout()
pl.savefig("compare_sod.png", dpi=300)
