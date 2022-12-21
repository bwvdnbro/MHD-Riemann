import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pl

from hll_solver import solve_MHD_Riemann_problem
from riemannSolver import RiemannSolver

# cosmology parameters
# we assume an Einstein-de Sitter cosmology with
#  H(a) = 1/a * da/dt = H_0 * a^(-3/2)
a_begin = 0.001
a_end = 0.002
H_0 = 100.0


def cosmology_get_dt_rescaled_from_da(a0, a1):
    """
    Get the rescaled time interval (dt' = dt/a^2) corresponding to the given
    range in scale factors, assuming our Einstein-de Sitter cosmology
    """
    return 2.0 * (1.0 / np.sqrt(a0) - 1.0 / np.sqrt(a1)) / H_0


def cosmology_get_term_aX(a, X):
    """
    Get the cosmological correction factor A in the equation
      dV/dt = A*V
    if V depends on the scale factor with a power X, e.g. for V=rho,
      rho' = rho*a^3
    and X = 3.
    """
    return -X * H_0 * np.sqrt(a)


def cosmology_get_a_from_t(t):
    """
    Get the scale factor for the given time, assuming that t=0 corresponds to
    a know scale factor a_begin.
    """
    return (1.0 / np.sqrt(a_begin) - 0.5 * H_0 * t) ** (-2)


# set up the Riemann problem: the Dai & Woodward (1994) Sod-like setup
left_state = {
    "rho": 1.08,
    "p": 0.95,
    "u": 1.2,
    "v": 0.01,
    "w": 0.5,
    "Bx": 4.0 / np.sqrt(4.0 * np.pi),
    "By": 3.6 / np.sqrt(4.0 * np.pi),
    "Bz": 2.0 / np.sqrt(4.0 * np.pi),
}
right_state = {
    "rho": 1.0,
    "p": 1.0,
    "u": 0.0,
    "v": 0.0,
    "w": 0.0,
    "Bx": 4.0 / np.sqrt(4.0 * np.pi),
    "By": 4.0 / np.sqrt(4.0 * np.pi),
    "Bz": 2.0 / np.sqrt(4.0 * np.pi),
}
# some other parameters
gamma = 5.0 / 3.0
ncell = 800
xsize = 1.0
# range in rescaled time coordinate
# because of the invariance of the Euler equations with gamma=5/3 under the
# transformation to co-moving coordinates when using this rescaled time
# variable, th exact solution of the co-moving (hydro) Riemann problem is given
# by evaluating the exact Riemann solver at this time
t = cosmology_get_dt_rescaled_from_da(a_begin, a_end)

# get the exact solution from the hydro Riemann solver
# note that this solution will be different from the MHD solution; we want it
# for reference only
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

# now get the (approximate) solution from a simple Euler step integration using
# a HLL Riemann solver that does know about MHD
riemann_problem = {
    "left_state": left_state,
    "right_state": right_state,
    "time": t,
    "dt": 4.0e-5,
    "xsize": xsize,
    "gamma": gamma,
    "ncell": ncell,
    "solver": "HLL",
    "comoving": True,
    "a_from_t": cosmology_get_a_from_t,
    "comoving_term": cosmology_get_term_aX,
    "H_0": H_0,
}
xs_MHD, solution_MHD = solve_MHD_Riemann_problem(riemann_problem, True)

# plot the results
fig, ax = pl.subplots(3, 3, sharex=True, figsize=(9, 8))

ax[0][0].plot(xs, rhosol_hydro, label="exact hydro")
ax[0][0].plot(xs_MHD, solution_MHD["rho"], label="HLL")
ax[0][0].set_ylabel("rho")
ax[0][1].plot(xs, Psol_hydro, label="exact hydro")
ax[0][1].plot(xs_MHD, solution_MHD["p"], label="HLL")
ax[0][1].set_ylabel("p")
ax[0][2].axis("off")

ax[1][0].plot(xs, usol_hydro, label="exact hydro")
ax[1][0].plot(xs_MHD, solution_MHD["u"], label="HLL")
ax[1][0].set_ylabel("u")
ax[1][1].plot(xs, np.zeros(xs.shape), label="exact hydro")
ax[1][1].plot(xs_MHD, solution_MHD["v"], label="HLL")
ax[1][1].set_ylabel("v")
ax[1][2].plot(xs, np.zeros(xs.shape), label="exact hydro")
ax[1][2].plot(xs_MHD, solution_MHD["w"], label="HLL")
ax[1][2].set_ylabel("w")

ax[2][0].plot(xs, np.zeros(xs.shape), label="exact hydro")
ax[2][0].plot(xs_MHD, solution_MHD["Bx"], label="HLL")
ax[2][0].set_ylabel("Bx")
ax[2][1].plot(xs, np.zeros(xs.shape), label="exact hydro")
ax[2][1].plot(xs_MHD, solution_MHD["By"], label="HLL")
ax[2][1].set_ylabel("By")
ax[2][2].plot(xs, np.zeros(xs.shape), label="exact hydro")
ax[2][2].plot(xs_MHD, solution_MHD["Bz"], label="HLL")
ax[2][2].set_ylabel("Bz")

ax[2][0].set_xlabel("x")
ax[2][1].set_xlabel("x")
ax[2][2].set_xlabel("x")

ax[2][0].legend(loc="best")

ax[0][1].set_title(
    f"a range [{a_begin}, {a_end}], H_0=100, Einstein-de Sitter cosmology"
)

pl.tight_layout()
pl.savefig("compare_daiwoodward_cosmology.png", dpi=300)
