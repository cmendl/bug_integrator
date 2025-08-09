"""
Discrete Schrödinger equation example in section 6.1 of:
  Gianluca Ceruti, Jonas Kusch, Christian Lubich
  A rank-adaptive robust integrator for dynamical low-rank approximation
  BIT Numerical Mathematics 62, 1149-1174 (2022)
  https://doi.org/10.1007/s10543-021-00907-7
  https://github.com/JonasKu/publication-A-rank-adaptive-robust-integrator-for-dynamical-low-rank-approximation
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import ortho_group
import matplotlib.pyplot as plt


def hamiltonian(y):
    """
    Construct the Hamiltonian in Eq. (23).
    """
    n = y.shape[0]
    # tridiagonal matrix
    d = 2*np.identity(n) - np.diag(np.ones(n - 1), k = 1) - np.diag(np.ones(n - 1), k = -1)
    vcos = np.diag([1 - np.cos(2*np.pi*j/n) for j in range(-n//2, n//2)])
    return 0.5*(d @ y + y @ d.conj().T) + vcos @ y @ vcos.conj().T


def func(y):
    """
    Evaluate the right-hand side of the ordinary differential equation.
    """
    return -1j * hamiltonian(y)


def rk4(f, y, h: float):
    """
    Runge–Kutta method of order 4.
    """
    k1 = h*f(y)
    k2 = h*f(y + 0.5*k1)
    k3 = h*f(y + 0.5*k2)
    k4 = h*f(y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6


def flow_update_u(u0, s0, v0, h: float):
    """
    K-step in Eq. (7).
    """
    # right-hand side of the ordinary differential equation for the K-step
    fu = lambda k: func(k @ v0.conj().T) @ v0
    k1 = rk4(fu, u0 @ s0, h)
    u_hat, _ = np.linalg.qr(np.concatenate((k1, u0), axis=1), mode="reduced")
    m_hat = u_hat.conj().T @ u0
    return u_hat, m_hat


def flow_update_v(u0, s0, v0, h: float):
    """
    L-step in Eq. (8).
    """
    # right-hand side of the ordinary differential equation for the L-step
    fv = lambda l: func(u0 @ l.conj().T).conj().T @ u0
    l1 = rk4(fv, v0 @ s0.conj().T, h)
    v_hat, _ = np.linalg.qr(np.concatenate((l1, v0), axis=1), mode="reduced")
    n_hat = v_hat.conj().T @ v0
    return v_hat, n_hat


def flow_update_core(u_hat, s0_hat, v_hat, h: float):
    """
    S-step in Eq. (9).
    """
    fs = lambda s: u_hat.conj().T @ func(u_hat @ s @ v_hat.conj().T) @ v_hat
    return rk4(fs, s0_hat, h)


def retained_singular_values(s, tol: float):
    """
    Indices of retained singular values based on given tolerance.
    """
    sq_sum = 0
    r1 = len(s)
    for i in reversed(range(len(s))):
        sq_sum += s[i]**2
        if np.sqrt(sq_sum) > tol:
            break
        r1 = i
    return range(r1)


def time_step(u0, s0, v0, dt: float, rel_tol: float):
    """
    Perform a single time evolution step.
    """
    u_hat, m_hat = flow_update_u(u0, s0, v0, dt)
    v_hat, n_hat = flow_update_v(u0, s0, v0, dt)
    s_hat = flow_update_core(u_hat, m_hat @ s0 @ n_hat.conj().T, v_hat, dt)
    # truncate
    p_hat, sigma_hat, qh_hat = np.linalg.svd(s_hat)
    idx = retained_singular_values(sigma_hat, dt * rel_tol)
    s1 = np.diag(sigma_hat[idx])
    p1 = p_hat[:, idx]
    q1 = qh_hat[idx, :].conj().T
    u1 = u_hat @ p1
    v1 = v_hat @ q1
    return u1, s1, v1


def f_ref(_, yvec, n: int):
    """
    Right-hand side of the reference ordinary differential equation.
    """
    y = np.reshape(yvec, (n, n))
    dydt = func(y)
    return np.reshape(dydt, -1)

def flow_ref(y0, t: float):
    """
    Reference solution of the ordinary differential equation.
    """
    n = y0.shape[0]
    sol = solve_ivp(f_ref, [0, t], np.reshape(y0, -1), args=(n,), rtol=1e-10, atol=1e-10)
    return np.reshape(sol.y[:, -1], (n, n))


def initial_values(n: int, rng: np.random.Generator):
    """
    Construct the initial value matrices.
    """
    u0 = ortho_group.rvs(n, random_state=rng)
    v0 = ortho_group.rvs(n, random_state=rng)
    s0 = np.diag([0.1**i for i in range(1, n+1)])
    s0 /= np.linalg.norm(s0, ord='fro')
    return u0, s0, v0


def main():

    n = 100
    rel_tol = 1e-5

    rng = np.random.default_rng(42)
    u0, s0, v0 = initial_values(n, rng)
    # cast to complex types
    u0 = (1+0j) * u0
    s0 = (1+0j) * s0
    v0 = (1+0j) * v0
    # reference initial value
    y0 = u0 @ s0 @ v0.conj().T

    # overall simulation time
    tmax = 1

    # reference solution
    y_ref = flow_ref(y0, tmax)

    r = 12
    u0 = u0[:, :r]
    s0 = s0[:r, :r]
    v0 = v0[:, :r]

    # # exact solution when starting from low-rank approximation of initial value
    # y_ref_lowrank = flow_ref(u0 @ s0 @ v0.conj().T, tmax)
    # err_lowrank = np.linalg.norm(y_ref_lowrank - y_ref)
    # print("err_lowrank:", err_lowrank)
    # sigma_ref_lowrank = np.linalg.svd(y_ref_lowrank, compute_uv=False)
    # print("number of retained singular values:", len(retained_singular_values(sigma_ref_lowrank, 1e-6)))

    nsteps_list = np.array([10, 20, 50, 100, 200, 500, 1000])
    errlist = np.zeros(len(nsteps_list))
    ranks_list = []
    for i, nsteps in enumerate(nsteps_list):
        print(32 * '_')
        print("nsteps:", nsteps)
        u, s, v = u0, s0, v0
        ranks = [s.shape[0]]
        for _ in range(nsteps):
            u, s, v = time_step(u, s, v, tmax/nsteps, rel_tol)
            ranks.append(s.shape[0])
        ranks_list.append(ranks)
        print("s.shape (after time evolution):", s.shape)
        y1 = u @ s @ v.conj().T
        errlist[i] = np.linalg.norm(y1 - y_ref)
    print("tmax/nsteps_list:", tmax/nsteps_list)
    print("errlist:", errlist)

    # visualize approximation error in dependence of the time step
    plt.title(f"n = {n}, tmax = {tmax}, r = {r}, rel_tol = {rel_tol}")
    plt.loglog(tmax/nsteps_list, errlist)
    plt.xlabel(r"$\Delta t$")
    plt.ylabel("error")
    plt.savefig("time_integration_matrix_schrodinger_error.pdf")
    plt.show()

    # visualize time-dependent ranks
    for i, nsteps in enumerate(nsteps_list):
        plt.plot(np.linspace(0, tmax, nsteps + 1, endpoint=True), ranks_list[i], label=f"dt = {tmax/nsteps}")
    plt.title(f"n = {n}, tmax = {tmax}, r = {r}, rel_tol = {rel_tol}")
    plt.xlabel("time")
    plt.ylabel("rank")
    plt.legend()
    plt.savefig("time_integration_matrix_schrodinger_ranks.pdf")
    plt.show()


if __name__ == "__main__":
    main()
