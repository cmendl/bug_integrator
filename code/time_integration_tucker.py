"""
References:
- Gianluca Ceruti, Christian Lubich, Dominik Sulz
  Rank-adaptive time integration of tree tensor networks
  SIAM J. Numer. Anal. 61, 194-222 (2023)
- Christian Lubich, Bart Vandereycken, Hanna Walach
  Time integration of rank-constrained Tucker tensors
  SIAM J. Numer. Anal. 56, 1273-1290 (2018)
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def hamiltonian(y):
    """
    Construct the Hamiltonian of the discrete nonlinear Schrödinger equation in Eq. (7.1).
    """
    n = y.shape
    delta = 0.02
    eps = 0.01
    # normalize to [0, 1]^3 volume
    laplace = (
        # axis 0
        n[0]**2 * (
          np.roll(np.pad(y[:-1, :, :], pad_width=((0, 1), (0, 0), (0, 0))), shift= 1, axis=0)
        + np.roll(np.pad(y[ 1:, :, :], pad_width=((1, 0), (0, 0), (0, 0))), shift=-1, axis=0)
        - 2 * y) +
        # axis 1
        n[1]**2 * (
          np.roll(np.pad(y[:, :-1, :], pad_width=((0, 0), (0, 1), (0, 0))), shift= 1, axis=1)
        + np.roll(np.pad(y[:,  1:, :], pad_width=((0, 0), (1, 0), (0, 0))), shift=-1, axis=1)
        - 2 * y) +
        # axis 2
        n[2]**2 * (
          np.roll(np.pad(y[:, :, :-1], pad_width=((0, 0), (0, 0), (0, 1))), shift= 1, axis=2)
        + np.roll(np.pad(y[:, :,  1:], pad_width=((0, 0), (0, 0), (1, 0))), shift=-1, axis=2)
        - 2 * y))
    return -0.5 * delta * laplace + eps * np.abs(y)**2 * y


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


def single_mode_product(a, t, i: int):
    """
    Compute the i-mode product between the matrix `a` and tensor `t`.
    """
    t = np.tensordot(a, t, axes=(1, i))
    # original i-th dimension is now 0-th dimension; move back to i-th place
    t = np.transpose(t, list(range(1, i + 1)) + [0] + list(range(i + 1, t.ndim)))
    return t


def multi_mode_product(u_list, c):
    """
    Compute the multi-mode product between the matrices `u_list` and the core tensor `c`.
    """
    assert len(u_list) == c.ndim
    t = c
    for i in range(c.ndim):
        t = single_mode_product(u_list[i], t, i)
    return t


def matricize(a, i: int):
    """
    Compute the matricization of a tensor along the i-th axis.
    """
    s = (int(np.prod(a.shape[:i])), a.shape[i], int(np.prod(a.shape[i+1:])))
    a = a.reshape(s)
    a = a.transpose((1, 0, 2)).reshape((s[1], s[0]*s[2]))
    return a


def tensorize(a, shape, i: int):
    """
    Tensorize a matrix, undoing the matricization along the i-th axis.
    """
    s = (shape[i], int(np.prod(shape[:i])), int(np.prod(shape[i+1:])))
    return a.reshape(s).transpose((1, 0, 2)).reshape(shape)


def flow_update_basis(u0_list, c0, i: int, h: float):
    """
    Update and augment the i-th basis matrix.
    """
    # Algorithm 2 in "Rank-adaptive time integration of tree tensor networks"
    # logical dimensions
    n = [u0.shape[0] for u0 in u0_list]
    q0, s0 = np.linalg.qr(matricize(c0, i).T)
    s0 = s0.T
    u_prod = np.identity(1)
    for j in range(len(u0_list)):
        if j == i:
            continue
        u_prod = np.kron(u_prod, u0_list[j])
    v0 = (u_prod @ q0).T
    # right-hand side of the ordinary differential equation for the basis update
    fu = lambda k: matricize(func(tensorize(k @ v0, n, i)), i) @ v0.conj().T
    k1 = rk4(fu, u0_list[i] @ s0, h)
    u_hat, _ = np.linalg.qr(np.concatenate((k1, u0_list[i]), axis=1), mode="reduced")
    m_hat = u_hat.conj().T @ u0_list[i]
    return u_hat, m_hat


def flow_update_core(u_hat_list, s0_hat, h: float):
    """
    Augment and update and the core tensor.
    """
    # Algorithm 3 in "Rank-adaptive time integration of tree tensor networks"
    u_hat_ad_list = [u.conj().T for u in u_hat_list]
    fs = lambda s: multi_mode_product(u_hat_ad_list, func(multi_mode_product(u_hat_list, s)))
    return rk4(fs, s0_hat, h)


def retained_singular_values(s, tol: float):
    """
    Number of retained singular values based on given tolerance.
    """
    sq_sum = 0
    r1 = len(s)
    for i in reversed(range(len(s))):
        sq_sum += s[i]**2
        if np.sqrt(sq_sum) > tol:
            break
        r1 = i
    return r1


def higher_order_svd(t, tol: float, max_ranks):
    """
    Compute the higher-order singular value decomposition (Tucker format approximation) of `t`.
    """
    assert t.ndim == len(max_ranks)
    u_list = []
    s_list = []
    for i in range(t.ndim):
        a = matricize(t, i)
        u, sigma, _ = np.linalg.svd(a, full_matrices=False)
        chi = retained_singular_values(sigma, tol)
        if max_ranks[i] > 0:
            # truncate in case max_ranks[i] < chi
            chi = min(chi, max_ranks[i])
        u_list.append(u[:, :chi])
        s_list.append(sigma)
    # form the core tensor
    c = t
    for i in range(c.ndim):
        # apply Ui^\dagger to the i-th dimension
        c = single_mode_product(u_list[i].conj().T, c, i)
    return u_list, c, s_list


def time_step(u0_list, c0, dt: float, rel_tol: float):
    """
    Perform a rank-adaptive Tucker integration step.
    """
    u_hat_list = []
    m_hat_list = []
    for i in range(len(u0_list)):
        u_hat, m_hat = flow_update_basis(u0_list, c0, i, dt)
        u_hat_list.append(u_hat)
        m_hat_list.append(m_hat)
    s_hat = flow_update_core(u_hat_list, multi_mode_product(m_hat_list, c0), dt)
    # truncate based on tolerance
    p_list, c1, _ = higher_order_svd(s_hat, dt * rel_tol, s_hat.ndim * (-1,))
    u1_list = [u_hat_list[i] @ p_list[i] for i in range(len(u_hat_list))]
    return u1_list, c1


def f_ref(_, yvec, shape):
    """
    Right-hand side of the reference ordinary differential equation.
    """
    y = np.reshape(yvec, shape)
    dydt = func(y)
    return np.reshape(dydt, -1)

def flow_ref(y0, t: float):
    """
    Reference solution of the ordinary differential equation.
    """
    shape = y0.shape
    sol = solve_ivp(f_ref, [0, t], np.reshape(y0, -1), args=(shape,), rtol=1e-10, atol=1e-10)
    return np.reshape(sol.y[:, -1], shape)


def initial_value(n: int):
    """
    Construct the initial value tensor.
    """
    gamma1 = 0.2
    gamma2 = 0.1
    return np.array([[[
            np.exp(-((i/n - 0.75)**2 + (j/n - 0.25)**2 + (k/n - 0.1)**2) / gamma1**2) +
            np.exp(-((i/n - 0.25)**2 + (j/n - 0.75)**2 + (k/n - 0.8)**2) / gamma2**2)
        for k in range(n)]
        for j in range(n)]
        for i in range(n)])


def main():

    n = 25
    rel_tol = 5e-5

    r = 5

    # reference initial value
    y0 = initial_value(n)
    # Tucker format approximation
    u0_list, c0, _ = higher_order_svd(y0, 0., (r, r, r))
    # cast to complex types
    y0 = (1+0j) * y0
    u0_list = [(1+0j) * u0 for u0 in u0_list]
    c0 = (1+0j) * c0
    print("c0.shape:", c0.shape)

    err0 = np.linalg.norm(multi_mode_product(u0_list, c0) - y0)
    print("initial Tucker format approximation error:", err0)

    # overall simulation time
    tmax = 1

    # visualize initial value
    _, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.voxels(y0.real > 0.1, facecolors="#0000FF7F")
    plt.title("initial value")
    plt.show()

    # reference solution
    y_ref = flow_ref(y0, tmax)
    _, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.voxels(y_ref.real > 0.1, facecolors="#0000FF7F")
    ax.voxels(y_ref.imag > 0.1, facecolors="#00FF007F")
    plt.title(f"reference solution at t = {tmax}")
    plt.show()

    # require a minimum number of steps to avoid overflow due to Laplace term
    nsteps_list = np.array([50, 100, 200, 500, 1000])
    errlist = np.zeros(len(nsteps_list))
    ranks_list = []
    for i, nsteps in enumerate(nsteps_list):
        print(32 * '_')
        print("nsteps:", nsteps)
        u_list, c = u0_list, c0
        ranks = [max(c.shape)]
        for _ in range(nsteps):
            u_list, c = time_step(u_list, c, tmax/nsteps, rel_tol)
            ranks.append(max(c.shape))
        ranks_list.append(ranks)
        print("c.shape (after time evolution):", c.shape)
        y1 = multi_mode_product(u_list, c)
        errlist[i] = np.linalg.norm(y1 - y_ref)
        # visualize solution
        _, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.voxels(y1.real > 0.1, facecolors="#0000FF7F")
        ax.voxels(y1.imag > 0.1, facecolors="#00FF007F")
        plt.title(f"rank adaptive solution at t = {tmax}, time step dt = {tmax/nsteps}")
        plt.show()
    print("tmax/nsteps_list:", tmax/nsteps_list)
    print("errlist:", errlist)

    # visualize approximation error in dependence of the time step
    plt.title(f"n = {n}, tmax = {tmax}, r = {r}, rel_tol = {rel_tol}")
    plt.loglog(tmax/nsteps_list, errlist)
    plt.xlabel(r"$\Delta t$")
    plt.ylabel("error")
    plt.savefig("time_integration_tucker_error.pdf")
    plt.show()

    # visualize time-dependent ranks
    for i, nsteps in enumerate(nsteps_list):
        plt.plot(np.linspace(0, tmax, nsteps + 1, endpoint=True), ranks_list[i], label=f"dt = {tmax/nsteps}")
    plt.title(f"n = {n}, tmax = {tmax}, r = {r}, rel_tol = {rel_tol}")
    plt.xlabel("time")
    plt.ylabel("rank")
    plt.legend()
    plt.savefig("time_integration_tucker_ranks.pdf")
    plt.show()


if __name__ == "__main__":
    main()
