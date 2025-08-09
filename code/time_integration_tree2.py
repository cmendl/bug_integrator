"""
Rank-adaptive integrator for a tree tensor network specialized to a TTNO Hamiltonian

References:
- Gianluca Ceruti, Christian Lubich, Dominik Sulz
  Rank-adaptive time integration of tree tensor networks
  SIAM J. Numer. Anal. 61, 194-222 (2023)
- Christian Lubich, Bart Vandereycken, Hanna Walach
  Time integration of rank-constrained Tucker tensors
  SIAM J. Numer. Anal. 56, 1273-1290 (2018)
"""

from typing import Sequence
import numpy as np
from scipy.linalg import expm
from scipy import sparse
import matplotlib.pyplot as plt


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


def is_isometry(a, tol: float = 1e-10):
    """
    Whether the matrix `a` is an isometry.
    """
    a = np.asarray(a)
    assert a.ndim == 2
    return np.allclose(a.conj().T @ a, np.identity(a.shape[1]), rtol=tol, atol=tol)


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


def higher_order_svd(t, tol: float, max_ranks=None):
    """
    Compute the higher-order singular value decomposition (Tucker format approximation) of `t`.
    """
    assert (not max_ranks) or (t.ndim == len(max_ranks))
    u_list = []
    s_list = []
    for i in range(t.ndim):
        a = matricize(t, i)
        u, sigma, _ = np.linalg.svd(a, full_matrices=False)
        chi = retained_singular_values(sigma, tol)
        if max_ranks:
            # truncate in case max_ranks[i] < chi
            chi = min(chi, max_ranks[i])
        u_list.append(u[:, :chi])
        s_list.append(sigma)
    # form the core tensor by applying Ui^\dagger to the i-th dimension
    c = multi_mode_product([u.conj().T for u in u_list], t)
    return u_list, c, s_list


class TreeNode:
    def __init__(self, conn, children: Sequence):
        # using convention that physical axis or axes connecting to children come first, and then parent axis
        self.conn = np.asarray(conn)
        self.children = list(children)
        if children:
            assert self.conn.ndim == len(children) + 1

    @property
    def is_leaf(self) -> bool:
        """
        Whether the node is a leaf.
        """
        return not self.children

    def orthonormalize(self):
        """
        Orthonormalize the tree in-place.
        """
        for i, c in enumerate(self.children):
            r = c.orthonormalize()
            self.conn = single_mode_product(r, self.conn, i)
        tmat = self.conn.reshape((-1, self.conn.shape[-1]))
        if is_isometry(tmat):
            return np.identity(tmat.shape[1])
        q, r = np.linalg.qr(tmat, mode="reduced")
        self.conn = q.reshape(self.conn.shape[:-1] + (q.shape[1],))
        return r

    def to_full_tensor(self):
        """
        Convert the tree with root 'self' to a full tensor,
        with leading dimensions the physical dimensions and the last dimension the rank.
        """
        if self.is_leaf:
            return self.conn
        t = self.conn
        phys_dims = []
        for i, c in enumerate(self.children):
            ct = c.to_full_tensor()
            # record physical dimensions
            phys_dims = phys_dims + list(ct.shape[:-1])
            t = single_mode_product(ct.reshape(-1, ct.shape[-1]), t, i)
        return t.reshape(phys_dims + [t.shape[-1]])


def tree_vdot(chi: TreeNode, psi: TreeNode):
    """
    Compute the logical inner product `<chi | psi>` of two trees with the same topology.
    """
    assert len(chi.children) == len(psi.children)
    t = psi.conn
    for i in range(len(psi.children)):
        r = tree_vdot(chi.children[i], psi.children[i])
        t = single_mode_product(r, t, i)
    t = chi.conn.reshape((-1, chi.conn.shape[-1])).conj().T @ t.reshape((-1, t.shape[-1]))
    return t


def tree_operator_averages(chi: TreeNode, op: TreeNode, psi: TreeNode):
    """
    Compute the operator averages `<chi | op | psi>` on all subtrees,
    representing the result in a tree of the same topology.
    """
    if op.is_leaf:
        assert chi.is_leaf
        assert psi.is_leaf
        assert  op.conn.ndim == 3
        assert chi.conn.ndim == 2
        assert psi.conn.ndim == 2
        return TreeNode(np.einsum(chi.conn.conj(), (3, 0), op.conn, (3, 4, 1), psi.conn, (4, 2), (0, 1, 2)), [])
    t = np.kron(op.conn, psi.conn)
    assert len(chi.children) == len(op.children) and len(psi.children) == len(op.children)
    children = []
    for i in range(len(op.children)):
        child = tree_operator_averages(chi.children[i], op.children[i], psi.children[i])
        t = single_mode_product(child.conn.reshape((child.conn.shape[0], -1)), t, i)
        children.append(child)
    t = chi.conn.reshape((-1, chi.conn.shape[-1])).conj().T @ t.reshape((-1, t.shape[-1]))
    return TreeNode(t.reshape((chi.conn.shape[-1], op.conn.shape[-1], psi.conn.shape[-1])), children)


def generate_simple_tree(local_state, depth: int):
    """
    Generate a simple tree with internal bond dimension 1 representing a product state.
    """
    assert depth >= 0
    local_state = np.asarray(local_state)
    if depth == 0:
        return TreeNode(local_state.reshape((-1, 1)), [])
    return TreeNode(np.ones((1, 1, 1)), [generate_simple_tree(local_state, depth - 1) for _ in range(2)])


def binary_tree_from_state(state, depth: int, tol: float):
    """
    Construct a binary tree approximating a given state.
    """
    assert depth >= 0
    state = np.asarray(state)
    assert state.ndim == 2
    if depth == 0:
        return TreeNode(state, [])
    d = int(np.sqrt(state.shape[0]))
    assert state.shape[0] == d**2
    u_list, c, _ = higher_order_svd(state.reshape((d, d, state.shape[1])), tol)
    c = single_mode_product(u_list[-1], c, c.ndim - 1)
    # recursive function call on subtrees
    return TreeNode(c, [binary_tree_from_state(u, depth - 1, tol) for u in u_list[:-1]])


def interleave_local_operator_axes(op, d: int, nsites: int):
    """
    Interleave the local output and input axes of a linear operator.
    """
    op = np.asarray(op)
    assert op.ndim == 2
    assert op.shape == 2 * (d**nsites,)
    perm = sum(zip(range(nsites), range(nsites, 2*nsites)), ())
    return op.reshape((2*nsites) * (d,)).transpose(perm)


def separate_local_operator_axes(op):
    """
    Separate the local output and input axes of a linear operator,
    returning its matrix representation.
    """
    op = np.asarray(op)
    nsites = op.ndim // 2
    d = op.shape[0]
    assert op.shape == (2*nsites) * (d,)
    perm = list(range(0, 2*nsites, 2)) + list(range(1, 2*nsites, 2))
    return op.transpose(perm).reshape(2 * (d**nsites,))


def reshape_leaf_operators(node: TreeNode, d: int):
    """
    Reshape the leaf tensors of a tree to physical dimensions `(d, d)`.
    """
    if node.is_leaf:
        return TreeNode(node.conn.reshape((d, d, -1)), [])
    return TreeNode(node.conn, [reshape_leaf_operators(c, d) for c in node.children])


def binary_tree_from_operator(op, d: int, nsites: int, depth: int, tol: float):
    """
    Construct a binary tree approximating a given linear operator.
    """
    op_interleaved = interleave_local_operator_axes(op, d, nsites)
    tree = binary_tree_from_state(op_interleaved.reshape((-1, 1)), depth, tol)
    return reshape_leaf_operators(tree, d)


def rk4(f, y, h: float):
    """
    Runge–Kutta method of order 4.
    """
    k1 = h*f(y)
    k2 = h*f(y + 0.5*k1)
    k3 = h*f(y + 0.5*k2)
    k4 = h*f(y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6


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


def flow_update_basis(state: TreeNode, hamiltonian: TreeNode, avg: TreeNode, env_root, s0, dt: float):
    """
    Update and augment the basis matrix of the i-th subtree
    for a Schrödinger differential equation with Hamiltonian given as TTNO.

    TTNO-adapted Algorithm 5 in "Rank-adaptive time integration of tree tensor networks".
    """
    y = TreeNode(state.conn @ s0, state.children)
    if state.is_leaf:
        # right-hand side of the ordinary differential equation for the basis update
        f = lambda y: -1j * np.einsum(env_root, (1, 2, 3), hamiltonian.conn, (0, 4, 2), y, (4, 3), (0, 1))
        k1 = rk4(f, y.conn, dt)
        u_hat, _ = np.linalg.qr(np.concatenate((k1, state.conn), axis=1), mode="reduced")
        m_hat = u_hat.conj().T @ state.conn
        avg_hat = np.einsum(u_hat.conj(), (3, 0), hamiltonian.conn, (3, 4, 1), u_hat, (4, 2), (0, 1, 2))
        return TreeNode(u_hat, []), m_hat, TreeNode(avg_hat, [])
    else:
        y1, c0_hat, m_hat_children, avg_hat_children = time_step_subtree(y, hamiltonian, avg, env_root, dt)
        q_hat, _ = np.linalg.qr(np.concatenate((
            matricize(y1.conn, y1.conn.ndim - 1).T,
            matricize(c0_hat, c0_hat.ndim - 1).T), axis=1), mode="reduced")
        y1.conn = tensorize(q_hat.T, y1.conn.shape[:-1] + (q_hat.shape[1],), y1.conn.ndim - 1)
        # inner product between new augmented (orthonormal) state and input state
        m_hat = np.kron(y1.conn.conj(), state.conn)
        assert len(y1.children) == len(state.children)
        assert len(y1.children) == len(m_hat_children)
        for i in range(len(y1.children)):
            m_hat = single_mode_product(m_hat_children[i].reshape((1, -1)), m_hat, i)
            assert m_hat.shape[i] == 1
        m_hat = m_hat.reshape((y1.conn.shape[-1], state.conn.shape[-1]))
        # compute new averages (expectation values)
        avg_hat = np.kron(y1.conn.conj(), np.kron(hamiltonian.conn, y1.conn))
        assert len(avg_hat_children) == len(hamiltonian.children)
        for j in range(len(hamiltonian.children)):
            avg_hat = single_mode_product(avg_hat_children[j].conn.reshape((1, -1)), avg_hat, j)
            assert avg_hat.shape[j] == 1
        avg_hat = avg_hat.reshape((y1.conn.shape[-1], hamiltonian.conn.shape[-1], y1.conn.shape[-1]))
        return y1, m_hat, TreeNode(avg_hat, avg_hat_children)


def flow_update_connecting_tensor(c0_hat, avg_hat_children: Sequence[TreeNode], hamiltonian: TreeNode, env_root, dt: float):
    """
    Update the (already augmented) connecting tensor of a tree node
    for a Schrödinger differential equation with Hamiltonian given as TTNO.

    TTNO-adapted Algorithm 6 in "Rank-adaptive time integration of tree tensor networks".
    """
    # cannot be an empty list
    assert avg_hat_children
    assert len(hamiltonian.children) == len(avg_hat_children)
    def f(c):
        t = np.kron(hamiltonian.conn, c)
        for i in range(len(hamiltonian.children)):
            t = single_mode_product(avg_hat_children[i].conn.reshape((avg_hat_children[i].conn.shape[0], -1)), t, i)
        t = single_mode_product(env_root.reshape((env_root.shape[0], -1)), t, t.ndim - 1)
        return -1j * t
    # perform time evolution
    c1_hat = rk4(f, c0_hat, dt)
    return c1_hat


def compute_child_environment(state: TreeNode, hamiltonian: TreeNode, avg_children: Sequence[TreeNode], env_root, i: int):
    """
    Compute the environment tensor for the i-th child node
    after gauge-transforming the root node into an isometry towards the child.
    """
    q0, s0 = np.linalg.qr(matricize(state.conn, i).T, mode="reduced")
    s0 = s0.T
    q0ten = tensorize(q0.T, state.conn.shape[:i] + (q0.shape[1],) + state.conn.shape[i+1:], i)
    # project onto the orthonormalized tree without the current subtree
    env = np.kron(hamiltonian.conn, q0ten)
    for j in range(len(state.children)):
        if j == i:
            continue
        ac = avg_children[j].conn
        env = single_mode_product(ac.reshape((ac.shape[0], -1)), env, j)
    # upstream axis
    env = single_mode_product(env_root.reshape((env_root.shape[0], -1)), env, env.ndim - 1)
    # isolate the i-th virtual bond and contract all other axes
    env = q0.conj().T @ matricize(env, i).T
    env = env.reshape((q0.shape[1], hamiltonian.conn.shape[i], q0.shape[1]))
    return env, s0


def time_step_subtree(state: TreeNode, hamiltonian: TreeNode, avg_tree: TreeNode, env_root, dt: float):
    """
    Perform a recursive rank-augmenting TTN integration step on a (sub-)tree
    for a Schrödinger differential equation with Hamiltonian given as TTNO.

    TTNO-adapted Algorithm 4 in "Rank-adaptive time integration of tree tensor networks".
    """
    assert not state.is_leaf
    children_hat_list = []
    m_hat_children = []
    a_hat_children = []
    for i in range(len(state.children)):
        env, s0 = compute_child_environment(state, hamiltonian, avg_tree.children, env_root, i)
        c_hat, m_hat, a_hat = flow_update_basis(state.children[i], hamiltonian.children[i], avg_tree.children[i], env, s0, dt)
        children_hat_list.append(c_hat)
        m_hat_children.append(m_hat)
        a_hat_children.append(a_hat)
    # augment the initial connecting tensor
    c0_hat = multi_mode_product(m_hat_children + [np.identity(state.conn.shape[-1])], state.conn)
    c1_hat = flow_update_connecting_tensor(c0_hat, a_hat_children, hamiltonian, env_root, dt)
    return TreeNode(c1_hat, children_hat_list), c0_hat, m_hat_children, a_hat_children


def tree_time_step(state: TreeNode, hamiltonian: TreeNode, dt: float, rel_tol_trunc: float):
    """
    Perform a rank-augmenting TTN integration step
    for a Schrödinger differential equation with Hamiltonian given as TTNO.
    """
    # should be the actual roots of the trees
    assert state.conn.shape[-1] == 1
    assert hamiltonian.conn.shape[-1] == 1
    avg_tree = tree_operator_averages(state, hamiltonian, state)
    state, _, _, _ = time_step_subtree(state, hamiltonian, avg_tree, np.ones((1, 1, 1)), dt)
    state = truncate_tree(state, dt * rel_tol_trunc)
    return state


def truncate_tree(node: TreeNode, tol: float, max_rank: int = None):
    """
    Perform a rank truncation of a tree from root to leaves.
    """
    if node.is_leaf:
        return node
    u_list = []
    children_trunc = []
    for i in range(len(node.children)):
        u, sigma, _ = np.linalg.svd(matricize(node.conn, i), full_matrices=False)
        chi = retained_singular_values(sigma, tol)
        if max_rank:
            # truncate in case max_rank < chi
            chi = min(chi, max_rank)
        u = u[:, :chi]
        u_list.append(u)
        cit = single_mode_product(u.T, node.children[i].conn, node.children[i].conn.ndim - 1)
        # recursion to children
        children_trunc.append(truncate_tree(TreeNode(cit, node.children[i].children), tol, max_rank))
    # form the truncated core tensor
    c = node.conn
    for i in range(len(node.children)):
        # apply Ui^\dagger to the i-th dimension
        c = single_mode_product(u_list[i].conj().T, c, i)
    return TreeNode(c, children_trunc)


def construct_ising_1d_hamiltonian(nsites: int, J: float, h: float, g: float):
    """
    Construct the Ising Hamiltonian `sum J Z Z + h Z + g X`
    on a one-dimensional lattice as sparse matrix.
    """
    # Pauli-X and Z matrices
    sigma_x = sparse.csr_matrix([[0., 1.], [1.,  0.]])
    sigma_z = sparse.csr_matrix([[1., 0.], [0., -1.]])
    H = sparse.csr_matrix((2**nsites, 2**nsites), dtype=float)
    # interaction terms
    hint = sparse.kron(sigma_z, sigma_z)
    for j in range(nsites - 1):
        H += J * sparse.kron(sparse.identity(2**j),
                  sparse.kron(hint,
                              sparse.identity(2**(nsites-j-2))))
    # external field
    for j in range(nsites):
        H += sparse.kron(sparse.identity(2**j),
              sparse.kron(h*sigma_z + g*sigma_x,
                          sparse.identity(2**(nsites-j-1))))
    return H


def crandn(size=None, rng: np.random.Generator=None):
    """
    Draw random samples from the standard complex normal (Gaussian) distribution.
    """
    if rng is None:
        rng = np.random.default_rng()
    # 1/sqrt(2) is a normalization factor
    return (rng.normal(size=size) + 1j*rng.normal(size=size)) / np.sqrt(2)


def main1():

    # random number generator
    rng = np.random.default_rng(725)

    d = 3
    nsites = 4

    op = 0.25 * crandn(2 * (d**nsites,), rng)

    op_interleaved = interleave_local_operator_axes(op, d, nsites)
    print("op_interleaved.shape:", op_interleaved.shape)

    op2 = separate_local_operator_axes(op_interleaved)
    err_interleave = np.linalg.norm(op2 - op)
    print("err_interleave:", err_interleave)

    op_tree = binary_tree_from_operator(op, d=d, nsites=nsites, depth=2, tol=1e-8)
    op_tensor = op_tree.to_full_tensor()
    print("op_tensor.shape:", op_tensor.shape)
    print("op_tree.conn.shape:", op_tree.conn.shape)
    err_op = np.linalg.norm(op_interleaved - op_tensor.reshape(op_interleaved.shape))
    print("err_op:", err_op)

    psi = 0.25 * crandn(d**nsites, rng)
    print("np.linalg.norm(psi):", np.linalg.norm(psi))
    psi_tree = binary_tree_from_state(psi.reshape((-1, 1)), depth=2, tol=1e-8)
    err_state = np.linalg.norm(psi_tree.to_full_tensor().reshape(-1) - psi)
    print("err_state:", err_state)

    chi = 0.25 * crandn(d**nsites, rng)
    chi_tree = binary_tree_from_state(chi.reshape((-1, 1)), depth=2, tol=1e-8)

    avg = tree_operator_averages(chi_tree, op_tree, psi_tree)
    print("avg.conn:", avg.conn)
    print("avg.conn.shape:", avg.conn.shape)
    print("avg.children[1].conn.shape:", avg.children[1].conn.shape)

    avg_ref = np.vdot(chi, op @ psi)
    print("avg_ref:", avg_ref)
    err_avg = abs(avg.conn[0, 0, 0] - avg_ref)
    print("err_avg:", err_avg)


def main2():

    d = 2
    nsites = 8

    local_state = np.array([1., 0.], dtype=complex)
    y_init = generate_simple_tree(local_state, depth=3)
    print("y_init.conn.shape:", y_init.conn.shape)
    y_init_tensor = y_init.to_full_tensor()
    print("y_init_tensor.shape:", y_init_tensor.shape)
    y_init_vec = y_init_tensor.reshape(-1)
    print("np.linalg.norm(y_init_vec):", np.linalg.norm(y_init_vec))

    # visualize initial state
    plt.plot(y_init_vec.real, label="Re")
    plt.plot(y_init_vec.imag, label="Im")
    plt.xlabel("i")
    plt.ylabel("y[i]")
    plt.title("initial state")
    plt.legend()
    plt.show()

    hamiltonian_matrix = construct_ising_1d_hamiltonian(nsites, 1., 0., 1.).todense()
    print("hamiltonian_matrix.shape:", hamiltonian_matrix.shape)
    hamiltonian = binary_tree_from_operator(hamiltonian_matrix, d=d, nsites=nsites, depth=3, tol=1e-8)
    print("hamiltonian.conn.shape:", hamiltonian.conn.shape)
    # restore from tree and compare with original matrix
    hamiltonian_tensor = hamiltonian.to_full_tensor().reshape((2*nsites) * (d,))
    err_h = np.linalg.norm(hamiltonian_matrix - separate_local_operator_axes(hamiltonian_tensor))
    print("err_h:", err_h)

    avg_init = tree_operator_averages(y_init, hamiltonian, y_init)
    en_init = avg_init.conn[0, 0, 0].real
    print("en_init:", en_init)

    # dt = 0.1
    # time_step_subtree(y_init, hamiltonian, avg_init, np.array([[[1.]]]), dt)

    # overall simulation time
    tmax = 1

    # relative truncation tolerance
    rel_tol = 1e-4

    # reference solution
    y_ref = expm(-1j * tmax * hamiltonian_matrix) @ y_init_vec
    plt.plot(y_ref.real, label="Re")
    plt.plot(y_ref.imag, label="Im")
    plt.xlabel("i")
    plt.ylabel("y[i]")
    plt.title(f"reference time-evolved state at t = {tmax}")
    plt.legend()
    plt.show()

    nsteps_list = np.array([10, 20, 50, 100, 200, 500, 1000])
    err_list = np.zeros(len(nsteps_list))
    ranks_list = []
    ediff_list = []
    for i, nsteps in enumerate(nsteps_list):
        print(32 * '_')
        print("nsteps:", nsteps)
        y = y_init
        print("initial y.conn.shape:", y.conn.shape)
        dt = tmax/nsteps
        ranks = [max(y.conn.shape)]
        ediff = [0]
        for _ in range(nsteps):
            y = tree_time_step(y, hamiltonian, dt, rel_tol)
            ranks.append(max(y.conn.shape))
            avg_tree = tree_operator_averages(y, hamiltonian, y)
            ediff.append(abs(avg_tree.conn[0, 0, 0].real - en_init))
        print("final y.conn.shape:", y.conn.shape)
        ranks_list.append(ranks)
        ediff_list.append(ediff)
        err_list[i] = np.linalg.norm(y.to_full_tensor().reshape(-1) - y_ref)
        # visualize solution
        y1_vec =  y.to_full_tensor().reshape(-1)
        plt.plot(y1_vec.real, label="Re")
        plt.plot(y1_vec.imag, label="Im")
        plt.xlabel("i")
        plt.ylabel("y[i]")
        plt.title(f"rank adaptive solution at t = {tmax}, time step dt = {dt}")
        plt.legend()
        plt.show()
    print("tmax/nsteps_list:", tmax/nsteps_list)
    print("err_list:", err_list)

    # visualize approximation error in dependence of the time step
    plt.loglog(tmax/nsteps_list, err_list)
    plt.xlabel(r"$\Delta t$")
    plt.ylabel("error")
    plt.title(f"tmax = {tmax}, rel_tol = {rel_tol}")
    plt.savefig("time_integration_tree2_error.pdf")
    plt.show()

    # visualize time-dependent ranks
    for i, nsteps in enumerate(nsteps_list):
        plt.plot(np.linspace(0, tmax, nsteps + 1, endpoint=True), ranks_list[i], label=f"dt = {tmax/nsteps}")
    plt.title(f"tmax = {tmax}, rel_tol = {rel_tol}")
    plt.xlabel("time")
    plt.ylabel("rank")
    plt.legend()
    plt.savefig("time_integration_tree2_ranks.pdf")
    plt.show()

    # visualize time-dependent energy differences
    for i, nsteps in enumerate(nsteps_list):
        plt.semilogy(np.linspace(0, tmax, nsteps + 1, endpoint=True), ediff_list[i], label=f"dt = {tmax/nsteps}")
    plt.title(f"tmax = {tmax}, rel_tol = {rel_tol}")
    plt.xlabel("time")
    plt.ylabel("deviation from initial energy")
    plt.legend()
    plt.savefig("time_integration_tree2_energy.pdf")
    plt.show()


if __name__ == "__main__":
    main2()
