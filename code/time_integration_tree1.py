"""
Rank-adaptive integrator for a tree tensor network and generic (possibly nonlinear) right-hand side function

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
from scipy import sparse
from scipy.integrate import solve_ivp
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


def tree_vdot(node_a: TreeNode, node_b: TreeNode):
    """
    Compute the logical inner product of two trees with the same topology.
    """
    if node_a.is_leaf:
        assert node_b.is_leaf
        assert node_a.conn.ndim == 2
        return node_a.conn.conj().T @ node_b.conn
    t = np.kron(node_a.conn.conj(), node_b.conn)
    # contract inner product of child nodes into 't'
    assert len(node_a.children) == len(node_b.children)
    for i in range(len(node_a.children)):
        r = tree_vdot(node_a.children[i], node_b.children[i])
        t = single_mode_product(r.reshape((1, -1)), t, i)
        assert t.shape[i] == 1
    return t.reshape((node_a.conn.shape[-1], node_b.conn.shape[-1]))


def generate_simple_tree(local_state, depth: int):
    """
    Generate a simple tree with internal bond dimension 1 representing a product state.
    """
    assert depth >= 0
    local_state = np.asarray(local_state)
    if depth == 0:
        return TreeNode(local_state.reshape((-1, 1)), [])
    return TreeNode(np.ones((1, 1, 1)), [generate_simple_tree(local_state, depth - 1) for _ in range(2)])


def binary_tree_from_state(mat, depth: int, tol: float):
    """
    Construct a binary tree approximating a given state.
    """
    assert depth >= 0
    mat = np.asarray(mat)
    assert mat.ndim == 2
    if depth == 0:
        return TreeNode(mat, [])
    d = int(np.sqrt(mat.shape[0]))
    assert mat.shape[0] == d**2
    u_list, c, _ = higher_order_svd(mat.reshape((d, d, mat.shape[1])), tol)
    c = single_mode_product(u_list[-1], c, c.ndim - 1)
    # recursive function call on subtrees
    return TreeNode(c, [binary_tree_from_state(u, depth - 1, tol) for u in u_list[:-1]])


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


def flow_update_basis(node: TreeNode, i: int, f_tree, dt: float):
    """
    Update and augment the basis matrix of the i-th subtree.

    'f_tree' takes a TTN with the current node as root as input and returns the (projected) right-hand side of the differential equation.

    Algorithm 5 in "Rank-adaptive time integration of tree tensor networks".
    """
    assert not node.is_leaf
    q0, s0 = np.linalg.qr(matricize(node.conn, i).T)
    s0 = s0.T
    q0ten = tensorize(q0.T, node.conn.shape, i)

    def func_subtree(childnode):
        dnode = f_tree(TreeNode(q0ten, [childnode if j == i else node.children[j] for j in range(len(node.children))]))
        # project onto the orthonormalized tree without the current subtree
        t = np.kron(q0ten.conj(), dnode.conn)
        for j in range(len(dnode.children)):
            if j == i:
                continue
            r = tree_vdot(node.children[j], dnode.children[j])
            t = single_mode_product(r.reshape((1, -1)), t, j)
            assert t.shape[j] == 1
        # upstream axis
        t = single_mode_product(np.identity(q0ten.shape[-1]).reshape((1, -1)), t, t.ndim - 1)
        dchildtensor = single_mode_product(t.reshape(q0ten.shape[i], dnode.conn.shape[i]), dnode.children[i].conn, dnode.children[i].conn.ndim - 1)
        return TreeNode(dchildtensor, dnode.children[i].children)

    if node.children[i].is_leaf:
        # right-hand side of the ordinary differential equation for the basis update
        f = lambda y: func_subtree(TreeNode(y, [])).conn
        k1 = rk4(f, node.children[i].conn @ s0, dt)
        u_hat, _ = np.linalg.qr(np.concatenate((k1, node.children[i].conn), axis=1), mode="reduced")
        m_hat = u_hat.conj().T @ node.children[i].conn
        return TreeNode(u_hat, []), m_hat
    else:
        childnode_hat, c0_hat = tree_time_step(TreeNode(node.children[i].conn @ s0, node.children[i].children), func_subtree, dt)
        q_hat, _ = np.linalg.qr(np.concatenate((
            matricize(childnode_hat.conn, childnode_hat.conn.ndim - 1).T,
            matricize(c0_hat, c0_hat.ndim - 1).T), axis=1), mode="reduced")
        childnode_hat.conn = tensorize(q_hat.T, childnode_hat.conn.shape[:-1] + (q_hat.shape[1],), childnode_hat.conn.ndim - 1)
        m_hat = tree_vdot(childnode_hat, node.children[i])
        return childnode_hat, m_hat


def flow_update_connecting_tensor(tensor0_hat, children_hat_list, f_tree, dt: float):
    """
    Augment and update and the connecting tensor of a tree node.

    'f_tree' takes a TTN with the current node as root as input and returns the (projected) right-hand side of the differential equation.

    Algorithm 6 in "Rank-adaptive time integration of tree tensor networks".
    """
    # cannot be an empty list
    assert children_hat_list
    def f(conn):
        dnode = f_tree(TreeNode(conn, children_hat_list))
        dtensor = dnode.conn
        for i in range(len(dnode.children)):
            r = tree_vdot(children_hat_list[i], dnode.children[i])
            dtensor = single_mode_product(r, dtensor, i)
        return dtensor
    return rk4(f, tensor0_hat, dt)


def tree_time_step(node: TreeNode, f_tree, dt: float):
    """
    Perform a rank-augmenting TTN integration step.

    Algorithm 4 in "Rank-adaptive time integration of tree tensor networks".
    """
    assert not node.is_leaf
    children_hat_list = []
    m_hat_list = []
    for i in range(len(node.children)):
        c_hat, m_hat = flow_update_basis(node, i, f_tree, dt)
        children_hat_list.append(c_hat)
        m_hat_list.append(m_hat)
    # augment the initial connecting tensor
    c0_hat = multi_mode_product(m_hat_list + [np.identity(node.conn.shape[-1])], node.conn)
    c1_hat = flow_update_connecting_tensor(c0_hat, children_hat_list, f_tree, dt)
    return TreeNode(c1_hat, children_hat_list), c0_hat


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


def func_vec(_, y):
    """
    Right-hand side of the reference ordinary differential equation.
    """
    return func_vec.lin @ y + (func_vec.nonlin[0] @ y.conj()) * (func_vec.nonlin[1] @ y) * (func_vec.nonlin[2] @ y)


def func_tree(y):
    """
    Evaluate the right-hand side of the ordinary differential equation,
    using a tree representation of the state.
    """
    func_tree.counter += 1
    yvec = y.to_full_tensor().reshape(-1)
    dyvec = func_vec(0, yvec)
    # represent as tree
    depth = round(np.log2(np.log2(len(dyvec))))
    return binary_tree_from_state(dyvec.reshape((-1, 1)), depth, 1e-9)

func_tree.counter = 0


def flow_ref(y0, t: float):
    """
    Reference solution of the ordinary differential equation.
    """
    sol = solve_ivp(func_vec, [0, t], y0, rtol=1e-10, atol=1e-10)
    return sol.y[:, -1]


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
    rng = np.random.default_rng(316)

    # create a tree with random tensor entries
    t0 = TreeNode(0.5 * crandn((2, 3), rng), [])
    t1 = TreeNode(0.5 * crandn((3, 3), rng), [])
    t2 = TreeNode(0.5 * crandn((2, 2), rng), [])
    t3 = TreeNode(0.5 * crandn((2, 4), rng), [])
    t4 = TreeNode(0.5 * crandn((6, 2), rng), [])
    t5 = TreeNode(0.5 * crandn((3, 3, 7), rng), [t0, t1])
    t6 = TreeNode(0.5 * crandn((2, 4, 5), rng), [t2, t3])
    t7 = TreeNode(0.5 * crandn((7, 5, 2, 5), rng), [t5, t6, t4])

    t_tensor = t7.to_full_tensor()
    print("t_tensor.shape:", t_tensor.shape)
    print("np.linalg.norm(t_tensor):", np.linalg.norm(t_tensor))
    print("t_tensor[0, 0, 0, 0, 0, 0]:", t_tensor[0, 0, 0, 0, 0, 0])

    r = t7.orthonormalize()
    print("r:", r)
    print("t7.conn.shape:", t7.conn.shape)

    t_tensor_normalized = t7.to_full_tensor()
    print("np.linalg.norm(t_tensor_normalized):", np.linalg.norm(t_tensor_normalized))
    print(f"np.linalg.norm(t_tensor_normalized) - sqrt({t7.conn.shape[-1]}):", np.linalg.norm(t_tensor_normalized) - np.sqrt(t7.conn.shape[-1]))
    err = np.linalg.norm(t_tensor - single_mode_product(r.T, t_tensor_normalized, t_tensor_normalized.ndim - 1))
    print("err:", err)

    # should be identity after orthonormalization
    d = tree_vdot(t7, t7)
    print("d:", d)
    err = np.linalg.norm(d - np.identity(t7.conn.shape[-1]))
    print("err:", err)

    t_trunc = truncate_tree(t7, 0.1)
    print("t_trunc.conn.shape:", t_trunc.conn.shape)
    err_trunc = np.linalg.norm(t_trunc.to_full_tensor() - t_tensor_normalized)
    print("err_trunc:", err_trunc)


def main2():

    # random number generator
    rng = np.random.default_rng(294)

    state = crandn(256, rng)

    tree = binary_tree_from_state(state.reshape((-1, 1)), depth=3, tol=1e-5)
    print("tree.conn.shape:", tree.conn.shape)

    err = np.linalg.norm(tree.to_full_tensor().reshape(-1) - state)
    print("err:", err)


def main3():

    # random number generator
    rng = np.random.default_rng(823)

    local_state = np.array([1j, 0.])
    y_init = generate_simple_tree(local_state, depth=3)
    print("y_init.conn.shape:", y_init.conn.shape)
    y_init_tensor = y_init.to_full_tensor()
    print("y_init_tensor.shape:", y_init_tensor.shape)
    t_test_ref = np.array([1.])
    for _ in range(8):
        t_test_ref = np.kron(t_test_ref, local_state)
    print("t_test_ref.shape:", t_test_ref.shape)
    err_t = np.linalg.norm(y_init_tensor - t_test_ref.reshape(y_init_tensor.shape))
    print("err_t:", err_t)

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

    # Hamiltonian part and a weak non-linearity;
    # results are sensitive to the strength of the non-linearity
    func_vec.lin = -1j * construct_ising_1d_hamiltonian(8, 1., 0., 1.)
    func_vec.nonlin = [0.05 * crandn((256, 256), rng) for _ in range(3)]

    # overall simulation time
    tmax = 1

    # relative truncation tolerance
    rel_tol = 5e-4

    # reference solution
    y_ref = flow_ref(y_init_tensor.reshape(-1), tmax)
    plt.plot(y_ref.real, label="Re")
    plt.plot(y_ref.imag, label="Im")
    plt.xlabel("i")
    plt.ylabel("y[i]")
    plt.title(f"reference time-evolved state at t = {tmax}")
    plt.legend()
    plt.show()

    nsteps_list = np.array([10, 20, 50, 100, 200])
    err_list = np.zeros(len(nsteps_list))
    ranks_list = []
    for i, nsteps in enumerate(nsteps_list):
        print(32 * '_')
        print("nsteps:", nsteps)
        y = y_init
        print("initial y.conn.shape:", y.conn.shape)
        dt = tmax/nsteps
        ranks = [max(y.conn.shape)]
        for _ in range(nsteps):
            y, _ = tree_time_step(y, func_tree, dt)
            y = truncate_tree(y, dt * rel_tol)
            ranks.append(max(y.conn.shape))
        print("final y.conn.shape:", y.conn.shape)
        ranks_list.append(ranks)
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

    print("func_tree.counter:", func_tree.counter)

    # visualize approximation error in dependence of the time step
    plt.loglog(tmax/nsteps_list, err_list)
    plt.xlabel(r"$\Delta t$")
    plt.ylabel("error")
    plt.title(f"tmax = {tmax}, rel_tol = {rel_tol}")
    plt.savefig("time_integration_tree1_error.pdf")
    plt.show()

    # visualize time-dependent ranks
    for i, nsteps in enumerate(nsteps_list):
        plt.plot(np.linspace(0, tmax, nsteps + 1, endpoint=True), ranks_list[i], label=f"dt = {tmax/nsteps}")
    plt.title(f"tmax = {tmax}, rel_tol = {rel_tol}")
    plt.xlabel("time")
    plt.ylabel("rank")
    plt.legend()
    plt.savefig("time_integration_tree1_ranks.pdf")
    plt.show()


if __name__ == "__main__":
    main3()
