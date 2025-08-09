"""
Rank-adaptive integrator for a tree tensor network specialized to a TTNO Hamiltonian
and attaching physical legs also to inner nodes

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
    """
    Tree node, containing a physical axis also in inner connecting tensors.

    Convention for axis ordering:
      1. axes connecting to children
      2. physical axis or axes
      3. parent axis
    """
    def __init__(self, conn, children: Sequence):
        self.conn = np.asarray(conn)
        assert self.conn.ndim >= 2
        self.children = list(children)

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
        # local physical dimension
        phys_dims = phys_dims + list(self.conn.shape[len(self.children):-1])
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


def apply_local_operator(op, psi):
    """
    Apply a local operator represented as a connecting tensor
    by contracting its physical input axis with the physical state axis
    and taking Kronecker products of virtual bond dimensions.
    """
    # operator has a physical input and output axis
    assert op.ndim == psi.ndim + 1
    nc = psi.ndim - 2  # without physical axis and parent bond
    # contract physical axes of 'op' and 'psi' and take the Kronecker product of virtual bonds
    idx_op  = tuple(range(0, 2*nc, 2)) + (2*nc, 2*nc + 1, 2*nc + 2)
    idx_psi = tuple(range(1, 2*nc, 2)) + (2*nc + 1,       2*nc + 3)
    idx_t   = tuple(range(2*nc))       + (2*nc, 2*nc + 2, 2*nc + 3)
    t = np.einsum(op, idx_op, psi, idx_psi, idx_t, optimize=True)
    assert t.ndim == 2*nc + 3
    # flatten respective virtual bonds
    t = t.reshape(tuple(t.shape[2*i] * t.shape[2*i+1] for i in range(nc)) + (t.shape[2*nc], t.shape[2*nc+1]*t.shape[2*nc+2]))
    return t


def local_operator_averages(conn_chi, conn_op, conn_psi, avg_children: list):
    """
    Evaluate the local operator average `<chi | op | psi>`
    given the averages of the connected child nodes.
    """
    t = apply_local_operator(conn_op, conn_psi)
    for i in range(len(avg_children)):
        ac = avg_children[i]
        t = single_mode_product(ac.reshape((ac.shape[0], -1)), t, i)
    t = conn_chi.reshape((-1, conn_chi.shape[-1])).conj().T @ t.reshape((-1, t.shape[-1]))
    t = t.reshape((conn_chi.shape[-1], conn_op.shape[-1], conn_psi.shape[-1]))
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
    # contract physical dimensions and interleave remaining dimensions
    assert len(chi.children) == len(op.children) and len(psi.children) == len(op.children)
    nc = len(chi.children)
    avg_children = []
    for i in range(nc):
        avg_children.append(tree_operator_averages(chi.children[i], op.children[i], psi.children[i]))
    avg_conn = local_operator_averages(chi.conn, op.conn, psi.conn, [ac.conn for ac in avg_children])
    return TreeNode(avg_conn, avg_children)


def flatten_local_dimensions(local_dims) -> tuple:
    """
    Flatten a nested tuple of local dimensions.
    """
    if isinstance(local_dims, int):
        return (local_dims,)
    return sum([flatten_local_dimensions(ld) for ld in local_dims], ())


def multiply_local_dimensions(local_dims: tuple) -> int:
    """
    Compute the overall product of local dimensions.
    """
    return np.prod(flatten_local_dimensions(local_dims), dtype=int)


def square_local_dimensions(local_dims: tuple) -> tuple:
    """
    Return a new tuple of nested local dimensions with entrywise squared dimensions.
    """
    if isinstance(local_dims, int):
        return local_dims**2
    return tuple(square_local_dimensions(ld) for ld in local_dims)


def tree_from_state(state, local_dims, tol: float):
    """
    Construct a tree approximating a given state,
    with `local_dims` a recursively nested tuple of the form
    (dims_subtree_0, ..., dims_subtree_{n-1}, dim_current_node).
    """
    state = np.asarray(state)
    assert state.ndim == 2
    if len(local_dims) == 1:
        assert state.shape[0] == local_dims[0]
        return TreeNode(state, [])
    child_dims = tuple([multiply_local_dimensions(ld) for ld in local_dims[:-1]])
    u_list, c, _ = higher_order_svd(state.reshape(child_dims + (local_dims[-1], state.shape[1])), tol)
    # re-absorb unitaries for physical axis and parent bond
    c = single_mode_product(u_list[-1], c, c.ndim - 1)
    c = single_mode_product(u_list[-2], c, c.ndim - 2)
    # recursive function call on subtrees
    return TreeNode(c, [tree_from_state(u_list[i], local_dims[i], tol) for i in range(len(local_dims) - 1)])


def interleave_local_operator_axes(op, local_dims):
    """
    Interleave the local output and input axes of a linear operator.
    """
    op = np.asarray(op)
    local_dims = flatten_local_dimensions(local_dims)
    assert op.ndim == 2
    assert op.shape == 2 * (np.prod(local_dims, dtype=int),)
    nsites = len(local_dims)
    perm = sum(zip(range(nsites), range(nsites, 2*nsites)), ())
    return op.reshape(local_dims + local_dims).transpose(perm)


def separate_local_operator_axes(op):
    """
    Separate the local output and input axes of a linear operator,
    returning its matrix representation.
    """
    op = np.asarray(op)
    nsites = op.ndim // 2
    assert op.shape[::2] == op.shape[1::2]
    local_dims = op.shape[::2]
    perm = list(range(0, 2*nsites, 2)) + list(range(1, 2*nsites, 2))
    return op.transpose(perm).reshape(2 * (np.prod(local_dims, dtype=int),))


def reshape_nodes_as_operators(node: TreeNode):
    """
    Reshape the tensors of a tree to physical dimensions of the form `(d, d)`.
    """
    d = int(np.sqrt(node.conn.shape[-2]))
    assert node.conn.shape[-2] == d**2
    if node.is_leaf:
        return TreeNode(node.conn.reshape((d, d, node.conn.shape[-1])), [])
    return TreeNode(node.conn.reshape(node.conn.shape[:-2] + (d, d, node.conn.shape[-1])),
                    [reshape_nodes_as_operators(c) for c in node.children])


def tree_from_operator(op, local_dims, tol: float):
    """
    Construct a binary tree approximating a given linear operator,
    with `local_dims` a recursively nested tuple of the form
    (dims_subtree_0, ..., dims_subtree_{n-1}, dim_current_node).
    """
    op_interleaved = interleave_local_operator_axes(op, flatten_local_dimensions(local_dims))
    tree = tree_from_state(op_interleaved.reshape((-1, 1)), square_local_dimensions(local_dims), tol)
    return reshape_nodes_as_operators(tree)


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
        assert len(y1.children) == len(state.children)
        assert len(y1.children) == len(m_hat_children)
        m_hat = state.conn
        for i in range(len(y1.children)):
            m_hat = single_mode_product(m_hat_children[i], m_hat, i)
        m_hat = y1.conn.reshape((-1, y1.conn.shape[-1])).conj().T @ m_hat.reshape((-1, m_hat.shape[-1]))
        # compute new averages (expectation values)
        assert len(avg_hat_children) == len(hamiltonian.children)
        avg_hat = local_operator_averages(y1.conn, hamiltonian.conn, y1.conn, [ac.conn for ac in avg_hat_children])
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
        t = apply_local_operator(hamiltonian.conn, c)
        for i in range(len(hamiltonian.children)):
            ac = avg_hat_children[i].conn
            t = single_mode_product(ac.reshape((ac.shape[0], -1)), t, i)
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
    env = apply_local_operator(hamiltonian.conn, q0ten)
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
    c0_hat = multi_mode_product(m_hat_children + [np.identity(state.conn.shape[-2]), np.identity(state.conn.shape[-1])], state.conn)
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

    # tree overlap and normalization

    # random number generator
    rng = np.random.default_rng(410)

    # create a tree with random tensor entries
    t0 = TreeNode(0.5 * crandn((2, 3), rng), [])
    t1 = TreeNode(0.5 * crandn((3, 3), rng), [])
    t2 = TreeNode(0.5 * crandn((2, 2), rng), [])
    t3 = TreeNode(0.5 * crandn((2, 4), rng), [])
    t4 = TreeNode(0.5 * crandn((6, 2), rng), [])
    t5 = TreeNode(0.5 * crandn((3, 3, 2, 7), rng), [t0, t1])
    t6 = TreeNode(0.5 * crandn((2, 4, 3, 5), rng), [t2, t3])
    t7 = TreeNode(0.5 * crandn((7, 5, 2, 4, 5), rng), [t5, t6, t4])

    t_tensor = t7.to_full_tensor()
    print("t_tensor.ndim:", t_tensor.ndim)
    print("t_tensor.shape:", t_tensor.shape)
    print("t_tensor.size:", t_tensor.size)
    print("np.linalg.norm(t_tensor):", np.linalg.norm(t_tensor))
    print("t_tensor[0, 0, 0, 0, 0, 0, 0, 0, 0]:", t_tensor[0, 0, 0, 0, 0, 0, 0, 0, 0])

    # create another tree with random tensor entries
    s0 = TreeNode(0.5 * crandn((2, 2), rng), [])
    s1 = TreeNode(0.5 * crandn((3, 3), rng), [])
    s2 = TreeNode(0.5 * crandn((2, 2), rng), [])
    s3 = TreeNode(0.5 * crandn((2, 4), rng), [])
    s4 = TreeNode(0.5 * crandn((6, 3), rng), [])
    s5 = TreeNode(0.5 * crandn((2, 3, 2, 7), rng), [s0, s1])
    s6 = TreeNode(0.5 * crandn((2, 4, 3, 5), rng), [s2, s3])
    s7 = TreeNode(0.5 * crandn((7, 5, 3, 4, 5), rng), [s5, s6, s4])

    s_tensor = s7.to_full_tensor()
    print("s_tensor.ndim:", s_tensor.ndim)

    overlap = tree_vdot(t7, s7)
    print("overlap.shape:", overlap.shape)
    overlap_ref = t_tensor.reshape((-1, t_tensor.shape[-1])).conj().T @ s_tensor.reshape((-1, s_tensor.shape[-1]))
    err_overlap = np.linalg.norm(overlap - overlap_ref)
    print("err_overlap:", err_overlap)

    r = t7.orthonormalize()
    print("r.shape:", r.shape)
    print("t7.conn.shape:", t7.conn.shape)

    t_tensor_normalized = t7.to_full_tensor()
    print("np.linalg.norm(t_tensor_normalized):", np.linalg.norm(t_tensor_normalized))
    print(f"np.linalg.norm(t_tensor_normalized) - sqrt({t7.conn.shape[-1]}):", np.linalg.norm(t_tensor_normalized) - np.sqrt(t7.conn.shape[-1]))
    err_norm = np.linalg.norm(t_tensor - single_mode_product(r.T, t_tensor_normalized, t_tensor_normalized.ndim - 1))
    print("err_norm:", err_norm)

    # should be identity after orthonormalization
    d = tree_vdot(t7, t7)
    err_id = np.linalg.norm(d - np.identity(t7.conn.shape[-1]))
    print("err_id:", err_id)

    t_trunc = truncate_tree(t7, 0.2)
    print("t_trunc.conn.shape:", t_trunc.conn.shape)
    err_trunc = np.linalg.norm(t_trunc.to_full_tensor() - t_tensor_normalized)
    print("err_trunc:", err_trunc)


def main2():

    # test tree generation from a state

    # random number generator
    rng = np.random.default_rng(126)

    local_dims = ((1,), ((4,), (6,), 5), ((3,), 7), 2)
    print("multiply_local_dimensions(local_dims):", multiply_local_dimensions(local_dims))

    state = 0.01 * crandn((multiply_local_dimensions(local_dims), 3), rng)
    print("np.linalg.norm(state):", np.linalg.norm(state))
    t = tree_from_state(state, local_dims, 1e-8)
    print("t.conn.shape:", t.conn.shape)

    t_tensor = t.to_full_tensor()
    print("t_tensor.shape:", t_tensor.shape)
    err = np.linalg.norm(state.reshape(t_tensor.shape) - t_tensor)
    print("err:", err)


def main3():

    # internal virtual bond dimensions should be 1 for a product state

    d = 2
    nsites = 7

    local_dims = ((d,), ((d,), (d,), d), ((d,), d), d)
    print("multiply_local_dimensions(local_dims):", multiply_local_dimensions(local_dims))
    print("d**nsites:", d**nsites)

    # product state
    local_state = np.array([0.5, 0.5j*np.sqrt(3)])
    state = np.array([1.])
    for _ in range(nsites):
        state = np.kron(state, local_state)
    print("state.shape:", state.shape)
    print("np.linalg.norm(state):", np.linalg.norm(state))

    t = tree_from_state(state.reshape((-1, 1)), local_dims, 1e-8)
    print("t.conn.shape:", t.conn.shape)
    print("t.children[1].conn.shape:", t.children[1].conn.shape)

    t_tensor = t.to_full_tensor()
    print("t_tensor.shape:", t_tensor.shape)
    err = np.linalg.norm(state.reshape(t_tensor.shape) - t_tensor)
    print("err:", err)


def main4():

    # operator averages

    # random number generator
    rng = np.random.default_rng(452)

    local_dims = (((3,), (5,), 2), (1,), ((3,), 7), 2)
    d_full = multiply_local_dimensions(local_dims)
    print("d_full:", d_full)

    op = 0.25 * crandn(2 * (d_full,), rng)

    op_interleaved = interleave_local_operator_axes(op, local_dims)
    print("op_interleaved.shape:", op_interleaved.shape)

    op2 = separate_local_operator_axes(op_interleaved)
    err_interleave = np.linalg.norm(op2 - op)
    print("err_interleave:", err_interleave)

    op_tree = tree_from_operator(op, local_dims, tol=1e-8)
    op_tensor = op_tree.to_full_tensor()
    print("op_tensor.shape:", op_tensor.shape)
    print("op_tree.conn.shape:", op_tree.conn.shape)
    err_op = np.linalg.norm(op_interleaved - op_tensor.reshape(op_interleaved.shape))
    print("err_op:", err_op)

    psi = 0.25 * crandn(d_full, rng)
    print("np.linalg.norm(psi):", np.linalg.norm(psi))
    psi_tree = tree_from_state(psi.reshape((-1, 1)), local_dims, tol=1e-8)
    err_state = np.linalg.norm(psi_tree.to_full_tensor().reshape(-1) - psi)
    print("err_state:", err_state)

    chi = 0.25 * crandn(d_full, rng)
    chi_tree = tree_from_state(chi.reshape((-1, 1)), local_dims, tol=1e-8)

    avg = tree_operator_averages(chi_tree, op_tree, psi_tree)
    print("avg.conn:", avg.conn)
    print("avg.conn.shape:", avg.conn.shape)
    print("avg.children[0].conn.shape:", avg.children[0].conn.shape)

    avg_ref = np.vdot(chi, op @ psi)
    print("avg_ref:", avg_ref)
    err_avg = abs(avg.conn[0, 0, 0] - avg_ref)
    print("err_avg:", err_avg)


def main5():

    # overall integration demo

    d = 2
    nsites = 8
    local_dims = (((d,), ((d,), d), d), ((d,), (d,), (d,), 1), d)
    d_full = multiply_local_dimensions(local_dims)
    print("d_full:", d_full)

    y_init_vec_constr = np.zeros(d**nsites, dtype=complex)
    y_init_vec_constr[0] = 1
    y_init = tree_from_state(y_init_vec_constr.reshape((-1, 1)), local_dims, 1e-8)
    print("y_init.conn.shape:", y_init.conn.shape)
    y_init_tensor = y_init.to_full_tensor()
    print("y_init_tensor.shape:", y_init_tensor.shape)
    y_init_vec = y_init_tensor.reshape(-1)
    print("np.linalg.norm(y_init_vec):", np.linalg.norm(y_init_vec))
    err_init = np.linalg.norm(y_init_vec - y_init_vec_constr)
    print("err_init:", err_init)

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
    hamiltonian = tree_from_operator(hamiltonian_matrix, local_dims, tol=1e-8)
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
    plt.savefig("time_integration_tree3_error.pdf")
    plt.show()

    # visualize time-dependent ranks
    for i, nsteps in enumerate(nsteps_list):
        plt.plot(np.linspace(0, tmax, nsteps + 1, endpoint=True), ranks_list[i], label=f"dt = {tmax/nsteps}")
    plt.title(f"tmax = {tmax}, rel_tol = {rel_tol}")
    plt.xlabel("time")
    plt.ylabel("rank")
    plt.legend()
    plt.savefig("time_integration_tree3_ranks.pdf")
    plt.show()

    # visualize time-dependent energy differences
    for i, nsteps in enumerate(nsteps_list):
        plt.semilogy(np.linspace(0, tmax, nsteps + 1, endpoint=True), ediff_list[i], label=f"dt = {tmax/nsteps}")
    plt.title(f"tmax = {tmax}, rel_tol = {rel_tol}")
    plt.xlabel("time")
    plt.ylabel("deviation from initial energy")
    plt.legend()
    plt.savefig("time_integration_tree3_energy.pdf")
    plt.show()


if __name__ == "__main__":
    main5()
