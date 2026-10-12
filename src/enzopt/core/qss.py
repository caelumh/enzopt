"""Quasi-steady-state structure of an enzyme network (F-13: the PINN representation for any network).

The enzyme forms are fast and the small molecules slow (Et ≪ substrate). For networks whose steps are first order
in the enzyme forms (at most one enzyme form on each side, and either both sides or neither), the enzyme forms obey
a linear system at given small-molecule concentrations: dφ/dt = L(x) φ, where each step X + ligands ⇌ Y + ligands is
an edge X → Y with pseudo-first-order rate kf·Π(ligands) (and Y → X with kr·Π(ligands)). The quasi-steady distribution
is the null vector of L with Σφ = 1 (King–Altman for a ring; any graph here).

Inactivation steps: an enzyme form that no step leaves (``X -> ED``, ED appearing only on the right of one-way steps)
is a dead form. Its steps are kept out of the quasi-steady graph (``inact_edges``); the dead form is listed last in
``enzyme`` and the PINN carries the active fraction a(t) as one more slow variable, d ln a/dt = −Σ k·[ligands]·φ_X.

Several enzymes: the active forms may fall into several connected graphs (e.g. two enzymes of a cascade). Each graph
is one enzyme; its free form is the first of its forms in ``enzyme_species``, and its amount is Et for the graph of the
first form and the free form's initial value for the others (``totals``). φ is then Σ_c f_c φ_c, each φ_c summing to 1
over its graph and f_c = amount_c / Σ amounts (``qss_fractions(..., fractions)``). Inactivation steps need one enzyme.

Slow coordinates: the net small-molecule changes that leave the enzyme forms unchanged span a space of dimension d
(1 for one overall reaction; more with competing substrates or enzyme-free background steps). The small molecules are
x = x0 + V ξ + δ, with ξ the d reaction extents, V the overall reactions (then the background steps) and δ the
correction that puts the enzyme-bound amounts back into the conservation laws: C_s δ = −Et C_e (φ − e0).

Only numpy (KIMM-CAD compatible); the PINN evaluates the same structure with torch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .network import Network
from .thermo import TOL, _null_space, _rank, species_vector


class QssError(ValueError):
    """The network has no quasi-steady representation (message is shown to the user)."""


@dataclass
class Edge:
    step: int  # 0-based step index
    a: int  # enzyme form index (into ``enzyme``) on the left
    b: int  # on the right
    lf: list  # [(small index, order)] ligands on the left (forward rate)
    lr: list  # ligands on the right (reverse rate)
    reversible: bool


@dataclass
class QssStructure:
    species: list[str]
    small: list[str]
    enzyme: list[str]  # first = free enzyme
    small_idx: list[int]  # indices into species
    enzyme_idx: list[int]
    edges: list[Edge]
    background: list[int]  # enzyme-free steps (0-based)
    reactants: list[list]  # per step: [(species index, order)] on the left
    products: list[list]  # per step: on the right
    N_s: np.ndarray  # (n_s × n_steps) net change of the small molecules per step
    N_e: np.ndarray  # (n_e × n_steps)
    V: np.ndarray  # (n_s × d) slow basis
    W: np.ndarray  # (d × n_steps): dξ/dt = W r (projection of the small-molecule change onto V)
    labels: list[str]  # name of each slow coordinate
    Mcorr: np.ndarray  # (n_s × n_e): δ = Et · Mcorr (φ − e0)
    inact_edges: list = None  # steps into the dead form (kept out of ``edges``)
    n_dead: int = 0  # dead enzyme forms, listed last in ``enzyme`` (at most one)
    comp: list = None  # per active form: the index of its enzyme (connected graph); None = one enzyme
    refs: list = None  # per enzyme: the index (into ``enzyme``) of its free form

    @property
    def n_enzymes(self) -> int:
        return len(self.refs) if self.refs else 1

    def totals(self, initial: dict, Et: float) -> np.ndarray:
        """Amount of each enzyme in one experiment: Et for the first, the free form's initial value for the others."""
        if self.n_enzymes == 1:
            return np.array([float(Et)])
        return np.array([float(Et)] + [float(initial.get(self.enzyme[r], 0.0)) for r in self.refs[1:]])

    @property
    def d(self) -> int:
        return self.V.shape[1]

    @property
    def n_active(self) -> int:
        """Enzyme forms in the quasi-steady graph (all but the dead form)."""
        return len(self.enzyme) - self.n_dead

    def ranges(self, x0_small: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per slow coordinate the extent range [lo, hi] that keeps the small molecules ≥ 0 with the other
        coordinates at 0 (exact for one coordinate; with several only a scale, see ``pinn.general``)."""
        lo = np.zeros(self.d)
        hi = np.zeros(self.d)
        for k in range(self.d):
            v = self.V[:, k]
            neg = [x0_small[i] / -v[i] for i in range(len(v)) if v[i] < -TOL]
            pos = [-x0_small[i] / v[i] for i in range(len(v)) if v[i] > TOL]
            hi[k] = min(neg) if neg else 0.0
            lo[k] = max(pos) if pos else 0.0
            if hi[k] < lo[k]:
                hi[k] = lo[k]
        return lo, hi

    def small_scale(self, x0_list: list[np.ndarray]) -> np.ndarray:
        """Typical size of each small molecule over the experiments (input scaling of auxiliary networks)."""
        out = np.full(len(self.small), 1e-6)
        for x0 in x0_list:
            lo, hi = self.ranges(x0)
            reach = np.abs(self.V) @ np.maximum(np.abs(lo), np.abs(hi))
            out = np.maximum(out, x0 + reach)
        return out


def structure(network: Network, overall: list[str] | None = None) -> QssStructure:
    """Build the quasi-steady structure; raises QssError when a step is not first order in the enzyme forms."""
    ens = list(network.enzyme_species)
    if not ens:
        raise QssError("enzyme_species(효소 형태)가 없음 — 준정상 표현은 효소 형태가 필요함")
    # dead forms: no step leaves them (only on the right of one-way steps) → listed last, outside the graph
    leaves = set()
    for st in network.steps:
        leaves.update(s for s in st.lhs if s in ens)
        if st.reversible:
            leaves.update(s for s in st.rhs if s in ens)
    dead = [s for s in ens[1:] if s not in leaves]
    if len(dead) > 1:
        raise QssError(f"효소가 빠져나오지 못하는 형태가 둘 이상({', '.join(dead)}): 실활 형태는 하나만")
    ens = [s for s in ens if s not in dead] + dead
    species = list(network.species)
    idx = network.index()
    small = [s for s in species if s not in ens]
    s_pos = {s: i for i, s in enumerate(small)}
    e_pos = {s: i for i, s in enumerate(ens)}
    edges, background, reac, prod = [], [], [], []
    n_steps = len(network.steps)
    N = np.zeros((len(species), n_steps))
    for j, st in enumerate(network.steps):
        reac.append([(idx[s], int(c)) for s, c in st.lhs.items()])
        prod.append([(idx[s], int(c)) for s, c in st.rhs.items()])
        for s, c in st.lhs.items():
            N[idx[s], j] -= c
        for s, c in st.rhs.items():
            N[idx[s], j] += c
        el = [(s, c) for s, c in st.lhs.items() if s in e_pos]
        er = [(s, c) for s, c in st.rhs.items() if s in e_pos]
        if not el and not er:
            background.append(j)
            continue
        if len(el) != 1 or len(er) != 1 or el[0][1] != 1 or er[0][1] != 1:
            raise QssError(
                f"단계 {j + 1} '{st}': 양쪽에 효소 형태가 하나씩(계수 1) 있어야 함 — 준정상 표현은 효소에 1차인 단계만"
            )
        lf = [(s_pos[s], int(c)) for s, c in st.lhs.items() if s in s_pos]
        lr = [(s_pos[s], int(c)) for s, c in st.rhs.items() if s in s_pos]
        edges.append(Edge(j, e_pos[el[0][0]], e_pos[er[0][0]], lf, lr, st.reversible))
    small_idx = [idx[s] for s in small]
    enzyme_idx = [idx[s] for s in ens]
    N_s, N_e = N[small_idx], N[enzyme_idx]
    n_act = len(ens) - len(dead)
    inact = [e for e in edges if e.b >= n_act]
    edges = [e for e in edges if e.b < n_act]
    for e in inact:
        if np.any(N_s[:, e.step]):
            raise QssError(f"단계 {e.step + 1} '{network.steps[e.step]}': 실활 단계는 작은 분자를 바꾸지 않아야 함")
    # enzymes = connected graphs of the active forms; each free form (first of its graph) must reach every form of its
    # graph and be reached back (strongly connected), otherwise the quasi-steady distribution is not defined
    comp = _components(n_act, edges)
    refs = [min(i for i in range(n_act) if comp[i] == c) for c in range(max(comp) + 1)]
    if len(refs) > 1 and dead:
        raise QssError("효소가 둘 이상인 반응망의 실활 단계(X -> ED)는 아직 다루지 않음")
    reach = [False] * n_act
    for r in refs:
        for i, ok in enumerate(_strongly_connected_to(n_act, edges, r)):
            reach[i] = reach[i] or (ok and comp[i] == comp[r])
    if not all(reach):
        bad = [ens[i] for i, ok in enumerate(reach) if not ok]
        raise QssError(f"효소 형태 {', '.join(bad)}: 자유 효소와 오가는 경로가 없음(준정상 분포가 정해지지 않음)")
    # slow space: {N_s y : N_e y = 0}
    Zy = _null_space(N_e).T if N_e.size else np.eye(n_steps)
    S = N_s @ Zy if Zy.size else np.zeros((len(small), 0))
    d = _rank(S)
    if d == 0:
        raise QssError("느린 변수가 없음(작은 분자가 바뀌는 전체 반응이 없음)")
    cols, labels = [], []

    def add(v, label):
        cand = cols + [v]
        M = np.column_stack(cand)
        if _rank(M) > len(cols) and _rank(np.column_stack([S, v])) == d:
            cols.append(v)
            labels.append(label)

    for o in overall or []:
        add(species_vector(network, o)[small_idx], str(o))
    for j in background:
        add(N_s[:, j], str(network.steps[j]))
    if len(cols) < d:  # complete with an orthonormal basis of the remainder
        u, s, _ = np.linalg.svd(S)
        for k in range(d):
            if len(cols) == d:
                break
            v = u[:, k]
            v = v / np.abs(v).max()
            if v[np.argmax(np.abs(v) > TOL)] < 0:
                v = -v
            add(v, f"slow{len(cols) + 1}")
    if len(cols) < d:
        raise QssError("느린 변수 기저를 만들지 못함")
    V = np.column_stack(cols)
    W = np.linalg.pinv(V) @ N_s
    W[np.abs(W) < 1e-12] = 0.0
    # conservation laws C (rows, C N = 0) → bound-amount correction
    C = _null_space(N.T)
    C_s, C_e = C[:, small_idx], C[:, enzyme_idx]
    Mcorr = -np.linalg.pinv(C_s) @ C_e if C.size else np.zeros((len(small), len(ens)))
    Mcorr[np.abs(Mcorr) < 1e-12] = 0.0
    return QssStructure(
        species=species,
        small=small,
        enzyme=ens,
        small_idx=small_idx,
        enzyme_idx=enzyme_idx,
        edges=edges,
        background=background,
        reactants=reac,
        products=prod,
        N_s=N_s,
        N_e=N_e,
        V=V,
        W=W,
        labels=labels,
        Mcorr=Mcorr,
        inact_edges=inact,
        n_dead=len(dead),
        comp=comp if len(refs) > 1 else None,
        refs=refs if len(refs) > 1 else None,
    )


def _components(n: int, edges: list[Edge]) -> list[int]:
    """Connected graph (undirected) of each active form, numbered in order of first appearance."""
    nb = {i: set() for i in range(n)}
    for e in edges:
        nb[e.a].add(e.b)
        nb[e.b].add(e.a)
    comp = [-1] * n
    c = 0
    for i in range(n):
        if comp[i] >= 0:
            continue
        stack, comp[i] = [i], c
        while stack:
            x = stack.pop()
            for y in nb[x]:
                if comp[y] < 0:
                    comp[y] = c
                    stack.append(y)
        c += 1
    return comp


def _strongly_connected_to(n: int, edges: list[Edge], root: int = 0) -> list[bool]:
    """Forms that can reach form ``root`` and be reached from it."""
    fwd = {i: set() for i in range(n)}
    for e in edges:
        fwd[e.a].add(e.b)
        if e.reversible:
            fwd[e.b].add(e.a)
    bwd = {i: set() for i in range(n)}
    for a, bs in fwd.items():
        for b in bs:
            bwd[b].add(a)

    def visit(start, g):
        seen, stack = {start}, [start]
        while stack:
            x = stack.pop()
            for y in g[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        return seen

    f, b = visit(root, fwd), visit(root, bwd)
    return [i in f and i in b for i in range(n)]


def qss_fractions(
    st: QssStructure, x_small: np.ndarray, kf: np.ndarray, kr: np.ndarray, fractions: np.ndarray | None = None
) -> np.ndarray:
    """Quasi-steady distribution φ of the active enzyme (sums to 1; 0 for a dead form) at one small-molecule state
    (numpy reference). With several enzymes, ``fractions`` (amount of each ÷ total, ``QssStructure.totals``) weights
    the distribution of each enzyme's graph."""
    n = st.n_active
    if st.n_enzymes > 1:
        f = np.asarray(fractions if fractions is not None else np.full(st.n_enzymes, 1.0 / st.n_enzymes), float)
        out = np.zeros(len(st.enzyme))
        for c, r in enumerate(st.refs):
            members = [i for i in range(n) if st.comp[i] == c]
            sub = [e for e in st.edges if e.a in members]
            pos = {m: j for j, m in enumerate([r] + [m for m in members if m != r])}
            part = QssStructure(**{**st.__dict__, "edges": [_reindex(e, pos) for e in sub], "comp": None,
                                   "refs": None, "n_dead": 0, "enzyme": [st.enzyme[m] for m in pos]})  # fmt: skip
            phi_c = qss_fractions(part, x_small, kf, kr)
            for m, j in pos.items():
                out[m] = f[c] * phi_c[j]
        return out
    L = np.zeros((n, n))
    for e in st.edges:
        wf = kf[e.step] * np.prod([x_small[i] ** o for i, o in e.lf]) if e.lf else kf[e.step]
        L[e.b, e.a] += wf
        L[e.a, e.a] -= wf
        if e.reversible:
            wr = kr[e.step] * np.prod([x_small[i] ** o for i, o in e.lr]) if e.lr else kr[e.step]
            L[e.a, e.b] += wr
            L[e.b, e.b] -= wr
    A = -L[1:, 1:]
    b = L[1:, 0]
    psi = np.linalg.solve(A, b) if n > 1 else np.zeros(0)
    phi = np.concatenate([[1.0], psi])
    return np.concatenate([phi / phi.sum(), np.zeros(st.n_dead)])


def _reindex(e: Edge, pos: dict) -> Edge:
    return Edge(e.step, pos[e.a], pos[e.b], e.lf, e.lr, e.reversible)
