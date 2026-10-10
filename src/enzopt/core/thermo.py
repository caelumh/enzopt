"""Thermodynamic constraints on rate constants (Haldane–Wegscheider), derived from the network (F-13 G4).

Each reversible step j has log10 K_j = log10 kf_j − log10 kr_j. Detailed balance requires log K = Nᵀ ℓ for some ℓ (one
number per species), and the listed overall reactions fix ν_o · ℓ = log10 K′_o. Equivalently, for every combination of
reversible steps y whose net change N y = Σ_o c_o ν_o is a combination of overall reactions,

    y · log K = c · log10 K′            (internal cycles: c = 0, product of step constants = 1)

These are linear in log10 rate constants, so one k₋ per independent constraint is solved from the others. The choice
of dependent steps is made here (``prefer`` first, then the last steps): for LDH it is step 5, the old
``haldane_step``. The parameter vector of the identification is

    θ = log10 [kf (all steps), kr (reversible, not dependent), K′ (overall reactions that constrain a cycle)]

and log10 kr_dependent = A θ. Only numpy (KIMM-CAD compatible).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .network import Network, Step

TOL = 1e-9


class ThermoError(ValueError):
    """The overall reactions do not fit the network (message is shown to the user)."""


def _null_space(M: np.ndarray, tol: float = TOL) -> np.ndarray:
    """Orthonormal basis of the null space of M as rows (k × M.shape[1])."""
    if M.size == 0:
        return np.eye(M.shape[1])
    u, s, vt = np.linalg.svd(M)
    scale = s.max() if s.size else 1.0
    rank = int((s > tol * max(1.0, scale)).sum())
    return vt[rank:]


def _rank(M: np.ndarray, tol: float = TOL) -> int:
    if M.size == 0:
        return 0
    s = np.linalg.svd(M, compute_uv=False)
    return int((s > tol * max(1.0, s.max())).sum())


def species_vector(network: Network, text: str) -> np.ndarray:
    """Net change (rhs − lhs) of a reaction written like a step, over ``network.species``."""
    st = Step.parse(text)
    idx = network.index()
    v = np.zeros(len(network.species))
    for side, sign in ((st.lhs, -1.0), (st.rhs, 1.0)):
        for s, c in side.items():
            if s not in idx:
                raise ThermoError(f"전체 반응 '{text}'의 종 '{s}'이 species 목록에 없음")
            v[idx[s]] += sign * c
    return v


@dataclass
class ThermoLayout:
    """θ layout and the linear map to the dependent k₋ (see module docstring)."""

    n_steps: int
    reversible: list[bool]
    overall: list[str]
    dependent: list[int]  # 0-based step indices whose k₋ follows from the others
    free_kr: list[int]  # 0-based reversible steps whose k₋ is a parameter (θ order)
    overall_used: list[int]  # indices into ``overall`` whose K′ is a parameter (θ order)
    A: np.ndarray  # (len(dependent), n_theta): log10 kr_dependent = A @ θ
    constraints: np.ndarray = field(repr=False, default=None)  # rows (y over reversible steps, c over overall)
    rev_index: list[int] = field(default_factory=list)  # 0-based step index of each reversible step

    @property
    def n_theta(self) -> int:
        return self.n_steps + len(self.free_kr) + len(self.overall_used)

    @property
    def n_free_kr(self) -> int:
        return len(self.free_kr)

    def rates(self, theta) -> tuple[np.ndarray, np.ndarray]:
        """θ → (kf, kr) for all steps; kr = 0 for one-way steps."""
        th = np.asarray(theta, float)
        n = self.n_steps
        kf = 10.0 ** th[:n]
        kr = np.zeros(n)
        for j, i in enumerate(self.free_kr):
            kr[i] = 10.0 ** th[n + j]
        if self.dependent:
            dep = self.A @ th[: self.n_theta]
            for j, i in enumerate(self.dependent):
                kr[i] = 10.0 ** dep[j]
        return kf, kr

    def theta(self, kf, kr, Kp: list[float] | None = None) -> np.ndarray:
        """(kf, kr, K′ of the used overall reactions) → θ (dependent k₋ are dropped)."""
        kf = np.asarray(kf, float)
        kr = np.asarray(kr, float)
        vals = list(np.log10(kf)) + [np.log10(kr[i]) for i in self.free_kr]
        Kp = list(Kp or [])
        vals += [np.log10(Kp[o]) for o in self.overall_used]
        return np.array(vals, float)

    def residuals(self, kf, kr, Kp: list[float]) -> np.ndarray:
        """y · log10 K − c · log10 K′ for every constraint (0 when the rate constants are consistent)."""
        if self.constraints is None or not len(self.constraints):
            return np.zeros(0)
        kf = np.asarray(kf, float)
        kr = np.asarray(kr, float)
        logK = np.array([np.log10(kf[i]) - np.log10(kr[i]) for i in self.rev_index])
        m = len(self.rev_index)
        kap = np.array([np.log10(Kp[o]) if Kp[o] else 0.0 for o in range(len(self.overall))])
        Y, C = self.constraints[:, :m], self.constraints[:, m:]
        return Y @ logK - C @ kap

    def describe(self) -> str:
        if not self.dependent:
            return "열역학 제약 없음(가역 순환 없음)"
        steps = ", ".join(str(i + 1) for i in self.dependent)
        return f"종속 k₋: 단계 {steps} (순환 {len(self.dependent)}개에서 계산)"


def layout(network: Network, overall: list[str] | None = None, prefer: list[int] | None = None) -> ThermoLayout:
    """Constraints and θ layout for ``network`` with the overall reactions ``overall`` (step-like strings).

    ``prefer``: 1-based step numbers to make dependent first (e.g. the old ``haldane_step``)."""
    overall = [str(o) for o in (overall or [])]
    n = len(network.steps)
    rev = [st.reversible for st in network.steps]
    rev_index = [i for i in range(n) if rev[i]]
    m = len(rev_index)
    N = np.zeros((len(network.species), n))
    idx = network.index()
    for j, st in enumerate(network.steps):
        for s, c in st.lhs.items():
            N[idx[s], j] -= c
        for s, c in st.rhs.items():
            N[idx[s], j] += c
    O = np.column_stack([species_vector(network, o) for o in overall]) if overall else np.zeros((N.shape[0], 0))
    if overall and _rank(O) < len(overall):
        raise ThermoError("전체 반응들이 서로 독립이 아님(한 반응이 다른 반응들의 합)")
    M = np.hstack([N[:, rev_index], -O]) if m or overall else np.zeros((N.shape[0], 0))
    Z = _null_space(M) if M.shape[1] else np.zeros((0, 0))
    if len(Z):
        Y = Z[:, :m]
        if _rank(Y) < len(Z):  # some combination is a pure relation between overall reactions
            raise ThermoError("전체 반응들이 반응망의 단계와 맞지 않음(서로 종속)")
    else:
        Y = np.zeros((0, m))
    q = len(Z)
    # dependent steps: preferred ones first, then from the last reversible step backwards
    order = [rev_index.index(s - 1) for s in (prefer or []) if (s - 1) in rev_index]
    order += [k for k in reversed(range(m)) if k not in order]
    D: list[int] = []
    for k in order:
        if len(D) == q:
            break
        if _rank(Y[:, D + [k]]) > len(D):
            D.append(k)
    if len(D) < q:
        raise ThermoError("열역학 제약에서 종속 단계를 고르지 못함")
    D.sort()
    F = [k for k in range(m) if k not in D]
    C = Z[:, m:] if len(Z) else np.zeros((0, len(overall)))
    used = [o for o in range(len(overall)) if len(C) and np.abs(C[:, o]).max() > TOL]
    free_kr = [rev_index[k] for k in F]
    n_theta = n + len(free_kr) + len(used)
    A = np.zeros((q, n_theta))
    if q:
        YD_inv = np.linalg.inv(Y[:, D])
        G = YD_inv @ Y[:, F]  # (q × |F|)
        H = YD_inv @ C[:, used]  # (q × |used|)
        for a, k in enumerate(D):
            A[a, rev_index[k]] += 1.0  # log kf of the dependent step
        for b, k in enumerate(F):
            A[:, rev_index[k]] += G[:, b]  # + G (log kf_F − log kr_F)
            A[:, n + b] -= G[:, b]
        for c, _o in enumerate(used):
            A[:, n + len(free_kr) + c] -= H[:, c]
        A[np.abs(A - np.round(A)) < 1e-9] = np.round(A[np.abs(A - np.round(A)) < 1e-9])
    return ThermoLayout(
        n_steps=n,
        reversible=rev,
        overall=overall,
        dependent=[rev_index[k] for k in D],
        free_kr=free_kr,
        overall_used=used,
        A=A,
        constraints=Z if len(Z) else np.zeros((0, m + len(overall))),
        rev_index=rev_index,
    )


def consistent_kr(network: Network, kf, kr, overall: list[str], Kp: list[float], prefer=None) -> np.ndarray:
    """``kr`` with the dependent k₋ recomputed from the others and K′ (used to make synthetic truths consistent)."""
    lay = layout(network, overall, prefer)
    kf2, kr2 = lay.rates(lay.theta(kf, kr, Kp))
    return kr2
