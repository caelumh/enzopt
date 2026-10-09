"""Mass-action ODEs with an analytic Jacobian, solved with scipy (Radau / LSODA).

``dx/dt = N · r(x)``, ``r_j = k_j · Π_i x_i^{a_ij}``. The Jacobian is exact, which keeps the stiff enzyme
mechanisms (intermediates on the ms scale, substrates on the minute scale) fast in Radau.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .network import Network


class SimulationError(RuntimeError):
    pass


class MassAction:
    def __init__(self, species: list[str], reactions: list[tuple[dict, dict]]):
        self.species = list(species)
        idx = {s: i for i, s in enumerate(self.species)}
        m, n = len(reactions), len(self.species)
        self.A = np.zeros((m, n), dtype=int)  # reactant orders
        self.N = np.zeros((n, m))  # stoichiometric matrix
        for j, (a, b) in enumerate(reactions):
            for s, v in a.items():
                self.A[j, idx[s]] += int(v)
                self.N[idx[s], j] -= v
            for s, v in b.items():
                self.N[idx[s], j] += v
        # sparse description of each reaction's reactants: [(species index, order), ...]
        self._terms = [[(i, int(self.A[j, i])) for i in range(n) if self.A[j, i] > 0] for j in range(m)]
        self.n_reactions = m

    @classmethod
    def from_network(cls, network: Network) -> MassAction:
        return cls(network.species, network.reactions())

    def rates(self, x: np.ndarray, k: np.ndarray) -> np.ndarray:
        r = np.array(k, dtype=float, copy=True)
        for j, terms in enumerate(self._terms):
            for i, o in terms:
                r[j] *= x[i] if o == 1 else x[i] ** o
        return r

    def rhs(self, t, x, k):
        return self.N @ self.rates(x, k)

    def jac(self, t, x, k):
        dr = np.zeros((self.n_reactions, len(self.species)))
        for j, terms in enumerate(self._terms):
            for p, (i, o) in enumerate(terms):
                v = k[j] * o * (x[i] ** (o - 1) if o > 1 else 1.0)
                for q, (l, ol) in enumerate(terms):
                    if q != p:
                        v *= x[l] if ol == 1 else x[l] ** ol
                dr[j, i] = v
        return self.N @ dr


@dataclass
class SolverSettings:
    method: str = "Radau"
    rtol: float = 1e-8
    atol_enzyme_rel: float = 1e-8  # absolute tolerance for enzyme species = Et * this
    atol_other: float = 1e-11
    max_step: float = np.inf

    def refined(self, factor: float = 10.0) -> SolverSettings:
        """Tighter settings for the 'suspected numerical error' re-run (SC5-5: step / 10)."""
        return SolverSettings(
            self.method,
            self.rtol / factor,
            self.atol_enzyme_rel / factor,
            self.atol_other / factor,
            self.max_step / factor if np.isfinite(self.max_step) else np.inf,
        )


def atol_vector(species: list[str], enzyme_species: list[str], Et: float, settings: SolverSettings) -> np.ndarray:
    ens = set(enzyme_species)
    scale = Et if Et and Et > 0 else 1.0
    return np.array([scale * settings.atol_enzyme_rel if s in ens else settings.atol_other for s in species])


def simulate(
    model: MassAction,
    k: np.ndarray,
    x0: np.ndarray,
    t_eval: np.ndarray,
    atol: np.ndarray | float = 1e-11,
    settings: SolverSettings | None = None,
    t0: float = 0.0,
) -> np.ndarray:
    """Return the state at ``t_eval`` as an array (len(t_eval) x n_species)."""
    st = settings or SolverSettings()
    t_eval = np.asarray(t_eval, float)
    kw = dict(method=st.method, rtol=st.rtol, atol=atol, args=(np.asarray(k, float),))
    if st.method in ("Radau", "BDF", "LSODA"):
        kw["jac"] = model.jac
    if np.isfinite(st.max_step):
        kw["max_step"] = st.max_step
    sol = solve_ivp(model.rhs, (t0, float(t_eval[-1])), np.asarray(x0, float), t_eval=t_eval, **kw)
    if not sol.success:
        raise SimulationError(sol.message)
    return sol.y.T


def simulate_network(
    network: Network,
    k: np.ndarray,
    x0: np.ndarray,
    t_eval: np.ndarray,
    Et: float = 0.0,
    settings: SolverSettings | None = None,
    model: MassAction | None = None,
) -> np.ndarray:
    st = settings or SolverSettings()
    model = model or MassAction.from_network(network)
    atol = atol_vector(network.species, network.enzyme_species, Et, st)
    return simulate(model, k, x0, t_eval, atol=atol, settings=st)


def haldane_kr(kf, kr, Kp: float, solve_step: int, cycle_steps: list[int] | None = None) -> np.ndarray:
    """Return ``kr`` with ``kr[solve_step-1]`` set by the Haldane relation K' = Π kf / Π kr over ``cycle_steps``.

    Steps are 1-based. Default cycle: steps 1..solve_step.
    """
    kf = np.asarray(kf, float)
    kr = np.array(kr, float, copy=True)
    cyc = cycle_steps or list(range(1, solve_step + 1))
    others = [s for s in cyc if s != solve_step]
    kr[solve_step - 1] = np.prod(kf[[s - 1 for s in cyc]]) / (np.prod(kr[[s - 1 for s in others]]) * Kp)
    return kr
