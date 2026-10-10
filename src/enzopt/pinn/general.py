"""PINN for any enzyme network whose steps are first order in the enzyme forms (F-13), batched over seeds on the GPU.

Generalises the LDH ordered bi-bi PINN (decision 2026-10-09: quasi-steady enzyme distribution). Per (seed, training
experiment) one small tanh MLP gives the d reaction extents ξ(t) of the slow coordinates (``core.qss``: the overall
reactions, then enzyme-free background steps):

    ξ_k = lo_k + (hi_k − lo_k)·sigmoid(u_k)              (range that keeps the small molecules ≥ 0)
    x_small ≈ x0 + V ξ  → φ = quasi-steady enzyme distribution (graph null vector, Gaussian elimination)
    enzyme forms = Et·a(t)·φ,   x_small = x0 + V ξ + Et·a·Mcorr (φ − e0)   (bound amounts in the conservation laws)
    slow residual  dξ/dt = W r(x)   (small-molecule change projected onto V; = r₁ for LDH at quasi-steady state)

Rate constants: θ = log10 [kf, free kr, K′] with the dependent k₋ from the thermodynamic constraints (``core.thermo``).
Observations: any number of signals per experiment, each a linear combination of species with its own σ; the data
loss is the mean squared σ-normalised residual over all points. Options kept from the LDH PINN: enzyme inactivation
(one shared a(t)), the missing-step search (``discrepancy``: rate factor exp(g) or a dead-end factor, d = 1), and the
``full`` representation (enzyme fractions as network outputs + their balance residuals; the method's original form,
kept as the documented failed alternative).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from ..core.activity import activity_curve, activity_rate  # noqa: F401 (re-exported)
from ..core.qss import QssError, structure
from ..core.thermo import ThermoError, layout


@dataclass
class PinnExperiment:
    name: str
    initial: dict
    Et: float  # mM
    t_obs: np.ndarray
    a_obs: np.ndarray  # primary signal (``PinnConfig.observable``, σ = ``PinnConfig.sigma``)
    extra: list = field(default_factory=list)  # further signals: [(name, t, y, sigma, {species: coefficient})]


@dataclass
class PinnConfig:
    seeds: list[int]
    adam_iters: int = 2000
    lbfgs_iters: int = 300
    width: int = 32
    depth: int = 3
    w_prior: float = 1e-2
    w_ic: float = 1.0
    w_phys_start: float = 1.0
    w_phys_final: float = 1e4
    sigma: float = 0.003
    eps340: float = 6.22
    prior_sd: float = 1.5
    init_jitter: float = 1.0  # initial log10 rate constants = prior centre + U(±init_jitter), per seed
    irreversible_last: bool = False  # LDH default network only (pinn.ldh): last step one-way
    device: str = "cpu"
    dtype: str = "float64"
    representation: str = "qss"  # "qss": enzyme forms at quasi-steady state; "full": network outputs
    fast_norm: str = "slow"  # full representation: 'slow' (substrate rate units) or 'gross' (CADKIMM)
    lr_theta: float = 2e-2
    lr_net: float = 3e-3
    n_col_uniform: int = 300
    n_col_early: int = 60
    cuda_graphs: bool = True  # capture each training step as one CUDA graph (removes launch overhead)
    # τ = t/T_e with each experiment's own last time (default) or one shared T_max (Holmes 1988 check, 2026-10-09)
    time_scale: str = "per_experiment"
    # enzyme activity a(t): "none" (a ≡ 1) or "shared" (one monotone a(t) for all experiments, core.activity)
    inactivation: str = "none"
    inact_k: int = 12  # number of sigmoid steps in h
    w_inact: float = 1e-2  # weight of h(T)² (prefers no loss of activity unless the data need it)
    # missing-step search (d = 1): "none", "rate_factor" (dξ/dt = J·exp(g(free small molecules))) or "dead_end"
    discrepancy: str = "none"
    disc_width: int = 16
    w_disc: float = 1e-2  # weight of mean g² relative to the physics weight (g only takes what the steps cannot)
    dead_end_form: str = "EN"  # with discrepancy "dead_end": enzyme form X and ligand Y of the candidate complex
    dead_end_ligand: str = "Lac"
    # the network (core.network.Network); None = LDH ordered bi-bi (pinn.ldh fills it in)
    network: object = None
    observable: dict | None = None  # primary signal {species: coefficient}; None = ε340·(NADH + EN + ENP)
    overall: list | None = None  # overall reactions (strings): slow coordinates and K′ constraints
    prefer_dependent: list | None = None  # 1-based steps to make dependent first (old haldane_step)
    # sigmoid range widened by this fraction on each side (initial point off the saturated edge). 0.05 fits the
    # appendix C networks closer to the ODE optimum but changes which basin a single run reaches in float32 vs float64
    # (AC-08), so it is off by default and a project option (identification.pinn.range_margin)
    range_margin: float = 0.0


@dataclass
class PinnResult:
    seed: int
    theta: np.ndarray
    kf: np.ndarray
    kr: np.ndarray
    Kp: float | None
    loss_parts: dict
    history: list = field(default_factory=list)
    activity: dict | None = None  # inactivation: {"T", "c", "s", "w"} for activity_curve()
    discrepancy: dict | None = None  # missing-step search: g and the states at collocation points (pipeline.suggest)
    dead_end: dict | None = None  # candidate step: {"form", "ligand", "Ki_mM"}
    Kp_all: list | None = None  # K′ of every overall reaction that constrains a cycle (θ order)


def supports(network, overall: list[str] | None = None) -> str | None:
    """None if the general PINN can represent ``network``; otherwise the reason (shown to the user)."""
    try:
        structure(network, overall)
        layout(network, overall)
    except (QssError, ThermoError) as e:
        return str(e)
    return None


class BatchedMLP(torch.nn.Module):
    """B independent MLPs R → R^out; returns outputs and d(outputs)/dτ by tangent propagation."""

    def __init__(self, B: int, width: int, depth: int, out_dim: int, gens: list[torch.Generator], dtype, device):
        super().__init__()
        dims = [1] + [width] * depth + [out_dim]
        self.Ws = torch.nn.ParameterList()
        self.bs = torch.nn.ParameterList()
        for din, dout in zip(dims[:-1], dims[1:], strict=True):
            bound = 1.0 / math.sqrt(din)  # torch.nn.Linear default init
            W = torch.empty(B, din, dout, dtype=torch.float64)
            b = torch.empty(B, 1, dout, dtype=torch.float64)
            for i, g in enumerate(gens):
                W[i].uniform_(-bound, bound, generator=g)
                b[i].uniform_(-bound, bound, generator=g)
            self.Ws.append(torch.nn.Parameter(W.to(device=device, dtype=dtype)))
            self.bs.append(torch.nn.Parameter(b.to(device=device, dtype=dtype)))

    def forward(self, tau: torch.Tensor):
        # tau: (B, n, 1)
        h, dh = tau, torch.ones_like(tau)
        last = len(self.Ws) - 1
        for i, (W, b) in enumerate(zip(self.Ws, self.bs, strict=True)):
            z = torch.baddbmm(b, h, W)
            dz = torch.bmm(dh, W)
            if i < last:
                h = torch.tanh(z)
                dh = (1.0 - h * h) * dz
            else:
                h, dh = z, dz
        return h, dh


def _conservation(qs) -> np.ndarray:
    """Conservation-law rows C (C N = 0) over all species of the structure's network."""
    from ..core.thermo import _null_space

    n = len(qs.species)
    N = np.zeros((n, qs.N_s.shape[1]))
    N[qs.small_idx, :] = qs.N_s
    N[qs.enzyme_idx, :] = qs.N_e
    return _null_space(N.T)


def _mul(a, b):
    return b if a is None else (a if b is None else a * b)


def _add(a, b):
    return b if a is None else (a if b is None else a + b)


class GeneralPinn:
    """Seeds × training experiments, trained together; each seed has its own rate constants."""

    def __init__(self, exps: list[PinnExperiment], prior_center: np.ndarray, cfg: PinnConfig):
        if cfg.network is None:
            raise ValueError("PinnConfig.network가 없음(pinn.ldh.train_pinn은 LDH 반응망을 채워 넣음)")
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        self.dt = torch.float32 if cfg.dtype == "float32" else torch.float64
        net = cfg.network
        self.network = net
        self.qs = structure(net, cfg.overall)
        self.lay = layout(net, cfg.overall, cfg.prefer_dependent)
        qs, lay = self.qs, self.lay
        # orient each slow coordinate so that the primary signal grows with it: for LDH ξ is then the free NADH
        # coordinate of the original LDH PINN (same sigmoid range and initial point), which keeps its training path
        prim0 = cfg.observable or {"NADH": cfg.eps340, "EN": cfg.eps340, "ENP": cfg.eps340}
        c_s = np.array([float(prim0.get(s, 0.0)) for s in qs.small])
        sgn = np.where(c_s @ qs.V < 0, -1.0, 1.0)
        qs.V = qs.V * sgn
        # anchors: one observed small molecule per slow coordinate. Its free concentration is exactly x0 + V ξ (the
        # bound-amount correction goes to the other molecules) and the slow residual is its exact balance, so for
        # LDH ξ is the free NADH and dξ/dt = −r₁ as in the original LDH PINN (decision 2026-10-10)
        seen = np.zeros(len(qs.small))
        for e in exps:
            for _n, _t, _y, _s, cf in e.extra:
                seen += np.array([abs(float(cf.get(s, 0.0))) for s in qs.small])
        prio = sorted(range(len(qs.small)), key=lambda i: (-abs(c_s[i]), -seen[i], i))
        anchors: list[int] = []
        for i in prio:
            if len(anchors) == qs.d:
                break
            if np.linalg.matrix_rank(qs.V[anchors + [i], :]) > len(anchors):
                anchors.append(i)
        VA = qs.V[anchors, :]
        qs.W = np.linalg.solve(VA, qs.N_s[anchors, :])
        qs.W[np.abs(qs.W) < 1e-12] = 0.0
        others = [i for i in range(len(qs.small)) if i not in anchors]
        C = _conservation(qs)
        Mc = np.zeros((len(qs.small), len(qs.enzyme)))
        if C.size and others:
            Mc[others, :] = -np.linalg.pinv(C[:, [qs.small_idx[i] for i in others]]) @ C[:, qs.enzyme_idx]
        Mc[np.abs(Mc) < 1e-12] = 0.0
        qs.Mcorr = Mc
        self.anchors = [qs.small[i] for i in anchors]
        self.exps = exps
        S, E = len(cfg.seeds), len(exps)
        self.S, self.E, self.B = S, E, S * E
        self.n_steps = len(net.steps)
        self.d, self.ns, self.ne = qs.d, len(qs.small), len(qs.enzyme)
        # one slow coordinate: ξ in a sigmoid range (positivity built in); several: ξ = scale·(u(τ) − u(0)), exact
        # initial condition and no box (per-coordinate boxes cut off states such as NADH formed by one reaction and
        # used by another; boxes over the whole feasible polytope are too wide to resolve the extents)
        self.lin = self.d > 1
        self.n_theta = lay.n_theta
        self.inact = cfg.inactivation == "shared"
        self.dead = cfg.discrepancy == "dead_end"
        self.disc = cfg.discrepancy == "rate_factor"
        if (self.dead or self.disc) and self.d != 1:
            raise ValueError("빠진 단계 탐색(discrepancy)은 전체 반응이 하나일 때만")
        n_inact = cfg.inact_k if self.inact else 0
        self.inact_sl = slice(self.n_theta, self.n_theta + n_inact)  # θ columns: core, inactivation, log10 K_i
        self.de_col = self.n_theta + n_inact
        pc = np.asarray(prior_center, float)[: self.n_theta]
        if len(pc) != self.n_theta:
            raise ValueError(f"사전분포 중심 {len(pc)}개 ≠ 매개변수 {self.n_theta}개")
        self.prior_c = torch.tensor(pc, dtype=self.dt, device=self.dev)
        T_all = [max([float(e.t_obs.max())] + [float(np.max(x[1])) for x in e.extra]) for e in exps]
        self.T = float(max(T_all))
        T_e = [T_all[i] if cfg.time_scale == "per_experiment" else self.T for i in range(E)]
        gens, theta0 = [], []
        for sd in cfg.seeds:
            for _ in range(E):
                g = torch.Generator().manual_seed(int(sd) * 1000 + len(gens) % E)
                gens.append(g)
            jit = np.random.default_rng(100 + int(sd)).uniform(-cfg.init_jitter, cfg.init_jitter, len(pc))
            # inactivation weights (extra θ columns) start near zero: softplus(−5) ≈ 0.007, a(T) ≈ 0.95
            extra = [np.full(n_inact, -5.0), np.array([1.0]) if self.dead else np.zeros(0)]  # K_i starts at 10 mM
            theta0.append(np.concatenate([pc + jit, *extra]))
        self.theta = torch.nn.Parameter(torch.tensor(np.array(theta0), dtype=self.dt, device=self.dev))
        out_dim = self.d if cfg.representation == "qss" else self.d + self.ne
        self.net = BatchedMLP(self.B, cfg.width, cfg.depth, out_dim, gens, self.dt, self.dev)

        def T_(a):
            return torch.tensor(np.asarray(a, float), dtype=self.dt, device=self.dev)

        # structure constants
        self.Vt = T_(qs.V.T)  # (d, ns)
        self.Wt = T_(qs.W.T)  # (n_steps, d)
        self.McT = T_(qs.Mcorr.T)  # (ne, ns)
        e0 = np.zeros(self.ne)
        e0[0] = 1.0
        self.e0 = T_(e0)
        order = qs.small + qs.enzyme
        self.perm = torch.tensor([order.index(s) for s in net.species], dtype=torch.long, device=self.dev)
        self.sp_index = {s: i for i, s in enumerate(net.species)}
        self.A_dep = T_(lay.A) if lay.dependent else None
        self.kr_free_idx = torch.tensor(lay.free_kr, dtype=torch.long, device=self.dev)
        self.kr_dep_idx = torch.tensor(lay.dependent, dtype=torch.long, device=self.dev)
        # per-experiment constants, repeated for each seed: index b = s*E + e
        x0s, lo, span, msk, p0, Et = [], [], [], [], [], []
        for e in exps:
            x0 = np.array([float(e.initial.get(s, 0.0)) for s in qs.small])
            l, h = qs.ranges(x0)
            sp = h - l
            ok = sp > 1e-9
            if not ok.any():
                raise ValueError(f"실험 {e.name}: 반응이 진행될 수 없음(기질·생성물의 초기 농도 확인)")
            # optional margin (range_margin): the initial point ξ = 0 usually sits on an edge of the feasible range (no
            # product yet), where sigmoid(u) saturates and the network can hardly leave it fast enough
            mg = 0.0 if self.lin else float(cfg.range_margin)
            l = np.where(ok, l - mg * sp, l)
            sp = np.where(ok, sp * (1.0 + 2.0 * mg), 0.0)
            x0s.append(x0)
            lo.append(l)
            span.append(sp)
            msk.append(ok.astype(float))
            p0.append(np.clip(np.where(ok, (0.0 - l) / np.where(ok, sp, 1.0), 0.5), 1e-4, 1 - 1e-4))
            Et.append(e.Et)

        def rep(v, width):
            a = np.tile(np.asarray(v, float).reshape(E, width), (S, 1))
            return torch.tensor(a, dtype=self.dt, device=self.dev).reshape(self.B, 1, width)

        if self.lin:  # ξ = scale·(u(τ) − u(0)): the range of each coordinate is only its scale (all coordinates live)
            sp_all = np.array(span, float)
            sp_max = sp_all.max(axis=0)
            span = [np.where(s_ > 1e-9, s_, sp_max) for s_ in sp_all]
            msk = [(s_ > 1e-9).astype(float) for s_ in span]
        self.x0s = rep(x0s, self.ns)
        self.lo, self.span, self.maskd, self.p0 = (rep(v, self.d) for v in (lo, span, msk, p0))
        self.n_valid = self.maskd.sum(dim=-1, keepdim=True).clamp_min(1.0)
        self.Et = rep(Et, 1)
        self.Tb = rep(T_e, 1)  # time scale per (seed, experiment)
        tc = np.unique(
            np.concatenate([np.linspace(0, 1, cfg.n_col_uniform), np.geomspace(1e-4, 0.05, cfg.n_col_early)])
        )
        self.tau_col = torch.tensor(tc, dtype=self.dt, device=self.dev).reshape(1, -1, 1).expand(self.B, -1, -1)
        # observation points of every signal, packed per experiment
        prim = cfg.observable or {"NADH": cfg.eps340, "EN": cfg.eps340, "ENP": cfg.eps340}
        pts = []
        for e in exps:
            rows = [(t, y, cfg.sigma, prim) for t, y in zip(e.t_obs, e.a_obs, strict=True)]
            for _name, tt, yy, sg, coef in e.extra:
                rows += [(t, y, float(sg), coef) for t, y in zip(tt, yy, strict=True)]
            pts.append(rows)
        nmax = max(len(r) for r in pts)
        n_sp = len(net.species)
        tau_obs, y_obs, s_obs, mask = (np.zeros((E, nmax)) for _ in range(4))
        coef = np.zeros((E, nmax, n_sp))
        for i, rows in enumerate(pts):
            for j, (t, y, sg, cf) in enumerate(rows):
                tau_obs[i, j] = t / T_e[i]
                y_obs[i, j] = y
                s_obs[i, j] = sg
                mask[i, j] = 1.0
                for s, c in cf.items():
                    coef[i, j, self.sp_index[s]] = float(c)
        s_obs[mask == 0] = 1.0

        def tile3(a):
            return torch.tensor(np.tile(a, (S, 1)), dtype=self.dt, device=self.dev).unsqueeze(-1)

        self.tau_obs, self.a_obs, self.s_obs, self.mask = map(tile3, (tau_obs, y_obs, s_obs, mask))
        self.coef_obs = torch.tensor(np.tile(coef, (S, 1, 1)), dtype=self.dt, device=self.dev)
        self.n_obs = self.mask.sum(dim=1, keepdim=True)
        self.tau0 = torch.zeros(self.B, 1, 1, dtype=self.dt, device=self.dev)
        if self.inact:
            K = cfg.inact_k
            self.inact_c = torch.tensor(np.linspace(0.0, 1.0, K), dtype=self.dt, device=self.dev)
            self.inact_s = 0.6 / max(1, K - 1)
            self.inact_u = self.Tb / self.T  # τ → u = t/T_max, per (seed, experiment)
        self.disc_params = torch.nn.ParameterList()
        if self.disc:
            scale = qs.small_scale([np.asarray(x) for x in x0s])
            self.disc_scale = T_(scale).reshape(1, 1, self.ns)
            w = cfg.disc_width
            shapes = [(self.ns, w), (1, w), (w, w), (1, w), (w, 1), (1, 1)]
            init = []
            for k, (a, b) in enumerate(shapes):
                t = torch.zeros(S, a, b, dtype=torch.float64)
                if k in (0, 2):  # last layer starts at zero: g = 0, i.e. the steps as they are
                    for i, sd in enumerate(cfg.seeds):
                        t[i].uniform_(
                            -1.0 / math.sqrt(a),
                            1.0 / math.sqrt(a),
                            generator=torch.Generator().manual_seed(7919 * int(sd) + k),
                        )
                init.append(torch.nn.Parameter(t.to(device=self.dev, dtype=self.dt)))
            self.disc_params = torch.nn.ParameterList(init)
        if self.dead:
            self.de_form = qs.enzyme.index(cfg.dead_end_form)
            self.de_lig = qs.small.index(cfg.dead_end_ligand)

    # ------------------------------------------------------------------ model pieces
    def disc_g(self, xs: torch.Tensor) -> torch.Tensor:
        """g at the given free small-molecule states (B, n, ns) → (B, n, 1); one network per seed."""
        x = xs / self.disc_scale
        n = x.shape[1]
        x = x.reshape(self.S, self.E * n, self.ns)
        W1, b1, W2, b2, W3, b3 = self.disc_params
        h = torch.tanh(torch.baddbmm(b1, x, W1))
        h = torch.tanh(torch.baddbmm(b2, h, W2))
        return torch.baddbmm(b3, h, W3).reshape(self.B, n, 1)

    def activity_weights(self, theta: torch.Tensor):
        """(B, 1, K) non-negative step weights, repeated over each seed's experiments; None without inactivation."""
        if not self.inact:
            return None
        w = torch.nn.functional.softplus(theta[:, self.inact_sl])
        return w.unsqueeze(1).expand(self.S, self.E, w.shape[1]).reshape(self.B, 1, w.shape[1])

    def activity(self, tau: torch.Tensor, wB: torch.Tensor) -> torch.Tensor:
        u = tau * self.inact_u
        c, s = self.inact_c, self.inact_s
        h = (wB * (torch.sigmoid((u - c) / s) - torch.sigmoid(-c / s))).sum(dim=-1, keepdim=True)
        return torch.exp(-h)

    def rates(self, theta: torch.Tensor):
        """kf, kr per seed (S, n_steps); dependent k₋ from the thermodynamic constraints, 0 for one-way steps."""
        n = self.n_steps
        th = theta[:, : self.n_theta]
        kf = 10.0 ** th[:, :n]
        kr = torch.zeros_like(kf)
        nf = len(self.lay.free_kr)
        if nf:
            kr = kr.index_copy(1, self.kr_free_idx, 10.0 ** th[:, n : n + nf])
        if self.A_dep is not None:
            kr = kr.index_copy(1, self.kr_dep_idx, 10.0 ** (th @ self.A_dep.T))
        return kf, kr

    def qss_fractions(self, xs: torch.Tensor, kfB: torch.Tensor, krB: torch.Tensor) -> torch.Tensor:
        """Quasi-steady enzyme distribution (B, n, ne) at free small-molecule states xs (B, n, ns).

        Forms 1..ne−1 relative to the free enzyme solve (−L_red) ψ = L[1:, 0]; −L_red is column diagonally dominant,
        so Gaussian elimination needs no pivoting. Rates are scaled by their maximum first (φ is invariant)."""
        rate: dict = {}
        for e in self.qs.edges:
            wf = kfB[..., e.step : e.step + 1]
            for i, o in e.lf:
                x = xs[..., i : i + 1]
                wf = wf * (x if o == 1 else x**o)
            rate[(e.a, e.b)] = _add(rate.get((e.a, e.b)), wf)
            if e.reversible:
                wr = krB[..., e.step : e.step + 1]
                for i, o in e.lr:
                    x = xs[..., i : i + 1]
                    wr = wr * (x if o == 1 else x**o)
                rate[(e.b, e.a)] = _add(rate.get((e.b, e.a)), wr)
        vals = list(rate.values())
        vals = torch.broadcast_tensors(*vals)
        scale = torch.stack(vals, dim=0).amax(dim=0).clamp_min(1e-30)
        rate = {k: v / scale for k, v in zip(rate.keys(), vals, strict=True)}
        n = self.ne
        if n == 1:
            return torch.ones_like(scale)
        out_tot = [None] * n
        for (i, _j), v in rate.items():
            out_tot[i] = _add(out_tot[i], v)
        # A[r][c] for r, c in 1..n−1 (row r: balance of form r), b[r] = rate 0 → r
        A = {r: {} for r in range(1, n)}
        for r in range(1, n):
            A[r][r] = out_tot[r]
        for (i, j), v in rate.items():
            if i >= 1 and j >= 1:
                A[j][i] = _add(A[j].get(i), -v)
        b = {r: rate.get((0, r)) for r in range(1, n)}
        for p in range(1, n):
            piv = A[p][p]
            for r in range(p + 1, n):
                arp = A[r].get(p)
                if arp is None:
                    continue
                f = arp / piv
                for c, apc in A[p].items():
                    if c > p:
                        A[r][c] = _add(A[r].get(c), -f * apc)
                if b[p] is not None:
                    b[r] = _add(b[r], -f * b[p])
                A[r].pop(p, None)
        psi = [None] * n
        for p in range(n - 1, 0, -1):
            acc = b[p]
            for c, apc in A[p].items():
                if c > p and psi[c] is not None:
                    acc = _add(acc, -apc * psi[c])
            psi[p] = None if acc is None else acc / A[p][p]
        one = torch.ones_like(scale)
        cols = [one] + [torch.zeros_like(scale) if v is None else v for v in psi[1:]]
        tot = cols[0]
        for c in cols[1:]:
            tot = tot + c
        return torch.cat(cols, dim=-1) / tot

    def states(self, tau: torch.Tensor, kfB=None, krB=None, wB=None):
        """Full state X (B, n, n_species), free small molecules, φ, dξ/dτ, sigmoid(u), dφ/dτ (full repr.)."""
        out, dout = self.net(tau)
        d = self.d
        u, du = out[..., :d], dout[..., :d]
        if self.lin:
            u0 = self.net(self.tau0[: tau.shape[0]])[0][..., :d]
            sig = u
            xi = self.span * (u - u0)
            dxi = self.span * du
        else:
            sig = torch.sigmoid(u)
            xi = self.lo + self.span * sig
            dxi = self.span * sig * (1.0 - sig) * du
        Et = self.Et if wB is None else self.Et * self.activity(tau, wB)
        xs_app = self.x0s + xi @ self.Vt
        if self.cfg.representation == "qss":
            phi = self.qss_fractions(xs_app.clamp_min(0.0), kfB, krB)
            dphi = None
        else:
            v, dv = out[..., d:], dout[..., d:]
            phi = torch.softmax(v, dim=-1)
            dphi = phi * (dv - (phi * dv).sum(dim=-1, keepdim=True))
        forms = Et * phi
        xs = xs_app + Et * ((phi - self.e0) @ self.McT)
        X = torch.cat([xs, forms], dim=-1).index_select(-1, self.perm)
        return X, xs, phi, dxi, sig, dphi

    def step_rates(self, X: torch.Tensor, kfB: torch.Tensor, krB: torch.Tensor):
        """Forward and backward rates of every step (lists of (B, n, 1))."""
        f, g = [], []
        for j in range(self.n_steps):
            fj = kfB[..., j : j + 1]
            for i, o in self.qs.reactants[j]:
                x = X[..., i : i + 1]
                fj = fj * (x if o == 1 else x**o)
            f.append(fj)
            if self.network.steps[j].reversible:
                gj = krB[..., j : j + 1]
                for i, o in self.qs.products[j]:
                    x = X[..., i : i + 1]
                    gj = gj * (x if o == 1 else x**o)
            else:
                gj = torch.zeros_like(fj)
            g.append(gj)
        return f, g

    def loss(self, w_phys: float):
        """Per-seed loss vector (S,) and per-seed parts."""
        cfg = self.cfg
        n = self.n_steps
        kf, kr = self.rates(self.theta)
        kfB = kf.unsqueeze(1).expand(self.S, self.E, n).reshape(self.B, 1, n)  # (B,1,n), b = s*E + e
        krB = kr.unsqueeze(1).expand(self.S, self.E, n).reshape(self.B, 1, n)
        wB = self.activity_weights(self.theta)
        # data
        X, _, _, _, _, _ = self.states(self.tau_obs, kfB, krB, wB)
        pred = (X * self.coef_obs).sum(dim=-1, keepdim=True)
        L_data = ((((self.a_obs - pred) / self.s_obs) ** 2) * self.mask).sum(dim=1) / self.n_obs.squeeze(1)
        # physics at collocation points
        X, xs, phi, dxi, _, dphi = self.states(self.tau_col, kfB, krB, wB)
        T = self.Tb
        f, g = self.step_rates(X.clamp_min(0.0), kfB, krB)  # the widened range may dip just below zero
        r = torch.cat([f[j] - g[j] for j in range(n)], dim=-1)  # (B, n_col, n_steps)
        J = r @ self.Wt  # (B, n_col, d)
        s_slow = self.span.clamp_min(1e-12) / T
        if self.disc:
            gd = self.disc_g(xs)
            J = J * torch.exp(gd)
        elif self.dead:
            Ki = (10.0 ** self.theta[:, self.de_col]).unsqueeze(1).expand(self.S, self.E).reshape(self.B, 1, 1)
            occ = phi[..., self.de_form : self.de_form + 1] * xs[..., self.de_lig : self.de_lig + 1].clamp_min(0.0) / Ki
            J = J / (1.0 + occ)
        res = ((dxi / T - J) / s_slow) ** 2 * self.maskd
        L_slow = (res.mean(dim=1, keepdim=True).sum(dim=-1, keepdim=True) / self.n_valid).reshape(self.B, 1)
        L_fast = torch.zeros_like(L_slow)
        if cfg.representation == "full":
            floor = self.Et * 1e-3
            Ne = self.qs.N_e
            for i in range(self.ne - 1):
                net_flux = gross = None
                for j in range(n):
                    c = Ne[i, j]
                    if c != 0:
                        net_flux = _add(net_flux, c * (f[j] - g[j]))
                        gross = _add(gross, abs(c) * (f[j] + g[j]))
                if net_flux is None:
                    continue
                scale = (gross + floor) if cfg.fast_norm == "gross" else s_slow[..., :1]
                L_fast = L_fast + (((self.Et * dphi[..., i : i + 1] / T - net_flux) / scale) ** 2).mean(dim=1)
        # initial condition: ξ(0) = 0
        if self.lin:  # exact by construction
            L_ic = torch.zeros_like(L_slow)
        else:
            _, _, _, _, sig0, _ = self.states(self.tau0, kfB, krB)
            L_ic = ((((sig0 - self.p0) / 0.01) ** 2) * self.maskd).sum(dim=-1) / self.n_valid.squeeze(-1)
            L_ic = L_ic.reshape(self.B, 1)

        def per_seed(x):
            return x.reshape(self.S, self.E).mean(dim=1)

        Ld, Ls, Lf, Li = per_seed(L_data), per_seed(L_slow), per_seed(L_fast), per_seed(L_ic)
        Lp = ((self.theta[:, : self.n_theta] - self.prior_c) ** 2).sum(dim=1) / (2 * cfg.prior_sd**2)
        if self.dead:  # log10 K_i around 10 mM
            Lp = Lp + (self.theta[:, self.de_col] - 1.0) ** 2 / (2 * cfg.prior_sd**2)
        total = Ld + w_phys * (Ls + Lf) + cfg.w_ic * Li + cfg.w_prior * Lp
        parts = dict(data=Ld, slow=Ls, fast=Lf, ic=Li, prior=Lp)
        if self.inact:  # total loss of activity at the end of the longest experiment, h(1)
            c, s = self.inact_c, self.inact_s
            w = torch.nn.functional.softplus(self.theta[:, self.inact_sl])
            h_end = (w * (torch.sigmoid((1.0 - c) / s) - torch.sigmoid(-c / s))).sum(dim=1)
            parts["inact"] = h_end**2
            total = total + cfg.w_inact * parts["inact"]
        if self.disc:
            parts["disc"] = per_seed((gd**2).mean(dim=1))
            total = total + w_phys * cfg.w_disc * parts["disc"]
        return total, parts

    # ------------------------------------------------------------------ training
    def _use_graphs(self) -> bool:
        return self.dev.type == "cuda" and self.cfg.cuda_graphs

    def _capture(self, fn):
        """Capture ``fn()`` (forward + backward [+ optimizer step]) as a CUDA graph; returns (graph, outputs)."""
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                fn(warmup=True)
        torch.cuda.current_stream().wait_stream(side)
        for p in self._params():
            p.grad = None
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = fn(warmup=False)
        return graph, outs

    def train(self, progress: Callable[[float, str], None] | None = None) -> list[PinnResult]:
        cfg = self.cfg
        hist = [[] for _ in range(self.S)]
        graphs = self._use_graphs()

        def lr(v):
            return torch.tensor(v, dtype=self.dt, device=self.dev) if graphs else v

        opt = torch.optim.Adam(
            [
                {"params": [self.theta], "lr": lr(cfg.lr_theta)},
                {"params": list(self.net.parameters()), "lr": lr(cfg.lr_net)},
                {"params": list(self.disc_params), "lr": lr(cfg.lr_net)},
            ],
            capturable=graphs,
        )
        n_ramp = max(1, int(0.7 * cfg.adam_iters))
        half = max(1, cfg.adam_iters // 2)
        lw0, lw1 = math.log(cfg.w_phys_start), math.log(cfg.w_phys_final)
        self._w = torch.tensor(cfg.w_phys_start, dtype=self.dt, device=self.dev)

        def adam_step(warmup: bool = False):
            if warmup or not graphs:
                opt.zero_grad(set_to_none=True)
            total, parts = self.loss(self._w)
            total.sum().backward()
            opt.step()
            return total, parts

        t0 = time.time()
        it = 0
        graph = outs = None
        if graphs and cfg.adam_iters > 4:
            graph, outs = self._capture(adam_step)  # the 2 warm-up calls are real steps
            it = 2
        while it < cfg.adam_iters:
            w = math.exp(lw0 + (lw1 - lw0) * min(1.0, it / n_ramp))
            self._w.fill_(w)
            if it == half:
                for gr in opt.param_groups:
                    if torch.is_tensor(gr["lr"]):
                        gr["lr"].mul_(0.5)
                    else:
                        gr["lr"] *= 0.5
            if graph is not None:
                graph.replay()
                total, parts = outs
            else:
                total, parts = adam_step()
            if it % 100 == 0 or it == cfg.adam_iters - 1:
                self._log(hist, it, total, parts)
                if progress:
                    frac = (it + 1) / (cfg.adam_iters + cfg.lbfgs_iters * 1.5)
                    progress(frac, f"Adam {it + 1}/{cfg.adam_iters} ({time.time() - t0:.0f} s)")
            it += 1
        total = parts = None  # drop references to the captured autograd graph before the next capture
        del graph, outs
        self._w.fill_(cfg.w_phys_final)
        if cfg.lbfgs_iters > 0:
            self._lbfgs(cfg.lbfgs_iters, hist, progress)
        for p in self._params():
            p.grad = None
        with torch.no_grad():
            total, parts = self.loss(self._w)
        self._log(hist, "final", total, parts)
        kf, kr = self.rates(self.theta.detach())
        th = self.theta.detach().double().cpu().numpy()
        kf = kf.detach().double().cpu().numpy()
        kr = kr.detach().double().cpu().numpy()
        act = [None] * self.S
        if self.inact:
            w = torch.nn.functional.softplus(self.theta.detach()[:, self.inact_sl]).double().cpu().numpy()
            c = self.inact_c.double().cpu().numpy().tolist()
            act = [{"kind": "steps", "T": self.T, "c": c, "s": self.inact_s, "w": w[i].tolist()} for i in range(self.S)]
        disc = self._discrepancy_points() if self.disc else [None] * self.S
        n, nf = self.n_steps, len(self.lay.free_kr)
        out = []
        for i, sd in enumerate(cfg.seeds):
            Kp_all = [float(10 ** th[i, n + nf + c]) for c in range(len(self.lay.overall_used))]
            r = PinnResult(
                int(sd),
                th[i].copy(),
                kf[i].copy(),
                kr[i].copy(),
                Kp_all[0] if Kp_all else None,
                hist[i][-1][2],
                hist[i],
                act[i],
                Kp_all=Kp_all,
            )
            r.discrepancy = disc[i]
            if self.dead:
                Ki = float(10 ** th[i, self.de_col])
                r.dead_end = {"form": cfg.dead_end_form, "ligand": cfg.dead_end_ligand, "Ki_mM": Ki}
            out.append(r)
        return out

    def _discrepancy_points(self, every: int = 3) -> list[dict]:
        """Per seed: g, the slow rate J (``r1``), the free small molecules and the fractions φ at collocation points."""
        n = self.n_steps
        with torch.no_grad():
            kf, kr = self.rates(self.theta)
            kfB = kf.unsqueeze(1).expand(self.S, self.E, n).reshape(self.B, 1, n)
            krB = kr.unsqueeze(1).expand(self.S, self.E, n).reshape(self.B, 1, n)
            X, xs, phi, _, _, _ = self.states(self.tau_col, kfB, krB, self.activity_weights(self.theta))
            g = self.disc_g(xs)
            f, gg_ = self.step_rates(X, kfB, krB)
            J = torch.cat([f[j] - gg_[j] for j in range(n)], dim=-1) @ self.Wt

            def np_(x):
                return x.detach().double().cpu().numpy()[:, ::every, 0]

            t = np_(self.tau_col * self.Tb)
            cols = {s: np_(xs[..., i : i + 1]) for i, s in enumerate(self.qs.small)}
            ph = {s: np_(phi[..., i : i + 1]) for i, s in enumerate(self.qs.enzyme)}
            gv, Jv = np_(g), np_(J[..., :1])
        out = []
        for i in range(self.S):
            rows = range(i * self.E, (i + 1) * self.E)
            out.append(
                {
                    "experiment": [self.exps[b - i * self.E].name for b in rows for _ in range(t.shape[1])],
                    "t": t[list(rows)].ravel().tolist(),
                    **{k: v[list(rows)].ravel().tolist() for k, v in cols.items()},
                    **{f"phi_{k}": v[list(rows)].ravel().tolist() for k, v in ph.items()},
                    "r1": Jv[list(rows)].ravel().tolist(),
                    "g": gv[list(rows)].ravel().tolist(),
                }
            )
        return out

    def _log(self, hist, it, total, parts):
        tv = total.detach().double().cpu().numpy()
        pv = {k: v.detach().double().cpu().numpy() for k, v in parts.items()}
        for i in range(self.S):
            hist[i].append((it, float(tv[i]), {k: float(pv[k][i]) for k in pv}))

    def _params(self):
        """Per-seed tensors first (θ, discrepancy network; leading dim S), then the per-(seed, experiment) MLPs."""
        return [self.theta] + list(self.disc_params) + list(self.net.parameters())

    @property
    def _n_seed_params(self) -> int:
        return 1 + len(self.disc_params)

    def _lbfgs(self, iters, hist, progress):
        """Batched L-BFGS: every seed keeps its own history and step length (seeds stay independent)."""
        params = self._params()
        S = self.S
        E = self.E

        k_seed = self._n_seed_params

        def seed_view(tensors):
            """Flatten parameter-shaped tensors into (S, n_per_seed)."""
            parts = [t.reshape(S, -1) for t in tensors[:k_seed]]
            for t in tensors[k_seed:]:
                parts.append(t.reshape(S, E, -1).reshape(S, -1))
            return torch.cat(parts, dim=1)

        sizes = [p[0].numel() if i < k_seed else p.reshape(S, E, -1)[0].numel() for i, p in enumerate(params)]

        def assign(x):
            off = 0
            with torch.no_grad():
                for i, p in enumerate(params):
                    n = sizes[i]
                    p.copy_(x[:, off : off + n].reshape(p.shape))
                    off += n

        def loss_and_grad(warmup: bool = False):
            for p in params:
                p.grad = None
            total, parts = self.loss(self._w)
            total.sum().backward()
            return total, parts

        graph = None
        if self._use_graphs():
            graph, (g_total, g_parts) = self._capture(loss_and_grad)
            g_grads = [p.grad for p in params]

        def evaluate(x):
            assign(x)
            if graph is not None:
                graph.replay()
                return g_total.detach().clone(), seed_view(g_grads).clone(), g_parts
            total, parts = loss_and_grad()
            grad = seed_view([p.grad for p in params]).detach()
            return total.detach(), grad, parts

        with torch.no_grad():
            x = seed_view([p.detach() for p in params]).clone()
        f, g, _ = evaluate(x)
        m = 20
        s_hist, y_hist, rho_hist = [], [], []
        step0 = torch.ones(S, 1, dtype=x.dtype, device=x.device)
        active = torch.ones(S, dtype=torch.bool, device=x.device)
        for k in range(iters):
            q = g.clone()
            alphas = []
            for s_, y_, r_ in reversed(list(zip(s_hist, y_hist, rho_hist, strict=True))):
                a = r_ * (s_ * q).sum(dim=1, keepdim=True)
                q = q - a * y_
                alphas.append(a)
            if y_hist:
                sy = (s_hist[-1] * y_hist[-1]).sum(dim=1, keepdim=True)
                yy = (y_hist[-1] * y_hist[-1]).sum(dim=1, keepdim=True)
                gamma = torch.where(yy > 0, sy / yy, torch.ones_like(sy))
            else:
                gamma = 1.0 / g.abs().sum(dim=1, keepdim=True).clamp_min(1e-12) * torch.ones_like(step0)
            z = gamma * q
            for (s_, y_, r_), a in zip(zip(s_hist, y_hist, rho_hist, strict=True), reversed(alphas), strict=True):
                b = r_ * (y_ * z).sum(dim=1, keepdim=True)
                z = z + s_ * (a - b)
            d = -z
            gd = (g * d).sum(dim=1, keepdim=True)
            bad = gd >= 0
            d = torch.where(bad, -g, d)
            gd = torch.where(bad, -(g * g).sum(dim=1, keepdim=True), gd)
            # backtracking Armijo line search, per seed
            t = step0.clone()
            accepted = torch.zeros(S, dtype=torch.bool, device=x.device)
            x_new, f_new, g_new = x.clone(), f.clone(), g.clone()
            for _ in range(12):
                xt = torch.where(accepted.unsqueeze(1), x_new, x + t * d)
                ft, gt, _ = evaluate(xt)
                ok = (ft <= f + 1e-4 * t.squeeze(1) * gd.squeeze(1)) & torch.isfinite(ft) & ~accepted
                x_new = torch.where(ok.unsqueeze(1), xt, x_new)
                f_new = torch.where(ok, ft, f_new)
                g_new = torch.where(ok.unsqueeze(1), gt, g_new)
                accepted = accepted | ok
                if bool(accepted[active].all()):
                    break
                t = torch.where(accepted.unsqueeze(1), t, t * 0.5)
            upd = accepted & active
            s_k = torch.where(upd.unsqueeze(1), x_new - x, torch.zeros_like(x))
            y_k = torch.where(upd.unsqueeze(1), g_new - g, torch.zeros_like(x))
            sy = (s_k * y_k).sum(dim=1, keepdim=True)
            if bool((sy > 1e-10).any()):
                good = sy > 1e-10
                s_hist.append(torch.where(good, s_k, torch.zeros_like(s_k)))
                y_hist.append(torch.where(good, y_k, torch.zeros_like(y_k)))
                rho_hist.append(torch.where(good, 1.0 / sy.clamp_min(1e-10), torch.zeros_like(sy)))
                if len(s_hist) > m:
                    s_hist.pop(0)
                    y_hist.pop(0)
                    rho_hist.pop(0)
            x = torch.where(upd.unsqueeze(1), x_new, x)
            f = torch.where(upd, f_new, f)
            g = torch.where(upd.unsqueeze(1), g_new, g)
            active = active & (accepted | (k < 3))
            if progress and (k % 20 == 0 or k == iters - 1):
                frac = (self.cfg.adam_iters + 1.5 * (k + 1)) / (self.cfg.adam_iters + 1.5 * iters)
                progress(frac, f"L-BFGS {k + 1}/{iters}")
            if not bool(active.any()):
                break
        total, _, parts = evaluate(x)
        self._log(hist, "lbfgs", total, parts)
        graph = None  # release the captured graph


def cuda_usable() -> bool:
    """A CUDA device is visible and accepts a small allocation (is_available() alone can be True with none visible)."""
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            return False
        torch.zeros(1, device="cuda").add_(1).item()
        return True
    except (RuntimeError, AssertionError):
        return False


def resolve_device(device: str = "auto", precision: str = "auto") -> tuple[str, str, str | None]:
    """Pick (device, dtype, note). GPU → float32 by default; no GPU → CPU float64 with a note (spec F6)."""
    note = None
    if device in ("auto", "cuda"):
        if cuda_usable():
            dev = "cuda"
        else:
            dev = "cpu"
            note = "GPU를 쓸 수 없어 CPU(float64)로 전환"
    else:
        dev = "cpu"
    if precision == "auto":
        dtype = "float32" if dev == "cuda" else "float64"
    else:
        dtype = precision
    return dev, dtype, note


def train_pinn(
    exps: list[PinnExperiment],
    prior_center: np.ndarray,
    cfg: PinnConfig,
    progress: Callable[[float, str], None] | None = None,
) -> list[PinnResult]:
    if cfg.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    model = GeneralPinn(exps, prior_center, cfg)
    return model.train(progress)
