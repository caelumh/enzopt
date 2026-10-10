"""PINN for the LDH ordered bi-bi mechanism (9 species, 5 reversible steps), batched over seeds on the GPU.

Ported from CADKIMM ``analysis/pinn_full_mechanism.py`` and reorganised for speed:

* one small tanh MLP per (seed, training experiment), all evaluated as one batched tensor program;
* time derivatives by forward-mode tangent propagation through the MLP (no double backward);
* float32 on the GPU by default, float64 on the CPU as reference/fallback (spec F6).

Representation (exact positivity of the slow variable's range, enzyme total and the 4 conservation laws):
    NADH = lo + (hi − lo)·sigmoid(u),   φ = softmax(v),   (E, EN, ENP, EQL, EQ) = Et·φ
    NAD = Ntot − NADH − (EN+ENP+EQL+EQ),  Lac = C3 + EQ + NAD,  Pyr = Stot − Lac − ENP − EQL
Loss per seed = data + w_phys·(slow + fast residuals) + w_ic·IC + w_prior·prior, averaged over experiments.

Change from CADKIMM (decision 2026-10-09, see PROGRESS.md): the slow residual is scaled by the experiment's
characteristic rate (hi − lo)/T instead of Ntot/100 per s, and w_phys is ramped up during Adam to a large final
value. With the CADKIMM scaling the network fitted the data while the ODE residual stayed ≈ 25 %, so the learned
rate constants did not reproduce the data when the ODE was re-solved.

Enzyme inactivation (option ``inactivation: shared``): the active enzyme is Et·a(t), t = time since the enzyme was
added, with one a(t) shared by all experiments and learned with the rate constants, without choosing its shape:
a = exp(−h(t/T)), h = Σ_k w_k [σ((u − c_k)/s) − σ(−c_k/s)], w_k = softplus(·) ≥ 0, so a(0) = 1 and a never rises.
In the residual it only rescales the enzyme forms, so the cost of learning a whole function is K more numbers per
seed and no ODE solve (a classical fit of the same function has to integrate the ODE for every trial of its K
coefficients).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

SPECIES = ["E", "NADH", "EN", "Pyr", "ENP", "EQL", "Lac", "EQ", "NAD"]
STEPS = [
    ({"E": 1, "NADH": 1}, {"EN": 1}),
    ({"EN": 1, "Pyr": 1}, {"ENP": 1}),
    ({"ENP": 1}, {"EQL": 1}),
    ({"EQL": 1}, {"EQ": 1, "Lac": 1}),
    ({"EQ": 1}, {"E": 1, "NAD": 1}),
]
ENZYME = ["E", "EN", "ENP", "EQL", "EQ"]
N_THETA_WR = 10  # log10 kf(5), log10 kr1..kr4, log10 K'
N_THETA_IRR = 9  # last step one-way: no K'


def matches_network(network) -> bool:
    """True if ``network`` is the LDH ordered bi-bi mechanism (step order as in STEPS)."""
    if sorted(network.species) != sorted(SPECIES) or len(network.steps) != len(STEPS):
        return False
    for st, (a, b) in zip(network.steps, STEPS, strict=True):
        if st.lhs != a or st.rhs != b:
            return False
    return all(st.reversible for st in network.steps[:4])


@dataclass
class PinnExperiment:
    name: str
    initial: dict
    Et: float  # mM
    t_obs: np.ndarray
    a_obs: np.ndarray


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
    irreversible_last: bool = False
    device: str = "cpu"
    dtype: str = "float64"
    representation: str = "qss"  # "qss": enzyme forms at quasi-steady state (King–Altman); "full": network logits
    fast_norm: str = "slow"  # 'slow': enzyme residuals in units of the substrate rate; 'gross': CADKIMM
    lr_theta: float = 2e-2
    lr_net: float = 3e-3
    n_col_uniform: int = 300
    n_col_early: int = 60
    cuda_graphs: bool = True  # capture each training step as one CUDA graph (removes launch overhead)
    # τ = t/T_e with each experiment's own last time (default) or one shared T_max. Per experiment keeps short runs
    # from crowding into the first few collocation points (Holmes 1988 check, 2026-10-09: set 172 error 11.9 → 5.1 %)
    time_scale: str = "per_experiment"
    # enzyme activity a(t): "none" (a ≡ 1) or "shared" (one monotone a(t) for all experiments, see module docstring)
    inactivation: str = "none"
    inact_k: int = 12  # number of sigmoid steps in h
    w_inact: float = 1e-2  # weight of h(T)² (prefers no loss of activity unless the data need it)


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


def activity_curve(act: dict | None, t) -> np.ndarray:
    """a(t) of a PinnResult (1 everywhere without inactivation)."""
    t = np.asarray(t, float)
    if not act:
        return np.ones_like(t)
    u = t[..., None] / act["T"]
    c, s, w = np.asarray(act["c"]), float(act["s"]), np.asarray(act["w"])
    h = (w * (1.0 / (1.0 + np.exp(-(u - c) / s)) - 1.0 / (1.0 + np.exp(c / s)))).sum(axis=-1)
    return np.exp(-h)


def activity_rate(act: dict | None, t) -> np.ndarray:
    """λ(t) = −d ln a/dt: the first-order loss rate of every enzyme form that gives a(t) (1/s)."""
    t = np.asarray(t, float)
    if not act:
        return np.zeros_like(t)
    u = t[..., None] / act["T"]
    c, s, w = np.asarray(act["c"]), float(act["s"]), np.asarray(act["w"])
    sg = 1.0 / (1.0 + np.exp(-(u - c) / s))
    return (w * sg * (1.0 - sg) / s).sum(axis=-1) / act["T"]


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


class LdhPinn:
    """Seeds × training experiments, trained together; each seed has its own rate constants."""

    def __init__(self, exps: list[PinnExperiment], prior_center: np.ndarray, cfg: PinnConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        self.dt = torch.float32 if cfg.dtype == "float32" else torch.float64
        self.exps = exps
        S, E = len(cfg.seeds), len(exps)
        self.S, self.E, self.B = S, E, S * E
        self.n_theta = N_THETA_IRR if cfg.irreversible_last else N_THETA_WR
        self.inact = cfg.inactivation == "shared"
        pc = np.asarray(prior_center, float)[: self.n_theta]
        self.prior_c = torch.tensor(pc, dtype=self.dt, device=self.dev)
        self.T = float(max(e.t_obs.max() for e in exps))
        T_e = [float(e.t_obs.max()) if cfg.time_scale == "per_experiment" else self.T for e in exps]
        gens = []
        theta0 = []
        for sd in cfg.seeds:
            for _ in range(E):
                g = torch.Generator().manual_seed(int(sd) * 1000 + len(gens) % E)
                gens.append(g)
            jit = np.random.default_rng(100 + int(sd)).uniform(-cfg.init_jitter, cfg.init_jitter, len(pc))
            # inactivation weights (extra θ columns) start near zero: softplus(−5) ≈ 0.007, a(T) ≈ 0.95
            theta0.append(np.concatenate([pc + jit, np.full(cfg.inact_k if self.inact else 0, -5.0)]))
        self.theta = torch.nn.Parameter(torch.tensor(np.array(theta0), dtype=self.dt, device=self.dev))
        self.net = BatchedMLP(
            self.B, cfg.width, cfg.depth, 1 if cfg.representation == "qss" else 6, gens, self.dt, self.dev
        )
        # per-experiment constants, repeated for each seed: index b = s*E + e
        lo, hi, Ntot, Stot, C3, Et, p0 = [], [], [], [], [], [], []
        for e in exps:
            ic = {s: float(e.initial.get(s, 0.0)) for s in ("NADH", "NAD", "Pyr", "Lac")}
            nt = ic["NADH"] + ic["NAD"]
            st = ic["Pyr"] + ic["Lac"]
            D = ic["NADH"] - ic["Pyr"]
            l, h = max(0.0, D), min(nt, st + D)
            if h - l < 1e-9:
                raise ValueError(f"실험 {e.name}: NADH 범위가 0(보존량 확인)")
            lo.append(l)
            hi.append(h)
            Ntot.append(nt)
            Stot.append(st)
            C3.append(ic["Lac"] - ic["NAD"])
            Et.append(e.Et)
            p0.append(float(np.clip((ic["NADH"] - l) / (h - l), 1e-4, 1 - 1e-4)))

        def rep(v):
            return torch.tensor(np.tile(np.array(v, float), S), dtype=self.dt, device=self.dev).reshape(-1, 1, 1)

        self.lo, self.hi, self.Ntot, self.Stot, self.C3, self.Et = map(rep, (lo, hi, Ntot, Stot, C3, Et))
        self.p0 = rep(p0)
        self.Tb = rep(T_e)  # time scale per (seed, experiment)
        tc = np.unique(
            np.concatenate([np.linspace(0, 1, cfg.n_col_uniform), np.geomspace(1e-4, 0.05, cfg.n_col_early)])
        )
        self.tau_col = torch.tensor(tc, dtype=self.dt, device=self.dev).reshape(1, -1, 1).expand(self.B, -1, -1)
        nmax = max(len(e.t_obs) for e in exps)
        tau_obs = np.zeros((E, nmax))
        a_obs = np.zeros((E, nmax))
        mask = np.zeros((E, nmax))
        for i, e in enumerate(exps):
            n = len(e.t_obs)
            tau_obs[i, :n] = e.t_obs / T_e[i]
            a_obs[i, :n] = e.a_obs
            mask[i, :n] = 1.0
        self.tau_obs = torch.tensor(np.tile(tau_obs, (S, 1)), dtype=self.dt, device=self.dev).unsqueeze(-1)
        self.a_obs = torch.tensor(np.tile(a_obs, (S, 1)), dtype=self.dt, device=self.dev).unsqueeze(-1)
        self.mask = torch.tensor(np.tile(mask, (S, 1)), dtype=self.dt, device=self.dev).unsqueeze(-1)
        self.n_obs = self.mask.sum(dim=1, keepdim=True)
        self.tau0 = torch.zeros(self.B, 1, 1, dtype=self.dt, device=self.dev)
        if self.inact:
            K = cfg.inact_k
            self.inact_c = torch.tensor(np.linspace(0.0, 1.0, K), dtype=self.dt, device=self.dev)
            self.inact_s = 0.6 / max(1, K - 1)
            self.inact_u = self.Tb / self.T  # τ → u = t/T_max, per (seed, experiment)

    def activity_weights(self, theta: torch.Tensor):
        """(B, 1, K) non-negative step weights, repeated over each seed's experiments; None without inactivation."""
        if not self.inact:
            return None
        w = torch.nn.functional.softplus(theta[:, self.n_theta :])
        return w.unsqueeze(1).expand(self.S, self.E, w.shape[1]).reshape(self.B, 1, w.shape[1])

    def activity(self, tau: torch.Tensor, wB: torch.Tensor) -> torch.Tensor:
        u = tau * self.inact_u
        c, s = self.inact_c, self.inact_s
        h = (wB * (torch.sigmoid((u - c) / s) - torch.sigmoid(-c / s))).sum(dim=-1, keepdim=True)
        return torch.exp(-h)

    # ------------------------------------------------------------------ model pieces
    def rates(self, theta: torch.Tensor):
        """kf, kr per seed (S, 5). k₋5 from Haldane (WR) or 0 (one-way last step)."""
        kf = 10.0 ** theta[:, :5]
        kr4 = 10.0 ** theta[:, 5:9]
        if self.cfg.irreversible_last:
            kr5 = torch.zeros_like(kf[:, :1])
        else:  # Haldane in log space (torch.prod's backward syncs, which breaks CUDA graph capture)
            kr5 = 10.0 ** (
                theta[:, :5].sum(dim=1, keepdim=True) - theta[:, 5:9].sum(dim=1, keepdim=True) - theta[:, 9:10]
            )
        return kf, torch.cat([kr4, kr5], dim=1)

    def qss_fractions(self, NADH, Pyr, Lac, NAD, kfB, krB):
        """Quasi-steady-state enzyme fractions of the 5-form ring E→EN→ENP→EQL→EQ→E (King–Altman).

        Weight of form i = Σ over the 5 spanning trees of the ring (one edge removed) of the product of the rates
        pointing toward i. Rates are scaled by their maximum first (φ is invariant), so float32 cannot overflow.
        """
        a = [kfB[..., 0:1] * NADH, kfB[..., 1:2] * Pyr, kfB[..., 2:3], kfB[..., 3:4], kfB[..., 4:5]]  # i → i+1
        b = [krB[..., 0:1], krB[..., 1:2], krB[..., 2:3], krB[..., 3:4] * Lac, krB[..., 4:5] * NAD]  # i+1 → i
        ab = torch.broadcast_tensors(*(a + b))
        a, b = list(ab[:5]), list(ab[5:])
        scale = torch.stack(ab, dim=0).amax(dim=0).clamp_min(1e-30)
        a = [x / scale for x in a]
        b = [x / scale for x in b]
        n = 5
        w = []
        for i in range(n):
            tot = 0.0
            for m in range(n):  # removed edge (i+m, i+m+1)
                term = 1.0
                for j in range(1, m + 1):  # nodes i+1..i+m point backward (j → j-1)
                    term = term * b[(i + j - 1) % n]
                for j in range(m + 1, n):  # nodes i+m+1..i+n-1 point forward (j → j+1)
                    term = term * a[(i + j) % n]
                tot = tot + term
            w.append(tot)
        W = torch.cat(w, dim=-1)
        return W / W.sum(dim=-1, keepdim=True)

    def states(self, tau: torch.Tensor, kfB=None, krB=None, wB=None):
        out, dout = self.net(tau)
        u, du = out[..., :1], dout[..., :1]
        sig = torch.sigmoid(u)
        span = self.hi - self.lo
        NADH = self.lo + span * sig
        dNADH = span * sig * (1.0 - sig) * du
        Et = self.Et if wB is None else self.Et * self.activity(tau, wB)
        if self.cfg.representation == "qss":
            NAD0 = self.Ntot - NADH
            Lac0 = self.C3 + NAD0
            Pyr0 = self.Stot - Lac0
            phi = self.qss_fractions(NADH, Pyr0.clamp_min(0.0), Lac0.clamp_min(0.0), NAD0.clamp_min(0.0), kfB, krB)
            dphi = None
        else:
            v, dv = out[..., 1:], dout[..., 1:]
            phi = torch.softmax(v, dim=-1)
            dphi = phi * (dv - (phi * dv).sum(dim=-1, keepdim=True))
        E_, EN, ENP, EQL, EQ = (Et * phi[..., i : i + 1] for i in range(5))
        NAD = self.Ntot - NADH - (EN + ENP + EQL + EQ)
        Lac = self.C3 + EQ + NAD
        Pyr = self.Stot - Lac - ENP - EQL
        S = dict(E=E_, NADH=NADH, EN=EN, Pyr=Pyr, ENP=ENP, EQL=EQL, Lac=Lac, EQ=EQ, NAD=NAD)
        return S, sig, dNADH, phi, dphi

    def loss(self, w_phys: float):
        """Per-seed loss vector (S,) and per-seed parts."""
        cfg = self.cfg
        kf, kr = self.rates(self.theta)
        kfB = kf.unsqueeze(1).expand(self.S, self.E, 5).reshape(self.B, 1, 5)  # (B,1,5), b = s*E + e
        krB = kr.unsqueeze(1).expand(self.S, self.E, 5).reshape(self.B, 1, 5)
        wB = self.activity_weights(self.theta)
        # data
        S, _, _, _, _ = self.states(self.tau_obs, kfB, krB, wB)
        a_pred = cfg.eps340 * (S["NADH"] + S["EN"] + S["ENP"])
        L_data = (((self.a_obs - a_pred) / cfg.sigma) ** 2 * self.mask).sum(dim=1) / self.n_obs.squeeze(1)  # (B,1)
        # physics at collocation points
        S, _, dN, phi, dphi = self.states(self.tau_col, kfB, krB, wB)
        T = self.Tb
        f, g = [], []
        for j, (a, b) in enumerate(STEPS):
            fj = kfB[..., j : j + 1]
            gj = krB[..., j : j + 1]
            for s in a:
                fj = fj * S[s]
            for s in b:
                gj = gj * S[s]
            f.append(fj)
            g.append(gj)
        r = [f[j] - g[j] for j in range(5)]
        s_slow = (self.hi - self.lo) / T
        L_slow = (((dN / T + r[0]) / s_slow) ** 2).mean(dim=1)
        floor = self.Et * 1e-3
        Et = self.Et
        bal = [
            (-r[0] + r[4], f[0] + g[0] + f[4] + g[4]),
            (r[0] - r[1], f[0] + g[0] + f[1] + g[1]),
            (r[1] - r[2], f[1] + g[1] + f[2] + g[2]),
            (r[2] - r[3], f[2] + g[2] + f[3] + g[3]),
        ]
        L_fast = torch.zeros_like(L_slow)
        for i, (net_flux, gsum) in enumerate(bal if cfg.representation == "full" else []):
            scale = (gsum + floor) if cfg.fast_norm == "gross" else s_slow
            L_fast = L_fast + (((Et * dphi[..., i : i + 1] / T - net_flux) / scale) ** 2).mean(dim=1)
        # initial condition on NADH
        _, sig0, _, _, _ = self.states(self.tau0, kfB, krB)
        L_ic = ((sig0 - self.p0) / 0.01) ** 2
        L_ic = L_ic.reshape(self.B, 1)

        def per_seed(x):
            return x.reshape(self.S, self.E).mean(dim=1)

        Ld, Ls, Lf, Li = per_seed(L_data), per_seed(L_slow), per_seed(L_fast), per_seed(L_ic)
        Lp = ((self.theta[:, : self.n_theta] - self.prior_c) ** 2).sum(dim=1) / (2 * cfg.prior_sd**2)
        total = Ld + w_phys * (Ls + Lf) + cfg.w_ic * Li + cfg.w_prior * Lp
        parts = dict(data=Ld, slow=Ls, fast=Lf, ic=Li, prior=Lp)
        if self.inact:  # total loss of activity at the end of the longest experiment, h(1)
            c, s = self.inact_c, self.inact_s
            w = torch.nn.functional.softplus(self.theta[:, self.n_theta :])
            h_end = (w * (torch.sigmoid((1.0 - c) / s) - torch.sigmoid(-c / s))).sum(dim=1)
            parts["inact"] = h_end**2
            total = total + cfg.w_inact * parts["inact"]
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
            w = torch.nn.functional.softplus(self.theta.detach()[:, self.n_theta :]).double().cpu().numpy()
            c = self.inact_c.double().cpu().numpy().tolist()
            act = [{"T": self.T, "c": c, "s": self.inact_s, "w": w[i].tolist()} for i in range(self.S)]
        out = []
        for i, sd in enumerate(cfg.seeds):
            Kp = None if cfg.irreversible_last else float(10 ** th[i, 9])
            r = PinnResult(int(sd), th[i].copy(), kf[i].copy(), kr[i].copy(), Kp, hist[i][-1][2], hist[i], act[i])
            out.append(r)
        return out

    def _log(self, hist, it, total, parts):
        tv = total.detach().double().cpu().numpy()
        pv = {k: v.detach().double().cpu().numpy() for k, v in parts.items()}
        for i in range(self.S):
            hist[i].append((it, float(tv[i]), {k: float(pv[k][i]) for k in pv}))

    def _params(self):
        return [self.theta] + list(self.net.parameters())

    def _lbfgs(self, iters, hist, progress):
        """Batched L-BFGS: every seed keeps its own history and step length (seeds stay independent)."""
        params = self._params()
        S = self.S
        E = self.E

        def seed_view(tensors):
            """Flatten parameter-shaped tensors into (S, n_per_seed)."""
            parts = [tensors[0].reshape(S, -1)]
            for t in tensors[1:]:
                parts.append(t.reshape(S, E, -1).reshape(S, -1))
            return torch.cat(parts, dim=1)

        sizes = [p[0].numel() if i == 0 else p.reshape(S, E, -1)[0].numel() for i, p in enumerate(params)]

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
    model = LdhPinn(exps, prior_center, cfg)
    return model.train(progress)
