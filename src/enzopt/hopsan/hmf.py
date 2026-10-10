"""Mass-action network → Hopsan .hmf (signal-block ODE), ported from CADKIMM ``hopsan/make_hmf.py``.

For each species: integrator ``X_<s>`` (initial state ``out#Value``) ← sum ``dXdt_<s>`` ← per-reaction gains
(k = ν·k_j) ← rate term (chain of SignalMultiply of the *delayed* concentrations ``Xd_<s>``). The unit delay
``Xd_<s>`` breaks the algebraic loop, so the scheme is explicit with O(Δt) error: Δt must be well below the fastest
time scale. Block names are an interface (parameter import and result reading rely on them).

Every block also gets a position (``<hopsangui><pose>``) so the model is readable when opened in the Hopsan GUI:
one row per species (sum → integrator → delay), the rate chain of reaction j in column j to the right, the gains
of reaction j in column j to the left. The CLI ignores positions.
"""

from __future__ import annotations

INIT_PARAM = "out#Value"  # SignalIntegrator2 initial output (state)
GAIN_PARAM = "k#Value"  # SignalGain gain
CONST_PARAM = "y#Value"  # SignalConstant value
ORIGIN = (2300.0, 2300.0)  # Hopsan GUI scene coordinates of the first species row (its examples sit near 2500)
ROW, COL = 120.0, 230.0


class HmfBuilder:
    def __init__(self, name, stop, timestep, nsamples=2048):
        self.name = name
        self.stop, self.timestep, self.nsamples = stop, timestep, nsamples
        self.components = []  # (typename, name, [(param, value)], [ports])
        self.connections = []  # (startcomp, startport, endcomp, endport)
        self.pose = {}  # name -> (x, y) relative to ORIGIN

    def place(self, name, x, y):
        self.pose[name] = (float(x), float(y))

    def add(self, typename, name, params=(), ports=()):
        self.components.append((typename, name, list(params), list(ports)))
        return name

    def connect(self, a, ap, b, bp):
        self.connections.append((a, ap, b, bp))

    def _poses(self):
        """Positions of every block; blocks nobody placed go in one row under the model."""
        pose = dict(self.pose)
        bottom = max((y for _, y in pose.values()), default=0.0) + 2 * ROW
        free = [name for _, name, _, _ in self.components if name not in pose]
        for i, name in enumerate(free):
            pose[name] = (i * 120.0, bottom)
        x0, y0 = ORIGIN
        return {n: (x0 + x, y0 + y) for n, (x, y) in pose.items()}

    def xml(self):
        pose = self._poses()
        xs = [x for x, _ in pose.values()] or [ORIGIN[0]]
        ys = [y for _, y in pose.values()] or [ORIGIN[1]]
        L = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<hopsanmodelfile hmfversion="0.4" hopsancoreversion="2.24.4" hopsanguiversion="2.24.4">',
            f'  <system typename="Subsystem" name="{self.name}" cqstype="UndefinedCQSType" '
            'subtypename="" locked="false">',
            f'    <simulationtime start="0" inherit_timestep="true" timestep="{self.timestep:g}" '
            f'stop="{self.stop:g}"/>',
            f'    <simulationlogsettings numsamples="{self.nsamples}" starttime="0"/>',
            "    <parameters/>",
            "    <aliases/>",
            "    <hopsangui>",
            '      <pose y="0" x="0" flipped="false" a="0"/>',
            '      <nametext visible="0" position="0"/>',
            f'      <viewport y="{(min(ys) + max(ys)) / 2:.0f}" x="{(min(xs) + max(xs)) / 2:.0f}" zoom="0.6"/>',
            '      <ports hidden="1"/>',
            '      <names hidden="0"/>',
            "    </hopsangui>",
            "    <objects>",
        ]
        for typename, name, params, ports in self.components:
            L.append(f'      <component typename="{typename}" name="{name}" cqstype="S" subtypename="" locked="false">')
            L.append("        <parameters>")
            for p, v in params:
                L.append(f'          <parameter type="double" name="{p}" value="{float(v)!r}" unit=""/>')
            L.append("        </parameters>")
            for pt in ports:
                L.append(f'        <port nodetype="NodeSignal" name="{pt}"/>')
            x, y = pose[name]
            L.append("        <hopsangui>")
            L.append(f'          <pose y="{y:.0f}" x="{x:.0f}" flipped="false" a="0"/>')
            L.append('          <nametext visible="1" position="0"/>')
            L.append("        </hopsangui>")
            L.append("      </component>")
        L.append("    </objects>")
        L.append("    <connections>")
        for a, ap, b, bp in self.connections:
            L.append(f'      <connect startcomponent="{a}" startport="{ap}" endcomponent="{b}" endport="{bp}"/>')
        L.append("    </connections>")
        L += ["  </system>", "</hopsanmodelfile>", ""]
        return "\n".join(L)


def build_mass_action(name, species, reactions, x0, stop, timestep, nsamples=2048, extras=None):
    """species: names; x0: {species: initial mM}; reactions: [(a, b, k)] or [(a, b, k, mod)].

    With ``mod`` the rate is k · mod(t) · Π[reactants] (e.g. a pump flow). ``extras(B)`` is called after the species
    blocks and returns {mod name: (component, port)}; species concentrations are ("X_<s>", "out").
    """
    B = HmfBuilder(name, stop, timestep, nsamples)
    row = {s: i for i, s in enumerate(species)}
    for s in species:
        B.add("SignalIntegrator2", "X_" + s, [(INIT_PARAM, x0.get(s, 0.0))], ["in", "out"])
        B.add("SignalUnitDelay", "Xd_" + s, [("out#Value", x0.get(s, 0.0))], ["in", "out"])
        B.connect("X_" + s, "out", "Xd_" + s, "in")
        B.add("SignalSum", "dXdt_" + s, [], ["in", "out"])
        B.connect("dXdt_" + s, "out", "X_" + s, "in")
        y = row[s] * ROW
        B.place("dXdt_" + s, 0, y)
        B.place("X_" + s, 110, y)
        B.place("Xd_" + s, 220, y)
    mods = extras(B) if extras else {}
    for j, rx in enumerate(reactions):
        a, b, k = rx[0], rx[1], rx[2]
        mod = rx[3] if len(rx) > 3 else None
        factors = [("Xd_" + s, "out") for s, c in a.items() for _ in range(int(c))]
        if mod is not None:
            factors.append(mods[mod])
        xr = 220 + COL * (j + 1)
        yr = min((row[s] for s in a if s in row), default=-1) * ROW
        if len(factors) == 0:
            src = B.add("SignalConstant", f"one_{j}", [(CONST_PARAM, 1.0)], ["y"])
            B.place(src, xr, yr)
            rate_comp, rate_port = src, "y"
        elif len(factors) == 1:
            rate_comp, rate_port = factors[0]
        else:
            prev_comp, prev_port = factors[0]
            for i, (fc, fp) in enumerate(factors[1:]):
                m = B.add("SignalMultiply", f"r{j}_m{i}", [], ["in1", "in2", "out"])
                B.place(m, xr + 90 * i, yr + 45 * i)
                B.connect(prev_comp, prev_port, m, "in1")
                B.connect(fc, fp, m, "in2")
                prev_comp, prev_port = m, "out"
            rate_comp, rate_port = prev_comp, prev_port
        net = {}
        for s, c in a.items():
            net[s] = net.get(s, 0.0) - c
        for s, c in b.items():
            net[s] = net.get(s, 0.0) + c
        for s, nu in net.items():
            if abs(nu) < 1e-15:
                continue
            g = B.add("SignalGain", f"r{j}_to_{s}", [(GAIN_PARAM, nu * k)], ["in", "k", "out"])
            B.place(g, -150 - 150 * j, row[s] * ROW)
            B.connect(rate_comp, rate_port, g, "in")
            B.connect(g, "out", "dXdt_" + s, "in")
    B.add("SignalSink", "Scope", [], ["in", "in_right", "in_bottom"])
    B.place("Scope", 110, -1.5 * ROW)
    for s in species:
        B.connect("X_" + s, "out", "Scope", "in")
    return B.xml()


def network_hmf(name, network, k, x0: dict, stop, timestep, nsamples=2048) -> str:
    """.hmf for an enzopt Network with rate vector ``k`` aligned with ``network.reactions()``."""
    rx = [(a, b, float(kj)) for (a, b), kj in zip(network.reactions(), k, strict=True)]
    return build_mass_action(name, network.species, rx, x0, stop, timestep, nsamples)


def write_params(path, x0: dict) -> None:
    """Parameter-import CSV that sets both the integrator state and the delay of every species."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        for s, v in x0.items():
            f.write(f"X_{s}#out#Value,{float(v)!r}\n")
            f.write(f"Xd_{s}#out#Value,{float(v)!r}\n")
