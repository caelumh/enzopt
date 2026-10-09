"""Mass-action network → Hopsan .hmf (signal-block ODE), ported from CADKIMM ``hopsan/make_hmf.py``.

For each species: integrator ``X_<s>`` (initial state ``out#Value``) ← sum ``dXdt_<s>`` ← per-reaction gains
(k = ν·k_j) ← rate term (chain of SignalMultiply of the *delayed* concentrations ``Xd_<s>``). The unit delay
``Xd_<s>`` breaks the algebraic loop, so the scheme is explicit with O(Δt) error: Δt must be well below the fastest
time scale. Block names are an interface (parameter import and result reading rely on them).
"""

from __future__ import annotations

INIT_PARAM = "out#Value"  # SignalIntegrator2 initial output (state)
GAIN_PARAM = "k#Value"  # SignalGain gain
CONST_PARAM = "y#Value"  # SignalConstant value


class HmfBuilder:
    def __init__(self, name, stop, timestep, nsamples=2048):
        self.name = name
        self.stop, self.timestep, self.nsamples = stop, timestep, nsamples
        self.components = []  # (typename, name, [(param, value)], [ports])
        self.connections = []  # (startcomp, startport, endcomp, endport)

    def add(self, typename, name, params=(), ports=()):
        self.components.append((typename, name, list(params), list(ports)))
        return name

    def connect(self, a, ap, b, bp):
        self.connections.append((a, ap, b, bp))

    def xml(self):
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
    for s in species:
        B.add("SignalIntegrator2", "X_" + s, [(INIT_PARAM, x0.get(s, 0.0))], ["in", "out"])
        B.add("SignalUnitDelay", "Xd_" + s, [("out#Value", x0.get(s, 0.0))], ["in", "out"])
        B.connect("X_" + s, "out", "Xd_" + s, "in")
        B.add("SignalSum", "dXdt_" + s, [], ["in", "out"])
        B.connect("dXdt_" + s, "out", "X_" + s, "in")
    mods = extras(B) if extras else {}
    for j, rx in enumerate(reactions):
        a, b, k = rx[0], rx[1], rx[2]
        mod = rx[3] if len(rx) > 3 else None
        factors = [("Xd_" + s, "out") for s, c in a.items() for _ in range(int(c))]
        if mod is not None:
            factors.append(mods[mod])
        if len(factors) == 0:
            src = B.add("SignalConstant", f"one_{j}", [(CONST_PARAM, 1.0)], ["y"])
            rate_comp, rate_port = src, "y"
        elif len(factors) == 1:
            rate_comp, rate_port = factors[0]
        else:
            prev_comp, prev_port = factors[0]
            for i, (fc, fp) in enumerate(factors[1:]):
                m = B.add("SignalMultiply", f"r{j}_m{i}", [], ["in1", "in2", "out"])
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
            B.connect(rate_comp, rate_port, g, "in")
            B.connect(g, "out", "dXdt_" + s, "in")
    B.add("SignalSink", "Scope", [], ["in", "in_right", "in_bottom"])
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
