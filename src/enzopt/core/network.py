"""Reaction networks: species, reaction steps (reversible pairs or one-way) and rate-constant layout.

A *step* is one line of ``project.yaml`` (``"E + NADH <=> EN"`` or ``"EQ -> E + NAD"``). A *reaction* is one
direction of a step. ``Network.reactions()`` expands steps into reactions with every reverse reaction right
after its forward reaction, which is the convention the CRNT validator relies on to find reversible pairs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

Complex = dict  # {species: stoichiometric coefficient}

_REV_ARROWS = ("<=>", "⇌", "<->")
_FWD_ARROWS = ("->", "→", "=>")
_EMPTY = {"", "0", "∅", "Ø", "{}"}
_TERM = re.compile(r"^\s*(\d+(?:\.\d+)?)?\s*\*?\s*([A-Za-z_][A-Za-z0-9_']*)\s*$")


class NetworkError(ValueError):
    """Raised for malformed reaction networks."""


def parse_complex(text: str) -> Complex:
    """Parse ``"2X + Y"`` into ``{"X": 2, "Y": 1}``. ``"0"``/``"∅"``/empty is the zero complex."""
    text = text.strip()
    if text in _EMPTY:
        return {}
    out: Complex = {}
    for term in text.split("+"):
        m = _TERM.match(term)
        if not m:
            raise NetworkError(f"복합체를 읽을 수 없음: '{text}' (항 '{term.strip()}')")
        coef = float(m.group(1)) if m.group(1) else 1.0
        if coef.is_integer():
            coef = int(coef)
        out[m.group(2)] = out.get(m.group(2), 0) + coef
    return out


def complex_str(c: Complex) -> str:
    if not c:
        return "0"
    parts = []
    for s, v in c.items():
        parts.append(s if v == 1 else f"{v:g}{s}")
    return " + ".join(parts)


@dataclass(frozen=True)
class Step:
    lhs: Complex
    rhs: Complex
    reversible: bool

    @classmethod
    def parse(cls, text: str) -> Step:
        for arrows, rev in ((_REV_ARROWS, True), (_FWD_ARROWS, False)):
            for arrow in arrows:
                if arrow in text:
                    left, right = text.split(arrow, 1)
                    return cls(parse_complex(left), parse_complex(right), rev)
        raise NetworkError(f"반응식에 화살표(<=> 또는 ->)가 없음: '{text}'")

    def __str__(self) -> str:
        return f"{complex_str(self.lhs)} {'<=>' if self.reversible else '->'} {complex_str(self.rhs)}"


@dataclass
class Network:
    """A mass-action reaction network.

    ``species`` keeps the order given in ``project.yaml`` (state-vector order for simulation).
    ``enzyme_species`` are the enzyme forms whose total is the enzyme amount ``Et``.
    """

    name: str
    species: list[str]
    steps: list[Step]
    enzyme_species: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        seen = set(self.species)
        if len(seen) != len(self.species):
            raise NetworkError(f"종 목록에 중복이 있음: {self.species}")
        for st in self.steps:
            for s in list(st.lhs) + list(st.rhs):
                if s not in seen:
                    raise NetworkError(f"반응식 '{st}'의 종 '{s}'이 species 목록에 없음")
        for s in self.enzyme_species:
            if s not in seen:
                raise NetworkError(f"enzyme_species의 '{s}'이 species 목록에 없음")

    @classmethod
    def from_strings(
        cls, name: str, species: list[str], steps: list[str], enzyme_species: list[str] | None = None
    ) -> Network:
        return cls(name, list(species), [Step.parse(s) for s in steps], list(enzyme_species or []))

    # ------------------------------------------------------------------ layout
    def reactions(self) -> list[tuple[Complex, Complex]]:
        """Directed reactions; each reverse reaction follows its forward reaction."""
        out = []
        for st in self.steps:
            out.append((dict(st.lhs), dict(st.rhs)))
            if st.reversible:
                out.append((dict(st.rhs), dict(st.lhs)))
        return out

    def rate_vector(self, kf, kr) -> np.ndarray:
        """Rate constants aligned with ``reactions()``. ``kr[i]`` is ignored for one-way steps."""
        kf = np.asarray(kf, float)
        kr = np.asarray(kr, float)
        if len(kf) != len(self.steps) or len(kr) != len(self.steps):
            raise NetworkError(f"속도 상수 개수가 단계 수({len(self.steps)})와 다름: kf {len(kf)}, kr {len(kr)}")
        out = []
        for i, st in enumerate(self.steps):
            out.append(kf[i])
            if st.reversible:
                out.append(kr[i])
        return np.array(out, float)

    @property
    def n_reactions(self) -> int:
        return sum(2 if st.reversible else 1 for st in self.steps)

    def index(self) -> dict[str, int]:
        return {s: i for i, s in enumerate(self.species)}

    # ------------------------------------------------------------------ variants
    def with_one_way(self, step_numbers: list[int], name: str | None = None) -> Network:
        """Copy with the given steps (1-based) made one-way (forward direction only)."""
        steps = []
        for i, st in enumerate(self.steps, start=1):
            steps.append(Step(st.lhs, st.rhs, False) if i in step_numbers else st)
        return Network(name or f"{self.name}_oneway", list(self.species), steps, list(self.enzyme_species))

    def with_extra_steps(
        self, extra: list[str], name: str | None = None, extra_species: list[str] | None = None
    ) -> Network:
        """Copy with extra steps appended (e.g. a dead-end complex). New species are appended to the species list."""
        species = list(self.species)
        new_steps = [Step.parse(s) for s in extra]
        for st in new_steps:
            for s in list(st.lhs) + list(st.rhs):
                if s not in species:
                    species.append(s)
        enzymes = list(self.enzyme_species) + [s for s in (extra_species or []) if s not in self.enzyme_species]
        return Network(name or f"{self.name}_ext", species, list(self.steps) + new_steps, enzymes)

    def describe(self) -> list[str]:
        return [f"{i}. {st}" for i, st in enumerate(self.steps, start=1)]


def x0_vector(network: Network, initial: dict, Et: float | None = None, free_enzyme: str | None = None) -> np.ndarray:
    """Initial state vector in ``network.species`` order. ``Et`` (mM) is put on ``free_enzyme``."""
    idx = network.index()
    x = np.zeros(len(network.species))
    for s, v in initial.items():
        if s in idx:
            x[idx[s]] = float(v)
    if Et is not None:
        fe = free_enzyme or (network.enzyme_species[0] if network.enzyme_species else None)
        if fe is None:
            raise NetworkError("효소량을 둘 자유 효소 종이 정해지지 않음(enzyme_species)")
        x[idx[fe]] = float(Et)
    return x
