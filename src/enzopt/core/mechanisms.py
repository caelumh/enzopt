"""Mechanism library: choose a mechanism type and name the molecules; the reaction steps are written for you.

``project.yaml``::

    network:
      mechanism:
        type: ordered_bibi                 # one of MECHANISMS (enzopt mechanisms lists them)
        substrates: [NADH, Pyr]            # binding order (A, B, C)
        products: [Lac, NAD]               # release order (P, Q, R)
        enzyme: E                          # free enzyme (default E)
        letters: {NADH: N, Pyr: P, Lac: L, NAD: Q}   # one letter per molecule for enzyme-form names
        add:                               # side steps, any number
          - dead_end: {form: EQ, ligand: Pyr}         # X + Y <=> XY: inhibitor, substrate inhibition, abortive
          - step: "EI + S <=> ESI"                    # any extra line written by hand
            enzyme_forms: [ESI]
          - background: true                          # the overall reaction without enzyme
      observable: ...
      thermodynamics: {Kp: 23}            # overall and haldane_step are filled in from the mechanism
      prior: ...                          # counts must match the generated steps

Enzyme-form names: the enzyme followed by the letters of what is bound, substrates in binding order (E + N → EN,
EN + P → ENP); a product complex lists the molecule that leaves last first (EQL: Q = NAD⁺ leaves after L = lactate).
With the LDH letters above, ``ordered_bibi`` writes exactly the hand-written LDH network, so the PINN accepts it.
Without ``letters`` each molecule's first letter is used; if two would share a letter, forms are written in full
(E_NADH_Pyr).

The steps of the catalytic cycle come first and ``haldane_step`` is the last of them (k₋ of that step from K').
Pure Python: part of the KIMM-CAD compatible core.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_']*$")


class MechanismError(ValueError):
    """Raised for an unusable mechanism block."""


@dataclass
class Mechanism:
    key: str
    title: str
    n_substrates: int
    n_products: int
    cycle: int | None  # number of catalytic-cycle steps (Haldane on the last); None = one-way cycle
    build: object  # (E, subs, prods, form, extra) -> list[str]
    note: str = ""


def _uni_irr(E, s, p, f, x):
    (S,), (P,) = s, p
    return [f"{E} + {S} <=> {f(S)}", f"{f(S)} -> {E} + {P}"]


def _uni(E, s, p, f, x):
    (S,), (P,) = s, p
    return [f"{E} + {S} <=> {f(S)}", f"{f(S)} <=> {E} + {P}"]


def _uni_iso(E, s, p, f, x):
    (S,), (P,) = s, p
    return [f"{E} + {S} <=> {f(S)}", f"{f(S)} <=> {f(P)}", f"{f(P)} <=> {E} + {P}"]


def _uni_bi(E, s, p, f, x):
    (A,), (P, Q) = s, p
    return [f"{E} + {A} <=> {f(A)}", f"{f(A)} <=> {f(Q, P)}", f"{f(Q, P)} <=> {f(Q)} + {P}", f"{f(Q)} <=> {E} + {Q}"]


def _ordered_bibi(E, s, p, f, x):
    (A, B), (P, Q) = s, p
    return [
        f"{E} + {A} <=> {f(A)}",
        f"{f(A)} + {B} <=> {f(A, B)}",
        f"{f(A, B)} <=> {f(Q, P)}",
        f"{f(Q, P)} <=> {f(Q)} + {P}",
        f"{f(Q)} <=> {E} + {Q}",
    ]


def _theorell_chance(E, s, p, f, x):
    (A, B), (P, Q) = s, p
    return [f"{E} + {A} <=> {f(A)}", f"{f(A)} + {B} <=> {f(Q)} + {P}", f"{f(Q)} <=> {E} + {Q}"]


def _random_bibi(E, s, p, f, x):
    (A, B), (P, Q) = s, p
    return _ordered_bibi(E, s, p, f, x) + [
        f"{E} + {B} <=> {f(B)}",
        f"{f(B)} + {A} <=> {f(A, B)}",
        f"{f(Q, P)} <=> {f(P)} + {Q}",
        f"{f(P)} <=> {E} + {P}",
    ]


def _ping_pong(E, s, p, f, x):
    (A, B), (P, Q) = s, p
    F = x["modified"]
    return [
        f"{E} + {A} <=> {f(A)}",
        f"{f(A)} <=> {F} + {P}",
        f"{F} + {B} <=> {f(B, base=F)}",
        f"{f(B, base=F)} <=> {E} + {Q}",
    ]


def _ordered_terter(E, s, p, f, x):
    (A, B, C), (P, Q, R) = s, p
    return [
        f"{E} + {A} <=> {f(A)}",
        f"{f(A)} + {B} <=> {f(A, B)}",
        f"{f(A, B)} + {C} <=> {f(A, B, C)}",
        f"{f(A, B, C)} <=> {f(R, Q, P)}",
        f"{f(R, Q, P)} <=> {f(R, Q)} + {P}",
        f"{f(R, Q)} <=> {f(R)} + {Q}",
        f"{f(R)} <=> {E} + {R}",
    ]


MECHANISMS: dict[str, Mechanism] = {
    m.key: m
    for m in [
        Mechanism(
            "uni_irreversible",
            "기질 1개, 비가역 (교과서 Michaelis–Menten)",
            1,
            1,
            None,
            _uni_irr,
            "한 방향 단계가 있어 약가역이 아님(등급 C). 실제 반응은 대개 가역이라 uni를 권함",
        ),
        Mechanism("uni", "기질 1개, 가역", 1, 1, 2, _uni),
        Mechanism("uni_iso", "기질 1개, 가역 + 효소 위 변환", 1, 1, 3, _uni_iso),
        Mechanism("uni_bi_ordered", "기질 1개 → 생성물 2개, 순서형", 1, 2, 4, _uni_bi),
        Mechanism("ordered_bibi", "기질 2개, 순서형 (LDH)", 2, 2, 5, _ordered_bibi),
        Mechanism("theorell_chance", "기질 2개, Theorell–Chance (가운데 복합체가 안 쌓임)", 2, 2, 3, _theorell_chance),
        Mechanism(
            "random_bibi",
            "기질 2개, 무작위형 (어느 기질이 먼저 붙어도 됨)",
            2,
            2,
            5,
            _random_bibi,
            "Haldane은 1~5단계 경로에만 걸린다. 갈라진 경로가 만드는 고리의 열역학 제약은 아직 적용하지 않음",
        ),
        Mechanism("ping_pong", "기질 2개, 핑퐁형 (첫 생성물이 나간 뒤 둘째 기질)", 2, 2, 4, _ping_pong),
        Mechanism("ordered_terter", "기질 3개, 순서형 (Ter Ter)", 3, 3, 7, _ordered_terter),
    ]
}


@dataclass
class Expanded:
    mechanism: str
    species: list[str]
    enzyme_species: list[str]
    steps: list[str]
    overall: str
    haldane_step: int | None
    notes: list[str] = field(default_factory=list)


def _names(v, what) -> list[str]:
    if isinstance(v, str):
        v = [v]
    out = [str(x) for x in (v or [])]
    for x in out:
        if not _NAME.match(x):
            raise MechanismError(f"{what}의 이름 '{x}': 영문자나 _로 시작하고 영문자·숫자·_·'만")
    return out


def expand(block: dict) -> Expanded:
    """The steps, species and enzyme forms of a ``network.mechanism`` block."""
    if not isinstance(block, dict):
        raise MechanismError("mechanism은 사전이어야 함(type, substrates, products …)")
    key = str(block.get("type", ""))
    if key not in MECHANISMS:
        raise MechanismError(f"type '{key}': {', '.join(MECHANISMS)} 중 하나")
    m = MECHANISMS[key]
    subs = _names(block.get("substrates"), "substrates")
    prods = _names(block.get("products"), "products")
    if len(subs) != m.n_substrates or len(prods) != m.n_products:
        raise MechanismError(
            f"{key}: 기질 {m.n_substrates}개, 생성물 {m.n_products}개가 필요(받은 것 {len(subs)}개, {len(prods)}개)"
        )
    E = _names(block.get("enzyme", "E"), "enzyme")[0]
    F = _names(block.get("modified_enzyme", "F"), "modified_enzyme")[0]
    mols = subs + prods
    if len(set(mols)) != len(mols) or E in mols:
        raise MechanismError("기질·생성물·효소 이름이 서로 겹침")
    letters = {str(k): str(v) for k, v in (block.get("letters") or {}).items()}
    for k in letters:
        if k not in mols and k not in _addon_ligands(block):
            raise MechanismError(f"letters의 '{k}'이 기질·생성물·곁가지 분자에 없음")
    for k in mols + _addon_ligands(block):
        letters.setdefault(k, k[0].upper())
    short = [letters[k] for k in mols + _addon_ligands(block)]
    full = len(set(short)) != len(short) or not all(_NAME.match(E + s) for s in short)

    def form(*bound, base=None):
        b = base or E
        return "_".join([b] + list(bound)) if full else b + "".join(letters[x] for x in bound)

    steps = m.build(E, subs, prods, form, {"modified": F})
    enzymes = [E]
    for st in steps:  # every name in the cycle that is not a substrate or product is an enzyme form
        for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_']*", st):
            if tok not in mols and tok not in enzymes:
                enzymes.append(tok)
    notes = [m.note] if m.note else []
    if full:
        notes.append("두 분자의 첫 글자가 같아 효소 형태를 전체 이름으로 씀. letters로 한 글자씩 정하면 짧아짐")
    for add in block.get("add") or []:
        steps_add, forms = _addon(add, form, subs, prods)
        steps += steps_add
        enzymes += [f for f in forms if f not in enzymes]
    species, seen = [], set()
    for st in steps:
        for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_']*", st):
            if tok not in seen:
                seen.add(tok)
                species.append(tok)
    enzymes = [s for s in species if s in enzymes]
    return Expanded(
        mechanism=key,
        species=species,
        enzyme_species=[E] + [s for s in enzymes if s != E],
        steps=steps,
        overall=f"{' + '.join(subs)} <=> {' + '.join(prods)}",
        haldane_step=m.cycle,
        notes=notes,
    )


def _addon_ligands(block: dict) -> list[str]:
    out = []
    for add in block.get("add") or []:
        if isinstance(add, dict) and isinstance(add.get("dead_end"), dict):
            lig = str(add["dead_end"].get("ligand", ""))
            if lig and lig not in out:
                out.append(lig)
    subs = [str(x) for x in block.get("substrates") or []]
    prods = [str(x) for x in block.get("products") or []]
    return [x for x in out if x not in subs + prods]


def _addon(add, form, subs, prods) -> tuple[list[str], list[str]]:
    if not isinstance(add, dict) or len(add) != 1 and not ("step" in add and set(add) <= {"step", "enzyme_forms"}):
        raise MechanismError(f"add 항목은 dead_end, step, background 중 하나: {add}")
    if "dead_end" in add:
        d = add["dead_end"]
        X, Y = _names(d.get("form"), "dead_end.form")[0], _names(d.get("ligand"), "dead_end.ligand")[0]
        name = _names(d.get("name"), "dead_end.name")[0] if d.get("name") else None
        if name is None:
            name = f"{X}_{Y}" if "_" in form(subs[0]) else X + form(Y)[len(form()) :]
        return [f"{X} + {Y} <=> {name}"], [name]
    if "step" in add:
        return [str(add["step"])], _names(add.get("enzyme_forms") or [], "step.enzyme_forms")
    if "background" in add:
        return ([f"{' + '.join(subs)} <=> {' + '.join(prods)}"], []) if add["background"] else ([], [])
    raise MechanismError(f"add 항목은 dead_end, step, background 중 하나: {add}")


def generic_example(key: str) -> Expanded:
    """The mechanism written with placeholder names (A, B, C → P, Q, R; S → P for one substrate)."""
    m = MECHANISMS[key]
    subs = ["S"] if m.n_substrates == 1 and m.n_products == 1 else ["A", "B", "C"][: m.n_substrates]
    prods = ["P", "Q", "R"][: m.n_products]
    return expand({"type": key, "substrates": subs, "products": prods})
