"""CRNT structure verdict (preconditions of theorem 149), ported from CADKIMM ``analysis/crnt_check.py``.

Input: reactions as ``(reactant complex, product complex)`` pairs, each complex a ``{species: coefficient}`` dict.
Output: linkage classes, strongly connected components, weak reversibility, deficiency and conservation laws.

The decision logic (weak reversibility = every linkage class equals one strongly connected component,
deficiency = complexes - linkage classes - rank) is fixed by the spec (E7) and must not be changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd

import numpy as np


def _key(c):
    return tuple(sorted((s, float(v)) for s, v in c.items() if v))


def build(rxns):
    complexes = {}
    for a, b in rxns:
        for c in (a, b):
            complexes.setdefault(_key(c), len(complexes))
    species = sorted({s for a, b in rxns for c in (a, b) for s in c})
    edges = [(complexes[_key(a)], complexes[_key(b)]) for a, b in rxns]
    return complexes, species, edges


def components(n, edges, directed):
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        if not directed:
            adj[v].add(u)
    seen, comps = set(), []
    for s in range(n):
        if s in seen:
            continue
        stack, comp = [s], set()
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            stack.extend(adj[u] - comp)
        seen |= comp
        comps.append(comp)
    return comps


def strongly_connected(n, edges):
    """Kosaraju. The first pass is an iterative post-order DFS (same order as the recursive original)."""
    adj = {i: [] for i in range(n)}
    radj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
        radj[v].append(u)
    order, seen = [], set()
    for root in range(n):
        if root in seen:
            continue
        seen.add(root)
        stack = [(root, iter(adj[root]))]
        while stack:
            u, it = stack[-1]
            advanced = False
            for w in it:
                if w not in seen:
                    seen.add(w)
                    stack.append((w, iter(adj[w])))
                    advanced = True
                    break
            if not advanced:
                order.append(u)
                stack.pop()
    seen, sccs = set(), []
    for u in reversed(order):
        if u in seen:
            continue
        stack, comp = [u], set()
        seen.add(u)
        while stack:
            x = stack.pop()
            comp.add(x)
            for w in radj[x]:
                if w not in seen:  # skip nodes already assigned to another SCC (CADKIMM fix 2026-10-09)
                    seen.add(w)
                    stack.append(w)
        sccs.append(comp)
    return sccs


def stoichiometric_matrix(rxns, species):
    idx = {s: i for i, s in enumerate(species)}
    N = np.zeros((len(species), len(rxns)))
    for j, (a, b) in enumerate(rxns):
        for s, v in a.items():
            N[idx[s], j] -= v
        for s, v in b.items():
            N[idx[s], j] += v
    return N


def conservation_laws(N, tol=1e-9):
    """Left null space basis: rows w with w·N = 0 (orthonormal, for numerical checks)."""
    if N.shape[1] == 0:
        return np.eye(N.shape[0])
    _, s, vt = np.linalg.svd(N.T)
    rank = int((s > tol).sum())
    return vt[rank:]


def conservation_laws_integer(N):
    """Integer basis of the same left null space (readable form such as 'total enzyme').

    Reduced row echelon form of N^T over the rationals, then each null vector scaled to integers.
    """
    A = [[Fraction(int(round(v))) for v in row] for row in N.T]  # reactions x species
    m, n = len(A), len(A[0]) if A else N.shape[0]
    piv, r = [], 0
    for c in range(n):
        if r == m:
            break
        p = next((i for i in range(r, m) if A[i][c] != 0), None)
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        pv = A[r][c]
        A[r] = [v / pv for v in A[r]]
        for i in range(m):
            if i != r and A[i][c] != 0:
                f = A[i][c]
                A[i] = [a - f * b for a, b in zip(A[i], A[r], strict=True)]
        piv.append(c)
        r += 1
    free = [c for c in range(n) if c not in piv]
    basis = []
    for fc in free:
        v = [Fraction(0)] * n
        v[fc] = Fraction(1)
        for i, pc in enumerate(piv):
            v[pc] = -A[i][fc]
        den = 1
        for x in v:
            den = den * x.denominator // gcd(den, x.denominator)
        iv = [int(x * den) for x in v]
        g = 0
        for x in iv:
            g = gcd(g, abs(x))
        iv = [x // g for x in iv] if g else iv
        if sum(x < 0 for x in iv) > sum(x > 0 for x in iv):
            iv = [-x for x in iv]
        basis.append(iv)
    return np.array(basis, dtype=int).reshape(len(basis), n)


def law_text(w, species) -> str:
    parts = []
    for c, sp in zip(w, species, strict=True):
        if c == 0:
            continue
        mag = "" if abs(c) == 1 else f"{abs(c)}·"
        sign = ("− " if c < 0 else "") if not parts else ("− " if c < 0 else "+ ")
        parts.append(f"{sign}{mag}{sp}")
    return " ".join(parts)


# ---------------------------------------------------------------------------------------------- verdict
GUARANTEE_FULL = "A"  # weakly reversible, deficiency 0
GUARANTEE_PERSIST = "B"  # weakly reversible, deficiency > 0
GUARANTEE_NONE = "C"  # not weakly reversible

NO_149_BANNER = "149 보장 없음: 멸종 가능"


@dataclass
class Verdict:
    name: str
    species: list[str]
    complexes: list[str]
    n_reactions: int
    linkage_classes: list[list[int]]
    sccs: list[list[int]]
    rank: int
    deficiency: int
    weakly_reversible: bool
    conservation_laws: list[list[int]]
    conservation_text: list[str]
    irreversible_complexes: list[str] = field(default_factory=list)
    edges: list[list[int]] = field(default_factory=list)

    @property
    def grade(self) -> str:
        if not self.weakly_reversible:
            return GUARANTEE_NONE
        return GUARANTEE_FULL if self.deficiency == 0 else GUARANTEE_PERSIST

    @property
    def guarantee_149(self) -> bool:
        return self.weakly_reversible

    @property
    def unique_equilibrium(self) -> bool:
        """Deficiency-zero theorem: one positive equilibrium per compatibility class (algebraic limit usable)."""
        return self.weakly_reversible and self.deficiency == 0

    def headline(self) -> str:
        if self.grade == GUARANTEE_FULL:
            return f"약가역 ✔ 결핍 {self.deficiency} → 멸종·발산 없음, 적합류마다 평형 하나"
        if self.grade == GUARANTEE_PERSIST:
            return f"약가역 ✔ 결핍 {self.deficiency} → 멸종·발산 없음 (평형 유일성 보장 없음)"
        return f"약가역 ✘ 결핍 {self.deficiency} → {NO_149_BANNER}"

    def guarantee_line(self) -> str:
        """One line for readers who do not know CRNT: what the verdict guarantees and under which premises."""
        prem = "전제: 질량작용, 일정한 속도 상수(등온), 닫힌 계, 양의 초기 농도"
        if self.grade == GUARANTEE_FULL:
            what = (
                "보장: 속도 상수가 무엇이든 어떤 종도 0으로 사라지거나 무한히 커지지 않고, "
                "보존량이 같은 실험은 같은 평형 하나로 간다"
            )
        elif self.grade == GUARANTEE_PERSIST:
            what = "보장: 속도 상수가 무엇이든 어떤 종도 0으로 사라지거나 무한히 커지지 않는다(평형 개수는 보장 밖)"
        else:
            what = "보장 없음: 한 방향 단계가 있어 어떤 종이 0으로 사라질 수 있다(정리 149 적용 불가)"
        return f"{what}. {prem}."

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "species": self.species,
            "complexes": self.complexes,
            "n_complexes": len(self.complexes),
            "n_reactions": self.n_reactions,
            "linkage_classes": self.linkage_classes,
            "n_linkage_classes": len(self.linkage_classes),
            "strongly_connected_components": self.sccs,
            "rank": self.rank,
            "deficiency": self.deficiency,
            "weakly_reversible": self.weakly_reversible,
            "conservation_laws": self.conservation_laws,
            "conservation_text": self.conservation_text,
            "irreversible_complexes": self.irreversible_complexes,
            "edges": self.edges,
            "grade": self.grade,
            "guarantee_149": self.guarantee_149,
            "unique_equilibrium": self.unique_equilibrium,
            "headline": self.headline(),
            "guarantee_line": self.guarantee_line(),
        }

    def report(self) -> str:
        lines = [
            f"[{self.name}] {self.headline()}",
            self.guarantee_line(),
            f"species: {self.species}",
            f"complexes: {len(self.complexes)}, linkage classes: {len(self.linkage_classes)}, "
            f"rank S: {self.rank}, deficiency: {self.deficiency}",
            f"weakly reversible: {self.weakly_reversible}",
            "149 판정: "
            + (
                "약가역 → 영속(멸종·발산 없음) 예측"
                if self.weakly_reversible
                else "약가역 아님 → 정리 적용 불가. 멸종 가능(한 방향 단계 존재)"
            ),
            "conservation laws (rows = coefficients over species):",
        ]
        lines += [f"  const = {t}" for t in self.conservation_text]
        if self.irreversible_complexes:
            lines.append(f"비가역이 걸린 복합체: {self.irreversible_complexes}")
        return "\n".join(lines)


def _complex_label(key) -> str:
    if not key:
        return "0"
    return " + ".join(s if v == 1 else f"{v:g}{s}" for s, v in key)


def verdict(rxns, name: str = "network") -> Verdict:
    complexes, species, edges = build(rxns)
    n = len(complexes)
    linkage = components(n, edges, directed=False)
    sccs = strongly_connected(n, edges)
    weakly_reversible = all(any(lc == scc for scc in sccs) for lc in linkage)
    N = stoichiometric_matrix(rxns, species)
    s = int(np.linalg.matrix_rank(N)) if N.size else 0
    deficiency = n - len(linkage) - s
    W = conservation_laws_integer(N)
    names = {v: k for k, v in complexes.items()}
    labels = [_complex_label(names[i]) for i in range(n)]
    bad = [labels[i] for lc in linkage if not any(lc == scc for scc in sccs) for i in sorted(lc)]
    return Verdict(
        name=name,
        species=species,
        complexes=labels,
        n_reactions=len(rxns),
        linkage_classes=[sorted(c) for c in linkage],
        sccs=[sorted(c) for c in sccs],
        rank=s,
        deficiency=deficiency,
        weakly_reversible=weakly_reversible,
        conservation_laws=[[int(x) for x in w] for w in W],
        conservation_text=[law_text(w, species) for w in W],
        irreversible_complexes=bad,
        edges=[[int(u), int(v)] for u, v in edges],
    )


def report(rxns, name: str = "network") -> str:
    return verdict(rxns, name).report()
