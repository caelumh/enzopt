"""AC-03: CRNT verdict of the LDH full mechanism, the design mechanism (EQP), pNP and the non-WR control."""

from conftest import LDH_ENZYME, LDH_SPECIES, LDH_STEPS

from enzopt.core.crnt import GUARANTEE_FULL, GUARANTEE_NONE, verdict
from enzopt.core.network import Network
from enzopt.core.project import load_project


def ldh() -> Network:
    return Network.from_strings("ldh", LDH_SPECIES, LDH_STEPS, LDH_ENZYME)


def test_ldh_full_mechanism(base):
    net = ldh()
    rx = net.reactions()
    assert len(net.species) == 9 and len(rx) == 10
    v = verdict(rx, "ldh")
    ref = base["verdict"]["ldh_full"]
    assert v.weakly_reversible is True
    assert v.deficiency == 0
    assert len(v.linkage_classes) == 3 == ref["linkage_classes"]
    assert len(v.complexes) == ref["complexes"] and v.rank == ref["rank"]
    assert len(v.conservation_laws) == 4
    assert v.grade == GUARANTEE_FULL and v.unique_equilibrium
    # integer conservation laws: total enzyme is one of them
    assert "E + EN + ENP + EQ + EQL" in v.conservation_text


def test_design_mechanism_with_dead_end(base):
    net = ldh().with_extra_steps(["EQ + Pyr <=> EQP"], extra_species=["EQP"])
    assert len(net.species) == 10
    v = verdict(net.reactions(), "design")
    ref = base["verdict"]["ldh_design_with_EQP"]
    assert v.weakly_reversible is True and v.deficiency == 0
    assert len(v.linkage_classes) == 4 == ref["linkage_classes"]
    assert len(v.complexes) == ref["complexes"] and v.rank == ref["rank"]
    assert len(v.conservation_laws) == 4
    # same integer laws as CADKIMM analysis/results/design/summary.txt
    assert v.conservation_text == [
        "E + EN + ENP + EQ + EQL + EQP",
        "E + EN + ENP + EQL + Lac − NAD",
        "EN + ENP + EQL + Lac + NADH",
        "E + EN + EQ − Lac − Pyr",
    ]


def test_pnp_not_weakly_reversible(base):
    pnp = Network.from_strings("pnp", ["E", "S", "ES", "P"], ["E + S <=> ES", "ES -> E + P"], ["E", "ES"])
    v = verdict(pnp.reactions(), "pnp")
    assert v.weakly_reversible is False  # regression: Kosaraju second-pass bug made pNP look WR (fixed 2026-10-09)
    assert v.grade == GUARANTEE_NONE and not v.guarantee_149
    assert len(v.complexes) == base["verdict"]["pnp"]["complexes"]
    assert "149 보장 없음" in v.headline()


def test_non_wr_control_last_step_one_way():
    ctrl = ldh().with_one_way([5])
    v = verdict(ctrl.reactions(), "control")
    assert v.weakly_reversible is False
    assert v.irreversible_complexes  # the complexes of the broken linkage class are named
    assert set(v.irreversible_complexes) == {"EQ", "E + NAD"}


def test_one_step_and_cadkimm_elementary():
    one = Network.from_strings("one", ["NADH", "NAD", "Pyr", "Lac"], ["Pyr + NADH <=> Lac + NAD"])
    v = verdict(one.reactions())
    assert v.weakly_reversible and v.deficiency == 0 and len(v.conservation_laws) == 3
    elem = Network.from_strings(
        "elem",
        ["E", "Pyr", "NADH", "EPN", "ELN", "Lac", "NAD"],
        ["E + Pyr + NADH <=> EPN", "EPN <=> ELN", "ELN <=> E + Lac + NAD"],
    )
    v = verdict(elem.reactions())
    assert v.weakly_reversible and v.deficiency == 0 and len(v.linkage_classes) == 1


def test_selkov_not_wr_positive_deficiency():
    sk = Network.from_strings("selkov", ["X", "Y"], ["X -> 0", "Y -> X", "2X + Y -> 3X", "0 -> Y"])
    v = verdict(sk.reactions())
    assert v.weakly_reversible is False and v.deficiency == 1


def test_example_project_networks(example_dir):
    p = load_project(example_dir)
    assert verdict(p.network.reactions()).weakly_reversible
    assert not verdict(p.control_network().reactions()).weakly_reversible
    d = verdict(p.design_network().reactions())
    assert d.weakly_reversible and d.deficiency == 0 and len(d.linkage_classes) == 4
