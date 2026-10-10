"""Choosing the reaction mechanism (`network.mechanism`, `enzopt mechanisms`) instead of writing the steps."""

import shutil

import pytest
from conftest import EXAMPLE, run_cli

from enzopt.core.crnt import verdict
from enzopt.core.mechanisms import MECHANISMS, MechanismError, expand, generic_example
from enzopt.core.network import Network
from enzopt.core.project import ProjectError, load_project

LDH = {
    "type": "ordered_bibi",
    "substrates": ["NADH", "Pyr"],
    "products": ["Lac", "NAD"],
    "letters": {"NADH": "N", "Pyr": "P", "Lac": "L", "NAD": "Q"},
}
GRADES = {
    "uni_irreversible": "C",
    "uni": "A",
    "uni_iso": "A",
    "uni_bi_ordered": "A",
    "ordered_bibi": "A",
    "theorell_chance": "A",
    "random_bibi": "B",
    "ping_pong": "A",
    "ordered_terter": "A",
}


def test_ldh_choice_writes_the_hand_written_network():
    x = expand(LDH)
    ex = load_project(EXAMPLE).network
    assert [str(s) for s in Network.from_strings("m", x.species, x.steps).steps] == [str(s) for s in ex.steps]
    assert sorted(x.species) == sorted(ex.species) and sorted(x.enzyme_species) == sorted(ex.enzyme_species)
    assert x.enzyme_species[0] == "E" and x.haldane_step == 5 and x.overall == "NADH + Pyr <=> Lac + NAD"


def test_ldh_choice_is_accepted_by_the_pinn():
    from enzopt.pinn.ldh import matches_network

    x = expand(LDH)
    assert matches_network(Network.from_strings("m", x.species, x.steps, x.enzyme_species))


@pytest.mark.parametrize("key", list(MECHANISMS))
def test_every_mechanism_expands_with_its_grade(key):
    x = generic_example(key)
    v = verdict(Network.from_strings(key, x.species, x.steps, x.enzyme_species).reactions(), key)
    assert v.grade == GRADES[key]
    # the enzyme total is conserved: the sum of the enzyme forms is orthogonal to every reaction vector
    import numpy as np

    from enzopt.core.massaction import MassAction

    net = Network.from_strings(key, x.species, x.steps, x.enzyme_species)
    e = np.array([1.0 if s in x.enzyme_species else 0.0 for s in net.species])
    assert np.allclose(e @ MassAction.from_network(net).N, 0.0)


def test_names_letters_and_side_steps():
    full = expand({"type": "ordered_bibi", "substrates": ["NADH", "Pyr"], "products": ["Lac", "NAD"]})
    assert "E_NADH" in full.species and full.notes  # NADH and NAD share N: written in full
    y = expand(
        {
            **LDH,
            "add": [
                {"dead_end": {"form": "EQ", "ligand": "Pyr"}},
                {"dead_end": {"form": "E", "ligand": "Oxa"}},
                {"step": "EO + Pyr <=> EOP", "enzyme_forms": ["EOP"]},
                {"background": True},
            ],
        }
    )
    assert y.steps[5:] == ["EQ + Pyr <=> EQP", "E + Oxa <=> EO", "EO + Pyr <=> EOP", "NADH + Pyr <=> Lac + NAD"]
    assert {"EQP", "EO", "EOP"} <= set(y.enzyme_species) and "Oxa" not in y.enzyme_species


@pytest.mark.parametrize(
    "block, what",
    [
        ({"type": "nope", "substrates": ["S"], "products": ["P"]}, "type"),
        ({"type": "ordered_bibi", "substrates": ["A"], "products": ["P", "Q"]}, "기질 2개"),
        ({"type": "uni", "substrates": ["S-1"], "products": ["P"]}, "이름"),
        ({"type": "uni", "substrates": ["S"], "products": ["S"]}, "겹침"),
    ],
)
def test_bad_mechanism_blocks(block, what):
    with pytest.raises(MechanismError, match=what):
        expand(block)


def _with_mechanism(tmp_path, mech_yaml: str):
    proj = tmp_path / "p"
    shutil.copytree(EXAMPLE, proj)
    y = (proj / "project.yaml").read_text(encoding="utf-8")
    start = y.index("  species:")
    end = y.index("  observable:")
    y = y[:start] + mech_yaml + y[end:]
    y = y.replace("    haldane_step: 5          # k₋5 = Π k₊ / (k₋1·k₋2·k₋3·k₋4·K')\n", "")
    y = y.replace("    overall: Pyr + NADH <=> Lac + NAD\n", "")
    (proj / "project.yaml").write_text(y, encoding="utf-8")
    return proj


MECH_YAML = """  mechanism:
    type: ordered_bibi
    substrates: [NADH, Pyr]
    products: [Lac, NAD]
    letters: {NADH: N, Pyr: P, Lac: L, NAD: Q}
"""


def test_project_with_a_chosen_mechanism(tmp_path):
    proj = _with_mechanism(tmp_path, MECH_YAML)
    p = load_project(proj)
    ex = load_project(EXAMPLE)
    assert [str(s) for s in p.network.steps] == [str(s) for s in ex.network.steps]
    assert p.haldane_step == 5 and p.thermodynamics["overall"] == "NADH + Pyr <=> Lac + NAD"
    assert p.mechanism.mechanism == "ordered_bibi"
    r = run_cli("check", proj)
    assert r.returncode == 0, r.stdout[-2000:]
    assert "예상 평형" in r.stdout


def test_mechanism_and_steps_together_is_an_error(tmp_path):
    proj = _with_mechanism(tmp_path, MECH_YAML + "  steps:\n    - E + NADH <=> EN\n")
    with pytest.raises(ProjectError, match="둘 중 하나"):
        load_project(proj)


def test_mechanisms_command():
    r = run_cli("mechanisms")
    assert r.returncode == 0 and all(k in r.stdout for k in MECHANISMS)
    r = run_cli("mechanisms", "ping_pong", "--substrates", "A,B", "--products", "P,Q")
    assert r.returncode == 0 and "F + B <=> FB" in r.stdout and "kr_log10" in r.stdout
