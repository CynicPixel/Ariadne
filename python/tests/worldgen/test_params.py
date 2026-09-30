import pytest
import yaml

from dataset_tools.worldgen.params import PresetError, load_preset, parse_preset


def test_defaults_validate():
    p = parse_preset({"name": "d"})
    assert p.clearance.rho_responder_m == 0.40


def test_unknown_keys_and_sections_fail():
    with pytest.raises(PresetError):
        parse_preset({"name": "x", "corridors": {"widht_m": [1.0, 1.2]}})
    with pytest.raises(PresetError):
        parse_preset({"name": "x", "corridor": {}})


def test_types_are_checked():
    with pytest.raises(PresetError):
        parse_preset({"name": "x", "lattice": {"nx": 4.5}})
    with pytest.raises(PresetError):
        parse_preset({"name": "x", "target": {"path_length_m": [10.0]}})
    with pytest.raises(PresetError):
        parse_preset({"name": "x", "search": {"max_attempts": True}})


def test_physical_constraints():
    # responder corridors must pass rho_responder on the grid
    with pytest.raises(PresetError):
        parse_preset({"name": "x", "corridors": {"width_m": [0.8, 1.2]}})
    # narrow passages must block a responder and admit the agent
    with pytest.raises(PresetError):
        parse_preset({"name": "x", "corridors": {"narrow_width_m": [0.55, 0.85]}})
    with pytest.raises(PresetError):
        parse_preset({"name": "x", "corridors": {"narrow_width_m": [0.40, 0.60]}})
    with pytest.raises(PresetError):
        parse_preset({"name": "x", "clearance": {"rho_agent_m": 0.5}})
    with pytest.raises(PresetError):
        parse_preset({"name": "x", "rooms": {"size_m": [3.5, 5.5]}})   # no wall to the cell border


def test_digest_ignores_formatting_but_not_values(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("# comment\nname: p\ntarget: {path_length_m: [25, 40]}\nlattice: {nx: 4}\n")
    b.write_text("lattice:\n  nx: 4\nname: p\ntarget:\n  path_length_m: [25.0, 40.0]\n")
    assert load_preset(a).digest() == load_preset(b).digest()
    c = tmp_path / "c.yaml"
    c.write_text(yaml.safe_dump({"name": "p", "target": {"path_length_m": [25, 41]}}))
    assert load_preset(c).digest() != load_preset(a).digest()
