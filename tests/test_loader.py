import pytest

from versor import LoadError, from_dict


def prog(chains):
    return {"version": "0.1", "name": "t", "chains": chains}


def chain(cid, vertices):
    return {"id": cid, "vertices": vertices}


def test_minimal_program_loads():
    p = from_dict(prog([chain(0, [
        {"id": 0, "out": [{"seg": [0.0, 0.0, -1.0], "to": 1}]},
        {"id": 1, "out": []},
    ])]))
    assert len(p.chains) == 1
    assert p.warnings == []


def test_branch_without_guard_rejected():
    with pytest.raises(LoadError, match="lack guards"):
        from_dict(prog([chain(0, [
            {"id": 0, "out": [
                {"seg": [1.0, 1.0, 0.0], "to": 1, "guard": [0, 1, 0]},
                {"seg": [0.0, 0.0, -1.0], "to": 1},
            ]},
            {"id": 1, "out": []},
        ])]))


def test_duplicate_chain_id_rejected():
    v = [{"id": 0, "out": []}]
    with pytest.raises(LoadError, match="duplicate chain id"):
        from_dict(prog([chain(0, v), chain(0, v)]))


def test_non_contiguous_chain_ids_rejected():
    v = [{"id": 0, "out": []}]
    with pytest.raises(LoadError, match="contiguous"):
        from_dict(prog([chain(0, v), chain(2, v)]))


def test_missing_entry_vertex_rejected():
    with pytest.raises(LoadError, match="entry vertex 0"):
        from_dict(prog([chain(0, [{"id": 1, "out": []}])]))


def test_dangling_edge_target_rejected():
    with pytest.raises(LoadError, match="does not exist"):
        from_dict(prog([chain(0, [
            {"id": 0, "out": [{"seg": [0.0, 0.0, -1.0], "to": 7}]},
        ])]))


def test_bad_seg_rejected():
    with pytest.raises(LoadError, match="expected"):
        from_dict(prog([chain(0, [
            {"id": 0, "out": [{"seg": [1.0, 2.0], "to": 0}]},
        ])]))


def test_non_unit_guard_normalized_with_warning():
    p = from_dict(prog([chain(0, [
        {"id": 0, "out": [
            {"seg": [1.0, 1.0, 0.0], "to": 1, "guard": [0, 2, 0]},
            {"seg": [0.0, 0.0, -1.0], "to": 1, "guard": [0, -2, 0]},
        ]},
        {"id": 1, "out": []},
    ])]))
    assert any("not unit norm" in w for w in p.warnings)
    assert p.chains[0].vertices[0][0].guard[1] == pytest.approx(1.0)


def test_guard_on_single_edge_warned():
    p = from_dict(prog([chain(0, [
        {"id": 0, "out": [{"seg": [0.0, 0.0, -1.0], "to": 1, "guard": [0, 1, 0]}]},
        {"id": 1, "out": []},
    ])]))
    assert any("single-edge" in w for w in p.warnings)


def test_dead_zone_lint_warning():
    # normalized x-component ~0.351: inside the dead zone under identity
    p = from_dict(prog([chain(0, [
        {"id": 0, "out": [{"seg": [0.351, 0.9364, 0.0], "to": 1}]},
        {"id": 1, "out": []},
    ])]))
    assert any("dead zone" in w for w in p.warnings)


def test_zero_length_segment_lint_warning():
    p = from_dict(prog([chain(0, [
        {"id": 0, "out": [{"seg": [0.0, 0.0, 0.0], "to": 1}]},
        {"id": 1, "out": []},
    ])]))
    assert any("zero-length" in w for w in p.warnings)
