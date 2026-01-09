from ase.build import bulk
import pytest
from omg.analysis import match_rmsds, ValidAtoms


@pytest.fixture
def c1():
    return ValidAtoms(bulk("Cu", "fcc", a=3.6))


@pytest.fixture
def c2():
    return ValidAtoms(bulk("NaCl", "rocksalt", a=4.1))


@pytest.fixture
def c3():
    return ValidAtoms(bulk("Al", "bcc", a=4.05))


@pytest.fixture
def c4():
    return ValidAtoms(bulk("CoCa", "zincblende", a=2.1))


def test_crystals_different(c1, c2, c3, c4):
    assert sum(r is not None for r in match_rmsds([c1], [c2], enable_progress_bar=False)[4]) == 0
    assert sum(r is not None for r in match_rmsds([c1], [c3], enable_progress_bar=False)[4]) == 0
    assert sum(r is not None for r in match_rmsds([c1], [c4], enable_progress_bar=False)[4]) == 0
    assert sum(r is not None for r in match_rmsds([c2], [c1], enable_progress_bar=False)[4]) == 0
    assert sum(r is not None for r in match_rmsds([c2], [c3], enable_progress_bar=False)[4]) == 0
    assert sum(r is not None for r in match_rmsds([c2], [c4], enable_progress_bar=False)[4]) == 0
    assert sum(r is not None for r in match_rmsds([c3], [c1], enable_progress_bar=False)[4]) == 0
    assert sum(r is not None for r in match_rmsds([c3], [c2], enable_progress_bar=False)[4]) == 0
    assert sum(r is not None for r in match_rmsds([c3], [c4], enable_progress_bar=False)[4]) == 0


def test_match_rate(c1, c2, c3, c4):
    assert sum(r is not None for r in match_rmsds(
        [c1, c2, c3, c4], [c1, c2, c3, c4], enable_progress_bar=False)[4]) == 4
    assert sum(r is not None for r in match_rmsds(
        [c1, c2], [c3, c4], enable_progress_bar=False)[4]) == 0
    assert sum(r is not None for r in match_rmsds(
        [c1, c2, c3, c4], [c1, c1, c1, c1], enable_progress_bar=False)[4]) == 1
    assert sum(r is not None for r in match_rmsds(
        [c1, c2, c1, c1, c2, c3, c1, c4, c4, c2, c1, c3, c4, c2, c1, c2, c4],
        [c1, c2, c3, c4, c1, c2, c3, c4, c1, c2, c3, c4, c1, c2, c3, c4, c1],
        enable_progress_bar=False)[4]) == 5
    assert sum(r is not None for r in match_rmsds(
        [c1, c2, c1, c1, c2, c3, c1, c4, c4, c2, c1, c3, c4, c2, c1, c2, c4],
        [c1, c2, c1, c2, c1, c2, c1, c2, c1, c2, c1, c2, c1, c2, c1, c2, c1],
        enable_progress_bar=False)[4]) == 9
    assert sum(r is not None for r in match_rmsds(
        [c1, c2],
        [c1, c2, c1, c1, c2, c3, c1, c4, c4, c2, c1, c3, c4, c2, c1, c2, c4],
        enable_progress_bar=False)[4]) == 2


if __name__ == '__main__':
    pytest.main([__file__])
