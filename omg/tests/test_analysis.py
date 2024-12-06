from ase.build import bulk
import pytest
from omg.analysis import match_rate_and_rmsd, unique_rate


@pytest.fixture
def c1():
    return bulk("Cu", "fcc", a=3.6)


@pytest.fixture
def c2():
    return bulk("NaCl", "rocksalt", a=4.1)


@pytest.fixture
def c3():
    return bulk("Al", "bcc", a=4.05)


@pytest.fixture
def c4():
    return bulk("CoCa", "zincblende", a=2.1)


def test_crystals_different(c1, c2, c3, c4):
    assert match_rate_and_rmsd([c1], [c2])[0] == 0.0
    assert match_rate_and_rmsd([c1], [c3])[0] == 0.0
    assert match_rate_and_rmsd([c1], [c4])[0] == 0.0
    assert match_rate_and_rmsd([c2], [c1])[0] == 0.0
    assert match_rate_and_rmsd([c2], [c3])[0] == 0.0
    assert match_rate_and_rmsd([c2], [c4])[0] == 0.0
    assert match_rate_and_rmsd([c3], [c1])[0] == 0.0
    assert match_rate_and_rmsd([c3], [c2])[0] == 0.0
    assert match_rate_and_rmsd([c3], [c4])[0] == 0.0


def test_match_rate(c1, c2, c3, c4):
    assert match_rate_and_rmsd([c1, c2, c3, c4], [c1, c2, c3, c4], ltol=0.2, stol=0.3, angle_tol=5.0)[0] == 1.0
    assert match_rate_and_rmsd([c1, c2], [c3, c4], ltol=0.2, stol=0.3, angle_tol=5.0)[0] == 0.0
    assert match_rate_and_rmsd([c1, c2, c3, c4], [c1, c1, c1, c1], ltol=0.2, stol=0.3, angle_tol=5.0)[0] == 1.0 / 4.0
    assert match_rate_and_rmsd([c1, c2, c1, c1, c2, c3, c1, c4, c4, c2, c1, c3, c4, c2, c1, c2, c4],
                               [c1, c2, c3, c4], ltol=0.2, stol=0.3, angle_tol=5.0)[0] == 1.0
    assert match_rate_and_rmsd([c1, c2, c1, c1, c2, c3, c1, c4, c4, c2, c1, c3, c4, c2, c1, c2, c4],
                               [c1, c2], ltol=0.2, stol=0.3, angle_tol=5.0)[0] == 11.0 / 17.0
    assert match_rate_and_rmsd([c1, c2], [c1, c2, c1, c1, c2, c3, c1, c4, c4, c2, c1, c3, c4, c2, c1, c2, c4],
                               ltol=0.2, stol=0.3, angle_tol=5.0)[0] == 1.0


def test_unique_rate(c1, c2, c3, c4):
    assert unique_rate([c1, c2, c3, c4], ltol=0.2, stol=0.3, angle_tol=5.0) == 1.0
    assert unique_rate([c1, c1, c1, c1], ltol=0.2, stol=0.3, angle_tol=5.0) == 1.0 / 4.0
    assert unique_rate([c1, c2, c1, c1, c2, c3, c1, c4, c4, c2, c1, c3, c4, c2, c1, c2, c4], ltol=0.2,
                       stol=0.3, angle_tol=5.0) == 4.0 / 17.0


if __name__ == '__main__':
    pytest.main([__file__])
