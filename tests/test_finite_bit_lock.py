import unittest
import torch

from grn.models.grn import update_finite_bit_lock


def step(candidate, remaining, lock_steps):
    return update_finite_bit_lock(
        torch.tensor(candidate, dtype=torch.bool),
        remaining,
        lock_steps,
    )


def test_three_step_lifetime_includes_detection_step():
    remaining, active, carried, expired = step([False, True], None, 3)
    assert remaining.tolist() == [0, 3]
    assert active.tolist() == [False, True]
    assert carried.tolist() == [False, False]
    assert expired.tolist() == [False, False]

    remaining, active, carried, expired = step(
        [False, False], remaining, 3
    )
    assert remaining.tolist() == [0, 2]
    assert active.tolist() == [False, True]
    assert carried.tolist() == [False, True]
    assert expired.tolist() == [False, False]

    remaining, active, carried, expired = step(
        [False, False], remaining, 3
    )
    assert remaining.tolist() == [0, 1]
    assert active.tolist() == [False, True]

    remaining, active, carried, expired = step(
        [False, False], remaining, 3
    )
    assert remaining.tolist() == [0, 0]
    assert active.tolist() == [False, False]
    assert expired.tolist() == [False, True]


def test_redetection_refreshes_lifetime():
    remaining, *_ = step([True], None, 3)
    remaining, *_ = step([False], remaining, 3)
    remaining, active, carried, expired = step([True], remaining, 3)
    assert remaining.tolist() == [3]
    assert active.tolist() == [True]
    assert carried.tolist() == [False]
    assert expired.tolist() == [False]


def test_one_step_lock_matches_instantaneous_candidate():
    remaining = None
    for candidate in ([True, False], [False, True], [False, False]):
        remaining, active, carried, _ = step(candidate, remaining, 1)
        assert active.tolist() == candidate
        assert not carried.any()


class FiniteBitLockTests(unittest.TestCase):
    def test_lifetime(self):
        test_three_step_lifetime_includes_detection_step()

    def test_refresh(self):
        test_redetection_refreshes_lifetime()

    def test_disabled(self):
        test_one_step_lock_matches_instantaneous_candidate()

    def test_matches_sliding_window_or(self):
        rng = torch.Generator().manual_seed(7)
        candidates = torch.rand((12, 3, 4), generator=rng) > 0.6
        for duration in (1, 2, 4, 20):
            remaining = None
            for index, candidate in enumerate(candidates):
                remaining, active, _, _ = update_finite_bit_lock(candidate, remaining, duration)
                expected = candidates[max(0, index-duration+1):index+1].any(dim=0)
                self.assertTrue(torch.equal(active, expected))


if __name__ == '__main__':
    unittest.main()
