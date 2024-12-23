# Imports
import torch
from omg.si.interpolants import *
from omg.si.corrector import Corrector, IdentityCorrector, PeriodicBoundaryConditionsCorrector


class OldLinearInterpolant():
    def __init__(self) -> None:
        pass

    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        return (1.0 - t) * x_0 + t * x_1

    def interpolate_derivative(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        return x_1 - x_0

    def get_corrector(self) -> Corrector:
        return IdentityCorrector()


class OldTrigonometricInterpolant():
    def __init__(self) -> None:
        pass
    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        return torch.cos(torch.pi * t / 2.0) * x_0 + torch.sin(torch.pi * t / 2.0) * x_1

    def interpolate_derivative(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        return (-torch.pi / 2.0 * torch.sin(torch.pi * t / 2.0) * x_0
                + torch.pi / 2.0 * torch.cos(torch.pi * t / 2.0) * x_1)

    def get_corrector(self) -> Corrector:
        return IdentityCorrector()


class OldPeriodicLinearInterpolant():
    def __init__(self) -> None:
        pass

    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        omega = 2.0 * torch.pi * (x_1 - x_0)
        out = t * torch.atan2(torch.sin(omega), torch.cos(omega)) / (2.0 * torch.pi)
        return out + x_0 - torch.floor(out + x_0)

    def interpolate_derivative(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        omega = 2.0 * torch.pi * (x_1 - x_0)
        return torch.atan2(torch.sin(omega), torch.cos(omega)) / (2.0 * torch.pi)

    def get_corrector(self) -> Corrector:
        return PeriodicBoundaryConditionsCorrector(min_value=0.0, max_value=1.0)

class OldEncoderDecoderInterpolant():
    def __init__(self) -> None:
        pass

    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        return (torch.cos(torch.pi * t) ** 2) * torch.where(t < 0.5, x_0, x_1)

    def interpolate_derivative(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        return (-2.0 * torch.cos(torch.pi * t) * torch.pi * torch.sin(torch.pi * t)
                * torch.where(t < 0.5, x_0, x_1))

    def get_corrector(self) -> Corrector:
        return IdentityCorrector()


class OldMirrorInterpolant():
    def __init__(self) -> None:
        pass

    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        return x_1.clone()

    def interpolate_derivative(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x_1)

    def get_corrector(self) -> Corrector:
        return IdentityCorrector()  # TODO: Sometimes you do want to have a Periodic Corrector here?


class OldScoreBasedDiffusionModelInterpolant():
    def __init__(self) -> None:
        pass

    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1.0 - (t ** 2)) * x_0 + t * x_1

    def interpolate_derivative(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        return -t / torch.sqrt(1.0 - (t ** 2)) * x_0 + x_1

    def get_corrector(self) -> Corrector:
        return IdentityCorrector()


class OldPeriodicScoreBasedDiffusionModelInterpolant():
    def __init__(self) -> None:
        self._corrector = PeriodicBoundaryConditionsCorrector(min_value=0.0, max_value=1.0)

    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        x_1prime = self._corrector.unwrap(x_0, x_1)
        x_t = torch.sqrt(1.0 - (t ** 2)) * x_0 + t * x_1prime
        return self._corrector.correct(x_t)

    def interpolate_derivative(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        x_1prime = self._corrector.unwrap(x_0, x_1)
        return -t / torch.sqrt(1.0 - (t ** 2)) * x_0 + x_1prime

    def get_corrector(self) -> Corrector:
        return self._corrector


class OldPeriodicTrigonometricInterpolant():
    def __init__(self) -> None:
        self._corrector = PeriodicBoundaryConditionsCorrector(min_value=0.0, max_value=1.0)

    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        x_1prime = self._corrector.unwrap(x_0, x_1)
        x_t = torch.cos(torch.pi * t / 2.0) * x_0 + torch.sin(torch.pi * t / 2.0) * x_1prime
        return self._corrector.correct(x_t)

    def interpolate_derivative(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        x_1prime = self._corrector.unwrap(x_0, x_1)
        return (-torch.pi / 2.0 * torch.sin(torch.pi * t / 2.0) * x_0
                + torch.pi / 2.0 * torch.cos(torch.pi * t / 2.0) * x_1prime)

    def get_corrector(self) -> Corrector:
        return self._corrector


class OldPeriodicEncoderDecoderInterpolant():
    """
    Periodic Encoder-decoder interpolant
    I(t, x_0, x_1) = cos^2(pi * t) * 1_[0, 0.5) * x_0 + cos^2(pi * t) * 1_(0.5, 1] * x_1 
    between points x_0 and x_1 on a torus 
    from two distributions p_0 and p_1 at times t.
    """

    def __init__(self) -> None:
        self._corrector = PeriodicBoundaryConditionsCorrector(min_value=0.0, max_value=1.0)

    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        x_1prime = self._corrector.unwrap(x_0, x_1)
        x_t = (torch.cos(torch.pi * t) ** 2) * torch.where(t < 0.5, x_0, x_1prime)
        return self._corrector.correct(x_t)

    def interpolate_derivative(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        x_1prime = self._corrector.unwrap(x_0, x_1)
        return (-2.0 * torch.cos(torch.pi * t) * torch.pi * torch.sin(torch.pi * t)
                * torch.where(t < 0.5, x_0, x_1prime))

    def get_corrector(self) -> Corrector:
        return self._corrector

def compare(interp: Interpolant, interp_os: object):
    """
    Function to compare interpolant with OS version
    :param interp:
        Stochastic interpolant.
    :type interp: SingleStochasticInterpolant
    :param interp_os:
        OS Stochastic Interpolant
    :type interp_os: SingleStochasticInterpolantOS
    """
    # Initialize
    initial_cond = torch.randn(size=(10,3))
    final_cond = torch.rand(size=(10,3))
    t = torch.rand(size=(10,1))

    # Interpolate
    interpolated_val = interp.interpolate(t, initial_cond, final_cond)
    os_interpolated_val = interp_os.interpolate(t, initial_cond, final_cond)
    interpolated_dt = interp.interpolate_derivative(t, initial_cond, final_cond)
    os_interpolated_dt = interp_os.interpolate_derivative(t, initial_cond, final_cond)

    # Return
    interp_match = torch.equal(interpolated_val, os_interpolated_val)
    dt_match = torch.equal(interpolated_dt, os_interpolated_dt)
    return interp_match, dt_match

# Script to compare interpolant output
if __name__ == '__main__':

    interp_list = [
        LinearInterpolant(), TrigonometricInterpolant(), PeriodicLinearInterpolant(),
        EncoderDecoderInterpolant(), MirrorInterpolant(), ScoreBasedDiffusionModelInterpolant(),
        PeriodicScoreBasedDiffusionModelInterpolant(), PeriodicTrigonometricInterpolant(),
        PeriodicEncoderDecoderInterpolant()
    ]
    old_interp_list = [
        OldLinearInterpolant(), OldTrigonometricInterpolant(), OldPeriodicLinearInterpolant(),
        OldEncoderDecoderInterpolant(), OldMirrorInterpolant(), OldScoreBasedDiffusionModelInterpolant(),
        OldPeriodicScoreBasedDiffusionModelInterpolant(), OldPeriodicTrigonometricInterpolant(),
        OldPeriodicEncoderDecoderInterpolant()
    ]
    for interp, old_interp in zip(interp_list, old_interp_list):
        outcome = compare(interp, old_interp)

        print(f'Interpolant: {interp.__class__.__name__}: int - {outcome[0]} ind_dot - {outcome[1]}')