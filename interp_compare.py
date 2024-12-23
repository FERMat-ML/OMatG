# Imports
import torch
from omg.si.single_stochastic_interpolant_os import *
from omg.si.single_stochastic_interpolant import *
from omg.si.interpolants import *

def compare(interp: SingleStochasticInterpolant, interp_os: SingleStochasticInterpolantOS):
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
    interpolated_val = interp.interpolate(t, initial_cond, final_cond, batch_indices=None)[0]
    os_interpolated_val = interp_os.interpolate(t, initial_cond, final_cond, batch_indices=None)[0]

    # Return
    #print(interpolated_val)
    #print(os_interpolated_val)
    interp_match = torch.equal(interpolated_val, os_interpolated_val)
    return interp_match

# Script to compare interpolant output
if __name__ == '__main__':

    interp_list = [
        LinearInterpolant(), TrigonometricInterpolant(), PeriodicLinearInterpolant(),
        EncoderDecoderInterpolant(), MirrorInterpolant(), ScoreBasedDiffusionModelInterpolant(),
        PeriodicScoreBasedDiffusionModelInterpolant(), PeriodicTrigonometricInterpolant(),
        PeriodicEncoderDecoderInterpolant()
    ]
    for interp in interp_list:
        si = SingleStochasticInterpolant(interp, gamma=None, epsilon=None, differential_equation_type='ODE')
        si_os = SingleStochasticInterpolantOS(interp, epsilon=None, differential_equation_type='ODE')
        outcome = compare(si, si_os)

        print(f'Interpolant: {interp.__class__.__name__}: {outcome}')