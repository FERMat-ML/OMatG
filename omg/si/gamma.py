import torch
from .abstracts import LatentGamma


class LatentGammaSqrt(LatentGamma):
    def __init__(self, a: float) -> None:
        """
        Gamma function gamma(t) = sqrt(a * t * (1 - t)) in the latent variable gamma(t) * z of a stochastic interpolant.
        """
        super().__init__()
        self._a = a

    def gamma(self, t: torch.tensor) -> torch.tensor:
        """
        Evaluate the gamma function gamma(t) in the latent variable gamma(t) * z at the times t.

        :param t:
            Times in [0,1].
        :type t: torch.tensor

        :return:
            Gamma function gamma(t).
        :rtype: torch.tensor
        """
        self._check_t(t)
        return torch.sqrt(self._a * t * (1.0 - t))

    def gamma_derivative(self, t: torch.tensor) -> torch.tensor:
        """
        Compute the derivative of the gamma function gamma(t) in the latent variable gamma(t) * z with respect to time.

        :param t:
            Times in [0,1].
        :type t: torch.tensor

        :return:
            Derivative of the gamma function.
        :rtype: torch.tensor
        """
        self._check_t(t)
        return self._a * (1.0 - 2.0 * t) / (2.0 * torch.sqrt(self._a * t * (1.0 - t)))


class LatentGammaEncoderDecoder(LatentGamma):
    def __init__(self) -> None:
        """
        Gamma function gamma(t) = sin^2(pi * t) in the latent variable gamma(t) * z of a stochastic interpolant.
        """
        super().__init__()

    def gamma(self, t: torch.tensor) -> torch.tensor:
        """
        Evaluate the gamma function gamma(t) in the latent variable gamma(t) * z at the times t.

        :param t:
            Times in [0,1].
        :type t: torch.tensor

        :return:
            Gamma function gamma(t).
        :rtype: torch.tensor
        """
        self._check_t(t)
        return torch.sin(torch.pi * t) ** 2

    def gamma_derivative(self, t: torch.tensor) -> torch.tensor:
        """
        Compute the derivative of the gamma function gamma(t) in the latent variable gamma(t) * z with respect to time.

        :param t:
            Times in [0,1].
        :type t: torch.tensor

        :return:
            Derivative of the gamma function.
        :rtype: torch.tensor
        """
        self._check_t(t)
        return 2.0 * torch.sin(torch.pi * t) * torch.pi * torch.cos(torch.pi * t)
