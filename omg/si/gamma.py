import torch
from .abstracts import LatentGamma


class LatentGammaSqrt(LatentGamma):
    """
    Gamma function gamma(t) = sqrt(a * t * (1 - t)) in the latent variable gamma(t) * z of a stochastic interpolant.

    :param a:
        Constant a > 0.
    :type a: float
    """

    def __init__(self, a: float) -> None:
        """Construct gamma function."""
        super().__init__()
        if a <= 0.0:
            raise ValueError("Constant a must be positive.")
        self._a = a

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the gamma function gamma(t) in the latent variable gamma(t) * z at the times t.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Gamma function gamma(t).
        :rtype: torch.Tensor
        """
        self._check_t(t)
        return torch.sqrt(self._a * t * (1.0 - t))

    def gamma_derivative(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the derivative of the gamma function gamma(t) in the latent variable gamma(t) * z with respect to time.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Derivative of the gamma function.
        :rtype: torch.Tensor
        """
        self._check_t(t)
        return self._a * (1.0 - 2.0 * t) / (2.0 * torch.sqrt(self._a * t * (1.0 - t)))

    def requires_antithetic(self) -> bool:
        """
        Whether the gamma function requires antithetic sampling because its derivative diverges as t -> 0  or t -> 1.

        :return:
            Whether the gamma function requires antithetic sampling.
        :rtype: bool
        """
        return True


class LatentGammaEncoderDecoder(LatentGamma):
    """
    Gamma function gamma(t) = sin^2(pi * t) in the latent variable gamma(t) * z of a stochastic interpolant.
    """

    def __init__(self) -> None:
        """Construct gamma function."""
        super().__init__()

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the gamma function gamma(t) in the latent variable gamma(t) * z at the times t.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Gamma function gamma(t).
        :rtype: torch.Tensor
        """
        self._check_t(t)
        return torch.sin(torch.pi * t) ** 2

    def gamma_derivative(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the derivative of the gamma function gamma(t) in the latent variable gamma(t) * z with respect to time.

        :param t:
            Times in [0,1].
        :type t: torch.Tensor

        :return:
            Derivative of the gamma function.
        :rtype: torch.Tensor
        """
        self._check_t(t)
        return 2.0 * torch.sin(torch.pi * t) * torch.pi * torch.cos(torch.pi * t)

    def requires_antithetic(self) -> bool:
        """
        Whether the gamma function requires antithetic sampling because its derivative diverges as t -> 0  or t -> 1.

        :return:
            Whether the gamma function requires antithetic sampling.
        :rtype: bool
        """
        return False
