***********************
Theoretical Background
***********************

OMatG implements the `stochastic interpolants (SIs) framework <https://arxiv.org/abs/2303.08797>`__ for the modeling and
generation of inorganic crystalline materials. SIs are a unifying framework for generative modeling that encompasses
flow-matching and diffusion-based methods as specific instances, while offering a more general and flexible approach
enabling the design of a broad class of novel generative models.

Stochastic Interpolants
========================

A stochastic interpolant :math:`x_t = x(t, x_0, x_1, z)=\alpha(t)\,x_0 + \beta(t)\,x_1 + \gamma(t)\,z` bridges samples :math:`x_0`
from a (trivial) base distribution to samples :math:`x_1` from the target data distribution. Here, :math:`t\in[0, 1]` represents
time and the random variable :math:`z` is drawn from a Gaussian distribution. The functional forms of :math:`\alpha`, :math:`\beta`, and
:math:`\gamma` are flexible, only subject to a few constraints that, amongst other things, ensure that
:math:`x(t=0, x_0, x_1, z) = x_0` and :math:`x(t=1, x_0, x_1, z) = x_1`.

The time-dependent probability density of the stochastic process :math:`x_t` can be realized either *via* deterministic
sampling through an ordinary differential equation (ODE) or stochastic sampling through a stochastic differential
equation (SDE), only requiring a sample :math:`x_0` from the base distribution. This enables generative modeling by evolving
samples from the base distribution to samples from the data distribution. Here, the required velocity term
:math:`b^\theta(t, x)` for both ODE- and SDE-based sampling can be learned from data by "averaging over many paired samples
:math:`(x_0, x_1)`." For SDE-based sampling, an additional denoiser :math:`z^\theta(t, x)` can be learned likewise.

The flexibility of the SI framework stems from the ability to tailor the choice of interpolants and choosing between
deterministic (ODE) and stochastic (SDE) sampling schemes (see figure below that visualizes the tunable components of
the SI framework for bridging samples from a base distribution (gray particles) to samples from a target distribution
(purple particles); figure taken from the `OMatG paper <https://openreview.net/forum?id=gHGrzxFujU>`__).

.. figure:: https://arxiv.org/html/2502.02582v1/x1.png
   :alt: stochastic interpolants
   :width: 400px
   :align: center

   Visualization of the stochastic interpolants framework

Application to Crystal Structures
==================================

OMatG defines a crystalline material of :math:`N` atoms by its unit cell that is described by three lattice vectors
:math:`\mathbf{L} \in \mathbb{R}^{3\times3}`, :math:`N` fractional coordinates :math:`\mathbf{X}\in[0,1)^{3\times N}` with periodic
boundary conditions, and :math:`N` discrete atomic species :math:`\mathbf{A}\in\mathbb{Z}^N_{>0}`. During training and generation,
all three components :math:`\{\mathbf{A}, \mathbf{X}, \mathbf{L}\}` are considered simultaneously. The SI framework is applied
to the continuous structural representations :math:`\{\mathbf{X}, \mathbf{L}\}` while the discrete atomic species :math:`\mathbf{A}`
are treated with `discrete flow matching <https://arxiv.org/abs/2402.04997>`__.
