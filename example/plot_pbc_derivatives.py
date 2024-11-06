from omg.si.interpolants import *
from omg.si.single_stochastic_interpolant import SingleStochasticInterpolant
from omg.si.gamma import *
import matplotlib, matplotlib.pyplot as plt
import torch

pbc_i = SingleStochasticInterpolant(
        interpolant=PeriodicLinearInterpolant(), 
        gamma=LatentGammaSqrt(1.0),
        epsilon=None,
        differential_equation_type='ODE',
        integrator_kwargs={'method':'rk4'}
        )

no_gamma_i = SingleStochasticInterpolant(
        interpolant=PeriodicLinearInterpolant(), 
        gamma=None,
        epsilon=None,
        differential_equation_type='ODE',
        integrator_kwargs={'method':'rk4'}
        )

natoms1 = 1
natoms = torch.tensor([natoms1])
x_0 = torch.rand(2) # x0 pos, 2D
x_1 = torch.rand(2) # x1 pos, 2D
# grabbing x_1prime from geodesic
diff = torch.abs(x_0-x_1)
x_1prime = torch.where(diff>=0.5,x_1 + torch.sign(x_0-0.5),x_1)
batch_pointer = torch.tensor([0])

# plotting x_0 and x_1 points
plt.figure(figsize=(10,10))
plt.plot([x_0[0],x_1[0]],[x_0[1],x_1[1]],linestyle='-',color='k',label='naive geodesic',zorder=0)
plt.scatter(x_1prime[0],x_1prime[1],s=50,color='blue',marker="*",label="x1p = x1 image")
plt.scatter(x_0[0],x_0[1],s=50,color='green',zorder=1,label="x0")
plt.scatter(x_1[0],x_1[1],s=50,color='red',zorder=1,label="x1")

# seed = torch.seed() 
# torch.manual_seed(0)
# z = torch.randn(x_0.shape)

random_z = torch.randn_like(x_0)

for t_ in torch.linspace(0,1,100):
    t = t_.repeat_interleave(natoms)

    seed = torch.seed() 
    torch.manual_seed(0)
    torch.randn_like = lambda _: random_z
    x_t, z = pbc_i.interpolate(t, x_0, x_1, batch_pointer)
    torch.randn_like = lambda _: -random_z
    x_t_neg, z_neg = pbc_i.interpolate(t, x_0, x_1, batch_pointer)
    x_t_no_gamma = no_gamma_i.interpolate(t, x_0, x_1, batch_pointer)[0]

    der = pbc_i._interpolate_derivative(t, x_0, x_1, z, batch_pointer)
    der_neg = pbc_i._interpolate_derivative(t, x_0, x_1, z_neg, batch_pointer)

    # plotting the non-geodesic derivative for positive z
    scale = 0.3
    plt.arrow(x_t[0], x_t[1], scale*der[0], scale*der[1], color='C3')
    # negative z
    plt.arrow(x_t_neg[0], x_t_neg[1], scale*der_neg[0], scale*der_neg[1], color='C4')

    # plotting the geodesic path
    plt.scatter(x_t_no_gamma[0],x_t_no_gamma[1],zorder=0,s=1,color='C1')
    # plotting the non-geodesic path for positive z
    plt.scatter(x_t[0],x_t[1],zorder=0,s=1,color='C4')
    # plotting the non-geodesic path for negative z
    plt.scatter(x_t_neg[0],x_t_neg[1],zorder=0,s=1,color='C3')

plt.plot([0,0],[0,1],linestyle='--',color='k')
plt.plot([0,1],[0,0],linestyle='--',color='k')
plt.plot([1,1],[0,1],linestyle='--',color='k')
plt.plot([0,1],[1,1],linestyle='--',color='k')

plt.xlim([-0.5,1.5])
plt.ylim([-0.5,1.5])
plt.legend(loc='upper right')
plt.savefig("pbc_SBDMI_.png")
