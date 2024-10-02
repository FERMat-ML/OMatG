from torch.nn import functional
from omg.globals import MAX_ATOM_NUM
from torch.distributions import Categorical
from torch_geometric.data import Data
import torch
from tqdm import tqdm

# copied from discrete integration
def discrete(x_t, x, t, dt, noise= 0.0):
    t = t[0]
    eps = torch.finfo(torch.float64).eps
    x_1_probs = functional.softmax(x, dim=-1)  # Shape (sum(n_atoms), MAX_ATOM_NUM).
    #print (torch.argmax(x_1_probs,-1))
    # Sample from distribution for every of the sum(n_atoms) elements.
    # Do not shift the atom type by one to get the real species. Instead shift x_t down.
    x_1_probs = x_1_probs.reshape((-1, MAX_ATOM_NUM))
    shifted_x_1 = Categorical(x_1_probs).sample()  # Shape (sum(n_atoms),)
    shifted_x_t = x_t - 1
    assert shifted_x_1.shape == x_t.shape == shifted_x_t.shape
    # Shape (sum(n_atoms), MAX_ATOM_NUM).
    shifted_x_1_hot = functional.one_hot(shifted_x_1, num_classes=MAX_ATOM_NUM)
    dpt = shifted_x_1_hot - (1.0 / MAX_ATOM_NUM)  # Shape (sum(n_atoms), MAX_ATOM_NUM).
    # Gather values from dpt based on shifted_x_t.
    dpt_xt = dpt.gather(-1, shifted_x_t[:, None]).squeeze(-1)  # Shape (sum(n_atoms),).

    # Compute pt: linear interpolation based on t.
    # TODO: consider adding functionality to use other types of interpolants
    # Shape (sum(n_atoms), MAX_ATOM_NUM).
    pt = (t * shifted_x_1_hot) + (1.0 - t) * (1.0 / MAX_ATOM_NUM)
    pt_xt = pt.gather(-1, shifted_x_t[:, None]).squeeze(-1)  # Shape (sum(n_atoms),).

    # Compute the rate R.
    # Shape (sum(n_atoms), MAX_ATOM_NUM).
    rate = functional.relu(dpt - dpt_xt[:, None]) / (MAX_ATOM_NUM * pt_xt[:, None])
    # Set p(x_t | x_1) = 0 or p(j | x_1) = 0 cases to zero.
    rate[(pt_xt == 0.0)[:, None].repeat(1, MAX_ATOM_NUM)] = 0.0
    rate[pt == 0.0] = 0.0

    # Add noise if present.
    rate_db = torch.zeros_like(rate)
    if noise > 0.0:
        rate_db[shifted_x_t == shifted_x_1] = 1.0
        rate_db[shifted_x_1 != shifted_x_t] = ((MAX_ATOM_NUM * t) + 1.0 - t) / (1.0 - t + eps)
        rate_db *= self._noise
    rate += rate_db

    # Compute step probabilities and sample
    step_probs = (rate * dt).clamp(max=1.0)  # Shape (sum(n_atoms), MAX_ATOM_NUM).
    step_probs.scatter_(-1, shifted_x_t[:, None], 0.0)
    step_probs.scatter_(-1, shifted_x_t[:, None], 1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)

    # Sample the next x_t
    #print ('step_probs',step_probs)
    x_t = Categorical(step_probs).sample() + 1
    return x_t


def rk(model, t, x, dt):
    # Performs RK4 integration
    k1 = model(x, t)
    # Only actually need remainder at very end
    x_k2 = Data(species = discrete(x.species, k1.species_b, t + dt /2 , dt / 2.), pos = torch.remainder(x.pos + dt * k1.pos_b / 2., 1.), cell = x.cell + dt * k1.cell_b / 2., n_atoms = x.n_atoms, batch = x.batch)  
    k2 = model(x_k2, t + dt/2)
    x_k3 = Data(species = discrete(x.species, k2.species_b, t + dt / 2, dt / 2), pos = torch.remainder(x.pos + dt * k2.pos_b / 2., 1.), cell = x.cell + dt * k2.cell_b / 2., n_atoms = x.n_atoms, batch = x.batch)  
    k3 = model(x_k3, t + dt/2)
    x_k4 = Data(species = discrete(x.species, k3.species_b, t, dt), pos = torch.remainder(x.pos + dt * k3.pos_b, 1.), cell = x.cell + dt * k3.cell_b, n_atoms = x.n_atoms, batch = x.batch)  
    k4 = model(x_k4, t + dt)
    # don't need division in species due to softmax
    x = Data(species = discrete(x.species, k1.species_b + 2 * k2.species_b + 2 * k3.species_b + k4.species_b, t, dt),
        pos = torch.remainder(x.pos + dt / 6. * (k1.pos_b + 2 * k2.pos_b + 2 * k3.pos_b + k4.pos_b), 1.), 
        cell = x.cell + dt / 6. * (k1.cell_b + 2 * k2.cell_b + 2 * k3.cell_b + k4.cell_b),
        n_atoms = x.n_atoms, batch = x.batch, ptr = x.ptr)

    return x

def integrate(method, model, x, steps = 100, min_t=0., max_t=1.):
    times = torch.linspace(min_t, max_t, steps)
    dt = times[1] - times[0]
    for time in tqdm(times[:-1]):
        time = time.repeat(len(x.n_atoms),)
        x = method(model, time, x, dt)
    return x

