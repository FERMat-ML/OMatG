def reshape_t(t: torch.Tensor, n_atoms: torch.Tensor, data_field: DataField) -> torch.Tensor:
 17     """
 18     Reshape the given tensor of times for every configuration of the batch so that it can be used for the given data field.
 19  
 20     For a batch size of batch_size, the data format for the different data fields is as follows:
 21     - species: torch.Tensor of shape (sum(n_atoms), ) containing the atomic numbers of the atoms in the configurations
 22     - cell: torch.Tensor of shape (batch_size, 3, 3) containing the cell vectors of the configurations
 23     - pos: torch.Tensor of shape (sum(n_atoms), 3) containing the atomic positions of the atoms in the configurations
 24  
 25     The returned tensor will have the same shape as the tensor of the given data field, and the correct time for every
 26     element of the data field tensor.
 27  
 28     :param t:
 29         Tensor of times for the configurations in the batch.
 30     :type t: torch.Tensor
 31     :param n_atoms:
 32         Tensor of the number of atoms in each configuration in the batch.
 33     :type n_atoms: torch.Tensor
 34     :param data_field:
 35         Data field for which the tensor of times should be reshaped.
 36     :type data_field: DataField
 37  
 38     :return:
 39         Tensor of times for the given data field.
 40     :rtype: torch.Tensor
 41     """
 42     assert len(t.shape) == len(n_atoms.shape) == 1
 43     t_per_atom = t.repeat_interleave(n_atoms)
 44     sum_n_atoms = int(n_atoms.sum())
 45     batch_size = len(t)
 46     if data_field == DataField.pos:
 47         return t_per_atom.repeat_interleave(3).reshape(sum_n_atoms, 3)
 48     elif data_field == DataField.cell:
 49         return t.repeat_interleave(3 * 3).reshape(batch_size, 3, 3)
 50     else:
 51         assert data_field == DataField.species
 52         return t_per_atom                                                        
