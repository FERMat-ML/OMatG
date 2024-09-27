def reshape_t(t: torch.Tensor, n_atoms: torch.Tensor, data_field: DataField) -> torch.Tensor:
     """ 
     Reshape the given tensor of times for every configuration of the batch so that it can be used for the given data field.
   
     For a batch size of batch_size, the data format for the different data fields is as follows:
     - species: torch.Tensor of shape (sum(n_atoms), ) containing the atomic numbers of the atoms in the configurations
     - cell: torch.Tensor of shape (batch_size, 3, 3) containing the cell vectors of the configurations
     - pos: torch.Tensor of shape (sum(n_atoms), 3) containing the atomic positions of the atoms in the configurations
  
     The returned tensor will have the same shape as the tensor of the given data field, and the correct time for every
    element of the data field tensor.
  
     :param t:
        Tensor of times for the configurations in the batch.
     :type t: torch.Tensor
     :param n_atoms:
         Tensor of the number of atoms in each configuration in the batch.
     :type n_atoms: torch.Tensor
     :param data_field:
         Data field for which the tensor of times should be reshaped.
     :type data_field: DataField
  
     :return:
         Tensor of times for the given data field.
     :rtype: torch.Tensor
     """
     assert len(t.shape) == len(n_atoms.shape) == 1
     t_per_atom = t.repeat_interleave(n_atoms)
     sum_n_atoms = int(n_atoms.sum())
     batch_size = len(t)
     if data_field == DataField.pos:
         return t_per_atom.repeat_interleave(3).reshape(sum_n_atoms, 3)
     elif data_field == DataField.cell:
         return t.repeat_interleave(3 * 3).reshape(batch_size, 3, 3)
     else:
         assert data_field == DataField.species
         return t_per_atom                                                        
