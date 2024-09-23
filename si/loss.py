# Imports
import torch

class LossFn(object):

    def __init__(self, switch):
        '''
        Initialization of loss function for SI

        @param switch : switch that indicates whether to learn
                        stochastic or deterministic model
        '''
        if switch.lower() in ('stochastic', 'deterministic'):
            self.switch = switch
        else:
            raise NotImplementedError

    def __call__(self, pred_list, gt_list):
        '''
        Call loss function

        @param pred_dict : dictionary of predictions
        @param gt_dict : dictionary of ground truth values
        @return loss_dict : dictionary of losses
        '''
        loss = []
        for pred, gt in zip(pred_list, gt_list):
            to_append = (pred ** 2) - (2 * pred * gt).mean()
            loss.append(to_append)

        # Return losses
        return torch.Tensor(loss)