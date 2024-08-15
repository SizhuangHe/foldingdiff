import torch
class IFMloss:
    def __init__(self, loss_type, include_x0=True, include_x1=True):
        self.loss_type = loss_type
        self.include_x0 = include_x0
        self.include_x1 = include_x1
    
    def _process(self, pred, mu_t):
        starting = 0
        ending = None
        if not self.include_x0:
            starting = 1
        if not self.include_x1:
            ending = -1
        return pred[:, starting:ending, :], mu_t[:, starting:ending, :]
    
    def _length(self, trajectory):
        # computes the length of each trajectory
        # works only for 1D cases
        batch_size, sequence_length, token_size = trajectory.shape
        assert token_size == 1
        diffs = torch.abs(torch.diff(trajectory.reshape(batch_size, sequence_length), dim=1))
        # Sum the absolute differences for each sequence
        lengths = torch.sum(diffs, dim=1)
        return lengths.unsqueeze(-1) # shape [btach_size, 1]
        
    def _compute_NLL(self, pred, mu_t):
        pred, mu_t = self._process(pred, mu_t)
        return torch.sum((pred - mu_t) ** 2)
    
    def _regularize_length(self, pred, mu_t, alpha = 0):
        pred, mu_t = self._process(pred, mu_t)
        return torch.sum(torch.sum((pred - mu_t) ** 2, dim=1) + alpha * self._length(pred))

    def _weigh_by_dist(self, pred, mu_t, x0, x1):
        assert x0 is not None 
        assert x1 is not None
        pred, mu_t = self._process(pred, mu_t)
        return torch.sum((pred - mu_t) ** 2) * torch.norm(pred-mu_t)
    
    def _only_x1(self, pred, mu_t):
        pred, mu_t = self._process(pred, mu_t)
        return torch.sum((pred[:, -1, :] - mu_t[:, -1, :]) ** 2) + torch.sum((pred[:, 0, :] - mu_t[:, 0, :]) ** 2)
    
    def _on_subset(self, pred, mu_t, indices):
        dim = pred.shape[-1]
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, dim)

        pred_subset = torch.gather(pred, 1, indices_expanded)
        mu_t_subset = torch.gather(mu_t, 1, indices_expanded)
        
        return torch.sum((pred_subset - mu_t_subset) ** 2)
    
    def loss(self, pred, mu_t, x0=None, x1=None, alpha=0.5, indices=None):
        if self.loss_type == "NLL":
            return self._compute_NLL(pred, mu_t)
        elif self.loss_type == "regularize_length":
            return self._regularize_length(pred, mu_t, alpha)
        elif self.loss_type == "weigh_by_dist":
            return self._weigh_by_dist(pred, mu_t, x0, x1)
        elif self.loss_type == "only_x1":
            assert self.include_x1 == True
            return self._only_x1(pred, mu_t)
        elif self.loss_type == "subset":
            return self._on_subset(pred, mu_t, indices)
        else:
            raise NotImplementedError