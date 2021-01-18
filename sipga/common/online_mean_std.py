import numpy as np
import torch


class OnlineMeanStd(torch.nn.Module):
    """Track mean and standard deviation of inputs with incremental formula."""

    def __init__(self, epsilon=1e-5, shape=()):
        super().__init__()
        self.mean = torch.nn.Parameter(torch.zeros(*shape), requires_grad=False)
        self.std = torch.nn.Parameter(torch.ones(*shape), requires_grad=False)
        self.count = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.eps = epsilon
        self.bound = 10
        self.S = torch.nn.Parameter(torch.zeros(*shape), requires_grad=False)

    @staticmethod
    def _convert_to_torch(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(x, float):
            x = torch.tensor([x])  # use [] to make tensor torch.Size([1])
        if isinstance(x, np.floating):
            x = torch.tensor([x])  # use [] to make tensor torch.Size([1])
        return x

    def forward(self, x, subtract_mean=True, clip=False):
        """Make input average free and scale to standard deviation."""
        is_numpy = isinstance(x, np.ndarray)
        x = self._convert_to_torch(x)
        assert x.shape[-1] == self.mean.shape[-1], \
            f'got shape={x.shape} but expected: {self.mean.shape}'
        if subtract_mean:
            x_new = (x - self.mean) / (self.std + self.eps)
        else:
            x_new = x / (self.std + self.eps)
        if clip:
            x_new = torch.clamp(x_new, -self.bound, self.bound)
        x_new = x_new.numpy() if is_numpy else x_new

        return x_new

    def update(self, x) -> None:
        """Update internals incrementally."""
        x = self._convert_to_torch(x)
        assert len(x.shape) == 1, 'Not implemented for dim > 1.'

        self.count.data += 1
        new_mean = self.mean + (x - self.mean) / self.count
        new_S = self.S + (x - self.mean) * (x - new_mean)

        # nn.Parameters cannot be updated directly, must use .data instead
        self.mean.data = new_mean
        self.std.data = torch.sqrt(new_S / self.count)
        self.S.data = new_S
