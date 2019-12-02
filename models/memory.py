import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Memory(nn.Module):

    def __init__(self, N, M):
        super(Memory, self).__init__()

        self.N = N
        self.M = M

        self.register_buffer('memory', torch.Tensor(N, M))
        # nn.init.constant_(self.memory, 1e-6)
        stdev = 1 / (np.sqrt(N + M))
        nn.init.uniform_(self.memory, -stdev, stdev)

    def size(self):
        return self.N, self.M

    def read(self, w):
        # print (w.size(), self.memory.size())
        assert w.size()[-1] == self.memory.size()[0]
        return torch.matmul(w, self.memory)

    def write(self, w, e, a):
        """write to memory (according to section 3.2 of NTM paper)."""
        # print ("before", self.memory)
        self.prev_mem = self.memory
        self.memory = torch.Tensor(self.N, self.M)
        erase = torch.matmul(w.unsqueeze(-1), e.unsqueeze(0))
        add = torch.matmul(w.unsqueeze(-1), a.unsqueeze(0))

        self.memory = self.prev_mem * (1 - erase) + add
        self.memory = self.memory.squeeze()
        # print ("after", w/)

    def address(self, k, β):
        """NTM Addressing (according to section 3.3).
        Returns a softmax weighting over the rows of the memory matrix.
        :param k: The key vector.
        :param β: The key strength (focus).
        :param g: Scalar interpolation gate (with previous weighting).
        :param s: Shift weighting.
        :param γ: Sharpen weighting scalar.
        :param w_prev: The weighting produced in the previous time step.
        """
        k = k.view(1, -1)
        wc = F.softmax(β * F.cosine_similarity(self.memory + 1e-4, k + 1e-4, dim=-1), dim=-1)

        return wc
