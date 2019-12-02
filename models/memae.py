import torch
from torch import nn
import torch.nn.functional as F
from .memory import Memory
from .autoencoder import *
from .heads import *
from .segnet import SegNet


class MEMAE(nn.Module):
    """A Neural Turing Machine."""

    def __init__(self, N=250, M=512, controller_size=512):
        """Initialize the NTM.
        :param num_inputs: External input size.
        :param num_outputs: External output size.
        :param controller: :class:`LSTMController`
        :param memory: :class:`NTMMemory`
        :param heads: list of :class:`NTMReadHead` or :class:`NTMWriteHead`
        Note: This design allows the flexibility of using any number of read and
              write heads independently, also, the order by which the heads are
              called in controlled by the user (order in list)
        """
        super(MEMAE, self).__init__()

        self.memory = Memory(N, M)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.segnet = SegNet()
        heads = nn.ModuleList([])
        heads += [
            NTMReadHead(self.memory, controller_size),
            NTMWriteHead(self.memory, controller_size)
        ]
        self.heads = heads

        self.N, self.M = self.memory.size()

        # Initialize the initial previous read values to random biases
        self.num_read_heads = 0
        self.init_r = []
        for head in heads:
            if head.is_read_head():
                init_r_bias = torch.randn(1, self.M) * 0.01
                self.register_buffer("read{}_bias".format(
                    self.num_read_heads), init_r_bias.data)
                self.init_r += [init_r_bias]
                self.num_read_heads += 1

        assert self.num_read_heads > 0, "heads list must contain at least a single read head"

        # Initialize a fully connected layer to produce the actual output:
        #   [controller_output; previous_reads ] -> output
        # self.fc = nn.Linear(64 + self.num_read_heads * self.M, num_outputs)
        # self.reset_parameters()

    # def create_new_state(self, batch_size):
    #     init_r = [r.clone().repeat(batch_size, 1) for r in self.init_r]
    #     controller_state = self.controller.create_new_state(batch_size)
    #     heads_state = [head.create_new_state(batch_size) for head in self.heads]

    #     return init_r, controller_state, heads_state

    # def reset_parameters(self):
    #     # Initialize the linear layer
    #     nn.init.xavier_uniform_(self.fc.weight, gain=1)
    #     nn.init.normal_(self.fc.bias, std=0.01)

    def forward(self, x, inference=False):
        """NTM forward function.
        :param x: input vector (batch_size x num_inputs)
        :param prev_state: The previous state of the NTM
        """
        # Unpack the previous state
        # prev_reads, prev_controller_state, prev_heads_states = prev_state

        # Use the controller to get an embeddings
        # inp = torch.cat([x] + prev_reads, dim=1)
        embeddings, indices = self.segnet(x)

        # embeddings = self.encoder(x)

        for head in self.heads:
            if head.is_read_head():
                read = head(embeddings)

            else:
                _ = head(embeddings)
        # print (read.shape)
        output = self.segnet(read.unsqueeze(-1).unsqueeze(-1), indices)
        # heads_states += [head_state]
        # output = self.decoder(read.unsqueeze(-1).unsqueeze(-1))
        # output = self.decoder(embeddings)
        return output
