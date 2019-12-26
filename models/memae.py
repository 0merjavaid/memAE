# import torch
# from torch import nn
# import torch.nn.functional as F
# from .memory import Memory
# from .autoencoder import *
# from .heads import *
# from .segnet import *


# class MEMAE(nn.Module):
#     """A Neural Turing Machine."""

#     def __init__(self, N=10, M=512, controller_size=512):
#         """Initialize the NTM.
#         :param num_inputs: External input size.
#         :param num_outputs: External output size.
#         :param controller: :class:`LSTMController`
#         :param memory: :class:`NTMMemory`
#         :param heads: list of :class:`NTMReadHead` or :class:`NTMWriteHead`
#         Note: This design allows the flexibility of using any number of read and
#               write heads independently, also, the order by which the heads are
#               called in controlled by the user (order in list)
#         """
#         super(MEMAE, self).__init__()

#         self.memory = Memory(N, M)
#         self.encoder = Encoder()
#         self.decoder = Decoder()
#         # self.segnet = get_model()
#         heads = nn.ModuleList([])
#         heads += [

#             NTMWriteHead(self.memory, controller_size),
#             NTMReadHead(self.memory, controller_size),
#         ]
#         self.heads = heads

#         self.N, self.M = self.memory.size()

#         # Initialize the initial previous read values to random biases
#         self.num_read_heads = 0
#         self.init_r = []
#         for head in heads:
#             if head.is_read_head():
#                 init_r_bias = torch.randn(1, self.M) * 0.01
#                 self.register_buffer("read{}_bias".format(
#                     self.num_read_heads), init_r_bias.data)
#                 self.init_r += [init_r_bias]
#                 self.num_read_heads += 1

#         assert self.num_read_heads > 0, "heads list must contain at least a single read head"

#         # Initialize a fully connected layer to produce the actual output:
#         #   [controller_output; previous_reads ] -> output
#         # self.fc = nn.Linear(64 + self.num_read_heads * self.M, num_outputs)
#         # self.reset_parameters()

#     # def create_new_state(self, batch_size):
#     #     init_r = [r.clone().repeat(batch_size, 1) for r in self.init_r]
#     #     controller_state = self.controller.create_new_state(batch_size)
#     #     heads_state = [head.create_new_state(batch_size) for head in self.heads]

#     #     return init_r, controller_state, heads_state

#     # def reset_parameters(self):
#     #     # Initialize the linear layer
#     #     nn.init.xavier_uniform_(self.fc.weight, gain=1)
#     #     nn.init.normal_(self.fc.bias, std=0.01)

#     def forward(self, x, inference=False):
#         """NTM forward function.
#         :param x: input vector (batch_size x num_inputs)
#         :param prev_state: The previous state of the NTM
#         """
#         # Unpack the previous state
#         # prev_reads, prev_controller_state, prev_heads_states = prev_state

#         # Use the controller to get an embeddings
#         # inp = torch.cat([x] + prev_reads, dim=1)
#         embeddings = self.encoder(x)
#         emd = embeddings.clone()         # embeddings = self.encoder(x)

#         for head in self.heads:
#             if head.is_read_head():
#                 read = head(emd)

#             else:
#                 if not inference:
#                     _ = head(emd.clone())

#         output = self.decoder(read)
#         return output


import torch
from torch import nn
import torch.nn.functional as F


class MEMAE(nn.Module):

    def __init__(self, num_memories):
        super(MEMAE, self).__init__()

        self.cls_loss_coef = 0  # cfg.cls_loss_coef

        self.num_memories = num_memories

        self.image_channel_size = 1

        self.addressing = "sparse"
        self.conv_channel_size = 16

        self.feature_size = self.conv_channel_size*4 * 6 * 6

        self.encoder = Encoder(image_channel_size=self.image_channel_size,
                               conv_channel_size=self.conv_channel_size,
                               )

        init_mem = torch.zeros(self.num_memories, self.feature_size)
        nn.init.kaiming_uniform_(init_mem)
        self.memory = nn.Parameter(init_mem)

        self.cosine_similarity = nn.CosineSimilarity(dim=2,)

        self.decoder = Decoder(
            image_channel_size=self.image_channel_size,
            conv_channel_size=self.conv_channel_size,
        )

        self.relu = nn.ReLU(inplace=True)

        if self.addressing == 'sparse':
            self.threshold = 1 / self.memory.size(0)
            self.epsilon = 1e-15

    def forward(self, x):
        batch, channel, height, width = x.size()

        z = self.encoder(x)

        ex_mem = self.memory.unsqueeze(0).repeat(batch, 1, 1)
        ex_z = z.unsqueeze(1).repeat(1, self.num_memories, 1)

        mem_logit = self.cosine_similarity(ex_z, ex_mem)
        mem_weight = F.softmax(mem_logit, dim=1)
        if self.addressing == 'soft':
            z_hat = torch.mm(mem_weight, self.memory)
        elif self.addressing == 'sparse':
            mem_weight = (self.relu(mem_weight - self.threshold) * mem_weight) \
                / (torch.abs(mem_weight - self.threshold) + self.epsilon)
            mem_weight = mem_weight / mem_weight.norm(p=1, dim=1) \
                .unsqueeze(1).expand(batch, self.num_memories)
            z_hat = torch.mm(mem_weight, self.memory)

        rec_x = self.decoder(z_hat)
        # rec_x = self.decoder(z)

        return rec_x

    def generate_from_memory(self, idx):
        z_hat = self.memory[idx]
        batch, _ = z_hat.size()

        rec_x = self.decoder(z_hat)

        if self.cls_loss_coef > 0.0:
            logit_x = self.classifier(rec_x)
        else:
            logit_x = torch.zeros(batch, self.num_classes).to(self.device)
        return dict(rec_x=rec_x, logit_x=logit_x)


class Encoder(nn.Module):

    def __init__(self, image_channel_size, conv_channel_size):
        super(Encoder, self).__init__()
        self.image_channel_size = image_channel_size
        self.conv_channel_size = conv_channel_size

        self.conv1 = nn.Conv2d(in_channels=self.image_channel_size,
                               out_channels=self.conv_channel_size,
                               kernel_size=1,
                               stride=2,
                               padding=1,
                               )

        self.bn1 = nn.BatchNorm2d(num_features=self.conv_channel_size,)

        self.conv2 = nn.Conv2d(in_channels=self.conv_channel_size,
                               out_channels=self.conv_channel_size*2,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               )

        self.bn2 = nn.BatchNorm2d(num_features=self.conv_channel_size*2,)

        self.conv3 = nn.Conv2d(in_channels=self.conv_channel_size*2,
                               out_channels=self.conv_channel_size*4,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               )

        self.bn3 = nn.BatchNorm2d(num_features=self.conv_channel_size*4,)

        self.conv4 = nn.Conv2d(in_channels=self.conv_channel_size*4,
                               out_channels=self.conv_channel_size*4,
                               kernel_size=3,
                               stride=2,
                               padding=0,
                               )

        self.bn4 = nn.BatchNorm2d(num_features=self.conv_channel_size*4,)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        batch, _, _, _ = x.size()
        # print (x.shape)

        x = x.view(batch, -1)
        return x


class Decoder(nn.Module):

    def __init__(self, image_channel_size, conv_channel_size):
        super(Decoder, self).__init__()
        self.image_channel_size = image_channel_size
        self.conv_channel_size = conv_channel_size

        self.deconv0 = nn.ConvTranspose2d(in_channels=self.conv_channel_size*4,
                                          out_channels=self.conv_channel_size*4,
                                          kernel_size=3,
                                          stride=2,
                                          padding=0,
                                          output_padding=0,
                                          )

        self.bn0 = nn.BatchNorm2d(num_features=self.conv_channel_size*4,)

        self.deconv1 = nn.ConvTranspose2d(in_channels=self.conv_channel_size*4,
                                          out_channels=self.conv_channel_size*2,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=0,
                                          )

        self.bn1 = nn.BatchNorm2d(num_features=self.conv_channel_size*2,)

        self.deconv2 = nn.ConvTranspose2d(in_channels=self.conv_channel_size*2,
                                          out_channels=self.conv_channel_size,
                                          kernel_size=2,
                                          stride=2,
                                          padding=1,
                                          output_padding=1,
                                          )

        self.bn2 = nn.BatchNorm2d(num_features=self.conv_channel_size,)

        self.deconv3 = nn.ConvTranspose2d(in_channels=self.conv_channel_size,
                                          out_channels=self.image_channel_size,
                                          kernel_size=2,
                                          stride=2,
                                          padding=0,

                                          )

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print (x.shape)
        x = x.view(-1, self.conv_channel_size*4, 6, 6)

        x = self.deconv0(x)
        x = self.bn0(x)
        x = self.relu(x)

        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.deconv3(x)
        x = self.sigmoid(x)
        # print (x.shape)
        return x
