import torch
from torch import nn
import torch.nn.functional as F


class MEMAE(nn.Module):
    """
    Memory augmented auto encoder. Autoencoder the creates embeddings
     then find most similar in the memory and recreates from memory
    """

    def __init__(self, num_memories):
        """

        Args:
            num_memories: number of memories to be used


        """

        super(MEMAE, self).__init__()

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
            self.epsilon = 1e-2

    def forward(self, x):
        batch, channel, height, width = x.size()
        # Create Encoding
        z = self.encoder(x)

        # Create copies of memories for the batch size
        ex_mem = self.memory.unsqueeze(0).repeat(batch, 1, 1)
        # To compare with all memeory locations
        ex_z = z.unsqueeze(1).repeat(1, self.num_memories, 1)
        # Find cosine similarity
        mem_logit = self.cosine_similarity(ex_z, ex_mem)

        # Softmax to get sum = 1
        mem_weight = F.softmax(mem_logit, dim=1)
        if self.addressing == 'soft':
            z_hat = torch.mm(mem_weight, self.memory)
        elif self.addressing == 'sparse':
            # Hard shrinkage operation as described in paper
            mem_weight = (self.relu(mem_weight - self.threshold) * mem_weight) \
                / (torch.abs(mem_weight - self.threshold) + self.epsilon)
            mem_weight = mem_weight / mem_weight.norm(p=1, dim=1) \
                .unsqueeze(1).expand(batch, self.num_memories)
            z_hat = torch.mm(mem_weight, self.memory)

        rec_x = self.decoder(z_hat)

        return rec_x


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
