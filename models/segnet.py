# # import torch.nn as nn
# # from torchsummary import summary
# # import torch


# # class conv2DBatchNormRelu(nn.Module):

# #     def __init__(
# #             self,
# #             in_channels,
# #             n_filters,
# #             k_size,
# #             stride,
# #             padding,
# #             bias=True,
# #             dilation=1,
# #             with_bn=True,
# #     ):
# #         super(conv2DBatchNormRelu, self).__init__()

# #         conv_mod = nn.Conv2d(int(in_channels),
# #                              int(n_filters),
# #                              kernel_size=k_size,
# #                              padding=padding,
# #                              stride=stride,
# #                              bias=bias,
# #                              dilation=dilation, )

# #         if with_bn:
# #             self.cbr_unit = nn.Sequential(conv_mod,
# #                                           nn.BatchNorm2d(int(n_filters)),
# #                                           nn.ReLU(inplace=True))
# #         else:
# #             self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

# #     def forward(self, inputs):
# #         outputs = self.cbr_unit(inputs)
# #         return outputs


# # class segnetDown2(nn.Module):

# #     def __init__(self, in_size, out_size):
# #         super(segnetDown2, self).__init__()
# #         self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
# #         self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
# #         self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

# #     def forward(self, inputs):
# #         outputs = self.conv1(inputs)
# #         outputs = self.conv2(outputs)
# #         unpooled_shape = outputs.size()
# #         outputs, indices = self.maxpool_with_argmax(outputs)
# #         return outputs, indices, unpooled_shape


# # class segnetDown3(nn.Module):

# #     def __init__(self, in_size, out_size):
# #         super(segnetDown3, self).__init__()
# #         self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
# #         self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
# #         self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
# #         self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

# #     def forward(self, inputs):
# #         outputs = self.conv1(inputs)
# #         outputs = self.conv2(outputs)
# #         outputs = self.conv3(outputs)
# #         unpooled_shape = outputs.size()
# #         outputs, indices = self.maxpool_with_argmax(outputs)
# #         return outputs, indices, unpooled_shape


# # class segnetUp2(nn.Module):

# #     def __init__(self, in_size, out_size):
# #         super(segnetUp2, self).__init__()
# #         self.unpool = nn.MaxUnpool2d(2, 2)
# #         self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
# #         self.conv2 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

# #     def forward(self, inputs, indices, output_shape):
# #         outputs = self.unpool(input=inputs, indices=indices,
# #                               output_size=output_shape)
# #         outputs = self.conv1(outputs)
# #         outputs = self.conv2(outputs)
# #         return outputs


# # class segnetUp3(nn.Module):

# #     def __init__(self, in_size, out_size):
# #         super(segnetUp3, self).__init__()
# #         self.unpool = nn.MaxUnpool2d(2, 2)
# #         self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
# #         self.conv2 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
# #         self.conv3 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

# #     def forward(self, inputs, indices, output_shape):
# #         outputs = self.unpool(input=inputs, indices=indices,
# #                               output_size=output_shape)
# #         outputs = self.conv1(outputs)
# #         outputs = self.conv2(outputs)
# #         outputs = self.conv3(outputs)
# #         return outputs


# # class SegNet(nn.Module):

# #     def __init__(self, n_classes=3, in_channels=3, is_unpooling=True):
# #         super(SegNet, self).__init__()

# #         self.in_channels = in_channels
# #         self.is_unpooling = is_unpooling

# #         self.down1 = segnetDown2(self.in_channels, 64)
# #         self.down2 = segnetDown2(64, 128)
# #         self.down3 = segnetDown3(128, 256)
# #         self.down4 = segnetDown3(256, 512)
# #         self.down5 = segnetDown3(512, 512)

# #         self.up5 = segnetUp3(512, 512)
# #         self.up4 = segnetUp3(512, 256)
# #         self.up3 = segnetUp3(256, 128)
# #         self.up2 = segnetUp2(128, 64)
# #         self.up1 = segnetUp2(64, n_classes)

# #     def forward(self, inputs, indices=None):
# #         if indices == None:
# #             down1, indices_1, unpool_shape1 = self.down1(inputs)
# #             down2, indices_2, unpool_shape2 = self.down2(down1)
# #             down3, indices_3, unpool_shape3 = self.down3(down2)
# #             down4, indices_4, unpool_shape4 = self.down4(down3)
# #             down5, indices_5, unpool_shape5 = self.down5(down4)
# #             return down5, [[indices_5, unpool_shape5], [indices_4, unpool_shape4], [indices_3, unpool_shape3], [indices_2, unpool_shape2], [indices_1, unpool_shape1]]
# #         else:
# #             indices = [[torch.zeros_like(i), j] for i, j in indices]
# #             up5 = self.up5(inputs, indices[0][0], indices[0][1])
# #             up4 = self.up4(up5, indices[1][0], indices[1][1])
# #             up3 = self.up3(up4, indices[2][0], indices[2][1])
# #             up2 = self.up2(up3, indices[3][0], indices[3][1])
# #             up1 = self.up1(up2, indices[4][0], indices[4][1])

# #             return up1

# #     def init_vgg16_params(self, vgg16):
# #         blocks = [self.down1, self.down2, self.down3, self.down4, self.down5]

# #         ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
# #         features = list(vgg16.features.children())

# #         vgg_layers = []
# #         for _layer in features:
# #             if isinstance(_layer, nn.Conv2d):
# #                 vgg_layers.append(_layer)

# #         merged_layers = []
# #         for idx, conv_block in enumerate(blocks):
# #             if idx < 2:
# #                 units = [conv_block.conv1.cbr_unit, conv_block.conv2.cbr_unit]
# #             else:
# #                 units = [
# #                     conv_block.conv1.cbr_unit,
# #                     conv_block.conv2.cbr_unit,
# #                     conv_block.conv3.cbr_unit,
# #                 ]
# #             for _unit in units:
# #                 for _layer in _unit:
# #                     if isinstance(_layer, nn.Conv2d):
# #                         merged_layers.append(_layer)

# #         assert len(vgg_layers) == len(merged_layers)

# #         for l1, l2 in zip(vgg_layers, merged_layers):
# #             if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
# #                 assert l1.weight.size() == l2.weight.size()
# #                 assert l1.bias.size() == l2.bias.size()
# #                 l2.weight.data = l1.weight.data
# #                 l2.bias.data = l1.bias.data


# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# import torchvision


# class SaveFeatures():
#     features = None

#     def __init__(self, m):
#         """
#         :param m:
#         """
#         self.hook = m.register_forward_hook(self.hook_fn)

#     def hook_fn(self, module, input, output):
#         """
#         :param module:
#         :param input:
#         :param output:
#         :return:
#         """
#         self.features = output

#     def remove(self):
#         """
#         :return:
#         """
#         self.hook.remove()


# class UnetBlock(nn.Module):

#     def __init__(self, up_in, x_in, n_out):
#         """
#         :param up_in:
#         :param x_in:
#         :param n_out:
#         """
#         super().__init__()
#         up_out = x_out = n_out
#         self.x_conv = nn.Conv2d(x_in, x_out, 1)
#         self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
#         self.bn = nn.BatchNorm2d(n_out)

#     def forward(self, up_p, x_p):
#         """
#         :param up_p:
#         :param x_p:
#         :return:
#         """
#         up_p = self.tr_conv(up_p)
#         x_p = self.x_conv(x_p)
#         cat_p = torch.cat([up_p, x_p], dim=1)
#         return self.bn(F.relu(up_p))


# class Unet34(nn.Module):

#     def __init__(self, rn):
#         """
#         :param rn:
#         """
#         super().__init__()
#         self.rn = rn
#         self.sfs = [SaveFeatures(rn[i]) for i in [2, 4, 5, 6]]
#         self.up1 = UnetBlock(512, 256, 256)
#         self.up2 = UnetBlock(256, 128, 256)
#         self.up3 = UnetBlock(256, 64, 256)
#         self.up4 = UnetBlock(256, 64, 256)
#         self.up5 = nn.ConvTranspose2d(256, 3, 2, stride=2)

#         self.final_relu = nn.ReLU()
#         self.final_bn = nn.BatchNorm2d(3)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.first_linear = nn.Linear(512, 512*49)

#     def forward(self, x, inference=False):
#         """
#         :param x:
#         :return:
#         """
#         if not inference:
#             x = F.relu(self.rn(x))
#             # x = self.avgpool(x)
#             print (x.shape)
#             return x
#         # print (x.shape)
#         else:
#             # x = self.first_linear(x.view(-1)).view(1, 512, 7, 7)
#             # print (x.shape)
#             x = self.up1(x, self.sfs[3].features)
#             x = self.up2(x, self.sfs[2].features)
#             x = self.up3(x, self.sfs[1].features)
#             x = self.up4(x, self.sfs[0].features)
#             x = self.up5(x)
#             x = self.final_relu(x)
#             x = self.final_bn(x)
#             return x
#         # return F.sigmoid(x[:, 0])

#     def close(self):
#         """
#         :return:
#         """
#         for sf in self.sfs:
#             sf.remove()


# def get_model():
#     m_base = nn.Sequential(
#         *(list(torchvision.models.resnet34(pretrained=True).children())[:8]))
#     model = Unet34(m_base).cuda()
#     return model
# # print (model(torch.zeros((1, 3, 224, 224)).cuda()).shape)


import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, image_channel_size=3, conv_channel_size=16):
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
                               out_channels=self.conv_channel_size*2,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               )

        self.bn3 = nn.BatchNorm2d(num_features=self.conv_channel_size*2,)

        self.relu = nn.ReLU(inplace=False)

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

        batch, _, _, _ = x.size()
        x = x.view(batch, -1)
        return x


class Decoder(nn.Module):

    def __init__(self, image_height=28, image_width=28, image_channel_size=3, conv_channel_size=16):
        super(Decoder, self).__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.image_channel_size = image_channel_size
        self.conv_channel_size = conv_channel_size

        self.deconv1 = nn.ConvTranspose2d(in_channels=self.conv_channel_size*2,
                                          out_channels=self.conv_channel_size*2,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=1,
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
                                          padding=1,
                                          )

        self.relu = nn.Sigmoid()
        self.bn3 = nn.BatchNorm2d(num_features=self.image_channel_size,)

    def forward(self, x):
        x = x.view(-1, self.conv_channel_size*2, 4, 4)

        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.deconv3(x)
        x = self.relu(x)
        # x = self.bn3(x)
        return x
