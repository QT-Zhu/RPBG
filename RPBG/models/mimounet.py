import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, relu=True, dilation=1, padding_mode='reflect', act_fun=nn.ELU, normalization=nn.BatchNorm2d):
        super().__init__()
        self.pad_mode = padding_mode
        self.filter_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        n_pad_pxl = int(self.dilation * (self.filter_size - 1) / 2)
        self.flag=relu

        # this is for backward campatibility with older model checkpoints
        self.block = nn.ModuleDict(
            {
                'conv_f': nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=n_pad_pxl),
                'act_f': act_fun(),
                'conv_m': nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=n_pad_pxl),
                'act_m': nn.Sigmoid(),
                'norm': normalization(out_channels)
            }
        )

    def forward(self, x, *args, **kwargs):
        if self.flag:
            features = self.block.act_f(self.block.conv_f(x))
        else:
            features = self.block.conv_f(x)
        mask = self.block.act_m(self.block.conv_m(x))
        output = features * mask
        output = self.block.norm(output)

        return output

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            GatedConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            GatedConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x

class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            GatedConv(8, out_plane//4, kernel_size=3, stride=1, relu=True),
            GatedConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            GatedConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            GatedConv(out_plane // 2, out_plane-8, kernel_size=1, stride=1, relu=True)
        )

        self.conv = GatedConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)

class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = GatedConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out

class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            GatedConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            GatedConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x3, x4):
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.conv(x)


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)


        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        return output

# MIMO-UNet with DAC, adapted from https://github.com/JOP-Lee/READ/blob/main/READ/models/unet.py
class MIMOUNet(nn.Module):
    r""" Rendering network with UNet architecture and multi-scale input.

    Args:
        num_input_channels: Number of channels in the input tensor or list of tensors. An integer or a list of integers for each input tensor.
        num_output_channels: Number of output channels.
        feature_scale: Division factor of number of convolutional channels. The bigger the less parameters in the model.
        num_res: Number of block resnet.
    """
    def __init__(
        self,
        num_input_channels=8, 
        num_output_channels=3,
        feature_scale=4,
        num_res=4

    ):
        super().__init__()

        self.feature_scale = feature_scale
        base_channel = 32

        base_channel = 32
        
        
        self.feat_extract = nn.ModuleList([
            GatedConv(num_input_channels, base_channel, kernel_size=3, relu=True, stride=1),
            GatedConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            GatedConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            GatedConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2),
            GatedConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2),
            GatedConv(base_channel, num_output_channels, kernel_size=3, relu=False, stride=1),
            GatedConv(base_channel*4, base_channel*8, kernel_size=3, relu=True, stride=2),
            GatedConv(base_channel*8, base_channel*4, kernel_size=4, relu=True, stride=2),
        ])

        self.SCM0 = SCM(base_channel * 8)
        self.SCM1 = SCM(base_channel * 4)
        self.SCM2 = SCM(base_channel * 2)
        
        self.FAM0 = FAM(base_channel * 8)
        self.FAM1 = FAM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 15, base_channel*1),
            AFF(base_channel * 15, base_channel*2),
            AFF(base_channel * 15, base_channel*4),
        ])
    
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
            EBlock(base_channel*8, num_res)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 8, num_res),
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            GatedConv(base_channel * 8, base_channel * 4, kernel_size=1, relu=True, stride=1),
            GatedConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            GatedConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1)

        ])

        self.ConvsOut = nn.ModuleList(
            [
                GatedConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                GatedConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.up =nn.Upsample(scale_factor=4, mode='bilinear')

        self.ffc = FFCResnetBlock(num_input_channels)
        self.ffc_2 = FFCResnetBlock(num_input_channels)
        self.ffc_4 = FFCResnetBlock(num_input_channels)


    def forward(self, *inputs, **kwargs):
        inputs = list(inputs)

        x =inputs[0]
        x_2 = inputs[1]
        x_4 = inputs[2]
        x_8 = inputs[3]

        x = self.ffc(x)[0]
        x_2 = self.ffc_2(x_2)[0]
        x_4 = self.ffc_4(x_4)[0]

        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)
        z8 = self.SCM0(x_8)

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        res3 = self.Encoder[2](z)

        z = self.feat_extract[6](res3)
        z = self.FAM0(z, z8)
        z = self.Encoder[3](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z13 = F.interpolate(res1, scale_factor=0.25)

        z21 = F.interpolate(res2, scale_factor=2)
        z23 = F.interpolate(res2, scale_factor=0.5)

        z32 = F.interpolate(res3, scale_factor=2)
        z31 = F.interpolate(res3, scale_factor=4)

        z43 = F.interpolate(z, scale_factor=2)
        z42 = F.interpolate(z43, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res1 = self.AFFs[0](res1, z21, z31, z41)
        res2 = self.AFFs[1](z12, res2, z32, z42) 
        res3 = self.AFFs[2](z13, z23, res3, z43)
        #z = self.ffc(z)[0]
        
        z = self.Decoder[0](z)
        

        z = self.feat_extract[7](z)
        
        z= self.up(z)
        z = torch.cat([z, res3], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)


        z = self.feat_extract[3](z)
        z= self.up(z)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)


        z = self.feat_extract[4](z)
        z= self.up(z)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[2](z)
        z = self.Decoder[3](z)
        z = self.feat_extract[5](z)

        return {'im_out':z}

class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output

class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)

        return out_xl, out_xg

class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin=0, ratio_gout=0,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g

class FFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, dilation=1,
                 **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g

        return out

if __name__ == '__main__':
    import time

    model = MIMOUNet().to('cuda')
    input = []
    img_sh = [1920,1072]
    sh_unit = 32
    img_sh = list(map(lambda a: a-a%sh_unit+sh_unit if a%sh_unit!=0 else a, img_sh))
    down = lambda a,b : a//2**b
    input.append(torch.zeros((1,8,down(img_sh[0],0), down(img_sh[1],0)), requires_grad=True).cuda())
    input.append(F.interpolate(input[0],scale_factor=0.5))
    input.append(F.interpolate(input[1],scale_factor=0.5))
    input.append(F.interpolate(input[2],scale_factor=0.5))

    model.eval()
    st = time.time()
    print(input[0].max(),input[0].min())
    with torch.set_grad_enabled(False):
        out = model(*input)
        print('model',time.time()-st)
    time.sleep(100)
    model.to('cpu')
