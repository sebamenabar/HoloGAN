import torch
import torch.nn as nn

from layers import AdaIN
from transformation_utils import (
    generate_inv_transform_matrix,
    transform_voxel_to_match_image,
)


class ObjectGenerator(nn.Module):
    def __init__(
        self,
        z_dim,
        w_dim,
        use_learnable_proj=True,  # TEMP: to replace with perspective projection
        w_shape=(4, 4, 4),
        upconv_filters=[128, 64],
        upconv_ks=[3, 3],  # for upconvolution, ks: kernel_size
        upconv_strides=[2, 2],
        upconv_out_padding=[1, 1],
        upconv_padding=[1, 1],
    ):
        super().__init__()

        self.z_dim = z_dim
        self.w_dim = w_dim
        self.w_shape = w_shape
        self.upconv_filters = upconv_filters
        self.use_learnable_proj = use_learnable_proj

        self.adain0 = AdaIN(w_dim, z_dim)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.w = nn.Parameter(torch.normal(0, 0.02, (1, w_dim, *w_shape)))

        in_channels = w_dim
        deconvs = []
        adains = []
        for filters, ks, stride, out_padding, padding in zip(
            upconv_filters,
            upconv_ks,
            upconv_strides,
            upconv_out_padding,
            upconv_padding,
        ):
            deconvs.append(
                nn.ConvTranspose3d(
                    in_channels=in_channels,
                    out_channels=filters,
                    kernel_size=ks,
                    stride=stride,
                    output_padding=out_padding,
                    padding=padding,
                )
            )
            adains.append(AdaIN(filters, z_dim))

            in_channels = filters

        self.deconvs = nn.ModuleList(deconvs)
        self.adains = nn.ModuleList(adains)

        if use_learnable_proj:
            self.proj1 = nn.ConvTranspose3d(
                in_channels=upconv_filters[-1],
                out_channels=upconv_filters[-1],
                kernel_size=3,
                stride=1,
                padding=1,
            )
            self.proj2 = nn.ConvTranspose3d(
                in_channels=upconv_filters[-1],
                out_channels=upconv_filters[-1],
                kernel_size=3,
                stride=1,
                padding=1,
            )
        else:
            print("USING 3D TRANSFORM FOR OBJECT PROJECTION")
        self.init_params()

    def init_params(self):
        for deconv in self.deconvs:
            nn.init.normal_(deconv.weight, std=0.02)
            nn.init.zeros_(deconv.bias)
        nn.init.normal_(self.proj1.weight, std=0.02)
        nn.init.zeros_(self.proj1.bias)
        nn.init.normal_(self.proj2.weight, std=0.02)
        nn.init.zeros_(self.proj2.bias)

    def forward(self, z, view_in=None):
        z_dim = z.size(-1)
        assert (
            z_dim == self.z_dim
        ), f"Input dimension ({z_dim}) does not match expected value ({self.z_dim})"

        if len(z.size()) == 3:
            bsz, num_objs, _ = z.size()
            z = z.view(bsz * num_objs, self.z_dim)
        else:
            bsz, _ = z.size()
            num_objs = 1
        if view_in is not None and len(view_in.size()) == 3:
            view_in = view_in.view(bsz * num_objs, 9)

        w = self.w.repeat(bsz * num_objs, 1, 1, 1, 1)
        h = self.adain0(w, z)
        h = self.lrelu(h)

        for deconv, adain in zip(self.deconvs, self.adains):
            h = deconv(h)
            h = adain(h, z)
            h = self.lrelu(h)

        if self.use_learnable_proj:
            h2_proj1 = self.proj1(h)
            h2_proj1 = self.lrelu(h2_proj1)

            h2_proj2 = self.proj2(h2_proj1)
            h2_proj2 = self.lrelu(h2_proj2)

            out = h2_proj2
        else:
            A = generate_inv_transform_matrix(view_in)
            A = A[:, :3]
            grid = nn.functional.affine_grid(A, h.size())
            h_rotated = nn.functional.grid_sample(h, grid)
            h_rotated = transform_voxel_to_match_image(h_rotated)

            out = h_rotated

        return out.view(bsz, num_objs, *out.size()[1:])


class Generator(nn.Module):
    def __init__(
        self,
        z_dim_bg=30,
        z_dim_fg=90,
        w_dim_bg=256,
        w_dim_fg=512,
        filters=[64, 64, 64],
        ks=[1, 4, 4],
        strides=[1, 2, 2],
        upconv_paddings=[0, 1, 1],
        upconv_out_paddings=[0, 0, 0],
        use_learnable_proj=True,
    ):
        super().__init__()

        self.z_dim_bg = z_dim_bg
        self.z_dim_fg = z_dim_fg

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.bg_generator = ObjectGenerator(z_dim_bg, w_dim_bg, use_learnable_proj)
        self.fg_generator = ObjectGenerator(z_dim_fg, w_dim_fg, use_learnable_proj)

        deconvs = []
        inst_norms = []
        prev_out_channels = 1024
        for f, k, s, p, op in zip(
            filters, ks, strides, upconv_paddings, upconv_out_paddings
        ):
            deconvs.append(
                nn.ConvTranspose2d(
                    in_channels=prev_out_channels,
                    out_channels=f,
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    output_padding=op,
                )
            )
            # inst_norms.append(nn.InstanceNorm2d(num_features=f))
            prev_out_channels = f

        self.deconvs = nn.ModuleList(deconvs)
        self.inst_norms = nn.ModuleList(inst_norms)

        self.out_conv = nn.ConvTranspose2d(
            # self.out_conv = nn.Conv2d(
            in_channels=prev_out_channels,
            out_channels=3,
            kernel_size=5,
            stride=1,
            padding=2,
            # output_padding=1,
        )

    def init_params(self):
        for deconv in self.deconvs:
            nn.init.normal_(deconv.weight, std=0.02)
            nn.init.zeros_(deconv.bias)
        nn.init.normal_(self.out_conv.weight, std=0.02)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, z_bg, z_fg, view_in_bg=None, view_in_fg=None):
        bsz = z_bg.size(0)

        bg = self.bg_generator(
            z_bg, view_in_bg,
        )  # (bsz, 1, height, width, depth, num_channels)
        fg = self.fg_generator(
            z_fg, view_in_fg,
        )  # (bsz, num_objects, height, width, depth, num_channels)
        composed_scene = torch.cat((bg, fg), axis=1)
        composed_scene = torch.max(composed_scene, dim=1)[0]

        h2_2d = composed_scene.view(bsz, 16 * 64, 16, 16)
        h = h2_2d
        # for deconv, inst_norm in zip(self.deconvs, self.inst_norms):
        for deconv in zip(self.deconvs,):
            # h = self.lrelu(inst_norm(deconv(h)))
            h = self.lrelu(deconv(h))

        h = torch.tanh(self.out_conv(h))

        return h


class Discriminator(nn.Module):
    def __init__(
        self, filters=[64, 128, 256, 512], ks=[5, 5, 5, 5], strides=[2, 2, 2, 2],
    ):
        super().__init__()

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        convs = []
        inst_norms = []
        prev_out_channels = 3
        for f, k, s in zip(filters, ks, strides):
            convs.append(
                nn.utils.spectral_norm(
                    nn.Conv2d(
                        in_channels=prev_out_channels,
                        out_channels=f,
                        kernel_size=k,
                        stride=s,
                    )
                )
            )
            # inst_norms.append(nn.InstanceNorm2d(f))
            prev_out_channels = f

        self.convs = nn.ModuleList(convs)
        self.inst_norms = nn.ModuleList(inst_norms)
        self.linear = nn.utils.spectral_norm(
            nn.Linear(in_features=prev_out_channels, out_features=1)
        )
        self.init_params()

    def init_params(self):
        for conv in self.convs:
            nn.init.normal_(conv.weight, std=0.02)
            nn.init.zeros_(conv.bias)
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        for conv in zip(self.convs,):
            # for conv, norm in zip(self.convs, self.inst_norms):
            x = conv(x)
            # x = norm(x)
            x = self.lrelu(x)

        return self.linear(x.view(x.size(0), -1))
