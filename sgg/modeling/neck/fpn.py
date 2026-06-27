import torch.nn as nn
import torch.nn.functional as F


class ConvModule(nn.Module):
    """
    A simplified convolution block implemented with pure PyTorch.

    This module applies convolution first, followed by an optional
    normalization layer and an optional activation layer.

    The execution order is:

        conv -> norm -> act

    Args:
        in_channels (int):
            Number of channels in the input feature map.

        out_channels (int):
            Number of channels produced by the convolution.

        kernel_size (int or tuple):
            Size of the convolving kernel.

        stride (int or tuple, optional):
            Stride of the convolution. Default is 1.

        padding (int or tuple, optional):
            Zero-padding added to both sides of the input. Default is 0.

        bias (bool, optional):
            Whether to use bias in the convolution layer. Default is True.

        use_norm (bool, optional):
            Whether to append a BatchNorm2d layer after convolution.
            Default is False.

        use_act (bool, optional):
            Whether to append a ReLU activation after normalization.
            Default is False.

    Shape:
        - Input: (B, C_in, H, W)
        - Output: (B, C_out, H_out, W_out)

    Example:
        >>> block = ConvModule(64, 128, kernel_size=3, padding=1,
        ...                    use_norm=True, use_act=True)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> y = block(x)
        >>> print(y.shape)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        use_norm=False,
        use_act=False,
    ):
        super().__init__()

        # Main convolution layer.
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        # Optional normalization layer.
        self.norm = nn.BatchNorm2d(out_channels) if use_norm else None

        # Optional activation layer.
        self.act = nn.ReLU(inplace=True) if use_act else None

    def forward(self, x):
        """
        Forward pass of the ConvModule.

        Args:
            x (Tensor):
                Input feature map of shape (B, C, H, W).

        Returns:
            Tensor:
                Output feature map after convolution, optional normalization,
                and optional activation.
        """
        x = self.conv(x)

        if self.norm is not None:
            x = self.norm(x)

        if self.act is not None:
            x = self.act(x)

        return x


class FPN(nn.Module):
    """
    Feature Pyramid Network (FPN) implemented with pure PyTorch.

    FPN is commonly used in detection and segmentation models to build
    a multi-scale feature pyramid from backbone feature maps. It creates
    lateral connections from different backbone stages and fuses them
    through a top-down pathway.

    Main steps:
        1. Apply 1x1 lateral convolutions to backbone features.
        2. Fuse high-level features into low-level features by upsampling.
        3. Apply 3x3 convolutions to generate final pyramid outputs.
        4. Optionally add extra pyramid levels.

    Args:
        in_channels (list[int]):
            Number of channels for each input feature map from the backbone.

        out_channels (int):
            Number of channels for each output feature map in the pyramid.

        num_outs (int):
            Number of output scales.

        start_level (int, optional):
            Index of the first backbone level to use. Default is 0.

        end_level (int, optional):
            Index of the last backbone level to use (inclusive behavior is
            handled internally as exclusive + 1). If set to -1, all input
            levels from start_level to the end are used. Default is -1.

        add_extra_convs (bool or str, optional):
            Whether to add extra convolution layers for extra output levels.

            Supported values:
                - False: do not add extra conv layers, use max pooling instead
                - True: same as "on_input"
                - "on_input": extra conv starts from the last backbone input
                - "on_lateral": extra conv starts from the last lateral feature
                - "on_output": extra conv starts from the last FPN output

            Default is False.

        relu_before_extra_convs (bool, optional):
            Whether to apply ReLU before extra convolution layers.
            Default is False.

        no_norm_on_lateral (bool, optional):
            Whether to disable normalization on lateral convolution blocks.
            Default is False.

        upsample_mode (str, optional):
            Interpolation mode used in the top-down pathway.
            Common values include "nearest" and "bilinear".
            Default is "nearest".

        scale_factor (float or None, optional):
            Scale factor used for upsampling. If None, the target spatial
            size of the previous level is used instead. Default is None.

    Inputs:
        inputs (list[Tensor]):
            A list of multi-level backbone feature maps.
            Each tensor should have shape (B, C_i, H_i, W_i).

    Returns:
        tuple[Tensor]:
            A tuple of multi-scale FPN feature maps. Each output tensor has
            shape (B, out_channels, H_i, W_i) or reduced spatial size for
            extra levels.

    Example:
        >>> fpn = FPN(
        ...     in_channels=[256, 512, 1024, 2048],
        ...     out_channels=256,
        ...     num_outs=5
        ... )
        >>> inputs = [
        ...     torch.randn(1, 256, 128, 128),
        ...     torch.randn(1, 512, 64, 64),
        ...     torch.randn(1, 1024, 32, 32),
        ...     torch.randn(1, 2048, 16, 16),
        ... ]
        >>> outputs = fpn(inputs)
        >>> for out in outputs:
        ...     print(out.shape)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        add_extra_convs=False,
        relu_before_extra_convs=False,
        no_norm_on_lateral=False,
        upsample_mode="nearest",
        scale_factor=None,
    ):
        super().__init__()

        # Basic validation for input channel settings.
        assert isinstance(in_channels, list)
        assert len(in_channels) > 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level
        self.end_level = end_level
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral

        # Upsampling behavior used in the top-down fusion path.
        self.upsample_mode = upsample_mode
        self.scale_factor = scale_factor

        # Determine the last backbone level used by FPN.
        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1

        # Normalize add_extra_convs to a unified internal form.
        if add_extra_convs is True:
            self.add_extra_convs = "on_input"
        elif isinstance(add_extra_convs, str):
            self.add_extra_convs = add_extra_convs
        else:
            self.add_extra_convs = False

        # Lateral 1x1 convs and output 3x3 convs.
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        # Build lateral and FPN conv blocks for each selected backbone level.
        for i in range(self.start_level, self.backbone_end_level):
            # Lateral 1x1 convolution used to align channel dimensions.
            l_conv = ConvModule(
                in_channels=in_channels[i],
                out_channels=out_channels,
                kernel_size=1,
                bias=True,
                use_norm=not no_norm_on_lateral,
                use_act=False,
            )

            # Output 3x3 convolution used after feature fusion.
            fpn_conv = ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
                use_norm=False,
                use_act=False,
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # Number of extra pyramid levels to add beyond backbone outputs.
        extra_levels = num_outs - self.backbone_end_level + self.start_level

        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                # The first extra conv may take input from the original backbone
                # feature map if configured as "on_input".
                if i == 0 and self.add_extra_convs == "on_input":
                    extra_in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    extra_in_channels = out_channels

                extra_fpn_conv = ConvModule(
                    in_channels=extra_in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True,
                    use_norm=False,
                    use_act=False,
                )

                self.fpn_convs.append(extra_fpn_conv)

        # Initialize module parameters.
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for convolution and normalization layers.

        Initialization rules:
            - Conv2d: Xavier uniform initialization for weights,
              zeros for bias.
            - BatchNorm2d / GroupNorm: ones for weights, zeros for bias.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        """
        Forward pass of the FPN.

        Args:
            inputs (list[Tensor]):
                Multi-level feature maps from the backbone.
                The number of input tensors must match ``len(self.in_channels)``.

        Returns:
            tuple[Tensor]:
                Multi-level FPN output feature maps.
        """
        assert len(inputs) == len(self.in_channels)

        # Step 1:
        # Build lateral features using 1x1 convolutions.
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # Step 2:
        # Build the top-down pathway by upsampling higher-level features
        # and adding them to lower-level lateral features.
        used_backbone_levels = len(laterals)

        for i in range(used_backbone_levels - 1, 0, -1):
            if self.scale_factor is not None:
                upsample = F.interpolate(
                    laterals[i],
                    scale_factor=self.scale_factor,
                    mode=self.upsample_mode,
                )
            else:
                prev_shape = laterals[i - 1].shape[2:]
                upsample = F.interpolate(
                    laterals[i],
                    size=prev_shape,
                    mode=self.upsample_mode,
                )

            laterals[i - 1] = laterals[i - 1] + upsample

        # Step 3:
        # Apply 3x3 convolutions to fused features to generate final outputs.
        outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels)
        ]

        # Step 4:
        # Add extra levels if more outputs are required.
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                # Use max pooling to downsample the last output repeatedly.
                for _ in range(self.num_outs - used_backbone_levels):
                    outs.append(
                        F.max_pool2d(
                            outs[-1],
                            kernel_size=1,
                            stride=2
                        )
                    )
            else:
                # Select the source feature for the first extra conv layer.
                if self.add_extra_convs == "on_input":
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == "on_lateral":
                    extra_source = laterals[-1]
                else:
                    extra_source = outs[-1]

                # First extra output level.
                outs.append(
                    self.fpn_convs[used_backbone_levels](extra_source)
                )

                # Additional extra output levels.
                for i in range(used_backbone_levels + 1, self.num_outs):
                    x = F.relu(outs[-1]) if self.relu_before_extra_convs else outs[-1]
                    outs.append(self.fpn_convs[i](x))

        return tuple(outs)