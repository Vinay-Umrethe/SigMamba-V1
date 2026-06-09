# Copyright (c) 2026 Vinay Umrethe
# This project is licensed under the MIT License - see the LICENSE file for details.

from transformers.configuration_utils import PretrainedConfig


class SigMambaConfig(PretrainedConfig):
    """
    Configuration setting for the SigMamba (SigLIP + Mamba) anomaly detection model.
    """

    model_type = "sigmamba"

    def __init__(
        self,
        feature_dim=1024,
        d_model=768,
        depth=8,
        seg_num=32,
        d_state=16,
        d_conv=4,
        expand=2,
        num_classes=1,
        vision_model_id="google/siglip2-large-patch16-384",
        **kwargs,
    ):
        """
        Args:
            feature_dim (int): Input feature dimension (e.g., 1024 for SigLIP).
            d_model (int): Internal Mamba dimension.
            depth (int): Number of Mamba layers.
            seg_num (int): Number of temporal segments to sample during training.
            d_state (int): SSM state dimension.
            d_conv (int): Local convolution width.
            expand (int): Block expansion factor.
            num_classes (int): Output dimension (1 for Anomaly Score).
            vision_model_id (str): Hugging Face ID for the vision encoder (SigLIP).
        """
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.depth = depth
        self.seg_num = seg_num
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.num_classes = num_classes
        self.vision_model_id = vision_model_id
        super().__init__(**kwargs)
