# Copyright (c) 2026 Vinay Umrethe
# This project is licensed under the MIT License - see the LICENSE file for details.

import numpy as np
import torch
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.utils import logging

logger = logging.get_logger(__name__)


class SigMambaImageProcessor(BaseImageProcessor):
    """
    Handles temporal sampling of feature sequences to a fixed size.
    Expects pre-extracted feature tensors as input.
    """

    model_input_names = ["features"]

    def __init__(self, seg_num=32, **kwargs):
        super().__init__(**kwargs)
        self.seg_num = seg_num

    def _sample_features(self, feat, length):
        """Uniformly samples the feature sequence to satisfy the target segments."""
        if isinstance(feat, torch.Tensor):
            feat = feat.numpy()
        if isinstance(feat, list):
            feat = np.array(feat)

        new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)
        r = np.linspace(0, len(feat), length + 1, dtype=np.int32)

        for i in range(length):
            if r[i] != r[i + 1]:
                new_feat[i, :] = np.mean(feat[r[i] : r[i + 1], :], 0)
            else:
                new_feat[i, :] = feat[r[i], :]

        return torch.from_numpy(new_feat)

    def preprocess(self, features, **kwargs):
        """
        Prepares feature sequences for model input.

        Returns:
            BatchFeature: Processed features shifted to PyTorch tensors.
        """
        if not isinstance(features, list):
            # If single item passed, wrap in list.
            features = [features]

        batch_output = []
        for feat in features:
            processed = self._sample_features(feat, self.seg_num)
            batch_output.append(processed)

        # Stack into batch.
        batch_tensor = torch.stack(batch_output)

        return BatchFeature(data={"features": batch_tensor}, tensor_type="pt")

    def __call__(self, features, **kwargs):
        return self.preprocess(features, **kwargs)
