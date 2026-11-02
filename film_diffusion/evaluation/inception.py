"""
InceptionV3 Feature and Logit Extractor for FID, IS, and KID Metrics.
"""

from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Inception_V3_Weights, inception_v3


class InceptionV3FeatureExtractor(nn.Module):
    """
    Pretrained InceptionV3 network adapted for extracting:
    1. 2048-dimensional intermediate feature representations (for FID and KID).
    2. 1000-dimensional classification logits/probabilities (for Inception Score).

    Automatically handles single-channel (grayscale) to 3-channel (RGB) expansion
    and bilinear spatial interpolation to Inception's native 299x299 resolution.

    Args:
        device (str or torch.device): Device on which Inception resides.
        use_fp16 (bool): Whether to run in half-precision.
    """

    def __init__(
        self,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        use_fp16: bool = False,
    ):
        super().__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.use_fp16 = use_fp16

        # Load weights safely
        try:
            weights = Inception_V3_Weights.DEFAULT
            self.inception = inception_v3(weights=weights, transform_input=False)
        except Exception:
            self.inception = inception_v3(pretrained=True, transform_input=False)

        self.inception.eval()
        self.inception.to(self.device)

        if use_fp16 and self.device.type == "cuda":
            self.inception = self.inception.half()

        # Freeze all parameters
        for param in self.inception.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract 2048-d pool3 features and classification logits.

        Args:
            x: Image tensor [B, C, H, W] normalized to [-1, 1] or [0, 1].

        Returns:
            Tuple of (features [B, 2048], logits [B, 1000])
        """
        x = x.to(self.device)

        # Convert grayscale (1-channel) to 3-channel RGB
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Normalize to Inception's expected input range [-1, 1] if given in [0, 1]
        if x.min() >= 0.0 and x.max() <= 1.0:
            x = x * 2.0 - 1.0

        # Resize to Inception native 299x299 resolution
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

        if self.use_fp16 and self.device.type == "cuda":
            x = x.half()

        # Inception V3 forward pass
        # Stage 1: Conv blocks
        x = self.inception.Conv2d_1a_3x3(x)
        x = self.inception.Conv2d_2a_3x3(x)
        x = self.inception.Conv2d_2b_3x3(x)
        x = self.inception.maxpool1(x)

        x = self.inception.Conv2d_3b_1x1(x)
        x = self.inception.Conv2d_4a_3x3(x)
        x = self.inception.maxpool2(x)

        # Stage 2: Inception blocks A
        x = self.inception.Mixed_5b(x)
        x = self.inception.Mixed_5c(x)
        x = self.inception.Mixed_5d(x)

        # Stage 3: Inception blocks B
        x = self.inception.Mixed_6a(x)
        x = self.inception.Mixed_6b(x)
        x = self.inception.Mixed_6c(x)
        x = self.inception.Mixed_6d(x)
        x = self.inception.Mixed_6e(x)

        # Stage 4: Inception blocks C
        x = self.inception.Mixed_7a(x)
        x = self.inception.Mixed_7b(x)
        x = self.inception.Mixed_7c(x)

        # Stage 5: Pooling -> 2048-dim features
        x = self.inception.avgpool(x)
        x = torch.flatten(x, 1)
        features = x.clone()

        # Stage 6: FC -> Logits for Inception Score
        x = self.inception.dropout(x)
        logits = self.inception.fc(x)

        return features.float(), logits.float()
