import json
import os

import torch
import torch.nn as nn
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import unwrap_model


class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer from Ganin & Lempitsky (2015)."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class DomainDiscriminator(nn.Module):
    """Lightweight discriminator on pooled CNN features."""

    def __init__(self, in_channels, hidden_dim=256, dropout=0.5):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        x = self.pool(x)
        return self.net(x).squeeze(-1)


class DomainAdaptiveDetectionModel(DetectionModel):
    """YOLO detection model with an auxiliary domain-adversarial head."""

    def __init__(self, cfg="yolo11n.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch, nc, verbose)
        self.domain_discriminator = None
        self.domain_map = {}
        self.feature_layer = 10
        self._domain_features = None
        self._domain_alignment_lambda = 0.0
        self._domain_map_path = None
        self._hook_handle = None
        self._input_channels = ch

    def _domain_hook(self, module, input, output):
        """Forward hook that caches the chosen layer's feature maps."""
        self._domain_features = output

    def setup_domain_alignment(self, args):
        """Configure domain alignment using trainer args."""
        self._domain_alignment_lambda = getattr(args, "domain_alignment_lambda", 0.0)
        self._domain_map_path = getattr(args, "domain_map_path", None)
        self.feature_layer = getattr(args, "domain_feature_layer", 10)
        self._domain_features = None

        # Remove old hook if layer changed.
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

        if 0 <= self.feature_layer < len(self.model):
            self._hook_handle = self.model[self.feature_layer].register_forward_hook(
                self._domain_hook
            )

        if self._domain_map_path and os.path.exists(self._domain_map_path):
            with open(self._domain_map_path, "r", encoding="utf-8") as f:
                self.domain_map = json.load(f)
        else:
            self.domain_map = {}

        if self._domain_alignment_lambda > 0.0 and self.domain_discriminator is None:
            self._build_discriminator()

    def _build_discriminator(self):
        """Infer feature channels and instantiate the domain head."""
        ch = self.yaml.get("channels", self._input_channels)
        device = next(self.model.parameters()).device
        with torch.no_grad():
            x = torch.zeros(1, ch, 64, 64, device=device)
            end = min(self.feature_layer + 1, len(self.model))
            for i in range(end):
                x = self.model[i](x)
            in_channels = x.shape[1]
        self.domain_discriminator = DomainDiscriminator(in_channels).to(device)

    def get_domain_labels(self, im_files):
        """Return 0/1 domain labels from a filename-to-domain mapping."""
        if not self.domain_map or not im_files:
            return None
        labels = []
        for f in im_files:
            base = os.path.basename(f)
            labels.append(float(self.domain_map.get(base, 0.0)))
        if not labels:
            return None
        return torch.tensor(labels, dtype=torch.float32, device=self._domain_features.device)

    def loss(self, batch, preds=None):
        """Compute detection loss plus adversarial domain alignment loss."""
        if not hasattr(self, "criterion") or self.criterion is None:
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"])

        det_loss, loss_items = self.criterion(preds, batch)
        if not self.training or self.domain_discriminator is None or self._domain_alignment_lambda <= 0.0:
            return det_loss, loss_items

        features = self._domain_features
        if features is None:
            return det_loss, loss_items

        domain_labels = self.get_domain_labels(batch.get("im_file", []))
        if domain_labels is None or domain_labels.numel() != features.shape[0]:
            return det_loss, loss_items

        domain_preds = self.domain_discriminator(features)
        domain_loss = nn.functional.binary_cross_entropy_with_logits(domain_preds, domain_labels)
        total_loss = det_loss + self._domain_alignment_lambda * domain_loss
        return total_loss, loss_items


class DomainAdaptiveDetectionTrainer(DetectionTrainer):
    """DetectionTrainer that builds the domain-adaptive model."""

    _CUSTOM_ARGS = {"domain_alignment_lambda", "domain_map_path", "domain_feature_layer"}

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return the domain-adaptive detection model."""
        model = DomainAdaptiveDetectionModel(
            cfg or self.args.model,
            ch=self.data["channels"],
            nc=self.data["nc"],
            verbose=verbose and RANK in {-1, 0},
        )
        if weights:
            model.load(weights)
        return model

    def set_model_attributes(self):
        super().set_model_attributes()
        model = unwrap_model(self.model)
        if hasattr(model, "setup_domain_alignment"):
            model.setup_domain_alignment(self.args)

    def get_validator(self):
        """Return a validator, stripping custom domain args from the config copy."""
        from copy import copy
        from ultralytics.models import yolo

        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        clean_args = copy(self.args)
        for key in self._CUSTOM_ARGS:
            if hasattr(clean_args, key):
                delattr(clean_args, key)
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=clean_args, _callbacks=self.callbacks
        )
