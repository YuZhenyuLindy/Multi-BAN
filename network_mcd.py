from typing import Optional, List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn


class ClassifierBase(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim: Optional[int] = -1, head: Optional[nn.Module] = None, feature_normal = False):
        super(ClassifierBase, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.feature_normal = feature_normal
        if bottleneck is None:
            self.bottleneck = nn.Identity()
            self._features_dim = backbone.out_features
        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            self._features_dim = bottleneck_dim

        if head is None:
            if self.feature_normal:
                self.head = nn.Sequential(nn.Dropout(0.5),
                        nn.Linear(self._features_dim, 1000),
                        nn.BatchNorm1d(1000),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(1000,1000),
                        nn.BatchNorm1d(1000),
                        nn.ReLU(),
                        nn.Linear(1000, num_classes, bias=False))
                self.head_aux = nn.Sequential(nn.Dropout(0.5),
                        nn.Linear(self._features_dim, 1000),
                        nn.BatchNorm1d(1000),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(1000,1000),
                        nn.BatchNorm1d(1000),
                        nn.ReLU(),
                        nn.Linear(1000, num_classes, bias=False))
            else:
                self.head = nn.Sequential(nn.Dropout(0.5),
                        nn.Linear(self._features_dim, 1000),
                        nn.BatchNorm1d(1000),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(1000,1000),
                        nn.BatchNorm1d(1000),
                        nn.ReLU(),
                        nn.Linear(1000, num_classes))
                self.head_aux = nn.Sequential(nn.Dropout(0.5),
                        nn.Linear(self._features_dim, 1000),
                        nn.BatchNorm1d(1000),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(1000,1000),
                        nn.BatchNorm1d(1000),
                        nn.ReLU(),
                        nn.Linear(1000, num_classes))

        else:
            self.head = head
            self.head_aux = head

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        f = self.backbone(x)
        f = f.view(-1, self.backbone.out_features)
        f = self.bottleneck(f)
        if self.feature_normal:
            f = nn.functional.normalize(f)
        predictions = self.head(f)
        predictions_aux = self.head_aux(f)
        return predictions, predictions_aux, f

    def get_parameters(self) -> List[Dict]:
        params = [
            {"params": self.backbone.parameters(), "lr_mult": 0.1},
            {"params": self.bottleneck.parameters(), "lr_mult": 1.},
            {"params": self.head.parameters(), "lr_mult": 1.},
            {"params": self.head_aux.parameters(), "lr_mult": 1.},
        ]
        return params


class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256 , args = None):
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        bottleneck = None
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim,feature_normal=args.feature_normal)
