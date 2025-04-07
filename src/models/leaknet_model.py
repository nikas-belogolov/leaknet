from module import BaseModule
from torch import optim
import torch

import torch.nn.functional as F

from .pooling import *
from .feature_extractor import FCNFeatureExtractor

from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np
import time

class LeakNetModel(BaseModule):
    """
        Multiple Instance Learning model for anomaly detection.
        
        Args:
            input_dim (int): Number of input features.
            aggregation_func (str): Aggregation function for the pooling layer.
            regularization_weight (float): Regularization weight for the regularization loss.
            
    """
    
    def __init__(
        self,
        input_dim,
        aggregation_func="mean",
        regularization_weight=0.001,
        dropout=0.1,
        d_attn=8,
        apply_positional_encoding=True,
        **kwargs
    ):
        super().__init__()
        
        self.feature_extractor = FCNFeatureExtractor(input_dim)
        
        self.classification_pooling = MILConjunctivePooling(
            self.feature_extractor.output_dim,
            aggregation_func=aggregation_func,
            dropout=dropout,
            apply_positional_encoding=apply_positional_encoding,
            d_attn=d_attn
        )
        
        # self.regression_pooling = MILConjunctivePooling(
        #     self.feature_extractor.output_dim,
        #     aggregation_func=aggregation_func
        # )
        
        self.regularization_weight = regularization_weight
        
        self.save_hyperparameters(
            input_dim=input_dim,
            regularization_weight=regularization_weight,
            d_attn=d_attn,
            dropout=dropout,
            aggregation_func=aggregation_func,
            apply_positional_encoding=apply_positional_encoding,
            **kwargs
        )

    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.feature_extractor(x)
        x = self.classification_pooling(x)
        return x["bag_logits"], torch.empty(1)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        bag_logits, instance_logits = self.forward(x)
        return self._compute_loss(bag_logits, instance_logits, y)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        bag_logits, instance_logits = self.forward(x)
        
        mFPR30 = self._calc_mfpr(x, y, window_size=30)
        mFPR45 = self._calc_mfpr(x, y, window_size=45)
        ROC_AUC_SCORE = roc_auc_score(y, bag_logits)
        
        self.log("mFPR@[0.5:0.95, 30]", mFPR30, on_step=True, on_epoch=True, prog_bar=True)
        self.log("mFPR@[0.5:0.95, 45]", mFPR45, on_step=True, on_epoch=True, prog_bar=True)
        self.log('roc_auc', ROC_AUC_SCORE, on_step=True, on_epoch=True, prog_bar=True)

        return self._compute_loss(bag_logits, instance_logits, y)
    
    
    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     bag_logits, instance_logits = self.forward(x)
    #     return self._compute_loss(bag_logits, instance_logits, y)
    
    @torch.jit.export
    def predict_step(self, batch):
        bag_logits, instance_logits = self.forward(batch)
        return torch.sigmoid(bag_logits)
    
    def _compute_loss(self, bag_classification_logits, bag_distance, labels):
        
        anomaly_label = labels
        
        # Anomaly classification loss
        classification_loss = F.binary_cross_entropy_with_logits(bag_classification_logits.squeeze(), anomaly_label)
        
        # Anomaly regression loss
        # regression_loss = F.mse_loss(bag_distance, distance_label)
        
        weights = torch.cat([param.view(-1) for param in self.parameters()])
            
        l1_loss = torch.abs(weights).sum()
        l2_loss = torch.square(weights).sum()
        
        # Elastic regularization loss
        regularization_loss = self.regularization_weight * (l1_loss + l2_loss)
        
        return classification_loss + regularization_loss
        # return classification_loss + regression_loss + regularization_loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        
        # scheduler = optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=self.hparams["learning_rate"],
        #     steps_per_epoch=len(self._trainer.train_loader),
        #     epochs=self._trainer.max_epochs
        # )
        
        # optimizer.param_groups[-1]['lr'] = scheduler.get_last_lr()[0]
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
        
    def _calc_mfpr(self, x, y, window_size):
        num_features = x.size(2)

        x = x.unfold(1, window_size, 1)
        y = y.repeat_interleave(x.size(1), dim=0)
        x = x.transpose(-1, -2)
        x = x.reshape(-1, window_size, num_features)
        y_logits, _ = self.forward(x)
        
        
        fprs = []
        
        # calculate FPR for different thresholds
        for i in np.linspace(0.5, 0.95, 10):
            y_pred = torch.sigmoid(y_logits) > i
            tn, fp, _, _ = confusion_matrix(y, y_pred).ravel()
            fpr = fp / (fp + tn)
            fprs.append(fpr)
        
        return np.mean(fprs)
        