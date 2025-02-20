from module import BaseModule
from torch import optim
import torch

import torch.nn.functional as F

from .pooling import *
from .feature_extractor import FCNFeatureExtractor

class LeakNetModel(BaseModule):
    """
        Multiple Instance Learning model for anomaly detection.
        
        Args:
            input_dim (int): Number of input features.
            aggregation_func (str): Aggregation function for the pooling layer.
            regularization_weight (float): Regularization weight for the regularization loss.
            
    """
    
    def __init__(self, input_dim, aggregation_func="mean", regularization_weight=0.0, lambda_coef=, **kwargs):
        super().__init__()
        
        self.feature_extractor = FCNFeatureExtractor(input_dim)
        
        self.classification_pooling = MILConjunctivePooling(
            self.feature_extractor.output_dim,
            aggregation_func=aggregation_func,
        )
        
        self.regression_pooling = MILConjunctivePooling(
            self.feature_extractor.output_dim,
            aggregation_func=aggregation_func
        )
        
        self.regularization_weight = regularization_weight

        self.save_hyperparameters(input_dim=input_dim, **self.pooling.hparams, **kwargs)

    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.feature_extractor(x)
        x = self.classification_pooling(x)
        
        # return logits, predicted distance
        return x["bag_logits"], torch.empty(1)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        bag_logits, instance_logits = self.forward(x)
        return self._compute_loss(bag_logits, instance_logits, y)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        bag_logits, instance_logits = self.forward(x)
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
        
        anomaly_label, distance_label = labels
        
        # Anomaly classification loss
        classification_loss = F.binary_cross_entropy_with_logits(bag_classification_logits.squeeze(), anomaly_label)
        
        # Anomaly regression loss
        regression_loss = F.mse_loss(bag_distance, distance_label)
        
        parameters = torch.cat([param.view(-1) for param in self.parameters()])
            
        l1_loss = self.compute_l1_loss(parameters)
        l2_loss = self.compute_l2_loss(parameters)
        
        # Elastic regularization loss
        regularization_loss = self.regularization_weight * (l1_loss + l2_loss)
        
        return classification_loss + regression_loss + regularization_loss
    
    def compute_l1_loss(self, w):
        return torch.abs(w).sum()
        
    def compute_l2_loss(self, w):
        return torch.square(w).sum()
    
    # def _compute_loss(self, pred, target):
    #     pred_label, pred_dist = pred
    #     true_label, true_dist = target

    #     pred_dist = torch.masked_select(pred_dist, true_label == 1)
    #     true_dist = torch.masked_select(true_dist, true_label == 1)
        
    #     loss_distance = F.mse_loss(pred_dist, true_dist) # calculate distance loss only for true anomalies
    #     loss_anomaly = F.binary_cross_entropy(pred_label, true_label)
        
    #     return loss_anomaly + self.lambda_distance * torch.nan_to_num(loss_distance)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                # 'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5),
                'scheduler': optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.hparams['learning_rate'],
                    steps_per_epoch=len(self.trainer.train_loader), 
                    epochs=self.trainer.max_epochs
                ),
                'interval': 'step'
            }
        }