import lightning as L
from .feature_extractor import FCNFeatureExtractor
from .pooling import MILConjunctivePooling
import torch
import torch.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score
from .visualize import plot_window_predictions
import numpy as np

class LeakNet(L.LightningModule):
    def __init__(
        self,
        input_dim,
        aggregation_func,
        regularization_weight,
        dropout,
        d_attn,
        apply_positional_encoding,
        learning_rate,
        test_threshold=0.9,
        test_window_size=30
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
        
        self.regularization_weight = regularization_weight
        
        self.save_hyperparameters(ignore=["test_threshold", "test_window_size"])
        
        self.test_window_size = test_window_size
        self.test_window_stride = 1
        self.test_threshold = test_threshold
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.feature_extractor(x)
        x = self.classification_pooling(x)
        return x["bag_logits"], x["instance_logits"]
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        bag_logits, instance_logits = self.forward(x)
        loss = self._compute_loss(bag_logits, instance_logits, y)
        self.log("loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        bag_logits, instance_logits = self.forward(x)
        
        loss = self._compute_loss(bag_logits, instance_logits, y)
        
        mfpr_30 = self._compute_mfpr(x, y, window_size=30)
        mfpr_45 = self._compute_mfpr(x, y, window_size=45)
        
        self.log_dict({
            "val_loss": loss,
            "mFPR@[0.5:0.95, 30]": mfpr_30,
            "mFPR@[0.5:0.95, 45]": mfpr_45,
            "roc_auc": roc_auc_score(y, bag_logits)
        }, prog_bar=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch):
        x, y_true = batch
        x = x.squeeze()
        windows = x.unfold(0, self.test_window_size, 1).transpose(1, 2)
        y_pred = self.predict_step(windows)
        plot_window_predictions(x, y_pred, y_true, window_size=self.test_window_size, stride=self.test_window_stride, threshold=self.test_threshold)
        
    @torch.jit.export
    def predict_step(self, batch):
        out, _ = self.forward(batch)
        return torch.sigmoid(out)
    
    def _compute_loss(self, bag_logits, instance_logits, labels: torch.Tensor):
        classification_loss = F.binary_cross_entropy_with_logits(bag_logits.squeeze(), labels)
        
        # Instance False Positive Loss
        mask = labels == 0
        instance_labels = labels.unsqueeze(1).repeat(1, instance_logits.size(-1))
        fp_loss = mask * F.binary_cross_entropy_with_logits(instance_logits.squeeze(), instance_labels)
        fp_loss = fp_loss.mean()
        
        # Elastic regularization loss
        weights = torch.cat([param.view(-1) for param in self.parameters()])
        l1_loss = torch.abs(weights).sum()
        l2_loss = torch.square(weights).sum()
        regularization_loss = self.regularization_weight * (l1_loss + l2_loss)
        
        return classification_loss + regularization_loss + fp_loss
    
    def _compute_mfpr(self, x, y, window_size):
        num_features = x.size(2)

        x = x.unfold(1, window_size, 1)
        y = y.repeat_interleave(x.size(1), dim=0)
        x = x.transpose(-1, -2)
        x = x.reshape(-1, window_size, num_features)
        y_logits, _ = self.forward(x)
        
        fprs = []
        
        for i in np.linspace(0.5, 0.95, 10):
            y_pred = torch.sigmoid(y_logits) > i
            tn, fp, _, _ = confusion_matrix(y, y_pred).ravel()
            fpr = fp / (fp + tn)
            fprs.append(fpr)
        
        return np.mean(fprs)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                "monitor": "val_loss"
            }
        }