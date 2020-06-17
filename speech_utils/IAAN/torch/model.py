import os

import torch
import torch.nn as nn
from torch import optim
import numpy as np
from tqdm import tqdm

from .ops import (
    AdditiveAttentionV1, AdditiveAttentionV2, SelfAttention,
    ScaledDotProductAttention, Mask, IAAMask, LayerNormWrapperV1,
    LabelSmoothing, Expand, ClassBalancedLoss
)


class IAAN(nn.Module):
    """
    Interaction-aware attention network as proposed in the paper:
        Yeh, S.-L., Lin, Y.-S., & Lee, C.-C. (2019). An Interaction-aware
        Attention Network for Speech Emotion Recognition in Spoken Dialogs.
        ICASSP 2019 - 2019 IEEE International Conference on Acoustics,
        Speech and Signal Processing (ICASSP), 6685–6689.
        https://doi.org/10.1109/ICASSP.2019.8683293

    """
    def __init__(self, features_dim=45, rnn_type="gru", rnn_kwargs=None,
                 attention_type="self_attention", attention_size=16,
                 iaa_size=64, num_linear_units=64, dropout_keep_prob=1.0,
                 num_classes=4):
        """Initialize the model.

        Parameters
        ----------
        features_dim : int
            Input dimension of the audio features. (default: 45)
        rnn_type : str
            One of ["rnn", "gru", "lstm"]. Type of RNN layer.
        rnn_kwargs : dict
            A dictionary of keyword arguments to be passed to the RNN-type
            layer constructor.
        attention_type : str
            One of ["self_attention", "additive_v1", "additive_v2",
            "scaled_dot"]. Type of the Attention layer.
        attention_size : int
            Attention layer's hidden size.
        iaa_size : int
            Interaction-aware attention layer's hidden size.
        num_linear_units : int
            Number of units of the second last linear layer.
        dropout_keep_prob : float
            Probability of keeping a connection in the dropout layer.
        num_classes : int
            Number of classes of the classification problem.

        """
        super(IAAN, self).__init__()

        rnn_type = rnn_type.lower()
        attention_type = attention_type.lower()
        # Cache data
        self.features_dim = features_dim
        self.rnn_type = rnn_type
        self.rnn_kwargs = rnn_kwargs
        self.attention_type = attention_type
        self.attention_size = attention_size
        self.iaa_size = iaa_size
        self.num_linear_units = num_linear_units
        self.dropout_keep_prob = dropout_keep_prob
        self.num_classes = num_classes
        # Verify validity
        if rnn_type not in ["rnn", "gru", "lstm"]:
            raise ValueError(
                "Invalid RNN layer type. Expect one of ['rnn', 'gru', 'lstm']."
                " Got {} instead.".format(rnn_type))
        if attention_type not in ["self_attention", "additive_v1",
                                  "additive_v2", "scaled_dot"]:
            raise ValueError(
                "Invalid attention layer type. Expect one of "
                "['self_attention', 'additive_v1', 'additive_v2', "
                "'scaled_dot'], got {} instead.".format(attention_type))
        # Layers

        # RNN-type layers
        rnn_mapping = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}
        # Turn off dropout when num_layers = 1
        if "num_layers" in rnn_kwargs and rnn_kwargs["num_layers"] == 1:
            rnn_kwargs["dropout"] = 0.0
        else:
            rnn_kwargs["dropout"] = 1 - self.dropout_keep_prob
        rnn_kwargs["input_size"] = features_dim
        self.rnn_size = rnn_kwargs["hidden_size"]

        self.rnn_center = rnn_mapping[rnn_type](**rnn_kwargs)
        self.rnn_target = rnn_mapping[rnn_type](**rnn_kwargs)
        self.rnn_opposite = rnn_mapping[rnn_type](**rnn_kwargs)
        self.rnn_output_size = (
            self.rnn_size * (2 if self.rnn_center.bidirectional else 1))

        # Masks
        self.center_mask = Mask(output_size=self.rnn_output_size)
        self.target_mask = Mask(output_size=self.rnn_output_size)
        self.opposite_mask = Mask(output_size=self.rnn_output_size)

        # Attention layers
        attention_mapping = {
            "self_attention": SelfAttention,
            "additive_v1": AdditiveAttentionV1,
            "additive_v2": AdditiveAttentionV2,
            "scaled_dot": ScaledDotProductAttention
        }
        if "additive" in attention_type:
            # Custom mask for additive attention
            self.target_mask_att = Mask(output_size=1)
            self.opposite_mask_att = Mask(output_size=1)
            # Center + Target
            self.att_ct = attention_mapping[attention_type](
                input_size=self.rnn_output_size, attention_size=attention_size)
            # Center + Opposite
            self.att_co = attention_mapping[attention_type](
                input_size=self.rnn_output_size, attention_size=attention_size)
        elif attention_type == "scaled_dot":
            self.att_target = attention_mapping[attention_type](
                input_size=self.rnn_output_size, attention_size=attention_size,
                dropout_keep_prob=self.dropout_keep_prob)
            self.att_opposite = attention_mapping[attention_type](
                input_size=self.rnn_output_size, attention_size=attention_size,
                dropout_keep_prob=self.dropout_keep_prob)
        else:
            self.att_target = attention_mapping[attention_type](
                input_size=self.rnn_output_size, attention_size=attention_size)
            self.att_opposite = attention_mapping[attention_type](
                input_size=self.rnn_output_size, attention_size=attention_size)
        # Batchnorm layers after attention layers
        self.batchnorm_ca = nn.BatchNorm1d(self.rnn_output_size)
        self.batchnorm_ta = nn.BatchNorm1d(self.rnn_output_size)
        self.batchnorm_oa = nn.BatchNorm1d(self.rnn_output_size)

        # Expand dimension
        self.expand_dim = Expand(axis=1)

        # Interaction-aware attention
        self.linear_c = nn.Linear(self.rnn_output_size, self.iaa_size)
        self.batchnorm_c = nn.BatchNorm1d(self.iaa_size)
        self.linear_p = nn.Linear(self.rnn_output_size, self.iaa_size)
        self.batchnorm_p = nn.BatchNorm1d(self.iaa_size)
        self.linear_r = nn.Linear(self.rnn_output_size, self.iaa_size)
        self.batchnorm_r = nn.BatchNorm1d(self.iaa_size)
        self.tanh = nn.Tanh()

        self.linear_u = nn.Linear(self.iaa_size, 1)

        self.iaa_mask = IAAMask()
        self.softmax = nn.Softmax(dim=-1)

        # Multi-layer perceptron
        self.linear_1 = nn.Linear(self.rnn_output_size * 3, num_linear_units)
        self.layer_norm = LayerNormWrapperV1(
            num_linear_units, dropout_keep_prob=dropout_keep_prob)
        self.batchnorm = nn.BatchNorm1d(num_linear_units)
        self.dropout = nn.Dropout(p=(1 - self.dropout_keep_prob))
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(num_linear_units, num_classes)

    def forward(self, center, target, opposite):
        """Forward pass logic.

        Parameters
        ----------
        center : torch.Tensor
            Center (utterance being predicted) of shape (B, T, D), where B is
            batch size, T is sequence length (along the time axis), and D is
            the features dimension.
        target : torch.Tensor
            Target (previous utterance of the current speaker) of the same
            shape as `center`.
        opposite : torch.Tensor
            Opposite (previous utterance of the current speaker) of the same
            shape as `center`.

        Returns
        -------
        torch.Tensor
            Logits.

        """
        # B: batch size, T: sequence length, H: rnn's hidden size
        # A: attention's hidden size, L: number of linear units
        # C: number of classes

        seq_len = center.shape[1]
        # Get masks
        center_mask = self.center_mask(center)  # (B, T, H)
        target_mask = self.target_mask(target)  # (B, T, H)
        opposite_mask = self.opposite_mask(opposite)  # (B, T, H)

        # Encode `center` tensor
        center_rnn = self.rnn_center(center)[0]  # (B, T, H)
        center_rnn = center_rnn * center_mask  # (B, T, H)

        # Encode `target` tensor
        target_rnn = self.rnn_target(target)[0]  # (B, T, H)
        if "additive" in self.attention_type:
            target_mask_att = self.target_mask_att(
                target).squeeze(-1)  # (B, H)
            target_att = self.att_ct(
                center_rnn, target_rnn, target_mask_att)  # (B, H)
        else:
            target_att = self.att_target(target_rnn)  # (B, H)

        target_att = self.dropout(self.batchnorm_ta(target_att))  # (B, H)
        target_att_expanded = self.expand_dim(
            target_att, size=seq_len)  # (B, T, H)
        target_att_expanded = target_att_expanded * target_mask  # (B, T, H)

        # Encode `opposite` tensor
        opposite_rnn = self.rnn_target(opposite)[0]  # (B, T, H)
        if "additive" in self.attention_type:
            opposite_mask_att = self.opposite_mask_att(
                opposite).squeeze(-1)  # (B, H)
            opposite_att = self.att_ct(
                center_rnn, opposite_rnn, opposite_mask_att)  # (B, H)
        else:
            opposite_att = self.att_target(opposite_rnn)  # (B, H)

        opposite_att = self.dropout(self.batchnorm_oa(opposite_att))  # (B, H)
        opposite_att_expanded = self.expand_dim(
            opposite_att, size=seq_len)  # (B, T, H)
        opposite_att_expanded = (
            opposite_att_expanded * opposite_mask)  # (B, T, H)

        # Interaction-aware attention
        center_linear = self.linear_c(center_rnn)  # (B, T, A)
        center_linear = self.dropout(self.batchnorm_c(
            center_linear.permute(0, 2, 1)).permute(0, 2, 1))  # (B, T, A)
        target_linear = self.linear_p(target_rnn)  # (B, T, A)
        target_linear = self.dropout(self.batchnorm_p(
            target_linear.permute(0, 2, 1)).permute(0, 2, 1))  # (B, T, A)
        opposite_linear = self.linear_r(opposite_rnn)  # (B, T, A)
        opposite_linear = self.dropout(self.batchnorm_r(
            opposite_linear.permute(0, 2, 1)).permute(0, 2, 1))  # (B, T, A)
        v = self.tanh(
            center_linear + target_linear + opposite_linear)  # (B, T, A)
        vu = self.linear_u(v).squeeze(-1)  # (B, T)

        vu_masked = self.iaa_mask(x=vu, y=center)  # (B, T)
        alphas = self.softmax(vu_masked)  # (B, T)
        center_att = (center_rnn * alphas.unsqueeze(-1)).sum(1)  # (B, H)
        center_att = self.batchnorm_ca(center_att)

        # Multi-layer perceptron
        att_concat = torch.cat(
            [center_att, target_att, opposite_att], dim=1)  # (B, H * 3)
        linear1 = self.linear_1(att_concat)  # (B, L)
        linear1 = self.dropout(self.relu(self.layer_norm(linear1)))  # (B, L)
        logits = self.linear_2(linear1)  # (B, C)

        return logits


class IAANTrainer:
    def __init__(self, save_dir, lr=0.001, weight_decay=1e-3, loss_type="ce",
                 cbl_kwargs=None, samples_per_cls=None, device="cuda",
                 features_dim=45, rnn_type="gru", rnn_kwargs=None,
                 attention_type="self_attention", attention_size=16,
                 iaa_size=64, num_linear_units=64, dropout_keep_prob=1.0,
                 num_classes=4):
        """Initialize the trainer.

        Parameters
        ----------
        save_dir : str
            Directory where the checkpoints will be saved.
        lr : float
            Learning rate.
        weight_decay : float
            Weight decay (L2 penalty)/
        loss_type : str
            One of ["ce", "sigmoid", "softmax", "focal"]. Type of loss.
                If "ce", use cross-entropy loss.
                Otherwise, use class-balanced loss with the corresponding loss.
                See `ClassBalancedLoss` for more details.
        cbl_kwargs : str
            A dictionary of keyword arguments to be passed to the
            class-balanced loss constructor. If `loss_type=None`, this option
            will be ignored.
        samples_per_cls : list
            List of integers representing the number of samples for each class.
        device : str
            Torch device in string format.

        features_dim : int
            Input dimension of the audio features. (default: 45)
        rnn_type : str
            One of ["rnn", "gru", "lstm"]. Type of RNN layer.
        rnn_kwargs : dict
            A dictionary of keyword arguments to be passed to the RNN-type
            layer constructor.
        attention_type : str
            One of ["self_attention", "additive_v1", "additive_v2",
            "scaled_dot"]. Type of the attention layer.
        attention_size : int
            Attention layer's hidden size.
        iaa_size : int
            Interaction-aware attention layer's hidden size.
        num_linear_units : int
            Number of units of the second last linear layer.
        dropout_keep_prob : float
            Probability of keeping a connection in the dropout layer.
        num_classes : int
            Number of classes of the classification problem.

        """
        # Verify validity
        if loss_type not in ["ce", "sigmoid", "softmax", "focal"]:
            raise RuntimeError(
                "Invalid loss type. Expect one of ['ce', 'sigmoid', 'softmax',"
                " 'focal'], got {} instead.".format(loss_type))
        if samples_per_cls is None and loss_type != "ce":
            raise RuntimeError("Class balaned loss is used but no number of "
                               "samples per class found.")
        # Initialize model
        self.device = torch.device(device)
        self.model = IAAN(
            features_dim=features_dim, rnn_type=rnn_type,
            rnn_kwargs=rnn_kwargs, attention_type=attention_type,
            attention_size=attention_size, iaa_size=iaa_size,
            num_linear_units=num_linear_units,
            dropout_keep_prob=dropout_keep_prob, num_classes=num_classes)
        self.model = self.model.to(device=self.device)

        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # Optimizer
        trainable_params = [param for param in self.model.parameters()
                            if param.requires_grad]
        self.optimizer = optim.Adam(
            trainable_params, lr=lr, weight_decay=weight_decay)
        # Loss
        if loss_type == "ce":
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = ClassBalancedLoss(
                samples_per_cls, loss_type=loss_type, **cbl_kwargs)

    def train_one_epoch(self, dataloader, progress=True):
        self.model.train()
        tot_loss = 0.0
        tot_preds = list()
        tot_labels = list()
        t = tqdm if progress else lambda x: x

        for center, target, opposite, labels in t(dataloader):
            # Send to correct device
            center = torch.from_numpy(center).to(
                dtype=torch.float32, device=self.device)
            target = torch.from_numpy(target).to(
                dtype=torch.float32, device=self.device)
            opposite = torch.from_numpy(opposite).to(
                dtype=torch.float32, device=self.device)
            labels = torch.from_numpy(labels).to(
                dtype=torch.int64, device=self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            logits = self.model(center, target, opposite)
            _, preds = torch.max(logits, 1)
            loss = self.criterion(logits, labels)

            # Update model
            loss.backward()
            self.optimizer.step()

            # Update info
            tot_loss += loss.item() * len(center)
            tot_preds.append(preds.cpu().numpy())
            tot_labels.append(labels.cpu().numpy())

        tot_loss = tot_loss / dataloader.num_samples
        tot_preds = np.concatenate(tot_preds)
        tot_labels = np.concatenate(tot_labels)
        return tot_loss, tot_preds, tot_labels

    def test_one_epoch(self, dataloader, progress=False):
        self.model.eval()
        tot_loss = 0.0
        tot_preds = list()
        tot_labels = list()
        t = tqdm if progress else lambda x: x

        with torch.no_grad():
            for center, target, opposite, labels in t(dataloader):
                # Send to correct device
                center = torch.from_numpy(center).to(
                    dtype=torch.float32, device=self.device)
                target = torch.from_numpy(target).to(
                    dtype=torch.float32, device=self.device)
                opposite = torch.from_numpy(opposite).to(
                    dtype=torch.float32, device=self.device)
                labels = torch.from_numpy(labels).to(
                    dtype=torch.int64, device=self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                logits = self.model(center, target, opposite)
                _, preds = torch.max(logits, 1)
                loss = self.criterion(logits, labels)

                # Update info
                tot_loss += loss.item() * len(center)
                tot_preds.append(preds.cpu().numpy())
                tot_labels.append(labels.cpu().numpy())

            tot_loss = tot_loss / dataloader.num_samples

        tot_preds = np.concatenate(tot_preds)
        tot_labels = np.concatenate(tot_labels)

        return tot_loss, tot_preds, tot_labels

    def save(self, epoch_idx):
        save_path = os.path.join(
            self.save_dir, "iaan_{}.pth".format(epoch_idx))
        torch.save(self.model.state_dict(), save_path)

    def load(self, epoch_idx):
        save_path = os.path.join(
            self.save_dir, "iaan_{}.pth".format(epoch_idx))
        self.model.load_state_dict(torch.load(save_path))
