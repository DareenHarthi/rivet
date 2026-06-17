# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

""" This ECAPA-TDNN implementation is adapted from https://github.com/speechbrain/speechbrain.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def ce_loss(logits, targets, reduction='none'):
    """
    cross entropy loss in pytorch.
    Args:
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        # use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
        reduction: the reduction argument
    """
    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)


class CELoss(nn.Module):
    """
    Wrapper for ce loss
    """
    def forward(self, logits, targets, reduction='none'):
        return ce_loss(logits, targets, reduction)
    
class NoiseMatrixLayer(torch.nn.Module):
    """Noise matrix layer for modeling label noise transition probabilities"""
    def __init__(self, num_classes, scale=1.0):
        super().__init__()
        self.num_classes = num_classes

        self.noise_layer = nn.Linear(self.num_classes, self.num_classes, bias=False)
        # initialization to identity matrix
        self.noise_layer.weight.data.copy_(torch.eye(self.num_classes))

        self.eye = None  # Will be initialized on first forward pass
        self.scale = scale

    def forward(self, x):
        if self.eye is None or self.eye.device != x.device:
            self.eye = torch.eye(self.num_classes, device=x.device)

        noise_matrix = self.noise_layer(self.eye)
        # Normalize to get valid probability transition matrix
        noise_matrix = F.normalize(noise_matrix, dim=0)
        noise_matrix = F.normalize(noise_matrix, dim=1)
        return noise_matrix * self.scale

def length_to_mask(length, max_len=None, dtype=None, device=None):
    assert len(length.shape) == 1

    if max_len is None:
        max_len = length.max().long().item()
    mask = torch.arange(
        max_len, device=length.device, dtype=length.dtype).expand(
            len(length), max_len) < length.unsqueeze(1)

    if dtype is None:
        dtype = length.dtype

    if device is None:
        device = length.device

    mask = torch.as_tensor(mask, dtype=dtype, device=device)
    return mask

def get_padding_elem(L_in: int, stride: int, kernel_size: int, dilation: int):
    if stride > 1:
        n_steps = math.ceil(((L_in - kernel_size * dilation) / stride) + 1)
        L_out = stride * (n_steps - 1) + kernel_size * dilation
        padding = [kernel_size // 2, kernel_size // 2]

    else:
        L_out = (L_in - dilation * (kernel_size - 1) - 1) // stride + 1

        padding = [(L_in - L_out) // 2, (L_in - L_out) // 2]
    return padding


class Conv1d(nn.Module):

    def __init__(
        self,
        out_channels,
        kernel_size,
        in_channels,
        stride=1,
        dilation=1,
        padding='same',
        groups=1,
        bias=True,
        padding_mode='reflect',
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.padding_mode = padding_mode

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            padding=0,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        if self.padding == 'same':
            x = self._manage_padding(x, self.kernel_size, self.dilation,
                                     self.stride)

        elif self.padding == 'causal':
            num_pad = (self.kernel_size - 1) * self.dilation
            x = F.pad(x, (num_pad, 0))

        elif self.padding == 'valid':
            pass

        else:
            raise ValueError(
                "Padding must be 'same', 'valid' or 'causal'. Got "
                + self.padding)

        wx = self.conv(x)

        return wx

    def _manage_padding(
        self,
        x,
        kernel_size: int,
        dilation: int,
        stride: int,
    ):
        L_in = x.shape[-1]
        padding = get_padding_elem(L_in, stride, kernel_size, dilation)
        x = F.pad(x, padding, mode=self.padding_mode)

        return x


class BatchNorm1d(nn.Module):
    def __init__(
        self,
        input_size,
        eps=1e-05,
        momentum=0.1,
    ):
        super().__init__()
        self.norm = nn.BatchNorm1d(
            input_size,
            eps=eps,
            momentum=momentum,
        )

    def forward(self, x):
        return self.norm(x)


class TDNNBlock(nn.Module):
    """An implementation of TDNN.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        activation=nn.ReLU,
        groups=1,
    ):
        super(TDNNBlock, self).__init__()
        self.conv = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
        )
        self.activation = activation()
        self.norm = BatchNorm1d(input_size=out_channels)

    def forward(self, x):
        return self.norm(self.activation(self.conv(x)))


class Res2NetBlock(torch.nn.Module):
    """An implementation of Res2NetBlock w/ dilation.
    """
    def __init__(
        self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1
    ):
        super(Res2NetBlock, self).__init__()
        assert in_channels % scale == 0
        assert out_channels % scale == 0

        in_channel = in_channels // scale
        hidden_channel = out_channels // scale

        self.blocks = nn.ModuleList(
            [
                TDNNBlock(
                    in_channel,
                    hidden_channel,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
                for i in range(scale - 1)
            ]
        )
        self.scale = scale

    def forward(self, x):
        y = []
        for i, x_i in enumerate(torch.chunk(x, self.scale, dim=1)):
            if i == 0:
                y_i = x_i
            elif i == 1:
                y_i = self.blocks[i - 1](x_i)
            else:
                y_i = self.blocks[i - 1](x_i + y_i)
            y.append(y_i)
        y = torch.cat(y, dim=1)
        return y


class SEBlock(nn.Module):
    """An implementation of squeeze-and-excitation block.
    """
    def __init__(self, in_channels, se_channels, out_channels):
        super(SEBlock, self).__init__()

        self.conv1 = Conv1d(
            in_channels=in_channels, out_channels=se_channels, kernel_size=1
        )
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = Conv1d(
            in_channels=se_channels, out_channels=out_channels, kernel_size=1
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, lengths=None):
        L = x.shape[-1]
        if lengths is not None:
            mask = length_to_mask(lengths * L, max_len=L, device=x.device)
            mask = mask.unsqueeze(1)
            total = mask.sum(dim=2, keepdim=True)
            s = (x * mask).sum(dim=2, keepdim=True) / total
        else:
            s = x.mean(dim=2, keepdim=True)

        s = self.relu(self.conv1(s))
        s = self.sigmoid(self.conv2(s))

        return s * x


class AttentiveStatisticsPooling(nn.Module):
    """This class implements an attentive statistic pooling layer for each channel.
    It returns the concatenated mean and std of the input tensor.
    """
    def __init__(self, channels, attention_channels=128, global_context=True):
        super().__init__()

        self.eps = 1e-12
        self.global_context = global_context
        if global_context:
            self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1)
        else:
            self.tdnn = TDNNBlock(channels, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = Conv1d(
            in_channels=attention_channels, out_channels=channels, kernel_size=1
        )

    def forward(self, x, lengths=None):
        """Calculates mean and std for a batch (input tensor).
        """
        L = x.shape[-1]

        def _compute_statistics(x, m, dim=2, eps=self.eps):
            mean = (m * x).sum(dim)
            std = torch.sqrt(
                (m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps)
            )
            return mean, std

        if lengths is None:
            lengths = torch.ones(x.shape[0], device=x.device)

        # Make binary mask of shape [N, 1, L]
        mask = length_to_mask(lengths * L, max_len=L, device=x.device)
        mask = mask.unsqueeze(1)

        # Expand the temporal context of the pooling layer by allowing the
        # self-attention to look at global properties of the utterance.
        if self.global_context:
            # torch.std is unstable for backward computation
            # https://github.com/pytorch/pytorch/issues/4320
            total = mask.sum(dim=2, keepdim=True).float()
            mean, std = _compute_statistics(x, mask / total)
            mean = mean.unsqueeze(2).repeat(1, 1, L)
            std = std.unsqueeze(2).repeat(1, 1, L)
            attn = torch.cat([x, mean, std], dim=1)
        else:
            attn = x

        # Apply layers
        attn = self.conv(self.tanh(self.tdnn(attn)))

        # Filter out zero-paddings
        attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=2)
        mean, std = _compute_statistics(x, attn)
        # Append mean and std of the batch
        pooled_stats = torch.cat((mean, std), dim=1)
        pooled_stats = pooled_stats.unsqueeze(2)

        return pooled_stats


class SERes2NetBlock(nn.Module):
    """An implementation of building block in ECAPA-TDNN, i.e.,
    TDNN-Res2Net-TDNN-SEBlock.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        res2net_scale=8,
        se_channels=128,
        kernel_size=1,
        dilation=1,
        activation=torch.nn.ReLU,
        groups=1,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TDNNBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            activation=activation,
            groups=groups,
        )
        self.res2net_block = Res2NetBlock(
            out_channels, out_channels, res2net_scale, kernel_size, dilation
        )
        self.tdnn2 = TDNNBlock(
            out_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            activation=activation,
            groups=groups,
        )
        self.se_block = SEBlock(out_channels, se_channels, out_channels)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(self, x, lengths=None):
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)

        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x, lengths)

        return x + residual

class ClassificationHead(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.net(x)
    
class ECAPA_TDNN(torch.nn.Module):
    """An implementation of the speaker embedding model in a paper.
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
    TDNN Based Speaker Verification" (https://arxiv.org/abs/2005.07143).
    """

    def __init__(
        self,
        input_size,
        device="cuda",
        lin_neurons=256,
        activation=torch.nn.ReLU,
        channels=[512, 512, 512, 512, 1536],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        attention_channels=128,
        res2net_scale=8,
        se_channels=128,
        global_context=True,
        groups=[1, 1, 1, 1, 1],
        age_bins=8
    ):

        super().__init__()
        assert len(channels) == len(kernel_sizes)
        assert len(channels) == len(dilations)
        self.channels = channels
        self.blocks = nn.ModuleList()

        noise_matrix_scale = 1.0
        self.age_noise_model = NoiseMatrixLayer(age_bins, scale=noise_matrix_scale)
        self.gender_noise_model = NoiseMatrixLayer(2, scale=noise_matrix_scale)

        # The initial TDNN layer
        self.blocks.append(
            TDNNBlock(
                input_size,
                channels[0],
                kernel_sizes[0],
                dilations[0],
                activation,
                groups[0],
            )
        )

        # SE-Res2Net layers
        for i in range(1, len(channels) - 1):
            self.blocks.append(
                SERes2NetBlock(
                    channels[i - 1],
                    channels[i],
                    res2net_scale=res2net_scale,
                    se_channels=se_channels,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    activation=activation,
                    groups=groups[i],
                )
            )

        # Multi-layer feature aggregation
        self.mfa = TDNNBlock(
            channels[-1],
            channels[-1],
            kernel_sizes[-1],
            dilations[-1],
            activation,
            groups=groups[-1],
        )

        # Attentive Statistical Pooling
        self.asp = AttentiveStatisticsPooling(
            channels[-1],
            attention_channels=attention_channels,
            global_context=global_context,
        )
        self.asp_bn = BatchNorm1d(input_size=channels[-1] * 2)

        # Final linear transformation
        self.fc_sid = Conv1d(
            in_channels=channels[-1] * 2,
            out_channels=512,
            kernel_size=1,
        )
        
        
        self.age_head    = ClassificationHead(512, age_bins)
        self.gender_head = ClassificationHead(512, 2)
        self.ce_loss = CELoss() 
        

    def forward(self, x, lengths=None):
        """Returns the embedding vector.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape (batch, time, channel).
        """
        # Minimize transpose for efficiency
        x = x.transpose(1, 2)

        xl = []
        for layer in self.blocks:
            try:
                x = layer(x, lengths=lengths)
            except TypeError:
                x = layer(x)
            xl.append(x)

        # Multi-layer feature aggregation
        x = torch.cat(xl[1:], dim=1)
        x = self.mfa(x)

        # Attentive Statistical Pooling
        x = self.asp(x, lengths=lengths)
        x = self.asp_bn(x)

        # Final linear transformation
        sid_emb = self.fc_sid(x)

        sid = sid_emb.transpose(1, 2)
        sid = sid.squeeze(1)
        
         
        # classification
        # age
        
        age_pred = self.age_head(sid)
        gender_pred = self.gender_head(sid)


        return sid, age_pred, gender_pred

    def _compute_accuracy(self, logits, labels):
        """Helper function to compute classification accuracy"""
        pred = torch.argmax(logits, dim=1)
        correct = (pred == labels).float().sum()
        accuracy = correct / labels.size(0) * 100.0
        return accuracy
    
    def get_embed(self, x, lengths=None):

        x = x.transpose(1, 2)

        xl = []
        for layer in self.blocks:
            try:
                x = layer(x, lengths=lengths)
            except TypeError:
                x = layer(x)
            xl.append(x)

        # Multi-layer feature aggregation
        x = torch.cat(xl[1:], dim=1)
        x = self.mfa(x)

        # Attentive Statistical Pooling
        x = self.asp(x, lengths=lengths)
        x = self.asp_bn(x)

        # Final linear transformation
        sid = self.fc_sid(x)

        sid = sid.transpose(1, 2)
        sid = sid.squeeze(1)
        

        return sid 
      

    def train_step(self, embeddings, age_labels, gender_labels, age_loss_fn, gender_loss_fn):
        """
        Normal training step without noisy label handling.
        Treats labels as clean ground truth.

        Args:
            embeddings: speaker embeddings from forward pass [B, embed_dim]
            age_labels: age class labels [B]
            gender_labels: gender class labels [B]
            age_loss_fn: age loss function (e.g., AgeLoss)
            gender_loss_fn: gender loss function (e.g., GenderLoss)

        Returns:
            loss_age: age classification loss
            loss_gender: gender classification loss
            age_acc: age classification accuracy
            gender_acc: gender classification accuracy
        """
        # Standard cross-entropy training
        loss_age, age_acc = age_loss_fn(embeddings, age_labels)
        loss_gender, gender_acc = gender_loss_fn(embeddings, gender_labels)

        return loss_age, loss_gender, age_acc, gender_acc

    def train_step_noisy(
        self,
        logits_x_w, logits_x_s,
       y,
        average_entropy_loss=True,
        label_type="age",
        num_classes=8
    ):
        if label_type=="age":
            noise_matrix = self.age_noise_model(logits_x_w)
        elif label_type=="gender":
            noise_matrix = self.gender_noise_model(logits_x_w)
        else:
            raise ValueError(f"Unknown label_type: {label_type!r} (expected 'age' or 'gender')")


        # noise_matrix *= 2
        
        # convert logits_w to probs
        probs_x_w = logits_x_w.softmax(dim=-1).detach()
             
        # convert logits_s to probs
        probs_x_s = logits_x_s.softmax(dim=-1)
        
        # compute forward-backward on graph x_w
        with torch.no_grad():
            # model p(y_hat | y, x) p(y|x)
            noise_matrix_col = noise_matrix.softmax(dim=-1)[:, y].detach().transpose(0, 1)
            em_y = probs_x_w * noise_matrix_col
            em_y = em_y / em_y.sum(dim=1, keepdim=True)

        # compute forward_backward on graph x_s
        em_probs_x_s = probs_x_s * noise_matrix_col
        em_probs_x_s = em_probs_x_s / em_probs_x_s.sum(dim=1, keepdim=True)
        
        # compute observed noisy labels
        noise_matrix_row = noise_matrix.softmax(dim=0)
        noisy_probs_x_w = torch.matmul(logits_x_w.softmax(dim=-1), noise_matrix_row)
        noisy_probs_x_w = noisy_probs_x_w / noisy_probs_x_w.sum(dim=-1, keepdims=True)

        # compute noisy loss 
        noise_loss = torch.mean(-torch.sum(F.one_hot(y, num_classes) * torch.log(noisy_probs_x_w), dim = -1))
        
        # compute em loss
        em_loss =  torch.mean(-torch.sum(em_y * torch.log(em_probs_x_s), dim=-1), dim=-1)
        
        # compute consistency loss
        con_loss = self.ce_loss(logits_x_s, probs_x_w, reduction='mean')
        
        # total loss
        loss = noise_loss + em_loss + con_loss
        
        # computer average entropy loss
        if average_entropy_loss:
            avg_prediction = torch.mean(logits_x_w.softmax(dim=-1), dim=0)
            prior_distr = 1.0/num_classes * torch.ones_like(avg_prediction)
            avg_prediction = torch.clamp(avg_prediction, min = 1e-6, max = 1.0)
            balance_kl =  torch.mean(-(prior_distr * torch.log(avg_prediction)).sum(dim=0))
            entropy_loss = 0.1 * balance_kl
            loss += entropy_loss

        return loss