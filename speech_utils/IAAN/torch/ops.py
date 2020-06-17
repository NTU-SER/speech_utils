import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class AdditiveAttentionV1(nn.Module):
    """Additive attention (Bahdanau attention), converted directly from
    TensorFlow version in the original repository.

    Parameters
    ----------
    input_size : int
        Features dimension of the inputs.
    attention_size : int
        Attention size.

    """
    def __init__(self, input_size, attention_size=16):
        super(AdditiveAttentionV1, self).__init__()
        # Cache info
        self.input_size = input_size
        self.attention_size = attention_size
        # Sub layers
        self.linear1 = nn.Linear(input_size, attention_size)
        self.linear2 = nn.Linear(input_size, attention_size)
        self.tanh = nn.Tanh()
        self.linear = nn.Linear(attention_size, 1)  # weighting params
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, values, values_mask):
        """Forward pass logic.

        Parameters
        ----------
        query : torch.Tensor
            Query tensor of shape (B, T, D), where B is batch size, T is
            sequence length (number of time steps) and D is features dimension
            (`self.input_size`).
        values : torch.Tensor
            Values tensor of the same shape as the query tensor.
        values_mask : torch.Tensor
            Just for compatibility with V2. This will be always ignored.

        Returns
        -------
        torch.Tensor
            Attention tensor of shape (B, D). The time axis (T) has been
            squeezed, meaning that the tensor itself computes most relevant
            data along the time axis for every features dimension.

        """
        if query.shape != values.shape:
            raise RuntimeError(
                "Size mismatch: {} and {}.".format(query.shape, values.shape))

        v = self.tanh(self.linear1(query) + self.linear2(values))  # (B, T, A)
        v = self.linear(v).squeeze(-1)  # (B, T)
        alphas = self.softmax(v)  # (B, T)

        # Weighting by scores computed and sum along the time axis
        outputs = (values * alphas.unsqueeze(-1)).sum(dim=1)  # (B, D)
        return outputs  # (B, D)


class AdditiveAttentionV2(nn.Module):
    """Yet another implementation of additive attention. In this
    implementation, the two tensor `query` and `values` are concatenated to
    compute the scores (as opposed to the other implementation where these
    tensors are passed through two linear layers independently). Moreover, the
    mask for the `values` tensor must be passed in forward call.

    Parameters
    ----------
    input_size : int
        Features dimension of the inputs.
    attention_size : int
        Attention size.

    """

    def __init__(self, input_size, attention_size=16):
        super(AdditiveAttentionV2, self).__init__()
        # Cache info
        self.linear1 = nn.Linear(input_size * 2, attention_size)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(attention_size, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, values, values_mask):
        """Forward pass logic.

        Parameters
        ----------
        query : torch.Tensor
            Query tensor of shape (B, T, D), where B is batch size, T is
            sequence length (number of time steps) and D is features dimension
            (`self.input_size`).
        values : torch.Tensor
            Values tensor of the same shape as the query tensor.
        values_mask : torch.Tensor
            Tensor of shape (B, T). Mask of the `values` tensor, indicating
            which time steps to ignore during training (because of padding).

        Returns
        -------
        torch.Tensor
            Attention tensor of shape (B, D). The time axis (T) has been
            squeezed, meaning that the tensor itself computes most relevant
            data along the time axis for every features dimension.

        """
        if query.shape != values.shape:
            raise RuntimeError(
                "Size mismatch: {} and {}.".format(query.shape, values.shape))

        features = torch.cat([query, values], dim=-1)  # (B, T, D * 2)
        v = self.linear1(features)  # (B, T, A)
        alphas = self.linear2(v).squeeze(-1)  # (B, T)

        # Apply mask, renormalize
        alphas = alphas * values_mask  # (B, T)
        alphas.div_(alphas.sum(-1, keepdim=True))  # (B, T)

        # Weighting by scores computed and sum along the time axis
        outputs = torch.bmm(alphas.unsqueeze(1), values).squeeze(1)  # (B, D)
        return outputs


class SelfAttention(nn.Module):
    """
    Attention implementation equivalent to Tensorflow version at
    `speech_utils.ACRNN.tf.model_utils.attention`.

    Parameters
    ----------
    input_size : int
        Features dimension of the inputs.
    attention_size : int
        Number of attention's hidden size.

    """
    def __init__(self, input_size, attention_size=16):
        super(SelfAttention, self).__init__()
        self.linear1 = nn.Linear(in_features=input_size,
                                 out_features=attention_size)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(in_features=attention_size, out_features=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inp):
        """Forward pass logic.

        Parameters
        ----------
        inp : tensor
            Input tensor of shape (B, T, D), where B is batch size, T is
            sequence length (number of time steps) and D is features dimension
            (a.k.a. `self.input_size`).

        Returns
        -------
        torch.Tensor
            Attention tensor of shape (B, D). The time axis (T) has been
            squeezed, meaning that the tensor itself computes most relevant
            data along the time axis for every features dimension.

        """
        dims = inp.size()

        v = self.tanh(self.linear1(inp))  # (B, T, A)
        v = self.linear2(v)  # (B, T, 1)
        # Reshape to apply softmax
        v = v.squeeze(-1)  # (B, T)
        alphas = self.softmax(v)  # (B, T)

        outputs = (inp * alphas.unsqueeze(-1)).sum(axis=1)  # (B, D)
        return outputs  # (B, D)


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention as described in:
        Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez,
        A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need.
        ArXiv:1706.03762 [Cs]. http://arxiv.org/abs/1706.03762


    Parameters
    ----------
    input_size : int
        Features dimension of the inputs.
    attention_size : int
        Attention size.
    dropout_keep_prob : float
        Probability of keeping a connection in the dropout layer.
    masking : bool
        Whether to apply the masking to the intermediate tensors or not.

    """
    def __init__(self, input_size, attention_size=64, dropout_keep_prob=1,
                 masking=False):
        super(ScaledDotProductAttention, self).__init__()
        # Cache info
        self.input_size = input_size
        self.attention_size = attention_size
        self.dropout_keep_prob = dropout_keep_prob
        self.masking = masking
        # Linear layers to generate Q (queries), K (keys) & V (values) tensors
        self.linearQ = nn.Linear(input_size, attention_size)
        self.linearK = nn.Linear(input_size, attention_size)
        self.linearV = nn.Linear(input_size, attention_size)
        # Mask
        if masking:
            self.K_mask = Mask()
            self.Q_mask = Mask()
        # Other layers
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=(1 - dropout_keep_prob))

    def forward(self, inp):
        """Forward pass logic.

        Parameters
        ----------
        inp : torch.Tensor
            Tensor of shape (B, T, D), where B is batch size, T is the sequence
            length (along the time axis), and D is the features dimension.

        Returns
        -------
        torch.Tensor
            Attention tensor of shape (B, D). The time axis (T) has been
            squeezed, meaning that the tensor itself computes most relevant
            data along the time axis for every features dimension.

        """
        # Compute Q (queries), K (keys) & V (values)
        Q = self.linearQ(inp)  # (B, T, A)
        K = self.linearK(inp)  # (B, T, A)
        V = self.linearV(inp)  # (B, T, A)

        # Batched matrix multiplication between Q and K_T
        matmul_QK = torch.bmm(Q, K.permute(0, 2, 1))  # (B, T, T)
        # Scale: divide by square root of `query` hidden size, which is
        # `attention_size`
        matmul_QK_scaled = matmul_QK / (self.attention_size ** 0.5)
        # Add K mask
        if self.masking:
            K_mask = self.K_mask(K)  # (B, T, T)
            K_paddings = torch.ones_like(matmul_QK_scaled) * 1e-8  # (B, T, T)
            K_paddings = K_paddings.to(device=inp.device)
            matmul_QK_scaled = torch.where(  # (B, T, T)
                torch.eq(K_mask, 0), K_paddings, matmul_QK_scaled)

        # Softmax to compute attention weights
        att_weights = self.softmax(matmul_QK_scaled)  # (B, T, T)
        # Add Q mask
        if self.masking:
            Q_mask = self.Q_mask(Q)  # (B, T, T)
            att_weights = Q_mask * att_weights  # (B, T, T)
        # Dropouts
        att_weights = self.dropout(att_weights)  # (B, T, T)

        # Compute outputs
        outputs = torch.bmm(att_weights, V)  # (B, T, A)
        outputs = torch.bmm(inp.permute(0, 2, 1), outputs)  # (B, D, A)
        outputs = outputs.sum(dim=-1)  # (B, D)

        return outputs


class Mask(nn.Module):
    """Mask the input tensor, indicating which time steps should be ignored
    during training (because of padding).

    Parameters
    ----------
    output_size : int or None
        If None:
            inputs: (B, C, D) -> outputs: (B, C, C)
        Otherwise:
            inputs: (B, C, D) -> outputs: (B, C, `output_size`)

    """
    def __init__(self, output_size=None):
        super(Mask, self).__init__()
        self.output_size = output_size

    def forward(self, inp):
        """Forward pass logic.

        Parameters
        ----------
        inp : torch.Tensor
            Input tensor of shape (B, C, D).

        Returns
        -------
        torch.Tensor
            Mask tensor of shape (B, C, C) or (B, C, `output_size`).

        """
        # Get output dimension
        dim = inp.shape[1] if self.output_size is None else self.output_size

        mask = torch.sign(torch.abs(inp.sum(dim=-1)))  # (B, C)
        mask = mask.unsqueeze(1).repeat(1, dim, 1)  # (B, `dim`, C)
        # Transpose if needed
        if self.output_size is not None:
            mask = mask.permute(0, 2, 1)  # (B, C, `dim`)
        return mask


class IAAMask(nn.Module):
    """Apply mask as implemented in the Interaction-aware attention of the
    TensoFlow version,

    """
    def __init__(self):
        super(IAAMask, self).__init__()

    def forward(self, x, y):
        """Forward pass logic.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be applied mask, shaped (B, C).
        y : type
            Tensor to calculate mask from, shaped (B, C, D)

        Returns
        -------
        torch.Tensor
            `x` shaped (B, C) after being mask by a mask calculated from `y`.

        """
        # Calculate mask
        mask_att = torch.sign(torch.abs((y.sum(-1))))  # (B, C)
        paddings = torch.ones_like(mask_att) * 1e-8  # (B, C)
        paddings = paddings.to(device=x.device)
        # Apply mask
        x = torch.where(torch.eq(mask_att, 0), paddings, x)  # (B, C)
        return x


class LayerNormWrapperV1(nn.Module):
    """Layer normalization wrapper, which performs drop out after normalizing.
    This makes use of the PyTorch implementation of layer normalization, which
    differs slightly to the TensorFlow version.

    Parameters
    ----------
    normalized_shape : array-like
        Array-like of integers, representing the dimensions on which the
        normalization will be performed. For example, if the input has shape
        (B, T, D, E), where B and T is dynamic, you may want to set
        `normalized_shape=(D,)` or `normalized_shape=(D, E)`.
    dropout_keep_prob : float
        Probability of keeping a connection in the dropout layer.
    **kwargs
        Optional keyword arguments to be passed to the layer normalization
        constructor.

    """
    def __init__(self, normalized_shape, dropout_keep_prob=1.0, **kwargs):
        super(LayerNormWrapperV1, self).__init__()
        # Cache info
        self.normalized_shape = normalized_shape
        self.dropout_keep_prob = dropout_keep_prob
        # Layers
        self.layer_norm = nn.LayerNorm(normalized_shape)
        self.dropout = nn.Dropout(p=(1 - dropout_keep_prob))

    def forward(self, inp):
        """Forward pass logic"""
        inp_normalized = self.layer_norm(inp)
        return self.dropout(inp_normalized)


class LabelSmoothing(nn.Module):
    """Label smoothing layer.

    """
    def __init__(self):
        super(LabelSmoothing, self).__init__()

    def forward(self, inp):
        """Forward pass logic.

        Parameters
        ----------
        inp : torch.Tensor
            Label tensor of shape (batch_size, num_classes).

        """
        features_dim = inp.shape[-1]
        inp_smoothed = (0.9 * inp) + (0.1 / features_dim)
        return inp_smoothed


class Expand(nn.Module):
    """Expand a tensor along a particular axis.

    """
    def __init__(self, size=None, axis=1):
        super(Expand, self).__init__()
        self.size = size
        self.axis = axis

    def forward(self, inp, size=None):
        """Forward pass logic.

        Parameters
        ----------
        inp : torch.Tensor
            Input tensor of shape (size_0, size_1, ..., size_a, ..., size_n),
            where `a` is `self.axis`.
        size : int (optional)
            Size of the dimension to be expanded. Need to pass either during
            construction or forward call.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (size_0, size_1, ..., size_(a-1), `size`,
            size_a, ..., size_n), where `a` is `self.axis` anf `size` is
            `self.size`.

        """
        if size is None:
            size = self.size
        if size is None:
            raise RuntimeError("Size not specified.")

        dims = inp.shape
        inp = inp.unsqueeze(self.axis)
        repeats = [size if i == self.axis else 1
                   for i in range(len(dims) + 1)]
        outputs = inp.repeat(repeats)
        return outputs


def focal_loss(input, target, alpha, gamma=2, reduction="none", eps=1e-8):
    """Compute focal loss.
    Adapted from:
        https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py

    Parameters
    ----------
    input : torch.Tensor
        Logits outputs of the model. Tensor of shape (B, N), where is batch
        size and N is number of classes.
    target : torch.Tensor
        Target predictions (i.e., groundtruths). Tensor of shape (B,).
    alpha : torch.Tensor or float
        If alpha is a tensor, it must be broadcastable to (B, N). The weighting
        factor(s) to compute focal loss.
    gamma : int or float
        Focusing parameter.
    reduction : str
        One of ["none", "mean", "sum"]. Mode of reduction to use.
        If "none", no reduction will be applied and the output tensor will have
        shape (B,).
        Otherwise, output tensor will be a scalar.
    eps : float
        For numerical stability.

    Returns
    -------
    torch.Tensor
        If `reduction="none"`, this will have shape (B,).
        Otherwise, this will be a scalar.

    """
    if reduction not in ["none", "mean", "sum"]:
        raise ValueError(
            "Invalid reduction mode. Expect one of ['none', 'mean', 'sum'], "
            "got {} instead.".format(reduction))

    n = input.size(0)  # batch size B
    out_size = (n,) + input.size()[2:]  # should be (B,)

    # Compute softmax over the classes axis
    input_soft = F.softmax(input, dim=1) + eps  # (B, N)

    # Create the labels one hot tensor
    target_one_hot = F.one_hot(target, num_classes=input.shape[1])  # (B, N)

    # Compute the actual focal loss
    weight = torch.pow(-input_soft + 1., gamma)  # (B, N)

    focal = -alpha * weight * torch.log(input_soft)  # (B, N)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)  # (B,)

    if reduction == "none":
        loss = loss_tmp  # (B,)
    elif reduction == "mean":
        loss = torch.mean(loss_tmp)  # scalar
    else:
        loss = torch.sum(loss_tmp)  # scalar

    return loss


class FocalLoss(nn.Module):
    """Compute focal loss.
    Adapted from:
        https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py

    Parameters
    ----------
    alpha : torch.Tensor or float
        If alpha is a tensor, it must be broadcastable to (B, N). The weighting
        factor(s) to compute focal loss.
        Note that alpha can be passed in the forward pass, allowing using
        dynamic weighting factor during training. If so, the existing alpha
        will be ignored.
    gamma : int or float
        Focusing parameter.
    reduction : str
        One of ["none", "mean", "sum"]. Mode of reduction to use.
        If "none", no reduction will be applied and the output tensor will have
        shape (B,).
        Otherwise, output tensor will be a scalar.
    eps : float
        For numerical stability.

    """
    def __init__(self, alpha=None, gamma=2.0, reduction="none", eps=1e-6):
        super(FocalLoss, self).__init__()
        # Cache info
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, input, target, alpha=None):
        """Forward pass logic.

        Parameters
        ----------
        input : torch.Tensor
            Logits outputs of the model. Tensor of shape (B, N), where is batch
            size and N is number of classes.
        target : torch.Tensor
            Target predictions (i.e., groundtruths). Tensor of shape (B,).
        alpha : torch.Tensor or float or None
            The weighting factor(s) to compute focal loss.
            If alpha is a tensor, it must be broadcastable to (B, N).
            If None, the cached alpha will be used.

        Returns
        -------
        torch.Tensor
            If `reduction="none"`, this will have shape (B,).
            Otherwise, this will be a scalar.

        """
        if alpha is None:
            alpha = self.alpha
        return focal_loss(input, target, alpha, self.gamma,
                          self.reduction, self.eps)


class ClassBalancedLoss(nn.Module):
    """
    Compute the Class Balanced Loss between `logits` and the ground truth
    `labels`. Class Balanced Loss:
        ((1 - beta)/(1 - beta^n)) * Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.

    Parameters
    ----------
    samples_per_cls : list
        Number of examples for each class.
    loss_type : str
        One of ["focal", "sigmoid", "softmax"].
    beta : float
        Hyperparameter for CBLoss.
    **kwargs : dict
        Custom keyword arguments to be passed to the chosen loss constructor
        (e.g., `gamma=1` for focal loss if `loss_type="focal"`).

    """
    def __init__(self, samples_per_cls, loss_type="softmax", beta=0.9999,
                 **kwargs):
        super(ClassBalancedLoss, self).__init__()

        self.samples_per_cls = samples_per_cls
        self.num_classes = len(samples_per_cls)
        self.beta = beta
        self.weights = self.calc_weights()
        self.loss_type = loss_type

        if loss_type == "sigmoid":
            self.loss = lambda x, y, z: F.binary_cross_entropy_with_logits(
                x, y, z, **kwargs)
        elif loss_type == "softmax":
            self.loss = lambda x, y, z: F.binary_cross_entropy(
                x, y, z, **kwargs)
        elif loss_type == "focal":
            self.loss = FocalLoss(**kwargs)

    def calc_weights(self):
        effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)  # (N,)
        weights = (1.0 - self.beta) / effective_num  # (N,)
        weights = weights / weights.sum() * self.num_classes  # (N,)

        weights = torch.tensor(weights).float().cuda()  # (N,)
        weights = weights.unsqueeze(0)  # (1, N)

        return weights

    def forward(self, y_hat, y):
        """Forward pass logic.

        Parameters
        ----------
        y_hat : tensor
            Tensor of shape (B, N) representing predictions, where B is batch
            size and N is the number of classes.
        y : tensor
            Tensor of shape (B,) representing target labels.

        Returns
        -------
        tensor
            A scalar tensor representing class balanced loss.

        """
        # Compute weights with respect to y
        y_one_hot = F.one_hot(y, self.num_classes).float()  # (B, N)
        weights = self.weights.repeat(y.shape[0], 1) * y_one_hot  # (B, N)
        weights = weights.sum(1)  # (B,)

        if self.loss_type == "focal":
            loss = self.loss(y_hat, y, weights.unsqueeze(-1))
        else:
            weights = weights.unsqueeze(1)
            weights = weights.repeat(1, self.num_classes)
            if self.loss_type == "softmax":
                y_hat = y_hat.softmax(dim=1)
            loss = self.loss(y_hat, y_one_hot, weights)
        return loss


"""
WARNING: Failed attempt. Don't use this implementation. I will try to fix this
in the future.
"""


class LayerNormWrapperV2(nn.Module):
    """Layer normalization wrapper, which performs drop out after normalizing.
    This implementation is adapted directly from TensorFlow implementation, but
    may not have an advantage of tracking running statistics.

    Parameters
    ----------
    normalized_shape : array-like
        Array-like of integers, representing the dimensions on which the
        normalization will be performed. For example, if the input has shape
        (B, T, D, E), where B and T is dynamic, you may want to set
        `normalized_shape=(D,)` or `normalized_shape=(D, E)`.
    dropout_keep_prob : float
        Probability of keeping a connection in the dropout layer.
    **kwargs
        Optional keyword arguments to be passed to the layer normalization
        constructor.

    """
    def __init__(self, begin_norm_axis=1, begin_params_axis=-1,
                 dropout_keep_prob=1.0):
        super(LayerNormWrapperV2, self).__init__()
        # Cache info
        self.begin_norm_axis = begin_norm_axis
        self.begin_params_axis = begin_params_axis
        # Trainable parameters, initialized when needed
        self.beta = None
        self.gamma = None

    def _initialize_parameters(self, params_shape):
        if self.beta is None or self.gamma is None:
            self.beta = nn.Parameter(torch.Tensor(params_shape))
            self.gamma = nn.Parameter(torch.Tensor(params_shape))

    def forward(self, inp, is_training):
        dims = inp.shape
        # Initialized when needed
        self._initialize_parameters(dims[self.begin_params_axis:])
        # Get mean and variance
        norm_dims = list(range(self.begin_norm_axis, len(dims)))
        mean = inp.mean(dim=norm_dims, keepdim=True)
        var = inp.var(dim=norm_dims, keepdim=True)
        # Normalize
        outputs = F.batch_norm(
            inp, mean, var, weight=self.gamma, bias=self.beta,
            training=is_training)
        return outputs
