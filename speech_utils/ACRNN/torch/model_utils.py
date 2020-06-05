import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from speech_utils.ACRNN.torch.data_utils import DatasetLoader


class Attention(nn.Module):
    """
    Attention implementation equivalent to Tensorflow's version at
    `speech_utils.ACRNN.tf.model_utils.attention`.

    Parameters
    ----------
    hidden_size : int
        Number of neurons of the input data.
    attention_size : int
        Number of attention's hidden size.

    """
    def __init__(self, hidden_size, attention_size):
        super(Attention, self).__init__()
        self.linear1 = nn.Linear(in_features=hidden_size,
                                 out_features=attention_size)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(in_features=attention_size, out_features=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Parameters
        ----------
        x : tensor
            Tensor of shape (B, H, W), where B is batch size, H is image height
            and W is image width. (For 3D-CRNN, x is a tensor of shape
            (B, T, L), where T is the number of time step and L is the LSTM
            hidden size (or 2 * the LSTM hidden size if LSTM is
            bi-directional)).

        Returns
        -------
        tensor
            Tensor of shape (B, W), where A is the attention size. (For
            3D-CRNN, output is a tensor of shape (B, L), where L is defined as
            above).

        """
        dims = x.size()
        x = x.view(-1, dims[-1])

        v = self.sigmoid(self.linear1(x))
        v = self.linear2(v)
        # Reshape to apply softmax
        v = v.view(dims[0], dims[1])
        v = self.softmax(v)

        v = v.unsqueeze(-1)
        x = x.view(*dims)

        output = (x * v).sum(axis=1)
        return output


class ACRNN(nn.Module):
    """
    Reference: "3-D Convolutional Recurrent Neural Networks with Attention
    Model for Speech Emotion Recognition"
    Authors: Chen Mingyi and
             He Xuanji and
             Yang Jing and
             Zhang Han.

    Adapted from
        https://github.com/xuanjihe/speech-emotion-recognition/blob/master/acrnn1.py

    3D-CRNN model wrapper with PyTorch. Take `inputs` as model input and
    return the model logits.

    Parameters
    ----------
    num_classes : int
    image_size : tuple
        Tuple of (image height, image width).
    num_filters_1 : int
        Number of output channels for the first convolutional layer.
    num_filters_2 : int
        Number of output channels for the rest of convolutional layers.
    kernel_size : tuple
        Kernel size of convolutional layers.
    negative_slope : float
        Negative slope for leaky ReLU activation function.
    maxpool_fsize : tuple
        Filter size of max pooling layer.
    num_linear_units : int
        Number of neurons in the first linear layer.
    lstm_hidden_size : int
        LSTM's hidden size.
    num_lstm_layers : int
        Number of LSTM layers.
    dropout_keep_prob : int
        Probability of keeping a connection.
    attention_size : int
    num_fcn_units : int
        Number of neurons in the second linear layer.

    """
    def __init__(self, num_classes=4, image_size=(300, 40), num_filters_1=128,
                 num_filters_2=256, kernel_size=(5, 3), negative_slope=0.01,
                 maxpool_fsize=(2, 4), num_linear_units=768,
                 lstm_hidden_size=128, num_lstm_layers=1, dropout_keep_prob=1,
                 attention_size=1, num_fcn_units=64):

        super(ACRNN, self).__init__()
        # Cache parameters
        self.params = {
            "padding": self._calc_padding(kernel_size),
            "linear1_in": image_size[1] // maxpool_fsize[1] * num_filters_2,
            "time_step": image_size[0] // maxpool_fsize[0],
            "num_linear_units": num_linear_units,
            "negative_slope": negative_slope
        }
        # Pre-defined
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.max_pool = nn.MaxPool2d(kernel_size=maxpool_fsize)
        self.dropout = nn.Dropout(p=(1 - dropout_keep_prob))
        # Apply batch normalization to input images
        self.batchnorm0 = nn.BatchNorm2d(num_features=3)
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=num_filters_1,
                               kernel_size=kernel_size, stride=1,
                               padding=self.params["padding"], bias=True)
        self.batchnorm1 = nn.BatchNorm2d(num_features=num_filters_1)
        # Second convolutional layers
        self.conv2 = nn.Conv2d(in_channels=num_filters_1,
                               out_channels=num_filters_2,
                               kernel_size=kernel_size, stride=1,
                               padding=self.params["padding"], bias=True)
        self.batchnorm2 = nn.BatchNorm2d(num_features=num_filters_2)
        # There are 4 similar convolutional layers
        # after the second convolutional layer
        self.conv3 = nn.Conv2d(in_channels=num_filters_2,
                               out_channels=num_filters_2,
                               kernel_size=kernel_size, stride=1,
                               padding=self.params["padding"], bias=True)
        self.batchnorm3 = nn.BatchNorm2d(num_features=num_filters_2)

        self.conv4 = nn.Conv2d(in_channels=num_filters_2,
                               out_channels=num_filters_2,
                               kernel_size=kernel_size, stride=1,
                               padding=self.params["padding"], bias=True)
        self.batchnorm4 = nn.BatchNorm2d(num_features=num_filters_2)

        self.conv5 = nn.Conv2d(in_channels=num_filters_2,
                               out_channels=num_filters_2,
                               kernel_size=kernel_size, stride=1,
                               padding=self.params["padding"], bias=True)
        self.batchnorm5 = nn.BatchNorm2d(num_features=num_filters_2)

        self.conv6 = nn.Conv2d(in_channels=num_filters_2,
                               out_channels=num_filters_2,
                               kernel_size=kernel_size, stride=1,
                               padding=self.params["padding"], bias=True)
        self.batchnorm6 = nn.BatchNorm2d(num_features=num_filters_2)
        # Linear layers after reshaping
        self.linear1 = nn.Linear(in_features=self.params["linear1_in"],
                                 out_features=num_linear_units)
        # Batchnorm
        self.batchnorm7 = nn.BatchNorm1d(num_features=num_linear_units)
        # Bidirectional-LSTM cells
        self.bilstm = nn.LSTM(
            input_size=num_linear_units, hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers, bias=True, batch_first=True,
            dropout=(1 - dropout_keep_prob), bidirectional=True)
        # Attention on the output of the LSTM cell
        self.attention = Attention(hidden_size=lstm_hidden_size * 2,
                                   attention_size=attention_size)
        # Linear layers after Attention
        self.linear2 = nn.Linear(in_features=lstm_hidden_size * 2,
                                 out_features=num_fcn_units)
        self.linear3 = nn.Linear(in_features=num_fcn_units,
                                 out_features=num_classes)

        # Initialize paramaters for convolutional layers with
        # kaiming normal distribution
        self.init_params()

    def _calc_padding(self, kernel_size):
        """
        Calculate `padding` for "same" padding for convolutional layer.
        Applicable for `stride=1` only.
        """
        ans = []
        for i in kernel_size:
            if i % 2 == 0:
                raise ValueError("Kernel size must be odd. Got "
                                 "{} instead".format(i))
            ans.append(i // 2)
        return ans

    def init_params(self):
        """
        Initialize parameters for convolutional layers with kaiming normal
        distribution.
        """
        def _init_params(m):
            if type(m) == nn.modules.conv.Conv2d:
                nn.init.kaiming_normal_(
                    m.weight, a=self.params["negative_slope"],
                    mode="fan_in", nonlinearity="leaky_relu")
        self.apply(_init_params)

    def forward(self, x):
        """
        Parameters
        ----------
        x : tensor
            Tensor of shape (B, C, H, W), where B is batch size, C is the
            numnber of channels, H is image height and W is image width.

        Returns
        -------
        tensor
            A tensor of shape (B, N), where N is the number of classes.
            This tensor is the output of the last linear layer (without
            applying softmax function).

        """
        batch_size = x.shape[0]
        # Apply batch normalization to input images
        x = self.batchnorm0(x)
        # First convolutional layer
        x = self.leaky_relu(self.conv1(x))
        x = self.batchnorm1(x)
        x = self.max_pool(x)
        # The rest of the convolutional layers
        x = self.leaky_relu(self.conv2(x))
        x = self.batchnorm2(x)
        x = self.leaky_relu(self.conv3(x))
        x = self.batchnorm3(x)
        x = self.leaky_relu(self.conv4(x))
        x = self.batchnorm4(x)
        x = self.leaky_relu(self.conv5(x))
        x = self.batchnorm5(x)
        x = self.leaky_relu(self.conv6(x))
        x = self.batchnorm6(x)
        # Convert back to channel_last
        x = x.permute([0, 2, 3, 1])
        # Reshape
        x = x.reshape(-1, self.params["linear1_in"])
        assert x.shape[0] == batch_size * self.params["time_step"]
        # Linear layer
        x = self.linear1(x)
        # Apply batchnorm, non-linearity then drop-out
        x = self.batchnorm7(x)
        x = self.dropout(self.leaky_relu(x))
        # Reshape. Note that in PyTorch, the input of LSTM
        # is of shape (seq_len, batch, input_features)
        x = x.view(-1, self.params["time_step"],
                   self.params["num_linear_units"])
        x = x.permute(1, 0, 2)
        assert x.shape[1] == batch_size
        # Bi-LSTM
        x, _ = self.bilstm(x)
        # Convert back to batch_first
        x = x.permute([1, 0, 2])
        assert x.shape[0] == batch_size
        assert x.shape[1] == self.params["time_step"]
        # Attention
        x = self.attention(x)
        assert x.shape[0] == batch_size
        # FCNs
        x = self.linear2(x)
        x = self.dropout(self.leaky_relu(x))
        x = self.linear3(x)

        return x


class FocalLoss(nn.Module):
    """
    Adapted from
        https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py

    Allow using dynamic alpha (weights) for each pass (instead in fixed alpha
    as in the original implementation.)

    Parameters
    ----------
    gamma : float
        Hyperparameter for focal loss.
    alpha : float, int, list or tensor
        Note that if alpha is passed in each pass then this default alpha value
        will be overwritten.
        If float or int: then alpha is the weight of the first class in a
        binary classification task.
        If list or tensor: then alpha is a list of weights of all classes in a
        multi-class classification task.
    size_average : bool
        If True, average the loss over the entire mini batch.
        If False, sum the loss over the entire mini batch.

    """
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        # For binary classification
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        # For multi-class classification
        elif isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target, alpha=None):
        """
        Parameters
        ----------
        input : tensor
            A tensor of shape (B, N), where B is batch size and N is the number
            of classes.
        target : tensor
            A tensor of shape (B,).
        alpha : list or tensor
            Custom weights for each class.
            If not None, this value will be used instead of `self.alpha`.

        Returns
        -------
        tensor
            Scalar tensor representing focal loss.

        """
        if input.dim() > 2:
            input = input.view(
                input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(
                -1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if alpha is None:
            alpha = self.alpha

        if alpha is not None:
            if alpha.type() != input.data.type():
                alpha = alpha.type_as(input.data)
            at = alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


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
        Custom keyword arguments to be passed to the loss chosen (e.g.,
        `gamma=1` for focal loss if `loss_type="focal"`).

    """
    def __init__(self, samples_per_cls, loss_type="softmax",
                 beta=0.9999, **kwargs):
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
        effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
        weights = (1.0 - self.beta) / effective_num
        weights = weights / weights.sum() * self.num_classes

        weights = torch.tensor(weights).float().cuda()
        weights = weights.unsqueeze(0)

        return weights

    def forward(self, y_hat, y):
        """
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
        y_one_hot = F.one_hot(y, self.num_classes).float()
        weights = self.weights.repeat(y.shape[0], 1) * y_one_hot
        weights = weights.sum(1)

        if self.loss_type == "focal":
            loss = self.loss(y_hat, y, weights)
        else:
            weights = weights.unsqueeze(1)
            weights = weights.repeat(1, self.num_classes)
            if self.loss_type == "softmax":
                y_hat = y_hat.softmax(dim=1)
            loss = self.loss(y_hat, y_one_hot, weights)
        return loss


def test(model, loss_function, test_dataset, batch_size, device,
         return_matrix=False):
    """Test a 3DCRNN model.

    Parameters
    ----------
    model
        PyTorch model.
    loss_function
    test_dataset : `speech_utils.ACRNN.torch.data_utils.TestLoader` instance
        The test dataset.
    batch_size : int
    device
    return_matrix : bool
        Whether to return the confusion matrix.

    Returns
    -------
    test_loss
        Description of returned object.

    """
    test_loss = 0
    test_preds_segs = []
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    for test_data_batch, test_labels_batch in test_loader:
        # Send to correct device
        test_data_batch = test_data_batch.to(device)
        test_labels_batch = test_labels_batch.to(device).long()
        # Forward
        test_preds_batch = model(test_data_batch)
        test_preds_segs.append(test_preds_batch.cpu())
        # Loss
        test_loss += loss_function(test_preds_batch, test_labels_batch)
    # Val average loss
    test_loss = test_loss.item() / test_dataset.n_samples
    # Accumulate results for val data
    test_preds_segs = np.vstack(test_preds_segs)
    test_preds = test_dataset.get_preds(test_preds_segs)
    # Make sure everything works properly
    assert len(test_preds) == test_dataset.n_actual_samples
    test_ua = test_dataset.accuracy(test_preds)
    test_ur = test_dataset.recall(test_preds)
    if return_matrix:
        test_conf = test_dataset.confusion_matrix(test_preds)
        return test_loss, test_ua * 100, test_ur * 100, test_conf
    else:
        return test_loss, test_ua * 100, test_ur * 100


def train(data, epochs, batch_size, learning_rate, random_seed=123,
          num_classes=4, dropout_keep_prob=1, save_path=None,
          loss_type="sigmoid", beta=0.9999, perform_test=False, **kwargs):
    """Train a 3DCRNN model in an epoch-based manner with PyTorch. Note that
    this function does not work with CPU (specifically, if you want to train on
    CPU, modify the module `ClassBalancedLoss` and this function.)

    Parameters
    ----------
    data : tuple
        Data extracted using `speech_utils.ACRNN.data_utils.extract_mel`.
    epochs : int
        Number of epochs.
    batch_size : int
    learning_rate : float
    random_seed : int
        Random seed for reproducibility.
    num_classes : int
        Number of classes.
    dropout_keep_prob : float
        The probability of keeping a connection in dropout.
    save_path : str
        Path to save the best model.
    loss_type : str
        One of ["ce", "sigmoid", "softmax", "focal"].
            If "ce", use the normal Cross Entropy Loss with weights computed as
            the number of sampels per class.
            Otherwise, use the Class Balanced Loss with the corresponding loss
            type.
    beta : float
        Hyperparameter for Class Balanced Loss. Default: 0.9999
    perform_test : bool
        Whether to test on test data at the end of training process.
    **kwargs
        Custom keyword arguments to pass to the ACRNN constructor.

    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if loss_type not in ["ce", "sigmoid", "softmax", "focal"]:
        raise ValueError("Invalid loss type. Expected one of "
                         "[\"ce\", \"sigmoid\", \"softmax\", \"focal\"]. Got "
                         "{} instead.".format(loss_type))
    # Load data into train and test set
    pre_process = lambda x: np.moveaxis(x, 3, 1).astype("float32")
    dataloader = DatasetLoader(
        data, num_classes=num_classes, pre_process=pre_process)
    train_dataset = dataloader.get_train_dataset()
    val_dataset = dataloader.get_val_dataset()
    test_dataset = dataloader.get_test_dataset()
    # Construct model
    device = torch.device("cuda:0")
    model = ACRNN(dropout_keep_prob=dropout_keep_prob, **kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model = model.to(device=device)
    # Loss
    _, samples_per_cls = np.unique(
        train_dataset.target, return_counts=True)
    if loss_type == "ce":  # cross-entropy:
        loss_function = nn.CrossEntropyLoss(weight=samples_per_cls)
    else:
        loss_function = ClassBalancedLoss(
            samples_per_cls=samples_per_cls, loss_type=loss_type, beta=beta)
    # For logging
    labs = ["Train", "Val", "Best Val"]
    loss_format = "{:.04f}"
    acc_format = "{:.02f}%"
    # Start training
    best_val_ua = 0
    for epoch in range(epochs):
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False)
        train_preds = []
        # Train one epoch
        for train_data_batch, train_labels_batch in train_loader:
            # Clear gradients
            optimizer.zero_grad()
            # Send data to correct device
            train_data_batch = train_data_batch.to(device)
            train_labels_batch = train_labels_batch.to(device).long()
            # Forward pass
            preds = model(train_data_batch)
            # Compute the loss, gradients, and update the parameters
            train_loss = loss_function(preds, train_labels_batch)
            train_loss.backward()
            optimizer.step()
            # Accumulate batch results
            train_preds.append(torch.argmax(preds, axis=1).cpu())
        # Evaluate training data
        train_loss = train_loss.item() / batch_size
        train_preds = np.concatenate(train_preds)
        train_ua = train_dataset.accuracy(train_preds) * 100
        train_ur = train_dataset.recall(train_preds) * 100
        # Evaluate validation data
        with torch.no_grad():
            val_loss, val_ua, val_ur = test(
                model, loss_function, val_dataset, batch_size=batch_size,
                device=device)
            # Update
            if val_ua > best_val_ua:
                best_val_ua = val_ua
                best_val_ur = val_ur
                best_val_loss = val_loss
                if save_path is not None:
                    torch.save(model.state_dict(), save_path)
            # Combine results
            loss = [loss_format.format(i) for i
                    in [train_loss, val_loss, best_val_loss]]
            ua = [acc_format.format(i) for i
                  in [train_ua, val_ua, best_val_ua]]
            ur = [acc_format.format(i) for i
                  in [train_ur, val_ur, best_val_ur]]

            loss = dict(zip(labs, loss))
            ua = dict(zip(labs, ua))
            ur = dict(zip(labs, ur))
            df = pd.DataFrame({"Loss": loss, "UA": ua, "UR": ur})
            print("*" * 40)
            print("Epoch #{}".format(epoch + 1))
            print(df)

    # Test at the end of training
    if perform_test:
        with torch.no_grad():
            model.load_state_dict(torch.load(save_path))
            test_loss, test_ua, test_ur, test_conf = test(
                model, loss_function, test_dataset, batch_size=batch_size,
                device=device, return_matrix=True)
            print("*" * 40)
            print("RESULTS ON TEST SET:")
            print("Loss:{:.4f}\tUnweighted Accuracy: {:.2f}\tUnweighted "
                  "Recall: {:.2f}".format(test_loss, test_ua, test_ur))
            print("Confusion matrix:\n{}".format(test_conf))
