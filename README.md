# speech_utils
An utilities package for Speech Emotion Recognition, including training scripts.

## Overview
This package has been developed to turn my works in the **U**ndergraduate **Re**search on **Ca**mpus (URECA) lab at NTU, under the supervision of Professor Chng Eng Siong, into an extensible and maintainable Speech Emotion Recognition API used internally by the research group.

## Dependencies

```
numpy
pandas
sklearn
keras
python_speech_features
```

### Additionally, you need to install a deep learning framework (either `TensorFlow` or `PyTorch`) in order to train and evaluate a model using respective DL framework.

## Installation

Build the project from source in development mode and allow the changes in source code to take effect immediately:
```
git clone https://github.com/NTU-SER/speech_utils
cd speech_utils/
pip install -e .
```

## Usage
Here is the list of SER architectures/models supported by the package:
- `ACRNN`: Chen, M., He, X., Yang, J., & Zhang, H. (n.d.). 3-D Convolutional Recurrent Neural Networks with Attention Model for Speech Emotion Recognition. 5.
- (work in progress...)

### 1. ACRNN:

Note that all scripts are well-documented. Use `python script_name.py --help` to see all arguments details.

Also note that both TensorFlow and PyTorch implementations have similar arguments and architecture. The performance, however, differs significantly that PyTorch version outperforms TensorFlow version most of the time. Moreover, training with PyTorch version takes up less memory and is faster than TensorFlow version. Therefore, PyTorch implementation is highly recommended to use.

1. Extract Mel features:
For example, execute:
```
python extract_mel.py /path/to/IEMOCAP_full_release /path/to/save/features.pkl --train_sess Session2 Session3 Session4 Session5 --test_sess Session1
```
if you want to read `Session1` data as validation and test data, and the rest of sessions as training data.

2. Train:

For example, execute:

```
python train_torch.py /path/to/features.pkl 200 --batch_size 96 --save_path /path/to/save/model.pth --perform_test --seed 136 --swap
```

  where `200` is the number of epochs. Execute the script with flag `--swap` to swap validation and test set. Exceute the script with flag `--perform_test` to perform testing during training.

3. Test:

This script is used to perform evaluation on a checkpoint. If you execute `train_torch.py` with `--perform_test`, you probably do not need to use this script.

For example, execute:

```
python test_torch.py /path/to/features.pkl /path/to/model.pth --batch_size 128
```

## Results:

### 1. ACRNN:

- Unweighted Accuracy: `66.03% ± 9.97%`
- Unweighted Recall: `65.99 ± 9.51%`
