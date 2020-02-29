import pickle

from tensorflow.compat.v1 import ConfigProto, InteractiveSession
import tensorflow as tf

from speech_utils.ARCNN.model_utils import train

config = ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
session = tf.Session(config=config).as_default()

def main():
    # Hyperparameters
    num_epochs = 1000
    batch_size = 64
    lr = 1e-5
    dropout_keep_prob = 1
    shuffle = False
    random_seed = None
    use_CBL = True
    # Load data
    data_path = "../data/ARCNN/features.pkl"
    with open(data_path, "rb") as fin:
        data = pickle.load(fin)
    # Train
    train(data, num_epochs, batch_size, lr,
          dropout_keep_prob=dropout_keep_prob,
          shuffle=shuffle, random_seed=None,
          use_CBL=use_CBL)

if __name__ == "__main__":
    main()
