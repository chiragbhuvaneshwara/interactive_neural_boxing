import tensorflow as tf
from train.nn.mann_keras.utils import save_network


class EpochWriter(tf.keras.callbacks.Callback):
    def __init__(self, path, Xmean, Ymean, Xstd, Ystd):
        super().__init__()
        self.path = path
        self.Xmean = Xmean
        self.Ymean = Ymean
        self.Xstd = Xstd
        self.Ystd = Ystd

    def on_epoch_end(self, epoch, logs=None):
        save_network(self.path % epoch, self.model, self.Xmean, self.Ymean, self.Xstd, self.Ystd)
        # print("\nModel saved to ", self.path % epoch)


class GatingChecker(tf.keras.callbacks.Callback):
    def __init__(self, X, batch_size):
        super().__init__()
        self.X = X
        self.batch_size = batch_size

    def on_epoch_begin(self, epoch, logs=None):
        get_variation_gating(self.model, self.X, self.batch_size)

def get_variation_gating(network, input_data, batch_size):
    gws = []
    r_lim = (input_data.shape[0] - 1) // batch_size
    for i in range(r_lim):
        bi = input_data[i * batch_size:(i + 1) * batch_size, :]
        out = network(bi)
        # TODO Test var for extracting gating outputs
        # gws.append(out[:, -6:])
        gws.append(out[:, -network.expert_nodes:])

    # print("\nChecking the gating variability: ")
    # print("  mean: ", np.mean(np.concatenate(gws, axis=0), axis=0))
    # print("  std: ", np.std(np.concatenate(gws, axis=0), axis=0))
    # print("  max: ", np.max(np.concatenate(gws, axis=0), axis=0))
    # print("  min: ", np.min(np.concatenate(gws, axis=0), axis=0))
    # print("")
