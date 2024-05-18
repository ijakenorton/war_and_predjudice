from transformer import masked_loss
import keras

def make_transformer(vocab_size):

    input_layer = keras.Input(shape=(vocab_size, ))
    first_dense = keras.layers.Dense(units=DIMENSION,activation=keras.activations.tanh,
                                     use_bias=False)
    output_layer = keras.layers.Dense(units=vocab_size,activation=keras.activations.softmax,
                                      use_bias=False)
    model = keras.Sequential([input_layer, first_dense, output_layer])
    model.compile(optimizer="adam",loss=keras.losses.categorical_crossentropy,
                  metrics=["accuracy"])

    return model
if __name__ == "__main__":

