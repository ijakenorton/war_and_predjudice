import path_utils
import tok_2_vec
from context import context
from context_types import ContextConfig
from predict_generator import predictTextDataGenerator

import os
import sys
import gzip
import keras
import pickle
import numpy as np
import tensorflow as tf
from transformer import (
    FixedEmbedding,
    PositionalEncoding,
    TransformerLayer,
    masked_loss,
    masked_accuracy,
)
import matplotlib.pyplot as plt


class BatchHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        super(BatchHistory, self).__init__()
        self.batch_losses = []
        self.batch_accuracies = []  # Adjust if you have other metrics

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.batch_losses.append(logs.get("loss"))
        self.batch_accuracies.append(
            logs.get("masked_accuracy")
        )  # Adjust based on your metrics

    def on_epoch_end(self, epoch, logs=None):
        # Here, you can also reset the batch history after each epoch if needed
        # Or save them to a file
        print(
            f"End of epoch {epoch}. Loss: {logs.get('loss')}, masked_accuracy: {logs.get('masked_accuracy')}"
        )


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def embedded_transformer():

    w = np.load(
        path_utils.embedding_path(context),
    )
    # Create a new sequential model
    model = keras.models.Sequential()
    model.add(FixedEmbedding(w, context["SEQ_LEN"]))
    model.add(
        PositionalEncoding(vec_dim=context["DIMENSION"], seq_len=context["SEQ_LEN"])
    )

    for l in range(context["NUM_LAYERS"]):
        model.add(
            TransformerLayer(
                vec_dim=context["DIMENSION"],
                key_dim=32,
                num_heads=context["NUM_HEADS"],
                dff=256,
            )
        )
    model.add(tf.keras.layers.Dense(context["VOCAB_SIZE"], activation="softmax"))

    learning_rate = CustomSchedule(context["DIMENSION"])
    opt = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    model.compile(optimizer=opt, loss=masked_loss, metrics=[masked_accuracy])
    return model


def embedded_fit(retrain=False):

    if os.path.isfile(path_utils.embedding_model_path(context)) and not retrain:
        return
    tokeniser = tok_2_vec.tokenise(path_utils.tokeniser_path(context))
    ids = tok_2_vec.load_encoding(context, path_utils.encoding_path(context), tokeniser)

    print("Loading data generator...")
    train_data = predictTextDataGenerator(
        ids=ids, seq_len=context["SEQ_LEN"], batch_size=32
    )
    batch_history = BatchHistory()
    model = embedded_transformer()
    model.summary()
    train_info = model.fit(
        train_data, epochs=context["EPOCHS"], callbacks=[batch_history]
    )

    model.save_weights(path_utils.embedding_model_path(context))
    history = train_info.history
    with gzip.open(
        path_utils.history_embedding_path(
            context,
        ),
        "w",
    ) as f:
        pickle.dump(history, f)


def predict():
    prompt = "It is a truth universally acknowledged"
    model = embedded_transformer()

    model.load_weights(path_utils.embedding_model_path(context))

    sys.stdout.flush()

    tokeniser = tok_2_vec.tokenise(path_utils.tokeniser_path(context))
    # Encode prompt to tokens
    tokens = tokeniser.encode(prompt)

    print(prompt, end="")

    for i in range(1, 100):
        # Check if prompt is more than seq_len, if so, truncate, grabbing the
        # last seq_len tokens
        if len(tokens) >= context["SEQ_LEN"]:
            tokens = tokens[-context["SEQ_LEN"] :]
        # Index of the last token, which is going to be the
        # index of the output stream that we are going to use for prediction
        j = len(tokens) - 1

        # If the prompt is less than context["SEQ_LEN""], pad it with zeros
        if len(tokens) < context["SEQ_LEN"]:
            x = np.concatenate(
                [tokens, np.zeros((context["SEQ_LEN"] - len(tokens)), dtype="int")],
                axis=0,
            )
        else:
            x = np.array(tokens)

        x = np.expand_dims(x, axis=0)

        # Compute output of the transformer
        y = model.predict(x, verbose="0")
        y = np.argmax(y[:, j, :])

        # Decode the token back to text
        t = tokeniser.decode(y)
        # Print it
        print(t, end="")
        sys.stdout.flush()
        # Apend the token (integer) to the prompot tokens
        tokens.append(y)

    print("\n")


def plot_history():
    history_path = path_utils.history_embedding_path(context)

    if os.path.isfile(history_path):
        with gzip.open(
            history_path, "rb"
        ) as f:  # Ensure you're opening in binary mode for pickle
            history = pickle.load(f)
    else:
        print("History file not found.")
        return  # Exit the function if the file does not exist

    print(history)
    fh = plt.figure()
    ph = fh.add_subplot(111)
    for key in history:
        ph.plot(history[key], label=key)
    ph.set_xlabel("Epoch")
    ph.set_ylabel("Loss")
    ph.set_ylim(
        [min(history["loss"]) - 0.5, max(history["loss"]) + 0.5]
    )  # Dynamically set based on data
    ph.legend(loc="upper right")

    out_path = path_utils.history_embedding_image_path(context)
    print("Saving to: ", out_path)

    plt.show()
    plt.savefig(out_path)  # Save before showing


def main():
    global context
    context["EPOCHS"] = 5

    dimensions = [32, 128, 512]
    context_windows = [3, 4, 10]
    layers = [2, 4, 6]
    heads = [4, 16, 32]

    for d, dim in enumerate(dimensions):
        for c, ctx in enumerate(context_windows):
            for l, layer in enumerate(layers):
                for h, head in enumerate(heads):
                    context["DIMENSION"] = dim
                    context["CONTEXT_WINDOW"] = ctx
                    context["NUM_HEADS"] = head
                    context["NUM_LAYERS"] = layer
                    print(
                        f"|Current context| {d + 1 * c+ 1 * l + 1 * h + 1} of {len(dimensions) * len(context_windows) * len(layers) * len(heads)}\n",
                        context,
                    )
                    embedded_fit()
                    predict()
                    plot_history()


if __name__ == "__main__":
    main()
