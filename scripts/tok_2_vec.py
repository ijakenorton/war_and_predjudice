import load_text
from tokeniser import Tokeniser, plot_tok2vec
import pickle
import os
import gzip
import numpy as np
from tqdm import tqdm
import path_utils
import plot_utils
from context_types import ContextConfig
from keras.models import Model
from keras.layers import Input, Dense, Activation
from context import context


def one_hot_skipgram_model(context):
    input_target = Input(
        shape=(context["VOCAB_SIZE"],), dtype="int8", name="target_input"
    )
    prediction_layer = Dense(context["DIMENSION"], activation="linear", use_bias=False)(
        input_target
    )
    output = Dense(context["VOCAB_SIZE"], activation="softmax")(prediction_layer)
    model = Model(inputs=input_target, outputs=output)

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def generate_skipgram_pairs_from_encodings(indexs, window_size=5):
    pairs = []
    for i in range(window_size, len(indexs) - window_size - 1):
        context_indices = range(i - window_size, i + window_size)
        target_word = indexs[i]
        for j in context_indices:
            if i != j:
                context_word = indexs[j]
                pairs.append((target_word, context_word))
    return pairs


def skipgram_generator(pairs, vocab_size, batch_size=1024):
    """Generate batches of skip-gram target, context pairs with one-hot encoding."""
    while True:
        np.random.shuffle(pairs)

        # Initialize batch data
        batch_target = np.zeros((batch_size, vocab_size), dtype=np.int8)
        batch_labels = np.zeros((batch_size, vocab_size), dtype=np.int8)

        current_batch_index = 0
        for target_index, context_index in pairs:
            # One-hot encode the target and context words
            batch_target[current_batch_index] = to_one_hot(target_index, vocab_size)
            batch_labels[current_batch_index] = to_one_hot(context_index, vocab_size)

            current_batch_index += 1
            if current_batch_index == batch_size:
                yield [batch_target], batch_labels
                # Reset batch data
                batch_target = np.zeros((batch_size, vocab_size), dtype=np.int8)
                batch_labels = np.zeros((batch_size, vocab_size), dtype=np.int8)
                current_batch_index = 0


def to_one_hot(index, vocab_size):
    one_hot = np.zeros(vocab_size)
    one_hot[index] = 1
    return one_hot


def make_text():
    return load_text.load_warandpeace() + "" + load_text.load_prideandprejudice()


def tokenise(path):
    tokeniser = Tokeniser(context["VOCAB_SIZE"])

    if os.path.isfile(path):
        tokeniser = tokeniser.load(path)
        return tokeniser

    tokeniser.train(text=make_text())
    tokeniser.save(path)

    return tokeniser


def tokenize_intervals(text_type="both"):
    for value in [500, 1000, 5000]:
        tokenise(f"../data/tokens/{text_type}/{value}")


def serialize_encodings(indexs, path):
    # print(f"writing to file {path}....")

    with open(path, "wb") as file:
        pickle.dump(indexs, file)


def load_encoding(context, path, tokeniser):

    if os.path.isfile(path):
        print(f"loading encoding from {path}")
        with open(path, "rb") as file:
            return pickle.load(file)

    indexs = tokeniser.encode(make_text())

    return indexs


def fit_skipgram(context_indices, retrain=False):
    if (
        os.path.isfile(path_utils.embedding(context))
        and os.path.isfile(path_utils.vec_model(context))
        and not retrain
    ):
        return

    skipgram_pairs = generate_skipgram_pairs_from_encodings(
        context_indices, context["CONTEXT_WINDOW"]
    )

    model = one_hot_skipgram_model(context)
    model.summary()

    num_pairs = len(skipgram_pairs)
    batch_size = 4096 * 2
    steps_per_epoch = num_pairs // batch_size

    train_info = model.fit(
        skipgram_generator(skipgram_pairs, context["VOCAB_SIZE"], batch_size),
        steps_per_epoch=steps_per_epoch,
        epochs=context["EPOCHS"],
    )

    embedding_weights = list(model.layers)[1].get_weights()[0]
    np.save(
        path_utils.embedding(context),
        embedding_weights,
    )

    history = train_info.history
    with gzip.open(
        path_utils.history(
            context,
        ),
        "w",
    ) as f:
        pickle.dump(history, f)

    model.save_weights(path_utils.vec_model(context))


def predict_n_words(target, n=10):
    tokeniser = tokenise(path_utils.tokeniser(context))
    model = one_hot_skipgram_model(context)
    print(f"loading model from {path_utils.vec_model(context)}...")
    model.load_weights(path_utils.vec_model(context))

    target_index = tokeniser.encode(target)
    print(f"target: {target}")
    target_vector = np.zeros((1, context["VOCAB_SIZE"]))
    target_vector[0, target_index] = 1
    predictions = model.predict([target_vector])[0]
    top_n_indices = np.argsort(predictions)[-n:][::-1]
    top_n_tokens = [tokeniser.decode(i) for i in top_n_indices]
    top_n_scores = [predictions[i] for i in top_n_indices]

    for token, score in zip(top_n_tokens, top_n_scores):
        print(f"{token}: {score:.4f}")

    return top_n_tokens, top_n_scores


def main():
    global context

    # dimensions = [64, 128, 256, 512]
    dimensions = [32]
    context_windows = [3, 4, 5]

    context["EPOCHS"] = 5

    tokeniser = tokenise(path_utils.tokeniser(context))
    context_indices = load_encoding(context, path_utils.encoding(context), tokeniser)
    index = 1
    for dim in dimensions:
        for ctx in context_windows:
            context["DIMENSION"] = dim
            context["CONTEXT_WINDOW"] = ctx
            print(
                f"|Current context| {index} of {len(dimensions) * len(context_windows)}\n",
                context,
            )
            # fit_skipgram(context_indices, True)
            # plot_utils.plot_vec(context)
            plot_utils.plot_tok_history(context)
            # predict_n_words("th", 20)

    # for i, dim in enumerate(dimensions):
    #     for j, ctx in enumerate(context_windows):
    #         context["DIMENSION"] = dim
    #         context["CONTEXT_WINDOW"] = ctx
    #         print(
    #             f"|Current context| {i * j} of {len(dimensions) * len(context_windows)}\n",
    #             context,
    #         )
    #         fit_skipgram(context_indices)
    #         plot_vec()
    #         plot_history()
    # vocab_size = context["VOCAB_SIZE"]

    # # Create necessary folders
    # if not os.path.isdir("../saved"):
    #     os.mkdir("../saved")

    # plot_vec()


if __name__ == "__main__":
    main()
