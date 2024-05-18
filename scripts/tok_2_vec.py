import load_text
from tokeniser import Tokeniser, plot_tok2vec
import pickle
import os
import numpy as np
from tqdm import tqdm

from keras.models import Model
from keras.layers import Input, Dense, Activation

context = {
    "DIMENSION": 32,
    "EPOCHS": 1,
    "VOCAB_SIZE": 5000,
    "CONTEXT_WINDOW": 5,
    "TEXT_TYPE": "both",
}


def one_hot_skipgram_model(vocab_size, embedding_dim):
    # Inputs are one-hot encoded vectors
    input_target = Input(shape=(vocab_size,), dtype="int8", name="target_input")
    input_context = Input(shape=(vocab_size,), dtype="int8", name="context_input")

    # Shared embedding layer for target and context
    shared_embedding = Dense(
        embedding_dim, activation="linear", use_bias=False, name="shared_embedding"
    )

    # Get embeddings for target and context
    target_embedding = shared_embedding(input_target)
    context_embedding = shared_embedding(input_context)

    # Compute logits using a dot product over all words in the vocabulary using another Dense layer
    # This Dense layer effectively replaces the manual dot product with weight matrix transposition
    prediction_layer = Dense(vocab_size, activation="linear", use_bias=False)
    target_logits = prediction_layer(target_embedding)

    # Apply softmax to logits to get a probability distribution
    output = Activation("softmax")(target_logits)

    # Complete model
    model = Model(inputs=[input_target, input_context], outputs=output)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def skipgram_generator(pairs, vocab_size, batch_size=1024):
    """Generate batches of skip-gram target, context pairs with one-hot encoding."""
    while True:
        np.random.shuffle(pairs)

        # Initialize batch data
        batch_target = np.zeros((batch_size, vocab_size), dtype=np.int8)
        batch_context = np.zeros((batch_size, vocab_size), dtype=np.int8)
        batch_labels = np.zeros((batch_size, vocab_size), dtype=np.int8)

        current_batch_index = 0
        for target_index, context_index in pairs:
            # One-hot encode the target and context words
            batch_target[current_batch_index] = to_one_hot(target_index, vocab_size)
            batch_context[current_batch_index] = to_one_hot(context_index, vocab_size)
            batch_labels[current_batch_index] = to_one_hot(context_index, vocab_size)

            current_batch_index += 1
            if current_batch_index == batch_size:
                yield [batch_target, batch_context], batch_labels
                # Reset batch data
                batch_target = np.zeros((batch_size, vocab_size), dtype=np.int8)
                batch_context = np.zeros((batch_size, vocab_size), dtype=np.int8)
                batch_labels = np.zeros((batch_size, vocab_size), dtype=np.int8)
                current_batch_index = 0


def to_one_hot(index, vocab_size):
    one_hot = np.zeros(vocab_size)
    one_hot[index] = 1
    return one_hot


def make_text(text_type):
    match text_type:
        case "both":
            return (
                load_text.load_warandpeace() + "" + load_text.load_prideandprejudice()
            )
        case "pride":
            return load_text.load_prideandprejudice()
        case "war":
            return load_text.load_warandpeace()


def tokenize(path, text_type="both"):
    tokeniser = Tokeniser(context["VOCAB_SIZE"])

    if os.path.isfile(path):
        print("loading tokeniser from file")
        tokeniser = tokeniser.load(path)
        return tokeniser

    tokeniser.train(text=make_text(text_type))
    tokeniser.save(path)

    return tokeniser


def tokenize_intervals(text_type="both"):
    for value in [500, 1000, 5000]:
        tokenize(f"../data/tokens/{text_type}/{value}")


def encoding_path(text_type):
    if not os.path.isdir(f"../data/encoding/war_and_peace"):
        os.makedirs(f"../data/encoding/war_and_peace")

    if not os.path.isdir(f"../data/encoding/pride_and_predjudice"):
        os.makedirs(f"../data/encoding/pride_and_predjudice")

    if not os.path.isdir(f"../data/encoding/both"):
        os.makedirs(f"../data/encoding/both")
    return f"../data/encoding/{text_type}/.indexs"


def tokenizer_path(text_type, value):
    if not os.path.isdir(f"../data/tokens/war_and_peace"):
        os.makedirs(f"../data/tokens/war_and_peace")

    if not os.path.isdir(f"../data/tokens/pride_and_predjudice"):
        os.makedirs(f"../data/tokens/pride_and_predjudice")

    if not os.path.isdir(f"../data/tokens/both"):
        os.makedirs(f"../data/tokens/both")
    return f"../data/tokens/{text_type}/{value}.json"


def embedding_path(text_type, epochs, dimension):
    if not os.path.isdir(f"../saved/embedding/war_and_peace"):
        os.makedirs(f"../saved/embedding/war_and_peace")

    if not os.path.isdir(f"../saved/embedding/pride_and_predjudice"):
        os.makedirs(f"../saved/embedding/pride_and_predjudice")

    if not os.path.isdir(f"../saved/embedding/both"):
        os.makedirs(f"../saved/embedding/both")

    return f"../saved/embedding/{text_type}/{epochs}_{dimension}_weights.npy"


def serialize_encodings(indexs, path):
    print(f"writing to file {path}....")

    with open(path, "wb") as file:
        pickle.dump(indexs, file)


def load_encoding(text_type, path, tokenizer):

    if os.path.isfile(path):
        print("loading encoding from file")
        with open(path, "rb") as file:
            return pickle.load(file)

    print("calculating encodings...")
    indexs = tokenizer.encode(make_text(text_type), verbose=True)
    serialize_encodings(indexs, path)

    return indexs


def generate_skipgram_pairs_from_encodings(indexs, window_size=5):
    pairs = []
    for i, target_word in tqdm(enumerate(indexs)):
        context_indices = range(
            max(i - window_size, 0), min(i + window_size + 1, len(indexs))
        )
        for j in context_indices:

            if i != j:
                context_word = indexs[j]
                pairs.append((target_word, context_word))
    return pairs


def fit_skipgram(context_indices):
    skipgram_pairs = generate_skipgram_pairs_from_encodings(
        context_indices, context["CONTEXT_WINDOW"]
    )

    model = one_hot_skipgram_model(
        context["vocab_size"], embedding_dim=context["DIMENSION"]
    )
    model.summary()

    num_pairs = len(skipgram_pairs)
    batch_size = 2048
    steps_per_epoch = num_pairs // batch_size

    model.fit(
        skipgram_generator(skipgram_pairs, context["VOCAB_SIZE"], batch_size),
        steps_per_epoch=steps_per_epoch,
        epochs=context["EPOCHS"],
    )

    embedding_weights = model.get_layer("shared_embedding").get_weights()[0]
    np.save(
        embedding_path(context["TEXT_TYPE"], context["EPOCHS"], context["DIMENSION"]),
        embedding_weights,
    )

    plot_vec()


def plot_vec():
    tokenizer = tokenize(tokenizer_path(context["TEXT_TYPE"], context["VOCAB_SIZE"]))
    embedding_weights = np.load(
        embedding_path(context["TEXT_TYPE"], context["EPOCHS"], context["DIMENSION"])
    )
    plot_tok2vec(embedding_weights, tokenizer.word_index, num_words_to_show=200)


def main():
    global context
    vocab_size = context["VOCAB_SIZE"]

    # Create necessary folders
    if not os.path.isdir("../saved"):
        os.mkdir("../saved")

    tokenizer = tokenize(tokenizer_path(context["TEXT_TYPE"], vocab_size))

    context_indices = load_encoding(
        context["TEXT_TYPE"], encoding_path(context["TEXT_TYPE"]), tokenizer
    )
    plot_vec()


if __name__ == "__main__":
    main()

# plot_tok2vec(w, word_index, num_words_to_show=100, show_word_tokens=[]):
# tokenize_intervals("both")


# model = token_2_vec(tokeniser.vocab_size)
