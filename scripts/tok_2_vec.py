import load_text
from tokeniser import Tokeniser, plot_tok2vec
import pickle
import os
import gzip
import numpy as np
from tqdm import tqdm
import path_utils
import matplotlib.pyplot as plt
from context_types import ContextConfig
from keras.models import Model
from keras.layers import Input, Dense, Activation
from context import context


def plot_history():
    history_path = path_utils.history_path(context)

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
    ph.plot(history["loss"], label="Loss")
    ph.set_xlabel("Epoch")
    ph.set_ylabel("Loss")
    ph.set_ylim(
        [min(history["loss"]) - 0.5, max(history["loss"]) + 0.5]
    )  # Dynamically set based on data
    ph.legend(loc="upper right")

    out_path = path_utils.history_image_path(context)
    print("Saving to: ", out_path)

    plt.savefig(out_path)  # Save before showing


def plot_vec():
    tokeniser = tokenise(path_utils.tokeniser_path(context))
    embedding_weights = np.load(
        path_utils.embedding_path(context),
    )

    plot_tok2vec(embedding_weights, tokeniser.word_index, num_words_to_show=200)


def plot_tok2vec(w, word_index, num_words_to_show=100, show_word_tokens=[]):
    """
    Plot the word vectors in 2D using t-SNE.  This method is dependent on nltk library (to select words to show).
    You can install the library with 'pip install nltk'

    param w: word vectors, numpy array of shape (vocab_size, vec_dim)
    param word_index: tokeniser word_index, a python dictionary that maps vocab_size words to integer ids
    param num_words_to_show: number of words to show in the plot (default 100)
    param show_word_tokens: list of tokens to show in the plot, if empty, the method will select up to 100 English words from the dictionary

    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    if len(show_word_tokens) == 0:
        try:
            import nltk
        except:
            raise RuntimeError(
                "Error: nltk is not installed. Please install nltk using 'pip install nltk'"
            )

        # Select up to 100 words from the tokens that are in the words dictionary
        try:
            nltk.data.find("corpora/words")
        except LookupError:
            nltk.download("words")

        english_dictionary = nltk.corpus.words.words()
        vocab_size = len(word_index)
        show_word_tokens = []
        words = list(word_index.keys())
        for i in range(vocab_size):
            word = words[i]
            if word[0] == " ":
                if len(word) > 4 and word[1:] in english_dictionary:
                    show_word_tokens += [i]
            else:
                if len(word) > 3 and word in english_dictionary:
                    show_word_tokens += [i]

            if len(show_word_tokens) >= num_words_to_show:
                break

    # Convert the tokens to vectors
    x = w[show_word_tokens]

    # Do dimension reduction from vec_dim to 2 (so we can plot the word points in 2D)
    tsne = TSNE(n_components=2, random_state=0)
    x_2d = tsne.fit_transform(x)

    # Plot vectors showing the first 100 tokens with text close to the point corresponding to the token
    plt.figure(figsize=(6, 5))
    for i in range(len(show_word_tokens)):
        plt.scatter(x_2d[i, 0], x_2d[i, 1], c="b")
        word = words[show_word_tokens[i]]
        plt.text(x_2d[i, 0], x_2d[i, 1], word, fontsize=9)
    plt.savefig(path_utils.vec_image_path(context))


def one_hot_skipgram_model(context):
    # Inputs are one-hot encoded vectors
    input_target = Input(
        shape=(context["VOCAB_SIZE"],), dtype="int8", name="target_input"
    )
    input_context = Input(
        shape=(context["VOCAB_SIZE"],), dtype="int8", name="context_input"
    )

    # Shared embedding layer for target and context
    shared_embedding = Dense(
        context["DIMENSION"],
        activation="linear",
        use_bias=False,
        name="shared_embedding",
    )

    # Get embeddings for target and context
    target_embedding = shared_embedding(input_target)

    # Compute logits using a dot product over all words in the vocabulary using another Dense layer
    # This Dense layer effectively replaces the manual dot product with weight matrix transposition
    prediction_layer = Dense(context["VOCAB_SIZE"], activation="linear", use_bias=False)
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


def make_text(context):
    match context["TEXT_TYPE"]:
        case "both":
            return (
                load_text.load_warandpeace() + "" + load_text.load_prideandprejudice()
            )
        case "pride":
            return load_text.load_prideandprejudice()
        case "war":
            return load_text.load_warandpeace()


def tokenise(path, text_type="both"):
    tokeniser = Tokeniser(context["VOCAB_SIZE"])

    if os.path.isfile(path):

        print(f"loading tokeniser from {path}")
        tokeniser = tokeniser.load(path)
        return tokeniser

    tokeniser.train(text=make_text(text_type))
    tokeniser.save(path)

    return tokeniser


def tokenize_intervals(text_type="both"):
    for value in [500, 1000, 5000]:
        tokenise(f"../data/tokens/{text_type}/{value}")


def serialize_encodings(indexs, path):
    print(f"writing to file {path}....")

    with open(path, "wb") as file:
        pickle.dump(indexs, file)


def load_encoding(context, path, tokeniser):

    if os.path.isfile(path):
        print(f"loading encoding from {path}")
        with open(path, "rb") as file:
            return pickle.load(file)

    print("calculating encodings...")
    indexs = tokeniser.encode(make_text(context), verbose=True)
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


def fit_skipgram(context_indices, retrain=False):
    if (
        os.path.isfile(path_utils.embedding_path(context))
        and os.path.isfile(path_utils.vec_model_path(context))
        and not retrain
    ):
        return

    skipgram_pairs = generate_skipgram_pairs_from_encodings(
        context_indices, context["CONTEXT_WINDOW"]
    )

    model = one_hot_skipgram_model(context)
    model.summary()

    num_pairs = len(skipgram_pairs)
    batch_size = 2048
    steps_per_epoch = num_pairs // batch_size

    train_info = model.fit(
        skipgram_generator(skipgram_pairs, context["VOCAB_SIZE"], batch_size),
        steps_per_epoch=steps_per_epoch,
        epochs=context["EPOCHS"],
    )

    embedding_weights = model.get_layer("shared_embedding").get_weights()[0]
    np.save(
        path_utils.embedding_path(context),
        embedding_weights,
    )

    history = train_info.history
    with gzip.open(
        path_utils.history_path(
            context,
        ),
        "w",
    ) as f:
        pickle.dump(history, f)

    model.save_weights(path_utils.vec_model_path(context))


def predict_n_words(target, tokens, n=10):
    tokeniser = tokenise(path_utils.tokeniser_path(context))
    model = one_hot_skipgram_model(context)
    print(f"loading model from {path_utils.vec_model_path(context)}...")
    model.load_weights(path_utils.vec_model_path(context))
    # Check if the token is in the tokeniser's vocabulary

    token_indexs = tokeniser.encode(tokens)
    print(f"token_indexs: {token_indexs}")
    target = tokeniser.encode(target)
    print(f"target: {target}")
    context_vector = np.zeros((1, context["VOCAB_SIZE"]))
    for ctx_token in token_indexs:
        if ctx_token in tokeniser.word_index:
            ctx_index = tokeniser.word_index[ctx_token]
            context_vector[0, ctx_index] = 1
    # Prepare the one-hot encoded input vector for the target token
    target_vector = np.zeros((1, context["VOCAB_SIZE"]))
    target_vector[0, target] = 1

    # Create a dummy context vector (same size, all zeros)

    # Predict using the model
    predictions = model.predict([target_vector, context_vector])[0]

    # Get the indices of the top n values
    top_n_indices = np.argsort(predictions)[-n:][::-1]

    # Decode indices to tokens
    top_n_tokens = [tokeniser.decode(i) for i in top_n_indices]
    top_n_scores = [predictions[i] for i in top_n_indices]

    # Print or return the top n words and their probabilities
    for token, score in zip(top_n_tokens, top_n_scores):
        print(f"{token}: {score:.4f}")

    return top_n_tokens, top_n_scores


def main():
    global context

    dimensions = [32, 64, 128, 256, 512]
    context_windows = [1, 2, 3, 4, 10]

    # dimensions = [32, 512]
    # context_windows = [3, 4, 10]
    tokeniser = tokenise(path_utils.tokeniser_path(context))
    context_indices = load_encoding(
        context, path_utils.encoding_path(context), tokeniser
    )
    for i, dim in enumerate(dimensions):
        for j, ctx in enumerate(context_windows):
            context["DIMENSION"] = dim
            context["CONTEXT_WINDOW"] = ctx
            print(
                f"|Current context| {i * j} of {len(dimensions) * len(context_windows)}\n",
                context,
            )
            fit_skipgram(context_indices, True)
            plot_vec()
            plot_history()
            predict_n_words("universally", "it is a truth acknowledged", 20)

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
