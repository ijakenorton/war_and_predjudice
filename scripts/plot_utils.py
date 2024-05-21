import numpy as np
import matplotlib.pyplot as plt
import gzip
import os
import pickle
import path_utils
import tok_2_vec
from context_types import ContextConfig
from sklearn.manifold import TSNE


def plot_tok_history(context: ContextConfig):
    history_path = path_utils.history(context)

    if os.path.isfile(history_path):
        with gzip.open(
            history_path, "rb"
        ) as f:  # Ensure you're opening in binary mode for pickle
            history = pickle.load(f)
    else:
        print("History file not found.")
        return  # Exit the function if the file does not exist

    fh = plt.figure()
    ph = fh.add_subplot(111)
    ph.plot(history["loss"], label="Loss")
    ph.set_xlabel("Epoch")
    ph.set_ylabel("Loss")
    ph.set_ylim(
        [min(history["loss"]) - 0.5, max(history["loss"]) + 0.5]
    )  # Dynamically set based on data
    ph.legend(loc="upper right")

    out_path = path_utils.history_image(context)
    print("Saving to: ", out_path)

    plt.savefig(out_path)  # Save before showing


def plot_vec(context: ContextConfig):
    tokeniser = tok_2_vec.tokenise(path_utils.tokeniser(context))
    embedding_weights = np.load(
        path_utils.embedding(context),
    )

    def plot_tok2vec(w, word_index, num_words_to_show=100, show_word_tokens=[]):
        """
        Plot the word vectors in 2D using t-SNE.  This method is dependent on nltk library (to select words to show).
        You can install the library with 'pip install nltk'

        param w: word vectors, numpy array of shape (vocab_size, vec_dim)
        param word_index: tokeniser word_index, a python dictionary that maps vocab_size words to integer ids
        param num_words_to_show: number of words to show in the plot (default 100)
        param show_word_tokens: list of tokens to show in the plot, if empty, the method will select up to 100 English words from the dictionary

        """

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
        plt.savefig(path_utils.vec_image(context))

    plot_tok2vec(embedding_weights, tokeniser.word_index, num_words_to_show=200)
