import path_utils
import tok_2_vec
from context import context
import find_matches
from model_classes import PredictTextDataGenerator, CustomSchedule

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


def embedded_transformer():

    w = np.load(
        path_utils.embedding(context),
    )
    # Create a new sequential model
    model = keras.models.Sequential()
    model.add(FixedEmbedding(w, context["SEQ_LEN"]))
    model.add(
        PositionalEncoding(vec_dim=context["DIMENSION"], seq_len=context["SEQ_LEN"])
    )

    for _ in range(context["NUM_LAYERS"]):
        model.add(
            TransformerLayer(
                vec_dim=context["DIMENSION"],
                key_dim=32,
                num_heads=context["NUM_HEADS"],
                dff=context["DFF"],
            )
        )
    model.add(tf.keras.layers.Dense(context["VOCAB_SIZE"], activation="softmax"))

    learning_rate = CustomSchedule(context["DIMENSION"])
    opt = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    model.compile(optimizer=opt, loss=masked_loss, metrics=[masked_accuracy])
    return model


def embedded_fit(ids, retrain=True):

    if os.path.isfile(path_utils.embedding_model(context)) and not retrain:
        return
    split_size = int(0.8 * len(ids))

    # Split the ids into training and validation sets
    train_ids = ids[:split_size]
    validation_ids = ids[split_size:]
    train_data = PredictTextDataGenerator(
        ids=train_ids, seq_len=context["SEQ_LEN"], batch_size=32
    )
    validation_data = PredictTextDataGenerator(
        ids=validation_ids, seq_len=context["SEQ_LEN"], batch_size=32
    )

    model = embedded_transformer()
    train_info = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=context["EPOCHS"],
    )

    model.save_weights(path_utils.embedding_model(context))
    history = train_info.history
    with gzip.open(
        path_utils.history_embedding(
            context,
        ),
        "w",
    ) as f:
        pickle.dump(history, f)


def predict(prompt):
    model = embedded_transformer()
    # model.summary()
    model.load_weights(path_utils.embedding_model(context))

    tokeniser = tok_2_vec.tokenise(path_utils.tokeniser(context))
    tokens = tokeniser.encode(prompt)

    output = prompt
    print(prompt, end="")

    for _ in range(1, 400):
        if len(tokens) >= context["SEQ_LEN"]:
            tokens = tokens[-context["SEQ_LEN"] :]
        j = len(tokens) - 1

        if len(tokens) < context["SEQ_LEN"]:
            x = np.concatenate(
                [tokens, np.zeros((context["SEQ_LEN"] - len(tokens)), dtype="int")],
                axis=0,
            )
        else:
            x = np.array(tokens)
        x = np.expand_dims(x, axis=0)
        y = model.predict(x, verbose="0")
        y = np.argmax(y[:, j, :])

        t = tokeniser.decode(y)
        output += t
        print(t, end="")
        sys.stdout.flush()
        tokens.append(y)

    print("\n")
    return output


def main():
    global context
    war_and_peace = "You can love a person dear to you with a human love"
    # , but an enemy can only be loved with divine
    # love."
    pride_and_predjudice = "It is a truth universally acknowledged"
    check_the_logs = "check the logs"

    layers = [6]
    heads = [16]
    dffs = [512]
    tokeniser = tok_2_vec.tokenise(path_utils.tokeniser(context))
    ids = tok_2_vec.load_encoding(context, path_utils.encoding(context), tokeniser)
    outputs = []

    index = 1
    for layer in layers:
        for head in heads:
            for diff in dffs:
                context["NUM_HEADS"] = head
                context["NUM_LAYERS"] = layer
                context["DFF"] = diff
                print(
                    f"|Current context| {index} of {len(layers) * len(heads) * len(dffs)}\n",
                    context,
                )
                embedded_fit(ids, retrain=False)
                index += 1
                outputs.append({"context": context, "text": predict(war_and_peace)})
                outputs.append(
                    {"context": context, "text": predict(pride_and_predjudice)}
                )
                outputs.append({"context": context, "text": predict(check_the_logs)})

    for output in outputs:
        matches = find_matches.find(output)

        for match in matches.items():
            print(match)


if __name__ == "__main__":
    main()
