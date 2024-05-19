import os
from context_types import ContextConfig


def encoding_path(context: ContextConfig):
    if not os.path.isdir(f"../data/encoding/war_and_peace"):
        os.makedirs(f"../data/encoding/war_and_peace")

    if not os.path.isdir(f"../data/encoding/pride_and_predjudice"):
        os.makedirs(f"../data/encoding/pride_and_predjudice")

    if not os.path.isdir(f"../data/encoding/both"):
        os.makedirs(f"../data/encoding/both")
    return f"../data/encoding/{context['TEXT_TYPE']}/.indexs"


def tokeniser_path(context: ContextConfig):
    if not os.path.isdir(f"../data/tokens/war_and_peace"):
        os.makedirs(f"../data/tokens/war_and_peace")

    if not os.path.isdir(f"../data/tokens/pride_and_predjudice"):
        os.makedirs(f"../data/tokens/pride_and_predjudice")

    if not os.path.isdir(f"../data/tokens/both"):
        os.makedirs(f"../data/tokens/both")
    return f"../data/tokens/{context[ 'TEXT_TYPE' ]}/{context[ 'VOCAB_SIZE' ]}"


def embedding_path(context: ContextConfig):
    if not os.path.isdir(f"../saved/embedding/war_and_peace"):
        os.makedirs(f"../saved/embedding/war_and_peace")

    if not os.path.isdir(f"../saved/embedding/pride_and_predjudice"):
        os.makedirs(f"../saved/embedding/pride_and_predjudice")

    if not os.path.isdir(f"../saved/embedding/both"):
        os.makedirs(f"../saved/embedding/both")

    return f"../saved/embedding/{context[ 'TEXT_TYPE' ]}/{context[ 'DIMENSION' ]}_{context['CONTEXT_WINDOW' ]}_weights.npy"


def vec_model_path(context: ContextConfig):
    if not os.path.isdir(f"../saved/vec_model/war_and_peace"):
        os.makedirs(f"../saved/vec_model/war_and_peace")

    if not os.path.isdir(f"../saved/vec_model/pride_and_predjudice"):
        os.makedirs(f"../saved/vec_model/pride_and_predjudice")

    if not os.path.isdir(f"../saved/vec_model/both"):
        os.makedirs(f"../saved/vec_model/both")

    return f"../saved/embedding_model/{context[ 'TEXT_TYPE' ]}/dim_{context[ 'DIMENSION']}_ctx_{context['CONTEXT_WINDOW' ]}.keras"


def vec_image_path(context: ContextConfig):
    if not os.path.isdir(f"../images/war_and_peace"):
        os.makedirs(f"../images/war_and_peace")

    if not os.path.isdir(f"../images/pride_and_predjudice"):
        os.makedirs(f"../images/pride_and_predjudice")

    if not os.path.isdir(f"../images/both"):
        os.makedirs(f"../images/both")
    return f"../images/{context[ 'TEXT_TYPE' ]}/dim_{context[ 'DIMENSION' ]}_ctx_{context['CONTEXT_WINDOW' ]}_embedding.svg"


def history_path(context: ContextConfig):
    if not os.path.isdir(f"../saved/history/war_and_peace"):
        os.makedirs(f"../saved/history/war_and_peace")

    if not os.path.isdir(f"../saved/history/pride_and_predjudice"):
        os.makedirs(f"../saved/history/pride_and_predjudice")

    if not os.path.isdir(f"../saved/history/both"):
        os.makedirs(f"../saved/history/both")

    return f"../saved/history/{context[ 'TEXT_TYPE' ]}/{context[ 'EPOCHS' ]}_{context[ 'DIMENSION']}.hist"


def history_image_path(context: ContextConfig):
    if not os.path.isdir(f"../images/history/war_and_peace"):
        os.makedirs(f"../images/history/war_and_peace")

    if not os.path.isdir(f"../images/history/pride_and_predjudice"):
        os.makedirs(f"../images/history/pride_and_predjudice")

    if not os.path.isdir(f"../images/history/both"):
        os.makedirs(f"../images/history/both")
    return f"../images/history/{context[ 'TEXT_TYPE' ]}/dim_{context[ 'DIMENSION' ]}_ctx_{context['CONTEXT_WINDOW' ]}_history.png"


def embedding_model_path(context: ContextConfig):
    if not os.path.isdir(f"../saved/embedding_model/war_and_peace"):
        os.makedirs(f"../saved/embedding_model/war_and_peace")

    if not os.path.isdir(f"../saved/embedding_model/pride_and_predjudice"):
        os.makedirs(f"../saved/embedding_model/pride_and_predjudice")

    if not os.path.isdir(f"../saved/embedding_model/both"):
        os.makedirs(f"../saved/embedding_model/both")

    return f"../saved/embedding_model/{context[ 'TEXT_TYPE']}/dim_{context['DIMENSION']}_layers_{context['NUM_LAYERS']}_seq_{context['SEQ_LEN']}_hds_{context['NUM_HEADS']}_ep_{context['EPOCHS' ]}.keras"


def history_embedding_path(context: ContextConfig):
    if not os.path.isdir(f"../saved/history_embedding/war_and_peace"):
        os.makedirs(f"../saved/history_embedding/war_and_peace")

    if not os.path.isdir(f"../saved/history_embedding/pride_and_predjudice"):
        os.makedirs(f"../saved/history_embedding/pride_and_predjudice")

    if not os.path.isdir(f"../saved/history_embedding/both"):
        os.makedirs(f"../saved/history_embedding/both")

    return f"../images/history_embedding/{context[ 'TEXT_TYPE']}/dim_{context['DIMENSION']}_layers_{context['NUM_LAYERS']}_seq_{context['SEQ_LEN']}_hds_{context['NUM_HEADS']}_ep_{context['EPOCHS']}.hist"


def history_embedding_image_path(context: ContextConfig):
    if not os.path.isdir(f"../images/history_embedding/war_and_peace"):
        os.makedirs(f"../images/history_embedding/war_and_peace")

    if not os.path.isdir(f"../images/history_embedding/pride_and_predjudice"):
        os.makedirs(f"../images/history_embedding/pride_and_predjudice")

    if not os.path.isdir(f"../images/history_embedding/both"):
        os.makedirs(f"../images/history_embedding/both")
    return f"../images/history_embedding/{context[ 'TEXT_TYPE' ]}/dim_{context['DIMENSION']}_layers_{context['NUM_LAYERS']}_seq_{context['SEQ_LEN']}_hds_{context['NUM_HEADS']}_ep_{context['EPOCHS' ]}_history.png"


def one_hot_model_path(context: ContextConfig):
    if not os.path.isdir(f"../saved/one_hot_model/war_and_peace"):
        os.makedirs(f"../saved/one_hot_model/war_and_peace")

    if not os.path.isdir(f"../saved/one_hot_model/pride_and_predjudice"):
        os.makedirs(f"../saved/one_hot_model/pride_and_predjudice")

    if not os.path.isdir(f"../saved/one_hot_model/both"):
        os.makedirs(f"../saved/one_hot_model/both")

    return f"../saved/one_hot_model/{context[ 'TEXT_TYPE']}/layers_{context['NUM_LAYERS']}_seq_{context['SEQ_LEN']}_hds_{context['NUM_HEADS']}_ep_{context['EPOCHS' ]}.keras"


def history_one_hot_path(context: ContextConfig):
    if not os.path.isdir(f"../saved/history_one_hot/war_and_peace"):
        os.makedirs(f"../saved/history_one_hot/war_and_peace")

    if not os.path.isdir(f"../saved/history_one_hot/pride_and_predjudice"):
        os.makedirs(f"../saved/history_one_hot/pride_and_predjudice")

    if not os.path.isdir(f"../saved/history_one_hot/both"):
        os.makedirs(f"../saved/history_one_hot/both")

    return f"../saved/one_hot_model/{context['TEXT_TYPE']}/layers_{context['NUM_LAYERS']}_seq_{context['SEQ_LEN']}_hds_{context['NUM_HEADS']}_ep_{context['EPOCHS']}.hist"


def history_one_hot_image_path(context: ContextConfig):
    if not os.path.isdir(f"../images/history_one_hot/war_and_peace"):
        os.makedirs(f"../images/history_one_hot/war_and_peace")

    if not os.path.isdir(f"../images/history_one_hot/pride_and_predjudice"):
        os.makedirs(f"../images/history_one_hot/pride_and_predjudice")

    if not os.path.isdir(f"../images/history_one_hot/both"):
        os.makedirs(f"../images/history_one_hot/both")
    return f"../images/history_one_hot/{context['TEXT_TYPE']}/layers_{context['NUM_LAYERS']}_seq_{context['SEQ_LEN']}_hds_{context['NUM_HEADS']}_ep_{context['EPOCHS']}_history.png"
