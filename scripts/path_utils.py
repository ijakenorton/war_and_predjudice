import os
from context_types import ContextConfig


def encoding(context: ContextConfig):

    if not os.path.isdir(f"../data/encoding/both"):
        os.makedirs(f"../data/encoding/both")
    return f"../data/encoding/{context['TEXT_TYPE']}/.indexs"


def tokeniser(context: ContextConfig):

    if not os.path.isdir(f"../data/tokens/both"):
        os.makedirs(f"../data/tokens/both")
    return f"../data/tokens/{context[ 'TEXT_TYPE' ]}/{context[ 'VOCAB_SIZE' ]}"


def embedding(context: ContextConfig):

    if not os.path.isdir(f"../saved/embedding/both"):
        os.makedirs(f"../saved/embedding/both")

    return f"../saved/embedding/{context[ 'TEXT_TYPE' ]}/{context[ 'DIMENSION' ]}_{context['CONTEXT_WINDOW' ]}_weights.npy"


def vec_model(context: ContextConfig):

    if not os.path.isdir(f"../saved/vec_model/both"):
        os.makedirs(f"../saved/vec_model/both")

    return f"../saved/vec_model/{context[ 'TEXT_TYPE' ]}/dim_{context[ 'DIMENSION']}_ctx_{context['CONTEXT_WINDOW' ]}.keras"


def vec_image(context: ContextConfig):

    if not os.path.isdir(f"../images/both"):
        os.makedirs(f"../images/both")
    return f"../images/{context[ 'TEXT_TYPE' ]}/dim_{context[ 'DIMENSION' ]}_ctx_{context['CONTEXT_WINDOW' ]}_embedding.svg"


def history(context: ContextConfig):

    if not os.path.isdir(f"../saved/history/both"):
        os.makedirs(f"../saved/history/both")

    return f"../saved/history/{context[ 'TEXT_TYPE' ]}/{context[ 'EPOCHS' ]}_{context[ 'DIMENSION']}.hist"


def history_image(context: ContextConfig):

    if not os.path.isdir(f"../images/history/both"):
        os.makedirs(f"../images/history/both")
    return f"../images/history/{context[ 'TEXT_TYPE' ]}/dim_{context[ 'DIMENSION' ]}_ctx_{context['CONTEXT_WINDOW' ]}_history.png"


def embedding_model(context: ContextConfig):

    if not os.path.isdir(f"../saved/embedding_model/both"):
        os.makedirs(f"../saved/embedding_model/both")

    return f"../saved/embedding_model/{context['TEXT_TYPE']}/dim_{context['DIMENSION']}_layers_{context['NUM_LAYERS']}_seq_{context['SEQ_LEN']}_hds_{context['NUM_HEADS']}_dff_{context['DFF' ]}.keras"


def history_embedding(context: ContextConfig):

    if not os.path.isdir(f"../saved/history_embedding/both"):
        os.makedirs(f"../saved/history_embedding/both")

    return f"../images/history_embedding/{context['TEXT_TYPE']}/dim_{context['DIMENSION']}_layers_{context['NUM_LAYERS']}_seq_{context['SEQ_LEN']}_hds_{context['NUM_HEADS']}_dff_{context['DFF']}.hist"


def history_embedding_image(context: ContextConfig):

    if not os.path.isdir(f"../images/history_embedding/both"):
        os.makedirs(f"../images/history_embedding/both")
    return f"../images/history_embedding/{context[ 'TEXT_TYPE']}/dim_{context['DIMENSION']}_layers_{context['NUM_LAYERS']}_seq_{context['SEQ_LEN']}_hds_{context['NUM_HEADS']}_dff_{context['DFF' ]}_history.png"


def one_hot_model(context: ContextConfig):

    if not os.path.isdir(f"../saved/one_hot_model/both"):
        os.makedirs(f"../saved/one_hot_model/both")

    return f"../saved/one_hot_model/{context['TEXT_TYPE']}/layers_{context['NUM_LAYERS']}_seq_{context['SEQ_LEN']}_hds_{context['NUM_HEADS']}_dff_{context['DFF']}.keras"


def history_one_hot(context: ContextConfig):

    if not os.path.isdir(f"../saved/history_one_hot/both"):
        os.makedirs(f"../saved/history_one_hot/both")

    return f"../saved/one_hot_model/{context['TEXT_TYPE']}/layers_{context['NUM_LAYERS']}_seq_{context['SEQ_LEN']}_hds_{context['NUM_HEADS']}_dff_{context['DFF']}.hist"
