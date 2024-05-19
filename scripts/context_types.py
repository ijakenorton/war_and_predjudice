from typing import TypedDict


class ContextConfig(TypedDict):
    DIMENSION: int
    EPOCHS: int
    VOCAB_SIZE: int
    CONTEXT_WINDOW: int
    SEQ_LEN: int
    TEXT_TYPE: str
    NUM_HEADS: int
    NUM_LAYERS: int
