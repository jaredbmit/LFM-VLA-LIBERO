"""Shared constants for VLA training and inference."""

ACTION_TOKEN = "<action>"
SYSTEM_PROMPT = (
    "You are a robot manipulation agent. Given an image of the current scene "
    "and a language instruction, predict the next action to execute."
)

HIDDEN_DIM = 2048
ACTION_DIM = 7
CHUNK_SIZE = 10
MAX_LENGTH = 256
