"""Host adapters: one thin wrapper per streaming world model family.

A host owns the streaming loop (cursor, action history, conditioning) and
exposes the uniform interface SaveBench and the demo server consume.
"""

from .base import StreamingHost

__all__ = ["StreamingHost"]
