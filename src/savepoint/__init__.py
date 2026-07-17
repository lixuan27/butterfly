"""SavePoint: a save button for neural games.

Serialize the emergent game state of a streaming world model — KV-cache,
rolling latent context, sampler RNG — into a portable ``.wsave`` file, and
restore it bit-for-bit to save / load / rewind / fork living neural worlds.
"""

from .state import WorldState, load_wsave, save_wsave

__version__ = "0.0.1"
__all__ = ["WorldState", "save_wsave", "load_wsave", "__version__"]
