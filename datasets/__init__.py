"""Local pretraining data tools.

The package deliberately does not depend on Hugging Face's package with the
same ``datasets`` import name.
"""

from .packed import PackedTokenCorpus

__all__ = ["PackedTokenCorpus"]
