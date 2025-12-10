import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from .BPEimpl import _run_bpe_trainning

__all__ =["_run_bpe_trainning"]