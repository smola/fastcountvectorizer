from .compat import CountVectorizer
from .fastcountvectorizer import FastCountVectorizer
from .version import version

__all__ = ["CountVectorizer", "FastCountVectorizer"]
__version__ = version
