# public interface
from .Regex import Regex


def match(pattern, string):
    """Try to apply the pattern at the start of the string, returning true
    or None if no match was found."""
    return Regex(pattern).match(string)


def fullmatch(pattern, string):
    """Try to apply the pattern to all of the string, returning true
    or None if no match was found."""
    return Regex(pattern).fullmatch(string)


def search(pattern, string):
    """Scan through string looking for a match to the pattern, returning true
    or None if no match was found."""
    return Regex(pattern).search(string)


def compile(pattern):
    "Compile a regular expression pattern, returning a pattern object."
    return Regex(pattern)
