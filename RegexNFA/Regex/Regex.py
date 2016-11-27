from .NFA import NFA


class Regex:
    """Regular Expression Objects
    """
    def __init__(self, pattern=''):
        self._pattern = pattern
        self._nfa = None

    def search(self, string='', *pos):
        """Scan through string looking for the first location
        where this regular expression produces a match, and return True if find match
        Return None if no position in the string matches the pattern;
        note that this is different from finding a zero-length match
        at some point in the string.
        """
        self._nfa = self.__get_nfa(self._pattern)
        return self.__do_match(string, pos)

    def match(self, string='', *pos):
        """If zero or more characters at the beginning of string match
        this regular expression, return True.
        Return None if the string does not match the pattern;
        note that this is different from a zero-length match.
        """
        if not self._pattern.startswith('^'):
            self._pattern = '^' + self._pattern
        self._nfa = self.__get_nfa(self._pattern)
        return self.__do_match(string, pos)

    def fullmatch(self, string='', *pos):
        """If the whole string matches this regular expression, return True
        Return None if the string does not match the pattern;
        note that this is different from a zero-length match.
        """
        if not self._pattern.startswith('^'):
            self._pattern = '^' + self._pattern
        if not self._pattern.endswith('$'):
            self._pattern += '$'
        self._nfa = self.__get_nfa(self._pattern)
        return self.__do_match(string, pos)

    @staticmethod
    def __get_nfa(pattern=''):
        if pattern.startswith('^') and pattern.endswith('$'):
            return NFA('(' + pattern[1:-1] + ')')
        elif pattern.startswith('^'):
            return NFA('(' + pattern[1:] + '.*)')
        elif pattern.endswith('$'):
            return NFA('(.*' + pattern[:-1] + ')')
        else:
            return NFA('(.*' + pattern + '.*)')

    def __do_match(self, string='', pos=()):
        assert len(pos) <= 2
        if len(pos) == 1:
            string = string[pos[0]:]
        elif len(pos) == 2:
            string = string[pos[0]:pos[1]]
        if self._nfa.recognizes(string):
            return True
        else:
            return None
