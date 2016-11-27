from collections import deque
from Digraph import Digraph
from DirectedDFS import DirectedDFS


class NFA:
    """This class provides a data type for creating a
    non-deterministic finite state automaton(NFA) from a regular expression
    and testing whether a given string is matched by that regular expression.
    """

    def __init__(self, s):
        self._ops = list()
        self._re = list(self.__convert(s))
        self._M = len(self._re)
        self._G = Digraph(self._M + 1)

        for i in range(self._M):
            lp = i  # left parenthesis(bracket, brace), used for closure

            # (), |
            if self._re[i] == '(' or self._re[i] == '|':
                self._ops.append(i)
            elif self._re[i] == ')':
                or_pos = list()
                _or = self._ops.pop()
                while self._re[_or] == '|':
                    or_pos.append(_or)
                    _or = self._ops.pop()
                lp = _or  # left parenthesis
                for pos in or_pos:
                    self._G.add_edge(lp, pos + 1)
                    self._G.add_edge(pos, i)

            # meta characters, support only convert meta character
            # \, ., |, *, (, ), +, [, ], {, }
            if i < self._M - 1 and self._re[i] == '\\':
                escape = '\\.*+?|()[]{}'  # '\\.|*()+[]{}'
                if escape.find(self._re[i + 1]):
                    self._G.add_edge(i, i + 1)
                else:
                    print("please don't use only one \\ "
                          "or \\(special character) like \\s,"
                          " which is not finish")

            # closure, and look forward to check
            # * closure, zero or more recognizes
            if i < self._M - 1 and self._re[i + 1] == '*':
                self._G.add_edge(lp, i + 1)
                self._G.add_edge(i + 1, lp)
            # + closure, one or more recognizes
            if i < self._M - 1 and self._re[i + 1] == '+':
                self._G.add_edge(i + 1, lp)
            # ? closure, zero or one recognizes
            if i < self._M - 1 and self._re[i + 1] == '?':
                self._G.add_edge(lp, i + 1)

            # keep moving
            if self._re[i] == '(' or \
                    self._re[i] == '*' or \
                    self._re[i] == ')' or \
                    self._re[i] == '+' or \
                    self._re[i] == '?':
                self._G.add_edge(i, i + 1)

    def recognizes(self, txt):
        pc = [0]
        # 0 is source, the state in start
        dfs = DirectedDFS(self._G, pc)
        pc.clear()
        # initialize the states collection, which the first state can arrived
        for v in range(self._G.V):
            if dfs.marked(v):
                pc.append(v)

        # calculate all of NFA states that txt[i+1] can arrived
        for i in range(len(txt)):
            recognizes = list()
            # calculate arrived states after recognizes
            for v in pc:
                if v < self._M:
                    if self._re[v] == txt[i] or self._re[v] == '.':
                        recognizes.append(v + 1)

            pc.clear()
            # calculate states, which epsilon transform can arrived after recognizes
            dfs = DirectedDFS(self._G, recognizes)
            for v in range(self._G.V):
                if dfs.marked(v):
                    pc.append(v)

        for v in pc:
            if v == self._M:
                return True
        return False

    def __convert(self, s):
        """using convert to straight implement some pattern
            like using (A|B|C) to implement [ABC]
            and AAAA* to A{3,}
        """
        seq = deque()

        i = 0
        length = len(s)
        while i < length:
            if s[i] == '\\':
                seq.append(s[i])    # add '\'
                seq.append(s[i + 1])    # add the character to convert
                i += 1
            elif s[i] == '[':   # [ABC] -> (A|B|C)
                seq.append('(')
                i += 1
                while s[i] != ']':
                    seq.append(s[i])
                    seq.append('|')
                    i += 1
                seq.pop()
                seq.append(')')
            elif s[i] == '{':  # A{3}->AAA, A{3,5}->AAAA?A?, A{3,}->AAAA*
                in_brace = ''
                num1, num2 = 0, 0
                multiple, _range, more = False, False, False    # {3},{3,5},{3,}
                # get content in brace
                i += 1
                while s[i] != '}':
                    in_brace += s[i]
                    i += 1
                # get the type of range
                if ',' in in_brace:
                    nums = in_brace.split(',')
                    num1 = int(nums[0])
                    if nums[1] == '':
                        more = True
                    elif nums[1] != '':
                        _range = True
                        num2 = int(nums[1])
                else:
                    multiple = True
                    num1 = int(in_brace)
                # get the basic unit used for multiple
                unit = list()
                if seq[-1] == ')':
                    unit.append(seq.pop())  # add ')'
                    if seq[-1] == '\\':
                        unit.append(seq.pop())  # add '\'
                    else:
                        lp_count = 0
                        rp_count = 1
                        while lp_count != rp_count:
                            if seq[-1] == ')':
                                rp_count += 1
                            elif seq[-1] == '(':
                                lp_count += 1
                            unit.append(seq.pop())
                else:
                    unit.append(seq.pop())

                # add multiple unit to seq
                def seq_add_unit(_seq, _unit):
                    for k in range(len(_unit) - 1, -1, -1):
                        _seq.append(_unit[k])

                while num1 > 0:
                    seq_add_unit(seq, unit)
                    num1 -= 1
                if multiple:
                    pass    # no-statement
                elif _range:
                    times = num2 - num1
                    while times > 0:
                        seq_add_unit(seq, unit)
                        seq.append('?')
                        times -= 1
                elif more:
                    seq_add_unit(seq, unit)
                    seq.append('*')
            else:
                seq.append(s[i])

            i += 1
        # generator result
        result = ''
        for ch in seq:
            result += ch
        return result

# unittest
if __name__ == '__main__':
    # normal
    re = '(A*B|AC)(D)'
    nfa = NFA('(' + re + ')')
    assert nfa.recognizes('ABD') is True
    assert nfa.recognizes('ACD') is True

    # '|' multiple-way or operator
    re = 'A(B|C|D)E'
    nfa = NFA('(' + re + ')')
    assert nfa.recognizes('ABE') is True
    assert nfa.recognizes('ACE') is True
    assert nfa.recognizes('ADE') is True

    # '+' recognizes model at least once
    re = 'A+B'
    nfa = NFA('(' + re + ')')
    assert nfa.recognizes('B') is False
    assert nfa.recognizes('AAB') is True

    # '?' recognizes model zero or once
    re = 'A?B'
    nfa = NFA('(' + re + ')')
    assert nfa.recognizes('B') is True
    assert nfa.recognizes('AAB') is False

    # '\' convert meta characters
    nfa = NFA('(' + r'3\.2' + ')')
    assert nfa.recognizes('3.2') is True

    # '[AEIOU]' character set
    nfa = NFA('(' + 'X[AEIOU]Y' + ')')
    assert nfa.recognizes('XOY') is True
    nfa = NFA('(' + 'X[[[]Y' + ')')
    assert nfa.recognizes('X[Y') is True
    assert nfa.recognizes('X[[Y') is False

    # A{3},A{3,5}, [ABC]{3,5}, (A|B){2,} etc. counted repeat
    nfa = NFA('(' + 'A{2}' + ')')
    assert nfa.recognizes('AA') is True
    nfa = NFA('(' + 'A{3,}' + ')')
    assert nfa.recognizes('AAAAA') is True   # this test is not passed
    nfa = NFA('(' + '[ABC]{2,4}' + ')')
    assert nfa.recognizes('AA') is True
    assert nfa.recognizes('ABC') is True
    assert nfa.recognizes('CCCC') is True
    nfa = NFA('(' + '(A|B){2,}' + ')')
    assert nfa.recognizes('AAAABBBB') is True

    # comprehensive
    nfa = NFA('(.*' + 'A*CB' + '.*)')
    assert nfa.recognizes('ACB') is True
    assert nfa.recognizes('CCCAACBCCCC') is True
    assert nfa.recognizes('CCCCC') is False
    assert nfa.recognizes('AAACCB') is True
    assert nfa.recognizes('CABC') is False

    nfa = NFA('(' + '([AB]|[CD])((A|B)|(C|D))' + ')')
    assert nfa.recognizes('AC') is True
    nfa = NFA('(' + '((A|B)|(C|D))((A|B)|(C|D))' + ')')
    assert nfa.recognizes('AC') is True
    nfa = NFA('(' + '((A|B)|[CD]){2}' + ')')
    assert nfa.recognizes('AC') is True
