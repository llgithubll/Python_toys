import Re


prog = Re.compile('[pj]ython')
result = prog.match('python')
if result:
    print('separate compile and match success')

result = Re.match('[pj]ython', 'jython')
if result:
    print('match success')

pattern = Re.compile("d")
if pattern.search("dog"):     # Match at index 0
    print('Found it')
assert Re.search('d', 'dog') == pattern.search('dog')
if not pattern.search("dog", 1):   # No match; search doesn't include the "d"
    print('Not found it')

pattern = Re.compile('[AEIOU]{3}(a|e|i|o|u){3,}')
assert pattern.match('AEIaei#') is True
assert pattern.match('AAAaaaa') is True

pattern = Re.compile('^([AEIOUaeiou]|[0123456789]|(@|#)){3,}$')
assert pattern.match('aaaa') is True
assert pattern.match('0@#A999') is True
assert pattern.match('@#') is None
