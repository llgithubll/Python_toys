# 非确定有限状态自动机实现的正则表达式引擎


### 说明
正则表达式是用来描述模式，然后进行模式匹配的有力工具。其简洁而强大的功能，给我们带来了很多方便。但是诸如Java，Perl，PHP，Python，Ruby...之类的编程语言都是用回溯（backtracking）来实现正则表达式的，在一些奇怪的输入下，会造成指数级的复杂度（eg:a?a?a?aaa)。然而使用自动机实现正则表达式引擎，简单优雅快速，不会出现任何病态的时间复杂度，当正则表达式长度为M，文本长度为N时，可以以O(M)的时间构造自动机，然后以O(MN)的时间完成匹配，在这里就用Python实现一个正则表达式引擎，并简单的封装成模块。

参考：
* [《算法第四版》5.4](http://algs4.cs.princeton.edu/54regexp/)
* [Regular Expression Matching Can Be Simple And Fast(but is slow in Java, Perl, PHP, Python, Ruby, ...)](https://swtch.com/~rsc/regexp/regexp1.html)

### 分析
基于非确定有限状态自动机（NFA）实现的正则表达式引擎主要是通过匹配过程驱动自动机的运行，即在有向图中遍历，然后根据能够遍历的情况来判断匹配状态。所以先通过正则表达式字符串建立状态有向图，构成自动机，而后将所有状态以括号起始，从左括号开始，状态0进行不扫描文本字符转换初始化可达状态集合，然后对集合中的每个状态，检查它是否可能与第一个字符匹配，检查匹配后得到NFA匹配第一个字符可达的状态集合，然后向该集合中加入该集合中任意状态通过不匹配字符可达的状态，这样就得到了匹配第一个字符后可达的所有状态集合，即可能匹配第二个输入字符的集合，以此类推。直到可达状态集合含有接受状态，表示匹配成功；或可达状态集合不含接受状态，NFA 停滞，表示匹配失败。


### 开发
整个开发过程，大致分成三个部分，第一部分构建辅助类，图的类与图的多点可达的抽象。第二部分是通过图的类和支持的正则表达式语法构造自动机，第三部分是将自动机封装成Re模块和Regex类。

#### 程序结构
* `__init__.py` ： 初始化 package
* `Digraph.py` ： 构造自动机的有向图
* `DirectedDFS.py` ： 通过深度优先搜索得到有向图多点可达信息，来得到自动机在匹配过程中可达状态的集合。
* `NFA.py` ： 实现非确定有限状态自动机，来完成字符串与正则表达式的匹配
* `Regex.py` ： 把自动机来封装成一个 Regex 类
* `Re.py` ： 利用 Regex 类来实现 Re 模块

#### 支持的正则表达式语法
* `.` ： 通配符，匹配任意字符（除了换行符）
* `*` ： 匹配０个或多个之前的模式
* `+` ： 匹配１个或多个之前的模式
* `?` ： 匹配０个或１个之前的模式
* `|` ： ‘或’，在不同的模式之间进行选择
* `()`： 可以将 `|` 包含在其中进行模式的选择，或模块化一类特定的模式
* `[]` ： 字符集，可以匹配字符集中的任意字符（暂不支持反字符集`[^abc]` 和 使用范围`[a-z]`）
* `{}`：`{3}` 匹配3次之前的模式。 `{3,}` 匹配3次或更多次之前的模式。 `{3,5}` 匹配3到5次之前的模式。
* `\\ `：转义字符，（暂时只支持对元字符 `\.*+?|()[]{}` 的匹配，不支持诸如`\s`，`\w`之类的对特殊字母进行转义）
* `^`：用在模式开始，表示从字符串的起始进行匹配
* `$`：用在模式的结尾，表示匹配到字符串的结尾

#### 程序实现
该程序的核心内容就是非确定有限状态自动机（NFA）。首先，抽象出图的类型`Digraph`来描述自动机，以正则表达式中的每个字符为顶点（状态），以边来表示转换。用`DirectedDFS`的搜索多点可达的信息，来记录自动机的进行转换过程中能够到达的状态。
然后，利用已经抽象出的图类型来构造NFA：NFA中的每一个顶点都是正则表达式中的一个字符，NFA中每一条边都代表着一种转换，我们用黑边表示匹配字符转换，红边表示不扫描文本字符的转换。

对于正则表达式支持的语法，我们可以得出下列几种基本的转换过程(鼠绘小彩图)：
* `*` ： 匹配０个或多个之前的模式

![A_zero_more.jpg](http://upload-images.jianshu.io/upload_images/2398333-b45a6a3e98e41f4b.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

* `+` ： 匹配１个或多个之前的模式

![A_one_more.jpg](http://upload-images.jianshu.io/upload_images/2398333-9d55bb2a010aab9a.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


* `?` ： 匹配０个或１个之前的模式

![A_zero_one.jpg](http://upload-images.jianshu.io/upload_images/2398333-6b81bb62bc4f12b0.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

* `|` ： ‘或’，在不同的模式之间进行选择

![or.jpg](http://upload-images.jianshu.io/upload_images/2398333-b649259388acf8cd.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

* `()`： 可以将 `|` 包含在其中进行模式的选择，或模块化一类特定的模式

![para.jpg](http://upload-images.jianshu.io/upload_images/2398333-618cfab2852e5df0.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

通过以上几种变换规则，就可以完成自动机的初始化工作。简要代码如下：
```
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
            self._G.add_edge(pos, i)    # meta characters, support only convert meta character

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
        self._G.add_edge(lp, i + 1)    # keep moving
    if self._re[i] == '(' or \
            self._re[i] == '*' or \
            self._re[i] == ')' or \
            self._re[i] == '+' or \
            self._re[i] == '?':
        self._G.add_edge(i, i + 1)
...
```

接着，来实现字符集和重复次数的功能。稍加思考就会发现 `[ABC]` 等价于 `(A|B|C)`，于是可以通过简单的一个转换就可以利用上面实现的`|` 与 `()` 来扩展出字符集这个功能（代码片段）：
```
elif s[i] == '[':   # [ABC] -> (A|B|C)
    seq.append('(')
    i += 1
    while s[i] != ']':
        seq.append(s[i])
        seq.append('|')
        i += 1
    seq.pop()
    seq.append(')')
...
```
但是，若要实现`A{3}`这样具有重复次数的能力，就不能通过上面简单的转换来得到结果，通过上面参考的[文章](https://swtch.com/~rsc/regexp/regexp1.html)，我们可以得到下面的指导，即实现重复能力，需要进行扩展：
> *Counted repetition*.Many regular expression implementations provide a countedrepetition operator{*n*} to match exactly *n*strings matching a pattern; {*n*,*m*} to match at least *n*but no more than*m*; and {*n*,} to match*n*or more.A recursive backtracking implementation can implementcounted repetition using a loop; an NFA or DFA-basedimplementation must expand the repetition:*e*{3} expands to *eee*;*e*{3,5} expands to*eeee*? *e*? ,and*e*{3,} expands to*eee*+.

这样可以通过将`{}`之前的模式扩展相应的次数即可实现此项功能，具体内容见代码。把之前的转换和本次扩展放到 NFA 类的一个函数`__convert(self, s)`中，来完成这两项功能。

此时，通过初始化和相关转换函数，就已经实现了正则表达式的基本语法，最后在利用一个函数来驱动自动机的执行：不断的取得并更新匹配字符之后的可达状态。简要代码：
```
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
...
```

### 测试
在开发过程中，对每一个文件都有独立的单元测试（本模块文件夹中单元测试的内容都被删除了，可以在github中的[RegexPrimary](https://github.com/llgithubll/Python_toys/tree/master/RegexNFA/RegexPrimary)里看到包含单元测试的开发版本），最后还有一个`test.py`对封装后的模块进行简单测试。下面只列出对自动机的测试部分，测试内容还是比较全面清晰的：
```
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
```

### 封装
最后，模仿着 Python [re模块](https://docs.python.org/3/library/re.html?highlight=re#module-re) 的接口实现一个类似的 Re 模块和 Regex 类，就基本完成了。

####   Re模块内容
* `Re.compile(pattern)` ： 把 pattern 编译成一个 Regex 对象，然后可以使用 Regex 对象的 match() 和 search() 方法。（Note：因为实现的不同，这里和标准库的行为一样，但过程不同，预先得到 Regex 对象并不会让后面的 match() 和 search() 更快）
* `Re.match(pattern, string)` ： 如果从 string 字符串的开始能够匹配成功，返回 True。否则，返回 None。
* `Re.fullmatch(pattern, string)` ： 如果整个 string 能够和 pattern 进行匹配，返回 True。否则，返回 None。
* `Re.search(pattern, string)` ： 扫描 string，如果发现能够匹配 pattern 的部分就返回

#### Regex 对象
* `Regex.search(string[, pos[, endpos]])` ：查找 string，如果发现能够匹配构造 Regex 对象的 pattern， 返回 True。否则，返回 None。可选指定字符串进行匹配的开始和结束位置。
* `Regex.match(string[, pos[, endpos]])` ： 如果从 string 字符串的开始能够匹配成功，返回 True。否则返回 None。可选指定字符串进行匹配的开始和结束位置。
* `Regex.fullmatch(string[, pos[, endpos]])` ： 如果整个 string 能够和 Regex 对象的 pattern 进行匹配， 返回 True。否则，返回 None。可选指定字符串进行匹配的开始与结束位置。

### 用法
使用 Python3.5+ 解释器

将 Regex 文件夹拷贝到程序的根目录中，然后使用`from Regex import Re`导入模块。

#### e.g.
```
Regex
test.py
```
`test.py:`
```
from Regex import Re

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
```
