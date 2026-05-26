import math

from symbolica import E, S
from symjit import compile_evaluator

x, a = S("x"), S("a")


def c(u):
    return sum((-1) ** (i // 2) * u**i / math.factorial(i) for i in range(0, 16, 2))


def s(u):
    return (1 - c(u) ** 2).sqrt()


def expr(x, a, n=5):
    u = x
    for _ in range(n):
        u = u - (s(u) - a) / c(u)
    return 4 * u


y = expr(x, a)
ev = y.evaluator({}, {}, [x, a])

t = [[0.123456, math.sqrt(2) / 2]]

print(len(ev.get_instructions()[0]))

print(math.pi)

p1 = ev.evaluate(t)
print(p1[0][0])

f = compile_evaluator(ev)
p2 = ev.evaluate(t)
print(p1[0][0])
