from symjit import Composer, compile_composer


def test_simple():
    cp = Composer(2, 2)
    s1 = cp.fadd(cp.arg(0), cp.arg(1))
    s2 = cp.fmul(cp.arg(0), cp.arg(1))
    cp.assign(cp.out(0), s1)
    cp.assign(cp.out(1), s2)

    f = compile_composer(cp)

    print(f(3, 4))


def test_recip():
    cp = Composer(2, 1)
    s1 = cp.fsub(cp.arg(0), cp.arg(1))
    s2 = cp.fadd(cp.arg(0), cp.arg(1))
    s2 = cp.fdiv(s1, s2)
    cp.assign(cp.out(0), s2)

    f = compile_composer(cp)

    print(f(5, 3))


def test_if_else():
    cp = Composer(2, 1)

    cond = cp.lt(cp.arg(0), cp.arg(1))

    b1 = cp.new_block()
    s1 = b1.fadd(b1.arg(0), b1.constant(10.0))

    b2 = cp.new_block()
    s2 = b2.fadd(b2.arg(0), b2.constant(3.0))

    cp.append_if_else(cond, b1, b2)
    t = cp.join(cond, s1, s2)
    cp.assign(cp.out(0), t)

    f = compile_composer(cp)

    print(f(3, 4))
    print(f(5, 4))


def test_complex():
    cp = Composer(3, 1)

    s1 = cp.fmul(cp.arg(1), cp.arg(2))
    s2 = cp.fadd(cp.arg(0), s1)
    s3 = cp.sin(s2)
    cp.assign(cp.out(0), s3)

    f = compile_composer(cp, dtype="complex128")
    print(f(-5, 2, 3))
    print(f(-24 + 0j, 4 - 3j, 4 + 3j))


def test_pi_viete(dtype="float64"):
    N = 21
    cp = Composer(1, 1)

    x = cp.arg(0)
    p = cp.new_temp()
    t = cp.new_temp()
    cp.assign(p, cp.constant(1.0))

    for i in range(N):
        cp.assign(t, x)

        for j in range(i):
            s1 = cp.sqrt(t)
            s2 = cp.fmul(x, s1)
            s3 = cp.fadd(x, s2)
            cp.assign(t, s3)

        s4 = cp.sqrt(t)
        s5 = cp.fmul(p, s4)
        cp.assign(p, s5)

    s6 = cp.fdiv(cp.constant(2.0), p)
    cp.assign(cp.out(0), s6)

    f = compile_composer(cp, dtype=dtype)
    print(f(1 / 2)[0][0])


def test_sum(dtype="float64"):
    cp = Composer(1, 1)
    x = cp.new_temp()
    s = cp.new_temp()
    cp.assign(s, cp.arg(0))

    b = cp.new_block()
    b.assign(s, b.fadd(s, x))

    cp.append_for(x, 1, 100, b)
    cp.assign(cp.out(0), s)

    # print(cp.get_instructions())

    f = compile_composer(cp, dtype=dtype)

    # print(f(0)[0][0])


def test_call(dtype="float64", direct=True):
    cp = Composer(2, 1)

    s1 = cp.call(lambda x: x**2 + 5, cp.arg(0))
    s2 = cp.call(lambda x, y: x**3 - y, cp.arg(0), cp.arg(1))
    cp.assign(cp.out(0), cp.fadd(s1, s2))

    f = compile_composer(cp, dtype=dtype, direct=direct)

    print(f(4, 7)[0][0])


#######################################################################

test_simple()
test_recip()
test_if_else()
test_complex()
test_pi_viete("float64")
test_pi_viete("complex128")
test_sum("float64")
test_sum("complex128")
test_call("float64")
test_call("complex128")
# test_call("float64", direct=False)
# test_call("complex128", direct=False)
