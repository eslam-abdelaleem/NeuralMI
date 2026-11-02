"""Exact chain-rule terms for the other conditioned quantities on the same realistic model."""
from reallike import RealLike

m = RealLike()


def show(name, A, B, C):
    J = m.mi(list(A) + list(C), B)
    M = m.mi(C, B)
    d = J - M
    print(f"  {name:34s} J={J:6.3f}  M={M:6.3f}  J-M={d:7.4f}  amp={(J + M) / abs(d):7.1f}")


for k in (5, 20):
    past = lambda v: [(v, s) for s in range(-k, 0)]
    print(f"history k={k}")
    show("transfer entropy X->Y", past('x'), [('y', 0)], past('y'))
    show("transfer entropy Y->X", past('y'), [('x', 0)], past('x'))
    show("directed information rate X->Y", past('x') + [('x', 0)], [('y', 0)], past('y'))
    show("instantaneous exchange", [('x', 0)], [('y', 0)], past('x') + past('y'))
