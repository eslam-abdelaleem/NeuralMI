import sys, time
sys.path.insert(0, sys.argv[1])            # scratchpad (reallike, harness)
from reallike import RealLike
from harness import run_rate

m = RealLike()
x, y = m.sample(20000, seed=0)
J, M, d = m.terms(5, 20)
print(f"exact h=5 L=20: J={J:.3f} M={M:.3f} J-M={d:.4f} nats; true rate {m.rate():.4f}")
t = time.time()
recs, msgs = run_rate(x, y, h=5, half_width=20, seed=0)
print(f"took {time.time() - t:.1f}s")
for r in recs:
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
print("warnings:")
for m_ in msgs:
    print("  -", m_)
