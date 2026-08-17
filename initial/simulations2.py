import random
from collections import Counter
import math

N = 1_000_000

counts = Counter()

for _ in range(N):
    S = sum(random.choice([-1, 1]) for _ in range(40))
    counts[S] += 1

print(f"Simulations: {N:,}")
print(f"Mean S: {sum(s * n for s, n in counts.items()) / N:.4f}")

variance = sum(((s ** 2) * n) for s, n in counts.items()) / N
print(f"Std dev S: {math.sqrt(variance):.4f}")
print()

print("S        Probability")
print("-" * 25)
for s in range(-22, 23, 2):
    print(f"{s:>3}      {counts[s] / N:.6%}")

print()

for bound in [10, 14, 18, 22]:
    prob = sum(n for s, n in counts.items() if abs(s) <= bound) / N
    print(f"P(|S| <= {bound:>2}) = {prob:.6%}")

print()
print(f"P(|S| > 22) = {sum(n for s, n in counts.items() if abs(s) > 22) / N:.6%}")
