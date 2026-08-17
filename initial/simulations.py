import random
from collections import Counter

N = 100_000

sums = []

for _ in range(N):
    coins = [random.choice([-1, 1]) for _ in range(40)]
    S = sum(coins)
    sums.append(S)

print("Mean:", sum(sums) / N)

counts = Counter(sums)

for s in sorted(counts):
    print(s, counts[s] / N)