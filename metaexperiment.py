# -*- coding: utf-8 -*-
import numpy as np
import experiment

NUM_TRIALS = 999

results = []

for _ in range(NUM_TRIALS):
    results.append(experiment.run())

tev = [r for r in results if r[5] > r[8]]
end_tev = 0
end_pre = 0
tot_tev = 0
tot_pre = 0

for r in results:
    if r[5] == r[8] or r[3] == r[6]:
        continue 
    if r[5] > r[8]:
        end_tev += 1
    else:
        end_pre += 1
    if r[3] > r[6]:
        tot_tev += 1
    else:
        tot_pre += 1

print(end_tev, end_pre)
print(tot_tev, tot_pre)
