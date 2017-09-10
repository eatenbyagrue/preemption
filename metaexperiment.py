# -*- coding: utf-8 -*- # import numpy as np
import experiment
import matplotlib.pyplot as plt

NUM_TRIALS = 9999

results = []

for _ in range(NUM_TRIALS):
    results.append(experiment.run())

tev_x = [r['tbc'] for r in results]
tev_y = [r['tes'] for r in results]

pre_x = [r['pbc'] for r in results]
pre_y = [r['pes'] for r in results]

print(sum(pre_y)/NUM_TRIALS, sum(tev_y)/NUM_TRIALS)
plt.scatter(tev_x, tev_y)
plt.scatter(pre_x, pre_y, marker='x')
plt.xlabel(r'$C^10(p)$')
plt.ylabel(r'$I_{BR}$')
plt.ylim(0.0, 0.2)
plt.show()
