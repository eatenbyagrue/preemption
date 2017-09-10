# -*- coding: utf-8 -*-
import helpers as ut
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

EXP_CRE_PARAM = (0.85, 0.005)
TRUST_RESOLUTION = 999
rho = np.linspace(0.0001, 0.9999, TRUST_RESOLUTION)
alpha, beta = ut.estimate_parameters(EXP_CRE_PARAM[0], EXP_CRE_PARAM[1])
print(alpha, beta)
pdf = stats.beta.pdf(rho, alpha, beta)

ax = plt.subplot()
ax.set_xlabel(r'$\rho$')
ax.set_ylabel(r'$\tau(\rho)$')
plt.tight_layout()
plt.plot(rho, pdf)
plt.show()
