# -*- coding: utf-8 -*-
import helpers as ut
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

EXP_CRE_PARAM = (0.8, 0.02)
TRUST_RESOLUTION = 999
rho = np.linspace(0.0001, 0.9999, TRUST_RESOLUTION)
alpha, beta = ut.estimate_parameters(EXP_CRE_PARAM[0], EXP_CRE_PARAM[1])

pdf = stats.beta.pdf(rho, alpha, beta)

plt.plot(rho, pdf)
plt.show()
