# -*- coding: utf-8 -*-
# Central functions of the modeling framework
# Mostly to do with updating the agent's credence


import scipy.integrate as integ


def expectation(pdf, rho):
    """Calculates the expected Value of the given pdf.

    Integration over x*pdf(x) with trapezoidal method.
    """

    e = integ.trapz(pdf*rho, rho)

    # Correct for inaccuracies that result in out of bounds.
    # Errors e.g. occur if the probability is very dense around a point,
    # but TRUST_RESOLUTION is not very high
    if (e > 1):
        e = 1
    if (e < 0):
        e = 0

    return e


def update_credence_preemption(c, ss, Trusts, rho):
    """Calculates the updated credence for an agent given new source messages
    According to the Preemption View:
    If the authority says Yay, believe with exp
    If the authority says Nay, believe with 1-exp

    """

    s = [s for s in ss if s['type']]
    # Is there any testimony included?
    if any(s):
        s = s[0]
        e = expectation(Trusts[s['to'], s['from'], :], rho)

        return e if s['msg'] else 1-e
    else:
        return c


def update_credence_tev(c, ss):
    """Calculates the updated credence for an agent given new source messages
    According to the Total Evidence View (Or just classically Bayesian).

    Takes old credence @cre, all new communications @ss. that source.
    Calculation according to Angere(ms) p.21 (NOT p.22).

    """

    pterm = c
    npterm = 1 - c
    for s in ss:
        msg = s['msg']
        e = s['e']
        pterm *= e if msg else 1-e
        npterm *= 1-e if msg else e

    new_c = pterm / (pterm + npterm)

    # staying regular
    if new_c > 0.9999999999:
        new_c = 0.9999999999
    if new_c < 0.0000000001:
        new_c = 0.0000000001
    return new_c


def update_trf(c, s, Trusts, rho):
    """Calculates the updated trust function given a new source message

    """

    msg = s['msg']
    e = s['e']
    nc = 1 - c
    ne = 1 - e
    trf = Trusts[s['to'], s['from'], :]
    new_trf = []

    if msg:
        dnom = e*c + ne*nc
        nom = (rho*c) + ((1-rho)*nc)
        new_trf = trf * (nom / dnom)
    else:
        dnom = e*nc + ne*c
        new_trf = trf * (((rho*nc) + ((1-rho)*c)) / dnom)

    return new_trf


def check_authority(strf, etrf, delta, threshold, rho):
    """ Checks whether an agent with @strf in her own inquiry and @etrf in the
    expert's reliability should regard the expert as an authority.

    """

    se = expectation(strf, rho)
    ee = expectation(etrf, rho)
    return ee - se > delta and ee > threshold
