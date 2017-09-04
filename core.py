# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as stats
import scipy.integrate as integ
import matplotlib.pyplot as plt

###############################################################################
# Parameters
# Model assumes target propostion P to be true
NUM_AGENTS = 3
TIME_STEPS = 49
TRUST_RESOLUTION = 99
ASSERT_THRESHOLD = 0.7999

###############################################################################
# Initialisations
rho = np.linspace(0.0001, 0.9999, TRUST_RESOLUTION)
Agents = 0
Trusts = 0
t = 0


def fcoin(bias):
    """Flips a true/false-coin with @bias to true
    """
    return np.random.binomial(1, bias)


def print3(T):
    # Prints 3dim Array T like this:
    # for each z, T[i,j,z] is a matrix with i rows and j columns
    print("Array with dimensions", str(np.shape(T)))
    for k in range(0, np.shape(T)[2]):
        print(T[:, :, k])


def setup():
    global Agents, Trusts

    ###########################################################################
    # SETUP AGENTS
    # Agents(i,j,t). i = agent, j = attribute, t = time
    # j = 0: activity, j = 1: aptitude, j = 2: credence
    # Starting Credences are uniformly distributed
    Agents = np.random.rand(NUM_AGENTS, 3, 1)
    # For this simple model, there is just one Expert
    # AGENT 0 = EXPERT
    # AGENT 1 = TOTAL EVIDENCE STRATEGY
    # AGENT 2 = PREEMPTION STRATEGY
    # The Expert is an expert
    Agents[0, :, 0] = [1, 0.9, 0.8]
    # The other agents aren't particularly good inquirer
    Agents[1:, 0, 0] = 0.9
    Agents[1:, 1, 0] = 0.55
    Agents[1:, 2, 0] = 0.55

    ###########################################################################
    # SETUP TRUST FUNCTIONS

    Trusts = np.zeros((NUM_AGENTS, NUM_AGENTS, TRUST_RESOLUTION))
    Trusts[0, 0, :] = stats.beta.pdf(rho, 5, 1.6)

    for i in range(1, NUM_AGENTS):
        # The agents recognized the expert
        Trusts[i, 0, :] = stats.beta.pdf(rho, 5, 1.2)
        # The agents are slightly overconfident
        Trusts[i, i, :] = stats.beta.pdf(rho, 5, 4)
        # The expert does not think very much of the other agents
        Trusts[0, i, :] = stats.beta.pdf(rho, 5, 5)


def expectation(pdf):
    """Calculates the expected Value of the given pdf.

    Integration over x*pdf(x) with trapezoidal method.
    """

    e = integ.trapz(pdf*rho, rho)

    # Correct for inaccuracies that result in out of bounds
    if (e > 1):
        e = 1
    if (e < 0):
        e = 0

    return e


def update_credence(c, ss):
    """Calculates the updated credence for an agent given new source messages

    Takes old credence @cre, all new communications @ss. that source.
    Calculation according to Angere(ms) p.21 (NOT p.22).

    """
    # e = exps[0]
    # msg = msgs[0]
    # nc = 1-c
    # ne = 1-e

    # if (msg):
        # new_c = c*e / (c*e + nc*ne)
    # else:
        # new_c = c*ne / (c*ne + nc*e)

    # return new_c

    pterm = c
    npterm = 1 - c
    for s in ss:
        msg = s['msg']
        e = s['e']
        pterm *= e if msg else 1-e
        npterm *= 1-e if msg else e

    new_c = pterm / (pterm + npterm)

    return new_c


def update_trf(c, s):
    """Calculates the updated trust function given a new source message

    """

    global Trusts, rho
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


def step(t):
    """This function describes a time step.

    1) With a Chance, the expert inquires. If inquired,
        1.1) Expert updates credence
        1.2) Expert updates trust function
        1.3) If sufficiently confident, states opinion to other agents
    2) Other Agents:
        1) With a chance, inquires
        2) Updates credence based on input (expert testimony, inquiry)
        3) Updates trust functions
    3) Voila!
    """
    global Agents, Trusts

    # Not very performant, better to initalize Agents with the right dimensions
    Agents = np.dstack((Agents, Agents[:, :, t]))

    # Keeping score of all the communications for this timestep
    S = []

    # Do ALL of the Inquiries first
    for i in range(NUM_AGENTS):
        act = Agents[i, 0, 0]
        apt = Agents[i, 1, 0]

        if fcoin(act):
            S.append({'msg': fcoin(apt),
                      'e': expectation(Trusts[i, i, :]),
                      'from': i,
                      'to': i
                      })
            # Add here the Experts communication to the other agentsq
            c = Agents[0, 2, t+1]
            if c > ASSERT_THRESHOLD or c < 1 - ASSERT_THRESHOLD:
                for i in range(1, NUM_AGENTS):
                    S.append({'msg': round(c),
                              'e': expectation(Trusts[i, 0, :]), 
                              'from': 0,
                              'to': i 
                              })

    # Do ALL of the updating now
    for i in range(NUM_AGENTS):
        # Get all communications for this particular agent
        # Not very performant, I recon
        ss = [s for s in S if s['to'] == i]

        # [print(trf) for trf in trfs]
        c = Agents[i, 2, t]

        if len(ss) > 0:
            Agents[i, 2, t+1] = update_credence(c, ss)

        for s in ss:
            Trusts[s['to'], s['from'], :] = update_trf(c, s)
        # inq = fcoin(eapt)
        # print(inq)
        # # The expected value of the expert's trust function
        # trf = Trusts[0,0,:]
        # e = expectation(trf)
        # Agents[0,2,t+1] = update_credence(ec, [inq], [e])
        # Trusts[0,0,:] = update_trf(ec,inq,trf,e)

        # if Agents[0,2,t+1] > ASSERT_THRESHOLD:
            # S.append[{ m : inq, FROM : 0, TO : 1 }]
            # S.append[{ m : inq, FROM : 0, TO : 2 }]

    # for i in range(1,NUM_AGENTS):
        # if fcoin(Agents[i,0,0]):
            # S.append[{ m : fcoin(Agents[i,1,0]), FROM:i, TO:i }]
        # if is_there_a_message:
            # msgs.append(msg)
        # if fcoin(act):
            # msgs.append(fcoin(apt))
        # Agents[i,2,t+1] = update_credence(c,msgs,


setup()

for i in range(0, 50):
    color = str(1 - (i/50))
    plt.subplot(3, 2, 1)
    plt.plot(rho, Trusts[0, 0, :], color=color)
    plt.subplot(3, 2, 3)
    plt.plot(rho, Trusts[1, 1, :], color=color)
    plt.subplot(3, 2, 5)
    plt.plot(rho, Trusts[2, 2, :], color=color)

    plt.subplot(3, 2, 2)
    plt.plot(Agents[0, 2, :])
    plt.subplot(3, 2, 4)
    plt.plot(Agents[1, 2, :])
    plt.subplot(3, 2, 6)
    plt.plot(Agents[2, 2, :])
    step(i)

plt.show()

