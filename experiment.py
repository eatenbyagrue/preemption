# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
###############################################################################
import helpers as ut
import core as co
###############################################################################
# Parameters
# Model assumes target propostion P to be true
NUM_AGENTS = 3
TIME_STEPS = 10
TRUST_RESOLUTION = 999
# Credence must be this high to assert p (or  non-p)
ASSERT_THRESHOLD = 0.8
# Expected Value of Trust function must be this high to recognize authority
AUTHORITY_THRESHOLD = 0.82
AUTHORITY_THRESHOLD_PARAM = (0.85, 0.005)
# Delta of Expected Values of Trust functions must be this high to recognize
# authority.
AUTHORITY_DELTA = 0.1
# Activity, Aptitude, Credence,
EXP_ACT = 0.9
EXP_APT = 0.85
EXP_CRE = 0.6
EXP_CRE_PARAM = (0.8, 0.02)
LAY_ACT = 0.5
LAY_APT = 0.5
#  Trust function parameters, Approx. normal distributed
# (mean, variance)
EXP_SELF_PARAM = (0.8, 0.005)
LAY_EXP_PARAM = (0.8, 0.01)
LAY_SELF_PARAM = (0.55, 0.01)
###############################################################################
# Initialisations
rho = np.linspace(0.0001, 0.9999, TRUST_RESOLUTION)
t = 0


def setup(time_steps=TIME_STEPS):
    """SETUP AGENTS & TRUST FUNCTIONS
    Agents(i,j,t). i = agent, j = attribute, t = time
    j = 0: activity,
    j = 1: aptitude,
    j = 2: credence,
    j = 3: preemption-strategy?
    j = 4: currently preempting evidence?
          if > 0, yes. if == 2, currently in 'preemption-mode'
          'preemption-mode': authority testimony preempts own evidence
    """

    global Agents, Trusts, TIME_STEPS
    TIME_STEPS = time_steps
    Agents = np.random.rand(NUM_AGENTS, 5, 1)
    # For this simple model, there is just one Expert
    # The Expert is an actual expert
    a, b = ut.estimate_parameters(EXP_CRE_PARAM[0], EXP_CRE_PARAM[1])
    ec = np.random.beta(a, b)
    Agents[0, :, 0] = [EXP_ACT, EXP_APT, ec, 0, 0]
    # The other agents aren't particularly good inquirer
    # ACTIVITY
    Agents[1:, 0, :] = 0.5
    # APTITUDE
    Agents[1:, 1, 0] = 0.5
    # CREDENCE is uniformly distributed
    # Agents[1:, 2, 0] = 0.6 
    Agents[1:, 2, 0] = np.random.uniform(1e-12, 1)
    # Agents 1 is a TEVer, 2 is a PREEMPTer
    Agents[1, 3, 0] = 0
    Agents[2, 3, 0] = 1
    # PREEMPTION MODE?
    Agents[1:, 4, 0] = 0

    ###########################################################################
    # SETUP TRUST FUNCTIONS

    Trusts = np.zeros((NUM_AGENTS, NUM_AGENTS, TRUST_RESOLUTION))

    alpha, beta = ut.estimate_parameters(EXP_SELF_PARAM[0], EXP_SELF_PARAM[1])

    Trusts[0, 0, :] = stats.beta.pdf(rho, alpha, beta)
    for i in range(1, NUM_AGENTS):
        # The agents recognized the expert
        a, b = ut.estimate_parameters(LAY_EXP_PARAM[0],
                                      LAY_EXP_PARAM[1])
        Trusts[i, 0, :] = stats.beta.pdf(rho, a, b)
        # The agents are slightly overconfident
        a, b = ut.estimate_parameters(LAY_SELF_PARAM[0],
                                      LAY_SELF_PARAM[1])
        Trusts[i, i, :] = stats.beta.pdf(rho, a, b)


def step(t):
    """This function describes a time step.

    1) With a Chance, the expert inquires. If so,
        1.1) Expert updates credence
        1.2) Expert updates trust function
        1.3) If sufficiently confident, states opinion to other agents
    2) Other Agents:
        1) With a chance, inquires, and maybe tells their opinion to others
        2) Updates credence based on input (expert testimony, inquiry)
            and strategy (preemption, tev)
        3) Updates trust functions
    3) Voila!
    """

    global Agents, Trusts

    # Add a new page to the the Agents' data for this timestep
    # Not very performant, better to initalize Agents with the right dimensions
    Agents = np.dstack((Agents, Agents[:, :, t]))

    # Keeping score of all the communications for this timestep
    S = []

    # Do ALL of the Communication and Inquiries first
    for i in range(NUM_AGENTS):
        act = Agents[i, 0, 0]
        apt = Agents[i, 1, 0]

        if ut.fcoin(act):
            # Cheating!
            # if 20 < t < 50:
                # inq = 0
            # else:
                # inq = fcoin(apt)
            inq = ut.fcoin(apt)
            S.append({'msg': inq,
                      'e': co.expectation(Trusts[i, i, :], rho),
                      'from': i,
                      'to': i,
                      'type': 0  # 0 = inquiry, 1 = testimony
                      })
            # print(S[-1])
            # Experts communication to the other agentsq
            if i == 0:
                c = Agents[i, 2, t+1]
                if c > ASSERT_THRESHOLD or c < 1 - ASSERT_THRESHOLD:
                    for j in range(1, NUM_AGENTS):
                        msg = round(c)
                        S.append({'msg': msg,
                                  'e': co.expectation(Trusts[j, i, :], rho),
                                  'from': i,
                                  'to': j,
                                  'type': 1  # 0 = inquiry, 1 = testimony
                                  })

    # Do ALL of the updating now
    for i in range(NUM_AGENTS):
        # Get all communications for this particular agent
        ss = [s for s in S if s['to'] == i]

        c = Agents[i, 2, t]

        # Update the agent's credence
        if len(ss) > 0:

            # The PREEMPTer is a normal bayesian, unless she recognizes an
            # expert as an
            # authority. Then she ignores her own evidence and instead adopts
            # the opinion of the authority. She still updates her trust
            # function.
            if Agents[i, 3, 0] > 0:  # Agent is a PREEMPTer
                # Check whether the agents recognized the expert as an auth.
                strf = Trusts[i, i, :]
                etrf = Trusts[i, 0, :]
                Agents[i, 4, t:] = co.check_authority(strf,
                                                      etrf,
                                                      AUTHORITY_DELTA,
                                                      AUTHORITY_THRESHOLD,
                                                      rho)
                if Agents[i, 4, t]:  # Agents is in PREEMPTION MODE
                    Agents[i, 2, t+1] = co.update_credence_preemption(c,
                                                                      ss,
                                                                      Trusts,
                                                                      rho)
                else:
                    Agents[i, 2, t+1] = co.update_credence_tev(c, ss)
            else:  # Agents is a TEVer
                Agents[i, 2, t+1] = co.update_credence_tev(c, ss)
                # if i == 0:
                # print('###############################################')
                # print('update timestep ', t)
                # print('c: ', c, 'msg: ', ss)
                # print('new_credence:', Agents[i, 2, t+1])

        # Update the agent's trust functions

        for s in ss:
            Trusts[s['to'], s['from'], :] = co.update_trf(c, s, Trusts, rho)


def total_score(creds):
    return sum([(1 - c)**2 for c in creds])


def end_score(creds):
    return (1 - creds[-1])**2


def begin_score(creds):
    return (1 - creds[0])**2


def drawstep(i, gs):
    ax = plt.subplot(gs[0, :2])
    color = 1 - (i/TIME_STEPS)*0.8 - 0.2
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel(r'$\tau_{\iota\varepsilon}(\rho)$')
    plt.plot(rho, Trusts[0, 0, :], color=str(color))
    # plt.axvline(x=co.expectation(Trusts[0, 0, :], rho), color=(1, color, 1))

    ax = plt.subplot(gs[1, 0])
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel(r'$\tau_{\iota\alpha}(\rho)$')
    plt.plot(rho, Trusts[1, 1, :], color=str(color))

    ax = plt.subplot(gs[1, 1])
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel(r'$\tau_{\varepsilon\alpha}(\rho)$')
    plt.plot(rho, Trusts[1, 0, :], color=str(color))

    ax = plt.subplot(gs[2, 0])
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel(r'$\tau_{\iota\beta}(\rho)$')
    plt.plot(rho, Trusts[2, 2, :], color=str(color))

    ax = plt.subplot(gs[2, 1])
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel(r'$\tau_{\varepsilon\beta}(\rho)$')
    plt.plot(rho, Trusts[2, 0, :], color=str(color))


def drawcredences(gs):
    ax = plt.subplot(gs[0, 2])
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$C_{\varepsilon}(p)$')
    plt.plot(Agents[0, 2, :])

    ax = plt.subplot(gs[1, 2])
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$C_{\alpha}(p)$')
    plt.plot(Agents[1, 2, :])

    ax = plt.subplot(gs[2, 2])
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$C_{\beta}(p)$')
    plt.plot(Agents[2, 2, :])
    plt.plot(Agents[2, 4, :])
    plt.tight_layout()
    plt.show()


def run():
    setup()
    # gs = gridspec.GridSpec(3, 3)
    for i in range(0, TIME_STEPS):
        # drawstep(i, gs)
        step(i)

    # drawcredences(gs)
    return {'ets': total_score(Agents[0, 2, :]),
            'ebs': begin_score(Agents[0, 2, :]),
            'ees': end_score(Agents[0, 2, :]),
            # TEV AGENT'S Scores
            'tbc': Agents[1, 2, 0],
            'tts': total_score(Agents[1, 2, :]),
            'tbs': begin_score(Agents[1, 2, :]),
            'tes': end_score(Agents[1, 2, :]),
            # PREEMPTION Agent's scores
            'pbc': Agents[2, 2, 0],
            'pts': total_score(Agents[2, 2, :]),
            'pbs': begin_score(Agents[2, 2, :]),
            'pes': end_score(Agents[2, 2, :]),
            }


run()
