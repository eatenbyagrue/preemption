# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as stats 
import scipy.integrate as integ 
import matplotlib.pyplot as plt

################################################################################
# Parameters
# Model assumes target propostion P to be true
NUM_AGENTS = 3
TIME_STEPS = 49
TRUST_RESOLUTION = 199

################################################################################
# Initialisations
rho = np.linspace(0.0001,0.9999,TRUST_RESOLUTION)
Agents = 0 
Trusts = 0

# Flips a true/false-coin with @bias to true
def fcoin(bias): 
    return np.random.binomial(1,bias)      
    
def print3(T):
    # Prints 3dim Array T like this:
    # for each z, T[i,j,z] is a matrix with i rows and j columns
    print("Array with dimensions",str(np.shape(T))) 
    for k in range(0,np.shape(T)[2]):
        print(T[:,:,k])

def setup():
    global Agents, Trusts
    
    ############################################################################
    # SETUP AGENTS
    # Agents(i,j,t). i = agent, j = attribute, t = time
    # j = 0: activity, j = 2: aptitude, j = 3: credence 
    # Starting Credences are uniformly distributed  
    Agents = np.random.rand(NUM_AGENTS, 3, 1)
    # For this simple model, there is just one Expert
    # AGENT 0 = EXPERT
    # AGENT 1 = TOTAL EVIDENCE STRATEGY
    # AGENT 2 = PREEMPTION STRATEGY
    # The Expert is an expert 
    Agents[0,:,0] = [1, 0.5, 0.9]         
    # The other agents aren't very clever Agents[1:,0,0] = 0.5
    Agents[1:,1,0] = 0.5
    Agents[1:,2,0] = 0.5

    # This is how you add New Timedata!
    # np.dstack((Agents,AddAgents)))

    ############################################################################
    # SETUP TRUST FUNCTIONS

    Trusts = np.zeros((NUM_AGENTS, NUM_AGENTS, TRUST_RESOLUTION))
    Trusts[0,0,:] = stats.beta.pdf(rho,5,0.6)
    integral = integ.trapz(Trusts[1,1,:],rho)
    for i in range(1,NUM_AGENTS):
        # The agents recognized the expert
        Trusts[i,0,:] = stats.beta.pdf(rho,5,1.2)
        # The agents are slightly overconfident
        Trusts[i,i,:] = stats.beta.pdf(rho,5,4)
        # The expert does not think very much of the other agents
        Trusts[0,i,:] = stats.beta.pdf(rho,5,5)
    
def expectation(pdf):

    e = integ.trapz(pdf*rho,rho)
    # Correct for inaccuracies that result in out of bounds
    if (e > 1): e = 1
    if (e < 0): e = 0
    return e

def update_credence(cre, msgs, trfs):
    # Calculate the expected values first
    # plt.plot(rho,trfs[0])
    # plt.show()
    exps = [expectation(trf) for trf in trfs]
    print(exps)
        

    pterm = cre 
    for i in range(0,len(msgs)):
        pterm *= exps[i] if msgs[i] else 1 - exps[i]
    
    npterm = 1 - cre
    for i in range(0,len(msgs)):
        npterm *= 1 - exps[i] if msgs[i] else exps[i]

    new_cre = pterm / (pterm + npterm)    
    print(new_cre)

    
def step():
    act = Agents[0,0,0]
    apt = Agents[0,1,0]
    cre = Agents[0,2,0]
    print3(Agents)
    if fcoin(act):
        new_cre = update_credence(cre, [fcoin(apt)], [Trusts[1,1,:]])
        # Agents[0,2,t+1] = new_cre
    


setup()
step()
