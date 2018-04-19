'''
Created on Apr 9, 2017

@author: Jon
'''
import sys
sys.path.append('./burlap.jar')
import java
from collections import defaultdict
from time import clock
from burlap.behavior.policy import Policy;
from burlap.assignment4 import BasicGridWorld;
from burlap.behavior.singleagent import EpisodeAnalysis;
from burlap.behavior.singleagent.auxiliary import StateReachability;
from burlap.behavior.singleagent.auxiliary.valuefunctionvis import ValueFunctionVisualizerGUI;
from burlap.behavior.singleagent.learning.tdmethods import QLearning;
from burlap.behavior.singleagent.planning.stochastic.policyiteration import PolicyIteration;
from burlap.behavior.singleagent.planning.stochastic.valueiteration import ValueIteration;
from burlap.behavior.valuefunction import ValueFunction;
from burlap.domain.singleagent.gridworld import GridWorldDomain;
from burlap.oomdp.core import Domain;
from burlap.oomdp.core import TerminalFunction;
from burlap.oomdp.core.states import State;
from burlap.oomdp.singleagent import RewardFunction;
from burlap.oomdp.singleagent import SADomain;
from burlap.oomdp.singleagent.environment import SimulatedEnvironment;
from burlap.oomdp.statehashing import HashableStateFactory;
from burlap.oomdp.statehashing import SimpleHashableStateFactory;
from burlap.assignment4.util import MapPrinter;
from burlap.oomdp.core import TerminalFunction;
from burlap.oomdp.core.states import State;
from burlap.oomdp.singleagent import RewardFunction;
from burlap.oomdp.singleagent.explorer import VisualExplorer;
from burlap.oomdp.visualizer import Visualizer;
from burlap.assignment4.util import BasicRewardFunction;
from burlap.assignment4.util import BasicTerminalFunction;
from burlap.assignment4.util import MapPrinter;
from burlap.oomdp.core import TerminalFunction;
from burlap.assignment4.EasyGridWorldLauncher import visualizeInitialGridWorld
from burlap.assignment4.util.AnalysisRunner import calcRewardInEpisode, simpleValueFunctionVis,getAllStates
from burlap.behavior.learningrate import ExponentialDecayLR, SoftTimeInverseDecayLR
import csv
import os
from collections import deque
import pickle

def frange(start, stop, step=1.0):
    ''' "range()" like function which accept float type'''
    i = start
    while i < stop:
        yield i
        i += step

def dumpCSV(nIter, times,rewards,steps,convergence,world,method):
    folder = './OUTPUT/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    fname = '{}{} {}.csv'.format(folder,world,method)
    iters = range(1,nIter+1)
    assert len(iters)== len(times)
    assert len(iters)== len(rewards)
    assert len(iters)== len(steps)
    assert len(iters)== len(convergence)
    with open(fname,'wb') as f:
        f.write('iter,time,reward,steps,convergence\n')
        writer = csv.writer(f,delimiter=',')
        writer.writerows(zip(iters,times,rewards,steps,convergence))
    
    
def runEvals(initialState,plan,rewardL,stepL):
    r = []
    s = []
    for trial in range(evalTrials):
        ea = plan.evaluateBehavior(initialState, rf, tf,200);
        r.append(calcRewardInEpisode(ea))
        s.append(ea.numTimeSteps())
    rewardL.append(sum(r)/float(len(r)))
    stepL.append(sum(s)/float(len(s))) 


def comparePolicies(policy1,policy2):
    assert len(policy1)==len(policy1)
    diffs = 0
    for k in policy1.keys():
        if policy1[k] != policy2[k]:
            diffs +=1
    return diffs

def mapPicture(javaStrArr):
    out = []
    for row in javaStrArr:
        out.append([])
        for element in row:
            out[-1].append(str(element))
    return out

def dumpPolicyMap(javaStrArr,fname):

    folder = './OUTPUT/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    fname = '{}{}'.format(folder,fname)

    pic = mapPicture(javaStrArr)
    with open(fname,'wb') as f:
        pickle.dump(pic,f)
    


if __name__ == '__main__':
    world = 'Easy'
    discount=0.99
    NUM_INTERVALS=MAX_ITERATIONS = 100;
    evalTrials = 50;
    userMap = [
              [ 0, 0, 0, -1, 0, 0, 0, 0, -5, 0],
              [ 0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
              [ 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
              [ 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
              [ 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
              [ 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
              [ 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
              [ 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
              [ 0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
              [ 0, -3, 0, 0, 0, 0, 0, 0, 0, 0],              
              ]
    n = len(userMap)
    tmp = java.lang.reflect.Array.newInstance(java.lang.Integer.TYPE,[n,n])
    for i in range(n):
        for j in range(n):
            tmp[i][j]= userMap[i][j]
    userMap = MapPrinter().mapToMatrix(tmp)
    maxX = maxY= n-1
    
    gen = BasicGridWorld(userMap,maxX,maxY)
    domain = gen.generateDomain()
    initialState = gen.getExampleState(domain);

    rf = BasicRewardFunction(maxX,maxY,userMap)
    tf = BasicTerminalFunction(maxX,maxY)
    env = SimulatedEnvironment(domain, rf, tf,initialState);
#    Print the map that is being analyzed
    print "/////{} Grid World Analysis/////\n".format(world)
    MapPrinter().printMap(MapPrinter.matrixToMap(userMap));

    hashingFactory = SimpleHashableStateFactory()
    increment = MAX_ITERATIONS/NUM_INTERVALS
    timing = defaultdict(list)
    rewards = defaultdict(list)
    steps = defaultdict(list)
    convergence = defaultdict(list)
    allStates = getAllStates(domain,rf,tf,initialState)
    # Value Iteration
    iterations = range(1,MAX_ITERATIONS+1)

    for discount in frange(0, 1, 0.1):
        Vname = 'Value d{:0.1f}'.format(discount)

        vi = ValueIteration(domain,rf,tf,discount,hashingFactory,-1, 1);
        vi.setDebugCode(0)
        vi.performReachabilityFrom(initialState)
        vi.toggleUseCachedTransitionDynamics(False)
        print "//{} {} Value Iteration Analysis//".format(Vname, world)
        timing[Vname].append(0)
        for nIter in iterations:
            startTime = clock()
            vi.runVI()
            timing[Vname].append(timing[Vname][-1]+clock()-startTime)
            p = vi.planFromState(initialState);
            convergence[Vname].append(vi.latestDelta)
            # evaluate the policy with evalTrials roll outs
            runEvals(initialState,p,rewards[Vname],steps[Vname])
            if nIter == 1:
                dumpPolicyMap(MapPrinter.printPolicyMap(allStates, p, gen.getMap()),'Value {} {} Iter {} Policy Map.pkl'.format(Vname, world,nIter))
            if vi.latestDelta <1e-6:
                dumpPolicyMap(MapPrinter.printPolicyMap(allStates, p, gen.getMap()),'Value {} {} Iter {} Policy Map.pkl'.format(Vname, world,nIter))
                break
        print "\n\n\n"
        dumpCSV(nIter, timing[Vname][1:], rewards[Vname], steps[Vname],convergence[Vname], world, Vname)
      
      
    for discount in frange(0, 1, 0.1):
        Pname = 'Policy d{:0.1f}'.format(discount)
        Pname2 = Pname + ' 2'

        pi = PolicyIteration(domain,rf,tf,discount,hashingFactory,1e-3,10, 1)
        pi.toggleUseCachedTransitionDynamics(False)
        print "//{} {} Policy Iteration Analysis//".format(Pname, world)
        timing[Pname].append(0)
        for nIter in iterations:
            startTime = clock()
            p = pi.planFromState(initialState);
            timing[Pname].append(timing[Pname][-1]+clock()-startTime)
            policy = pi.getComputedPolicy()
            current_policy = {state:policy.getAction(state).toString() for state in allStates}
            convergence[Pname2].append(pi.lastPIDelta)
            if nIter == 1:
                convergence[Pname].append(999)
            else:
                convergence[Pname].append(comparePolicies(last_policy,current_policy))
            last_policy = current_policy
            runEvals(initialState,p,rewards[Pname],steps[Pname])
            if nIter == 1:
                #simpleValueFunctionVis(pi, p, initialState, domain, hashingFactory, "Policy {} Iteration {}".format(Pname, nIter))
                dumpPolicyMap(MapPrinter.printPolicyMap(allStates, p, gen.getMap()),'Policy {} {} Iter {} Policy Map.pkl'.format(Pname, world,nIter))
            if convergence[Pname2][-1] <1e-6:
                dumpPolicyMap(MapPrinter.printPolicyMap(allStates, p, gen.getMap()),'Policy {} {} Iter {} Policy Map.pkl'.format(Pname, world,nIter))
                break
        MapPrinter.printPolicyMap(pi.getAllStates(), p, gen.getMap());
        print "\n\n\n"
        dumpCSV(nIter, timing[Pname][1:], rewards[Pname], steps[Pname],convergence[Pname2], world, Pname)
        #raise
      
    MAX_ITERATIONS=NUM_INTERVALS = MAX_ITERATIONS*10;
    increment = MAX_ITERATIONS/NUM_INTERVALS
    iterations = range(1,MAX_ITERATIONS+1)
    for lr in [0.1,0.9]:
        for qInit in [-100,0,100]:
            for epsilon in [0.1,0.3,0.5]:
                for discount in frange(0, 1, 0.1):
                    last10Chg = deque([99]*10,maxlen=10)
                    Qname = 'Q-Learning L{:0.1f} q{:0.1f} E{:0.1f} d{:0.1f}'.format(lr,qInit,epsilon,discount)
                    agent = QLearning(domain,discount,hashingFactory,qInit,lr,epsilon,300)
                    #agent.setLearningRateFunction(SoftTimeInverseDecayLR(1.,0.))
                    agent.setDebugCode(0)
                    print "//{} {} Iteration Analysis//".format(world,Qname)
                    for nIter in iterations:
                        if nIter % 50 == 0: print(nIter)
                        startTime = clock()
                        ea = agent.runLearningEpisode(env,300)
                        if len(timing[Qname])> 0:
                            timing[Qname].append(timing[Qname][-1]+clock()-startTime)
                        else:
                            timing[Qname].append(clock()-startTime)
                        env.resetEnvironment()
                        agent.initializeForPlanning(rf, tf, 1)
                        p = agent.planFromState(initialState)     # run planning from our initial state
                        last10Chg.append(agent.maxQChangeInLastEpisode)
                        convergence[Qname].append(sum(last10Chg)/10.)
                        # evaluate the policy with one roll out visualize the trajectory
                        runEvals(initialState,p,rewards[Qname],steps[Qname])
                        if nIter == 1 :
                            dumpPolicyMap(MapPrinter.printPolicyMap(allStates, p, gen.getMap()),'QL {} {} Iter {} Policy Map.pkl'.format(Qname,world,nIter))
                        if convergence[Qname][-1] <0.5:
		                    dumpPolicyMap(MapPrinter.printPolicyMap(allStates, p, gen.getMap()),'QL {} {} Iter {} Policy Map.pkl'.format(Qname,world,nIter));break
                    print "\n\n\n"
                    dumpCSV(nIter, timing[Qname], rewards[Qname], steps[Qname],convergence[Qname], world, Qname)
     