# Trenton Griffiths
# CS_5890
# Cart and Pole Reinorcement Learning Problem

import tensorflow as tf
import numpy as np
import math
import random
import gym
import matplotlib.pyplot as plt

random.seed(666)

#GLOBAL CONSTANTS
FORCE_MAG = 10
POLEMASS_LENGTH = 0.209*0.326
TOTALMASS = 0.711+0.209
MASSPOLE = 0.209
TAU = 0.02
GRAVITY = 9.8
LENGTH = 0.326
FOURTHIRDS = 4/3
GAMMA = 0.5
ALPHA = 0.5

lookUpTable = []

def fillTable():
    global lookUpTable
    for action in range(0,2):
        lookUpTable.append([])
        for theta in range(0,6):
            lookUpTable[action].append([])
            for thetaDot in range(0,3):
                lookUpTable[action][theta].append([])
                for x in range(0,3):
                    lookUpTable[action][theta][thetaDot].append([])
                    for xDot in range(0,3):
                        lookUpTable[action][theta][thetaDot][x].append(random.randint(-10,10))
                

class state:
    def __init__(self, x, xDot, theta, thetaDot):
        self.theta = theta
        self.thetaDot = thetaDot
        self.x = x
        self.xDot = xDot



def main():    
    attempt = 0
    listofAttempts = []
    listofAttemptsVariable = []
    fillTable()
    while attempt < 100:
        iterator = 0
        previousState = state(0, 0, 0, 0)
        currState = state(0, 0, 0, 0)
        action = True # True = right, False = left
        inBounds = True
        while (inBounds):        
            action = argMaxQ(currState)
            updateQTable(action, currState, previousState)
            previousState.x = currState.x
            previousState.xDot = currState.xDot
            previousState.theta = currState.theta
            previousState.thetaDot = currState.thetaDot
            cartPoleReaction(action, currState)
            inBounds = checkInBounds(currState)
            iterator += 1
            # print("Action: ", action, "     x: ", round(currState.x, 1), "        xDot: ", round(currState.xDot, 1), "      theta: ", round(currState.theta, 1), "    thetaDot: ", round(currState.thetaDot, 1), "      Iteration: ", iterator)
        attempt += 1
        print("Attempt: ", attempt, "   Iterations: ", iterator)
        listofAttempts.append(iterator)
        
        global lookUpTable
        lookUpTable = []
        global ALPHA, GAMMA, FORCE_MAG
        GAMMA = 10
        FORCE_MAG = .5 
        ALPHA = 10
        # random.seed(666)
        fillTable()
        iterator = 0
        currState = state(0, 0, 0, 0)
        action = True # True = right, False = left
        inBounds = True
        while (inBounds):
            previousState = currState
            action = argMaxQ(currState)
            updateQTable(action, currState, previousState)
            cartPoleReaction(action, currState)
            inBounds = checkInBounds(currState)
            iterator += 1
        listofAttemptsVariable.append(iterator)


    plt.plot(listofAttempts, "red", label = 'Control')
    plt.plot(listofAttemptsVariable, "blue", label = 'Variable')
    plt.title('Attempts vs. Iterations')
    plt.xlabel('Attempts')
    plt.ylabel('number of iterations')
    plt.legend()
    plt.show()


    



def checkInBounds(currState):
    if thetaToCatagory(currState) == 6 or thetaDotToCatagory(currState) == 3 or xToCatagory(currState) == 3 or xDotToCatagory(currState) == 3:
        return False
    else:
        return True


def updateQTable(action, currState, previousState):
    lookUpTable[int(action)][thetaToCatagory(currState)][thetaDotToCatagory(currState)][xToCatagory(currState)][xDotToCatagory(currState)] += ALPHA*(calcReward(currState) + GAMMA*(maxQ(action, currState)) - lookUpTable[int(action)][thetaToCatagory(previousState)][thetaDotToCatagory(previousState)][xToCatagory(previousState)][xDotToCatagory(previousState)])


def calcReward(currState):
    reward = 0

    if currState.theta >= -12 and currState.theta < -6:
        reward += -1
    elif currState.theta >= -6 and currState.theta < -1:
        reward += 2
    elif currState.theta >= -1 and currState.theta < 0:
        reward += 4
    elif currState.theta >= 0 and currState.theta < 1:
        reward += 4
    elif currState.theta >= 1 and currState.theta < 6:
        reward += 2
    elif currState.theta >= 6 and currState.theta <= 12:
        reward += -1

    if currState.x >= -2.4 and currState.x < -0.8:
        reward += -1
    elif currState.x >= -0.8 and currState.x < 0.8:
        reward += 3
    elif currState.x >= 0.8 and currState.x <= 2.4:
        reward += -1
    
    return reward


def maxQ(action, currState):
    ifLeft = qTable(False, currState)
    ifRight = qTable(True, currState)
    return max(ifLeft,ifRight)


def argMaxQ(currState):    
    ifLeft = qTable(False, currState)
    ifRight = qTable(True, currState)
    return False if ifLeft > ifRight else True


def cartPoleReaction(action, currState):
    force = FORCE_MAG if action > 0 else - FORCE_MAG
    temp = (force+POLEMASS_LENGTH*currState.thetaDot**2*math.sin(currState.theta))/TOTALMASS
    thetaAcc = (GRAVITY*math.sin(currState.theta) - math.cos(currState.theta)*temp)/(LENGTH*(FOURTHIRDS-MASSPOLE*(math.cos(currState.theta))**2*TOTALMASS))
    xAcc = temp-POLEMASS_LENGTH*thetaAcc*math.cos(currState.theta)/TOTALMASS

    currState.x += TAU*currState.xDot
    currState.xDot += TAU*xAcc
    currState.theta += TAU*currState.thetaDot
    currState.thetaDot += TAU*thetaAcc

    return currState


def qTable(action, currState):
    value = lookUpTable[int(action)][thetaToCatagory(currState)][thetaDotToCatagory(currState)][xToCatagory(currState)][xDotToCatagory(currState)]
    return value


def thetaToCatagory(currState):
    if currState.theta >= -12 and currState.theta < -6:
        return 0
    elif currState.theta >= -6 and currState.theta < -1:
        return 1
    elif currState.theta >= -1 and currState.theta < 0:
        return 2
    elif currState.theta >= 0 and currState.theta < 1:
        return 3
    elif currState.theta >= 1 and currState.theta < 6:
        return 4
    elif currState.theta >= 6 and currState.theta <= 12:
        return 5
    else:
        return 6


def thetaDotToCatagory(currState):
    infinity = 1000000
    if currState.thetaDot >= -infinity and currState.thetaDot < -50:
        return 0
    elif currState.thetaDot >= -50 and currState.thetaDot < 50:
        return 1
    elif currState.thetaDot >= 50 and currState.thetaDot <= infinity:
        return 2
    else:
        return 3
    

def xToCatagory(currState):
    if currState.x >= -2.4 and currState.x < -0.8:
        return 0
    elif currState.x >= -0.8 and currState.x < 0.8:
        return 1
    elif currState.x >= 0.8 and currState.x <= 2.4:
        return 2
    else:
        return 3


def xDotToCatagory(currState):
    infinity = 1000000
    if currState.x >= -infinity and currState.x < -0.5:
        return 0
    elif currState.x >= -0.5 and currState.x < 0.5:
        return 1
    elif currState.x >= 0.5 and currState.x <= infinity:
        return 2
    else:
        return 3


main()
