# Trenton Griffiths
# CS_5890
# Cart and Pole Reinorcement Learning Problem

import tensorflow as tf
import numpy as np
import math

#GLOBAL CONSTANTS
FORCE_MAG = 10
POLEMASS_LENGTH = 0.209*0.326
TOTALMASS = 0.711+0.209
MASSPOLE = 0.209
TAU = 0.02
GRAVITY = 9.8
LENGTH = 0.326
FOURTHIRDS = 4/3

class state:
    def __init__(self, x, xDot, theta, thetaDot):
        self.x = x
        self.xDot = xDot
        self.theta = theta
        self.thetaDot = thetaDot


def main():
    currState = state(0, 1, 0, 1)
    action = True # True = right, False = left
    iterator = 0

    while ((currState.theta > -90 and currState.theta < 90) and (currState.x > -10 and currState.x < 10)):
        cartPole(action, currState)

        if (currState.thetaDot < -1):
            action = False
        elif(currState.thetaDot > 1):
            action = True

        iterator += 1
        print("Action: ", action, "\nx: ", currState.x, "\nxDot: ", currState.xDot, "\ntheta: ", currState.theta,
             "\nthetaDot: ", currState.thetaDot, "\nIteration: ", iterator, "\n")


def cartPole(action, currState):
    force = FORCE_MAG if action > 0 else - FORCE_MAG
    temp = (force+POLEMASS_LENGTH*currState.thetaDot**2*math.sin(currState.theta))/TOTALMASS
    thetaAcc = (GRAVITY*math.sin(currState.theta) - math.cos(currState.theta)*temp)/(LENGTH*(FOURTHIRDS-MASSPOLE*(math.cos(currState.theta))**2*TOTALMASS))
    xAcc = temp-POLEMASS_LENGTH*thetaAcc*math.cos(currState.theta)/TOTALMASS

    currState.x += TAU*currState.xDot
    currState.xDot += TAU*xAcc
    currState.theta += TAU*currState.thetaDot
    currState.thetaDot += TAU*thetaAcc

    return currState


if __name__=="__main__":
    main()
