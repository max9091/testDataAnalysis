# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 10:35:38 2017

@author: max9091
"""

from TestAnalysis import testDataAnalysis as tda
import matplotlib.pyplot as plt
from numpy import unique as npUnique
from sys import argv as sysArgv
from os import path as osPath

def PlotResults(Assy):
    saveFig = False
    withScatter = True
    withExcluded = True
    dictTable = tda.dictList2DictTable(Assy.cTR.getRuns({'valid': True}))
    rRatios = npUnique(dictTable['R'])
    for ratio in rRatios:
        ### Plot piecewise defined line throw data points + statistics for each load level
        Assy.cTR.plotFNLine(ratio=ratio, withScatter=withScatter, withExcluded = withExcluded, saveFig=saveFig)
        
        ### Plot Basquin line with constant scatter over all data points
        Assy.cTR.plotFNLine_Basquin(ratio=ratio, withScatter=withScatter, withExcluded = withExcluded, saveFig=saveFig)
    
    plt.show() #to display plots if script is run from command window

if len(sysArgv)>1:
    myDir = str(sysArgv[1])
    if osPath.isdir(myDir):
        if myDir.endswith((osPath.sep,'/')):
            myDir = myDir[0:-1]
#        AssyName = myDir.split('/')[-1] #for debugging with spyder
        AssyName = myDir.split(osPath.sep)[-1]
        Assy = tda.testResults(AssyName)
        Assy.addCyclicTestResults()        
    else:
        print('The input path: <%s> does not exist!' %(myDir))  
        
if len(sysArgv)>2: #specific failures given as additional arguments
    fList = sysArgv[2:]
    Assy.cTR.defineValidResults(fList)
    PlotResults(Assy)

else:
    print('Too few input arguments! The AssyPath must be provided as first argument! Provide the valid failures as further arguments')