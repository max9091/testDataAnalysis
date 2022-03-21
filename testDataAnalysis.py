# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 10:35:38 2017

@author: max9091
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter
# from matplotlib2tikz import save as tikz_save
from tikzplotlib import save as tikz_save
import os
from scipy.stats import norm
import csv

class para:
    'Class to save parameters with value and unit'
    def __init__(self,val,unit):
        self.val = val
        self.unit = unit
    
    def __float__(self):
        return float(self.val)
    
    def __str__(self):
        return str(float(self.val))
    
    def printPara(self):
#        print('%.2f %s' %(self.val,self.unit))
        return '%.2f %s' %(self.val,self.unit)

class oneCyclicTestRun:
    'Class to save one cyclic test run'
    
    def __init__(self, SpNo, RunNo, StartTime, EndTime, F_o, F_u, R, f, RT, Nges, FailureLoc, Comments, valid = False):
        self.SpNo = SpNo
        self.RunNo = RunNo
        self.StartTime = StartTime
        self.EndTime = EndTime
        self.F_o = F_o
        self.F_u = F_u
        if (R - round(float(F_u)/float(F_o),6)):     #if the given R is different from the ratio F_u/F_o
            print('ERROR: Check given data of Specimen No. %d!!! --> F_u/F_o = %g != R = %g' %(SpNo,F_u/F_o,R))
            self.R = round(float(F_u)/float(F_o),6)
        else:
            self.R = R
        self.f = f
        self.RT = RT
        self.Nges = Nges
        self.FailureLoc = FailureLoc
        self.Comments = Comments
        self.valid = valid
        
    def __str__(self):
        return(';'.join([str(x) for x in list(self.__dict__.values())]))
#        return '%02d;%02d;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % list(self.__dict__.values())
        


class staticTestResults:
    testRuns = []
    
#    def __init__()

class cyclicTestResults:
    validFailures = []
    testRuns = []
    axisLimits = []
    AssyName = ''
    
#    def __init__(self,name,desciption = '',components = ''):
#        self.name = name
#        self.desciption = desciption
#        self.components = components
        
    def __str__(self):
        if len(self.testRuns) > 0:
            output = ';'.join(list(self.testRuns[0].__dict__.keys())) + '\n'
            output = output + '-;-;dd.mm.yyyy hh:mm;dd.mm.yyyy hh:mm;kN;kN;1;Hz;°C;1;-;-' + '\n' #toDo: read from object!
            for run in self.testRuns:
                output = output + str(run) + '\n'
            return output
        else:
            return None
    
#    def getRun(self,SpNo,RunNo): #--> replaced by new method getRuns(self, which)
#        for run in self.testRuns:
#            if run.SpecimenNo == SpNo and run.RunNo == RunNo: #if found
#                return run
#        #if not found --> return null
#        return None
            
#    def addNewRun(self,testRun):
#        self.testRuns.append(testRun)

    def getValidResults(self,SpNo = None):
        dataList = []  
        SpecimenFound = False
        for run in self.testRuns:
            if run.valid: 
                results4SpN = run.__dict__
#                results4SpN = {'SpNo': run.SpNo, 'R':run.R, 'FailureLoc': run.FailureLoc, 'F_o': run.F_o, 'Nges': run.Nges} #[R-Ratio, FailureLoc, F_o, N]
                if run.SpNo == SpNo:
                    dataList.append(results4SpN)
                    SpecimenFound = True
                elif SpNo == None:
                    dataList.append(results4SpN)
                        
        #if not found --> return null
        if SpNo != None and SpecimenFound == False: #SpecimenNo not found
            print('INFO: SpecimenNo not found!')
            return None
        #if found or all data
        
        return dataList
    
    def getRuns(self,which = {}):
        #function to get Runs with specific parameters defined in the dict which
        data = []
        #delete all items with value None's in which
        tempWhich = {} 
        for item in which.items():
            if item[1] != None:
                tempWhich[item[0]] = item[1]
        which = tempWhich
        del(tempWhich)
        #iterate over all testRuns and find matching runs for input which
        for run in self.testRuns:
            if set(which.items()) == which.items() & run.__dict__.items():
                data.append(run.__dict__)
        return data

    def defineValidResults(self,failureList):
        self.validFailures = failureList
        for run in self.testRuns:
            if run.FailureLoc in failureList:
                run.valid = True
            else:
                run.valid = False
    
        
    def calcLinearRegLine(self, ratio = None, failureList = np.array([]), valid = True):
#        data_F = np.array([])
#        data_N = np.array([])
#        data_failure = np.array([])
        if len(failureList) > 0:
            dataList = []
            for failure in failureList:
                temp = self.getRuns({'R': ratio, 'FailureLoc': failure, 'valid': valid})
                for entry in temp: dataList.append(entry)
        else:
            dataList = self.getRuns({'R': ratio, 'valid': valid})
        
        dataDict = dictList2DictTable(dataList) 
#        for d in dataList:
#            data_F = np.append(data_F, d['F_o'])
#            data_N = np.append(data_N, d['Nges'])
#            data_failure = np.append(data_failure,d['FailureLoc']) #--> if input FailureLoc = None --> returnvalue != NONE!!!
            
        kd_FN = np.polyfit(np.log10(dataDict['F_o']),np.log10(dataDict['Nges']),1)
        return kd_FN, dataDict


    def calcStatistics_4eachLL(self, ratio = None, failureList = None):   
        kd, data = self.calcLinearRegLine(ratio=ratio,failureList=failureList)
        data_FN = np.transpose(np.concatenate(([data['F_o']],[data['Nges']])))
         
        ### calc statistics for each load horizont
        all_lhF = np.unique(data['F_o'])
        all_lhF.sort() #.sort() gives ascending order by default
        
        data_pue = []
        
        for lhF in all_lhF:
            #calc Statistics for each load level if possible
            all_lhN = data_FN[np.where(data_FN[:,0] == lhF)][:,1]
            all_lhN.sort()
            if len(all_lhN) > 1:
                pue = calcPue(len(all_lhN))

                kd_pue = np.polyfit(pue,np.log10(all_lhN),1)  #linear regression by minimizing in cycle direction!
                p = np.poly1d(kd_pue)
#                linPue = np.arange(1,99)
#                logN = p(linPue)
#                plt.semilogx(10**logN,linPue,'k')
                
                N_90 = 10**p(90)
                N_10 = 10**p(10)               
                T_N = N_90/N_10
                
#                distFact = 2*1.281552   #c.f. Haibach p.32, and Wikipedia: Normalverteilung
#                logN_50_test2 = np.mean(np.log10(all_lhN))
#                print('logN_50 = ',10**logN_50_test2) # check: logN_50 and p(50) are equal!
#                print('p(50) = ',10**p(50)) # check: logN_50 and p(50) are equal!
#                n_test2 = len(all_lhN)
#                s_logN_test2 = np.sqrt(1/(n_test2-1)*np.sum((np.log10(all_lhN) - logN_50_test2)**2))
#                print('regline T_N = ',T_N) #zum testen
#                print('calc T_N = ',10**(-distFact*s_logN_test2)) #zum testen
#                print('factor = ',np.log10(1/T_N)/s_logN_test2) #zum testen
                
                data_pue.append({'F_o': lhF, 'kd_pue': kd_pue, 'T_N': T_N, 'N_90': N_90, 'N_50': 10**np.polyval(kd_pue,50), 'N_10': N_10, 'Nges': all_lhN})    
            elif len(all_lhN) == 1:
                data_pue.append({'F_o': lhF, 'kd_pue': None, 'T_N': None, 'N_90': None, 'N_50': all_lhN[0], 'N_10': None, 'Nges': all_lhN})
        
        data_pue.reverse() #so that the highest load is at the first position in the list
        
        return data_pue
    
    def plotStatistics_4eachLL(self, ratio = None, failureList = np.array([]), **options):
        if 'figHandles' in options:
            fig, axScatter = options.get('figHandles')
        else:    
            fig, axScatter = plt.subplots()
        
        if 'data_pue' in options:
            data_pue = options.get('data_pue')
        else:
            data_pue = self.calcStatistics_4eachLL(ratio = ratio, failureList = failureList)
            
        for lhF in data_pue:
            #plot Statistics for each load level if possible
            all_lhN = lhF['Nges']
            if len(all_lhN) > 1:
                all_lhN.sort()
                #Plot data points in probability net
                pue = calcPue(len(all_lhN))
                axScatter.semilogx(all_lhN,pue,'ko')
                #TODO: get the marker colors (different failures) from Woehlerline and put same colors! 

                #Calc and plot linear regression line in probability net
                p = np.poly1d(lhF['kd_pue'])
                linPue = np.arange(1,99)
                logN = p(linPue)
                axScatter.semilogx(10**logN,linPue,'r')
                
                N_90 = 10**p(90)
                N_10 = 10**p(10)
                
                T_N = N_90/N_10
                
                axScatter.text(10**p(50),50,'$T_N = $\n1:%1.3f' %(1/T_N), ha = 'left')
                
        
#        formatFNPlot(axScatter)
        axScatter.grid(1,'both')
        axScatter.yaxis.set_major_formatter(ScalarFormatter())
        axScatter.yaxis.set_minor_formatter(ScalarFormatter())
        axScatter.set_title('Probablility net for each load level')
        axScatter.set_xlabel('')
        axScatter.set_ylabel('Survival Probability P_ue in %')
        
        return fig, axScatter

    def calcStatistics_4allData(self, ratio = None, failureList = np.array([])):

        kd, data = self.calcLinearRegLine(ratio=ratio,failureList=failureList)
        data_FN = np.transpose(np.concatenate(([data['F_o']],[data['Nges']])))
        
        ### calc one scatter range for all data points 
        # (cf. Method of discrete load steps(Perlenschnurverfahren) -> Martin A. et al - 2011 - Zur Auswertung der Schwing..)
        data_N_oneLL = []
        data_F_oneLL = []
        F_oneLL = 10**np.mean(np.log10(data['F_o']))  #Accord. to DIN 50100:2016-12: free choise(Page 53) --> I choose the mean load level

        for dataPoint in data_FN:
            F_test = dataPoint[0]
            N_test = dataPoint[1]
            N_oneLL = N_test*(F_oneLL/F_test)**(kd[0]) #DIN 50100:2016-12 Equ.(36)
            data_N_oneLL.append(N_oneLL)
            data_F_oneLL.append(F_oneLL)

        #--> Calculation of scatter range T_N
        logN_50_oneLL = np.mean(np.log10(data_N_oneLL))  #DIN 50100:2016-12 Equ.(37)
        n = len(data_N_oneLL)
        s_logN = (n-1.74)/(n-2)*np.sqrt(1/(n-2)*np.sum((np.log10(data_N_oneLL) - logN_50_oneLL)**2)) # DIN 50100:2016-12 Equ.(38)+(39) or. cf.: Martin A. et al - 2011
        
#        distFact = 2*1.281552   #c.f. Haibach p.32,33, and Wikipedia: Normalverteilung C = 80% (range 10% - 90%)
        distFact = 2*norm.ppf(0.9) #norm.ppf(0.9) of scipy.stats package gives the scatter range for 90% of the standard normal distribution 
        
        T_N = 10**(-distFact*s_logN) # cf. Haibach p.33 (2.1-30)
        T_F = 10**(np.log10(T_N)/(-kd[0]))  # cf. Wiedemann p.844 (7.2-4): -kd[0] = -k --> because regression line was calculated with y=N over x=F
        
        data_scatterAll = {'s_logN': s_logN, 'T_N': T_N, 'T_F': T_F ,'kd': kd, 'data': data, 
                           'data_N_oneLL': data_N_oneLL, 'data_F_oneLL': data_F_oneLL, 
                           'logN_50_oneLL': logN_50_oneLL, 'F_oneLL': F_oneLL}
        
        return data_scatterAll
    
    
    def plotStatistics_4allData(self, ratio = None, failureList = np.array([]), **options):
        if 'figHandles' in options:
            fig, axScatter = options.get('figHandles')
        else:    
            fig, axScatter = plt.subplots()
        
        if 'data_scatter' in options:
            data_scatter = options.get('data_scatter')
        else:
            data_scatter = self.calcStatistics_4allData(ratio=ratio, failureList=failureList)
        
        kd = data_scatter['kd']
        F_D =  10**((np.log10(1e6) - kd[1])/kd[0])
        data = data_scatter['data']
        
        line1, = axScatter.loglog(data['Nges'],data['F_o'],'yo',)
        line1.set_label('data points')
        
        paraScatter = self.calcScatter4LineParameters(scatter=data_scatter)
        
        #Plot linear regression line
        p = np.poly1d(kd)
        linF = np.logspace(np.log10(min(data['F_o'])),np.log10(max(data['F_o'])),num=10)
        logN = p(np.log10(linF))
        line, = axScatter.loglog(10**logN,linF,'r-')   #plot linear regression line
        line.set_label('linear regression line\n'+ \
                       '($k = %.4f$, $k_{\max} = %.4f$, $k_{\min} = %.4f$,\n' %(-kd[0],max(np.abs(paraScatter['k'])),min(np.abs(paraScatter['k']))) + \
                       ' $F(N=1e6) = %.4f$,\n $F_{\max} = %.2f$, $F_{\min} = %.2f$)'%(F_D, max(paraScatter['F_D']),min(paraScatter['F_D'])))
        
        ### calc one scatter range for all data points 
        # (cf. Method of discrete load steps(Perlenschnurverfahren) -> Martin A. et al - 2011 - Zur Auswertung der Schwing..)
        data_N_oneLL = data_scatter['data_N_oneLL']
        data_F_oneLL = data_scatter['data_F_oneLL']
        F_oneLL = data_F_oneLL[0]

        
        #Plot data points at one single load level (mean load level)
        line2, = axScatter.loglog(data_N_oneLL,data_F_oneLL,'mx')       
        line2.set_label('data points on mean load level $F = %0.2f$' %F_oneLL)

        #--> Calculation of scatter range T_N for 2D-Data
        logN_50_oneLL = data_scatter['logN_50_oneLL']
#        n = len(data_N_oneLL)
#        s_logN = (n-1.74)/(n-2)*np.sqrt(1/(n-2)*np.sum((np.log10(data_N_oneLL) - logN_50_oneLL)**2)) # cf.: Martin A. et al - 2011
        s_logN = data_scatter['s_logN']
        
#        distFact = 2*1.281552   #c.f. Haibach p.32,33, and Wikipedia: Normalverteilung C = 80%
        
#        T_N = 10**(-distFact*s_logN) # cf. Haibach p.33 (2.1-30)
#        T_F = 10**(np.log10(T_N)/(-kd[0]))  # cf. Wiedemann p.844 (7.2-4): -kd[0] = -k --> because regression line was calculated with y=N over x=F
        T_N = data_scatter['T_N']
        T_F = data_scatter['T_F']
        
#        axScatter.text(10**logN_50_oneLL,F_oneLL,'T_N = 1:%1.3f' %(1/T_N), ha = 'left', va = 'bottom')
#        axScatter.text(10**logN_50_oneLL,F_oneLL,'T_F = 1:%1.3f' %(1/T_F), ha = 'left', va = 'top')
        
        #Plot Scatter T_N
        plotSc_T_N = [(10**logN_50_oneLL)/np.sqrt(T_N),(10**logN_50_oneLL)*np.sqrt(T_N)]
        lineScT_N, = axScatter.loglog(plotSc_T_N,[F_oneLL,F_oneLL],'m-',linewidth=3 ,marker='|', ms=10, alpha=0.4, solid_capstyle="butt")
        lineScT_N.set_label('Scatter $T_N = $1:%1.3f' %(1/T_N))
        axScatter.loglog(plotSc_T_N[0],[F_oneLL],'m',marker=5,ms=10,alpha=0.4)
        axScatter.loglog(plotSc_T_N[1],[F_oneLL],'m',marker=4,ms=10,alpha=0.4)
        
        #Plot Scatter T_F
        plotSc_T_F = [F_oneLL*np.sqrt(T_F),F_oneLL/np.sqrt(T_F)]
        lineScT_F, = axScatter.loglog([10**logN_50_oneLL,10**logN_50_oneLL],plotSc_T_F,'g-',linewidth=3 ,marker='_', ms=10, alpha=0.4, solid_capstyle="butt")
        lineScT_F.set_label('Scatter $T_F = $1:%1.3f' %(1/T_F))
        axScatter.loglog([10**logN_50_oneLL],plotSc_T_F[0],'g',marker=7,ms=10,alpha=0.4)
        axScatter.loglog([10**logN_50_oneLL],plotSc_T_F[1],'g',marker=6,ms=10,alpha=0.4)
        
        formatFNPlot(axScatter)
        axScatter.set_title('Shift all data points to mean load level (%s)' %self.AssyName)
        axScatter.set_xlabel('')
        axScatter.set_ylabel('$F_{\max}$ [kN]') #F_o = F_max
        
        data_scatterAll = {'s_logN': s_logN, 'T_N': T_N, 'T_F': T_F ,'kd': kd, 'data': data}
        
        return data_scatterAll
    

    def plotFNPoints(self, data = {}, **options):
        if len(data) == 0:
            data = dictList2DictTable(self.getValidResults())
        
        if 'failureList' in options and not len(options.get('failureList')) == 0:
            failureList = options.get('failureList')
        else:  
            failureList = np.unique(data['FailureLoc'])
                
        if 'figHandles' in options:
            fig, ax = options.get('figHandles')
            sepFig = False
        else:
            fig, ax = plt.subplots()
            sepFig = True
        
        if 'ratio' in options:
            ratio = options.get('ratio')
        else:
            ratio = np.unique(data['R'])
            
        if 'withSpNo' in options:
            withSpNo = options.get('withSpNo')
        else:
            withSpNo = False
        
        for failure in failureList:
            data_N = np.array([])
            data_F = np.array([])
            for i, dataPoint in enumerate(data['FailureLoc']):
                if dataPoint == failure:
                    data_N = np.append(data_N,data['Nges'][i])
                    data_F = np.append(data_F,data['F_o'][i])
            if len(data_N) and len(data_F):
                if failure != 'excluded':
                    line1, = ax.loglog(data_N,data_F,'o') #plot data points
                    line1.set_label('data points ({})'.format(failure))
                else:
                    line1, = ax.loglog(data_N,data_F,'rx') #plot excluded data points
                    line1.set_label('excluded points')
            del(data_N,data_F)
        
        if withSpNo:
            for i, SpNo in enumerate(data['SpNo']):
                ax.text(data['Nges'][i],data['F_o'][i],'{:g}'.format(SpNo),
                        fontsize=6,
                        ha='center',va='center')
        
        formatFNPlot(ax)
        ax.set_title('Data points of ' + self.AssyName)
        
        axisLimits = [ax.get_xlim()[0], ax.get_xlim()[1], ax.get_ylim()[0], ax.get_xlim()[1]]
        if sepFig:
            ax.text(axisLimits[0]**(1+0.01),axisLimits[2]**(1+0.01),
                    'R = %s' %('; R = '.join(str(x) for x in list(ratio))), 
                    VerticalAlignment='bottom', bbox=dict(facecolor='white', alpha=1));
        
        #TODO: Mark each failure point with specific color and return the markers
        
        return fig, ax
  
    
    def plotFNPoints_Excluded(self, **options):
        if 'figHandles' in options:
            fig, ax = options.get('figHandles')
        
        if 'withSpNo' in options:
            withSpNo = options.get('withSpNo')
        else:
            withSpNo = False
        
        which = {'valid': False}
        if 'ratio' in options:
            which['R'] = options.get('ratio')
            
        data = dictList2DictTable(self.getRuns(which)) #get excluded points (= not valid)!
        
        if 'FailureLoc' in data.keys():
            for i,FLoc in enumerate(data['FailureLoc']): data['FailureLoc'][i] = 'excluded'
        
            if 'figHandles' in options:
                fig, ax = self.plotFNPoints(data, figHandles = [fig, ax], withSpNo = withSpNo)
            else:
                fig, ax = self.plotFNPoints(data, withSpNo = withSpNo)
        else:
            return None
            
        return fig, ax

    def plotFNLine(self,axisLimits = None, ratio = None, failureList = np.array([]), valid = True, withScatter = False, **options):
        if 'saveFigName' in options:
            saveFig = True
            saveFigName = options.get('saveFigName')
        elif 'saveFig' in options:
            saveFig = options.get('saveFig')
            saveFigName = ''
        else:
            saveFig = False
        
        kd, data = self.calcLinearRegLine(ratio, failureList, True)
    
        if ratio == None:
            ratio = data['R'][0]
            if len(np.unique(data['R'])) > 1:
                print('WARNING: Mulitple ratios present in data but no specific ratio given for plotting! -> plotting ratio of first test')
        
        if len(failureList) < 1:
            failureList = np.unique(data['FailureLoc'])
            failureList.sort()  #sort in alphabetic order -> e.g. to always get the same filenames
        
        #Plot the figure
        if withScatter:
            fig, (axScatter, axFN) = plt.subplots(2,1,sharex = True, figsize=(12,8))
        else:
            fig, axFN = plt.subplots()
        
        #Plot data points
        self.plotFNPoints(data, failureList = failureList, figHandles = [fig, axFN])
        if 'withExcluded' in options:            
            self.plotFNPoints_Excluded(figHandles = [fig, axFN], ratio = ratio)
        
        
        #Plot 50% survival probability line
        data_pue = self.calcStatistics_4eachLL(ratio=ratio, failureList=failureList)
        
        data_F_o = getDictItemsFromList(data_pue,'F_o')
        data_N_50 = getDictItemsFromList(data_pue,'N_50')
        line2, = axFN.loglog(data_N_50,data_F_o,'r-')   #plot Woehler line (50% survival probability)
        line2.set_label('Woehlerline (P = 50%)')
        
#        
        if axisLimits != None:
            axFN.axis(axisLimits)         #take axisLimits from func. parameters
        elif self.axisLimits == []:
            axFN.axis('auto')             #take auto axisLimits (if not defined before)
        else:
            axFN.axis(self.axisLimits)    #take axisLimits from object variable (must be defined before)
        axisLimits = [axFN.get_xlim()[0], axFN.get_xlim()[1], axFN.get_ylim()[0], axFN.get_xlim()[1]]
        
        if withScatter:
            self.plotStatistics_4eachLL(data_pue=data_pue, figHandles = (fig, axScatter))
            scatter_str = '_withScatter_'
            
            data_N_90 = getDictItemsFromList(data_pue,'N_90')
            maskArray = data_N_90 != np.array(None) # 1) to delete None objects in array (see calcStatistics_4eachLL if all_lhN == 1)
                                                    # 2) assuming the same maskArray for N_10!
            data_N_10 = getDictItemsFromList(data_pue,'N_10')
            
            
            line_N_90, = axFN.loglog(data_N_90[maskArray],data_F_o[maskArray],'m--')
            line_N_90.set_label('90% Survival Probability')
            
            line_N_10, = axFN.loglog(data_N_10[maskArray],data_F_o[maskArray],'m-.')
            line_N_10.set_label('10% Survival Probability')
        else:
            scatter_str = '_'
        
        axFN.text(axisLimits[0]**(1+0.01),axisLimits[2]**(1+0.01),'R = %g' %(ratio), 
                  VerticalAlignment='bottom', bbox=dict(facecolor='white', alpha=1));
#                  
#                  
#        axFN.text(axisLimits[0]+axisRangeX**0.01,axisLimits[2]+axisRangeY**0.01,
#                'R = %s' %('; R = '.join(str(x) for x in list(ratio))), 
#                VerticalAlignment='bottom', bbox=dict(facecolor='white', alpha=1));
        
        formatFNPlot(axFN)
        axFN.set_title('Woehlerline of ' + self.AssyName)
        
        if saveFig:
            defaultPath = 'plots/'
            if saveFigName == '':
                fLocs_str = 'Failure=' + '+'.join(str(x) for x in list(failureList))
                saveFigName = 'PY_FNLine_%s_R=%g%s%s' %(self.AssyName,ratio,scatter_str,fLocs_str)
            
            fig.savefig(defaultPath + saveFigName + '.png')
            fig.savefig(defaultPath + saveFigName + '.pdf')
            tikz_save( defaultPath + saveFigName + '.tikz' )
                    
        if withScatter:
            return fig, (axScatter, axFN)
        else:
            return fig, axFN
            
    
    def plotFNLine_Basquin(self,axisLimits = None, ratio = None, failureList = np.array([]), valid = True, withScatter = False, **options):
        if 'figHandles' in options: #Plot the figure in the given figHandles
            if withScatter:
                fig, (axScatter, axFN) = options.get('figHandles')
            else:
                fig, axFN = options.get('figHandles')
                axScatter = None
        else: #Plot the figure in new figHandles
            if withScatter:
                fig, (axScatter, axFN) = plt.subplots(2,1,sharex = True,figsize=(12,8))
            else:
                fig, axFN = plt.subplots()
                axScatter = None
        if 'saveFigName' in options:
            saveFig = True
            saveFigName = options.get('saveFigName')
        elif 'saveFig' in options:
            saveFig = options.get('saveFig')
            saveFigName = ''
        else:
            saveFig = False
        if 'colorScheme' in options:
            cS = options.get('colorScheme')
        elif 'cS' in options:
            cS = options.get('cS')
        else:
            cS = {'regLine': 'r','dataPoints': 'failures','scatter': 'm'}
        
        if 'withSpNo' in options:
            withSpNo = options.get('withSpNo')
        else:
            withSpNo = False
            
        if 'LoadType' in options:
            LoadType = options.get('LoadType')
        else:
            LoadType = ('F_{\max}','kN')
        
        kd, data = self.calcLinearRegLine(ratio, failureList, valid)
        F_D =  10**((np.log10(1e6) - kd[1])/kd[0])
        
        if ratio == None:
            ratio = data['R'][0]
            if len(np.unique(data['R'])) > 1:
                print('WARNING: Mulitple ratios present in data but no specific ratio given for plotting! -> plotting ratio of first test')
        
        if len(failureList) < 1:
            failureList = np.unique(data['FailureLoc'])
            failureList.sort()  #sort in alphabetic order -> e.g. to always get the same filenames
        
            
        #Plot data points
        self.plotFNPoints(data, failureList = failureList, figHandles = (fig, axFN), withSpNo = withSpNo)
        if 'withExcluded' in options:            
            self.plotFNPoints_Excluded(figHandles = [fig, axFN], ratio = ratio, withSpNo = withSpNo)
        
        #Calc linear regression line (minimizing in cycle direction!)
         
        p = np.poly1d(kd)
        linF = np.logspace(np.log10(min(data['F_o'])),np.log10(max(data['F_o'])),num=10)
        logN = p(np.log10(linF))
        line2, = axFN.loglog(10**logN,linF,cS['regLine']+'-')   #plot linear regression line
        slopeVar = 'k'
        if LoadType[1]=='MPa':
            slopeVar = 'm'    
        line2.set_label('linear regression line\n($%s$ = %.2f, $%s(N=1e6)$ = %.2f %s)' %(slopeVar,-kd[0],LoadType[0],F_D,LoadType[1]))
        
        if axisLimits != None:
            axFN.axis(axisLimits)         #take axisLimits from func. parameters
        elif self.axisLimits == []:
            axFN.axis('auto')             #take auto axisLimits (if not defined before)
        else:
            axFN.axis(self.axisLimits)    #take axisLimits from object variable (must be defined before)
        axisLimits = [axFN.get_xlim()[0], axFN.get_xlim()[1], axFN.get_ylim()[0], axFN.get_xlim()[1]]
        
        axFN.text(axisLimits[0]**(1+0.01),axisLimits[2]**(1+0.01),'R = %g' %(ratio), 
                  VerticalAlignment='bottom', bbox=dict(facecolor='white', alpha=1));
                  
        if withScatter:
            scatter_str = '_withScatter_'
            scatter = self.calcStatistics_4allData(ratio=ratio, failureList=failureList, )
            self.plotStatistics_4allData(data_scatter = scatter, figHandles = (fig, axScatter))
            
            data_F_o = np.unique(data['F_o'])
            data_F_o.sort()
            
            data_N_50 = 10**p(np.log10(data_F_o))
            data_N_90 = data_N_50*np.sqrt(scatter['T_N'])
            data_N_10 = data_N_50/np.sqrt(scatter['T_N'])
            
            line_N_90, = axFN.loglog(data_N_90,data_F_o,cS['scatter']+'--')
            line_N_90.set_label('90% Survival Probability')
            
            line_N_10, = axFN.loglog(data_N_10,data_F_o,cS['scatter']+'-.')
            line_N_10.set_label('10% Survival Probability')
            
            if 'customSurvivalProb' in options:
                cSP = options.get('customSurvivalProb')
                scatter_str += 'cSP=%g_' %(cSP)
                
                N_cSP_oneLL = 10**(scatter['logN_50_oneLL'] - norm.ppf(cSP)*scatter['s_logN'])
                F_oneLL = scatter['F_oneLL']
                data_N_cSP = N_cSP_oneLL*(data_F_o/F_oneLL)**(kd[0])
                
                line_N_cSP, = axFN.loglog(data_N_cSP,data_F_o,cS['scatter']+':')
                line_N_cSP.set_label('%g%% Survival Probability\n$N(F=%g) = %d$, $N(F=%g) = %d$' %(cSP*100,data_F_o[-1],data_N_cSP[-1],data_F_o[0],data_N_cSP[0]))
                
        else:
            scatter_str = '_'
            
#            # TO CHECKK T_F
#            data_F_90 = data_F_o*np.sqrt(scatter['T_F'])
#            data_F_10 = data_F_o/np.sqrt(scatter['T_F'])
#            line_F_90, = axFN.loglog(data_N_50,data_F_90,'r--')
#            line_F_90.set_label('90% Survival Probability F_90')
#            line_F_10, = axFN.loglog(data_N_50,data_F_10,'r-.')
#            line_F_10.set_label('10% Survival Probability F_10')
        
        # Format the plot
        formatFNPlot(axFN,LoadType=LoadType)
        axFN.set_title('Basquinline of ' + self.AssyName)
        fig.tight_layout()
        
        if saveFig:
            defaultPath = 'plots/'
            if saveFigName == '':
                fLocs_str = 'Failure=' + '+'.join(str(x) for x in list(failureList))
                saveFigName = 'PY_BasquinLine_%s_R=%g%s%s' %(self.AssyName,ratio,scatter_str,fLocs_str)
            
            fig.savefig(defaultPath + saveFigName + '.png',dpi=300)
            fig.savefig(defaultPath + saveFigName + '.pdf')
            tikz_save(defaultPath + saveFigName + '.tikz')
        
        return fig, (axScatter, axFN)
    
    def saveFNLine_Basquin_parameter2file(self,ratio=None,failureList = np.array([]),F_kp = 1,**options):
        
        scatterData = self.calcStatistics_4allData(ratio=ratio,failureList=failureList)
        k = -scatterData['kd'][0]
        
        if ratio == None:
            ratio = scatterData['data']['R'][0]
            if len(np.unique(scatterData['data']['R'])) > 1:
                print('WARNING: Mulitple ratios present in data but no specific ratio given for plotting! -> plotting ratio of first test')
        
        if len(failureList) < 1:
            failureList = np.unique(scatterData['data']['FailureLoc'])
            failureList.sort()  #sort in alphabetic order -> e.g. to always get the same filenames
        
        paraData = {}
        paraData['k'] = k
        paraData['F_kp'] = F_kp*1e3 #in N  #--> F_kp given in kN but printed in N!!! due to further analysis together with MOPs!!!
        
        p = np.poly1d(scatterData['kd'])
        F_ref = 10 #kN
        N_ref = 10**(p(np.log10(F_ref)))
        
        N_kp_cSP50p = evalBasquinLine_4F(F_kp,k,F_ref,N_ref)      
        
        paraData['N_kp_cSP50p'] = N_kp_cSP50p
        paraData['N_kp_cSP10p'] = 10**(np.log10(N_kp_cSP50p) - norm.ppf(0.10)*scatterData['s_logN'])
        paraData['N_kp_cSP90p'] = 10**(np.log10(N_kp_cSP50p) - norm.ppf(0.90)*scatterData['s_logN'])
        paraData['N_kp_cSP95p'] = 10**(np.log10(N_kp_cSP50p) - norm.ppf(0.95)*scatterData['s_logN'])
        paraData['N_kp_cSP99p'] = 10**(np.log10(N_kp_cSP50p) - norm.ppf(0.99)*scatterData['s_logN'])
        
        defaultPath = 'plots/'
        fLocs_str = '' + '+'.join(str(x) for x in list(failureList))
        #Tube-neck Fatigue_TD_9A-I-13E002-S10_R001.csv
        basquinParameterFileName = '%s Fatigue_TD_%s_%s.csv' %(fLocs_str,self.AssyName,rratio2str(ratio))
        
        print(paraData)
        
        with open(os.path.join(os.getcwd(),defaultPath,basquinParameterFileName), 'w', newline='') as csvfile:
            csvWriter = csv.writer(csvfile, delimiter=';')
            for key, value in paraData.items():
                print(key+':'+str(value))
                csvWriter.writerow([key,value])
            
    
    def calcScatter4LineParameters(self,ratio=None, **options):
        if 'scatter' in options:
            scatterAll = options.get('scatter')
        else:
            scatterAll = self.calcStatistics_4allData(ratio = ratio)
        
        kd = scatterAll['kd']
        max_F_o = max(scatterAll['data']['F_o'])
        min_F_o = min(scatterAll['data']['F_o'])
        topMost_N_50 = 10**np.polyval(kd,np.log10(max_F_o))
        topMost_N_10 = topMost_N_50/np.sqrt(scatterAll['T_N'])  # --> corner of max_F_o with 10% Survival Probability
        bottomMost_N_50 = 10**np.polyval(kd,np.log10(min_F_o))
        bottomMost_N_90 = bottomMost_N_50*np.sqrt(scatterAll['T_N']) # --> corner of min_F_o with 90% Survival Probability
        k_min = (np.log10(bottomMost_N_90/topMost_N_10)/np.log10(min_F_o/max_F_o))
#        kd_check = np.polyfit(np.log10(np.array([max_F_o,min_F_o])),np.log10(np.array([topMost_N_10,bottomMost_N_90])),1) # TO CHECK k_min
        
        k_max = kd[0]
        
        N_D = 1e6
        F_D =  10**((np.log10(N_D) - kd[1])/kd[0])
        
        F_D_min = F_D*np.sqrt(scatterAll['T_F']) #F(N=1e6) with P=90%
        F_D_max = F_D/np.sqrt(scatterAll['T_F']) #F(N=1e6) with P=10%
        
#        return topMost_N_50
        return {'k': [k_min,k_max], 'F_D': [F_D_min, F_D_max]}
#        return max_F_o

    
def formatFNPlot(ax,**options):
    if 'LoadType' in options:
        LoadType = options.get('LoadType')
    else:
        LoadType = ('F_{\max}','kN')
        
    ax.set_xlabel('$N$ [cycles to failure]')
    ax.set_ylabel('$%s$ [%s]' %(LoadType[0],LoadType[1]))
    
    ax.grid(1,'both')
#    ax.yaxis.set_major_formatter(ScalarFormatter())
#    ax.yaxis.set_minor_formatter(ScalarFormatter())
    ax.legend() # plot the legend with labels defined in line.set_label()
    
    # from litepresence at https://stackoverflow.com/questions/21920233/matplotlib-log-scale-tick-label-number-formatting/33213196
    # user controls
    #####################################################
    sub_ticks = [10,11,12,14,16,18,22,25,35,45] # fill these midpoints
    sub_range = [-8,8] # from 100000000 to 0.000000001
    myformat = "%g" # standard float string formatting
    
    # set scalar and string format floats
    #####################################################
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(myformat))
    ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(ticker.FormatStrFormatter(myformat))
    
#    #force 'autoscale'
#    #####################################################
#    yd = [] #matrix of y values from all lines on plot
#    for n in range(len(ax.get_lines())):
#            line = ax.get_lines()[n]
#            yd.append((line.get_ydata()).tolist())
#    yd = [item for sublist in yd for item in sublist]
#    ymin, ymax = np.min(yd), np.max(yd)
#    ax.set_ylim([0.9*ymin, 1.1*ymax])
#    
    ymin, ymax = ax.get_ylim()
    # add sub minor ticks
    #####################################################
    set_sub_formatter=[]
    for i in sub_ticks:
        for j in range(sub_range[0],sub_range[1]):
            set_sub_formatter.append(i*10**j)
    k = []
    for l in set_sub_formatter:
        if ymin<l<ymax:
            k.append(l)
    ax.set_yticks(k)
    #####################################################
    
    
def calcPue(n):
    pue = np.array([])
    for j in range(1,n+1):
        pue = np.append(pue,(3*j - 1)/(3*n + 1)*100)
    pue = np.flip(pue,0)
    return pue

def getDictItemsFromList(oneDictList,dictKey):
    itemArray = np.array([])
    for oneDict in oneDictList:
        itemArray = np.append(itemArray,oneDict[dictKey])
    return itemArray

def dictList2DictTable(oneDictList):
    dictTable = {}
    for oneDict in oneDictList:
        for key in oneDict:
            if not key in dictTable:
                dictTable[key] = np.array([])
            dictTable[key] = np.append(dictTable[key],oneDict[key])
        
    return dictTable

def evalBasquinLine_4F(F,k,F_ref,N_ref): #evaluates the cylces N for a given load level F (of given BasquinLine with slope k and point F_ref,N_ref)
    N = N_ref*((F_ref/F)**k)
    return N
    
def evalBasquinLine_4N(N,k,F_ref,N_ref): #calculates the force F for a given cycle no. N (of given BasquinLine with slope k and point F_ref,N_ref)
    F = F_ref*((N_ref/N)**(1/k))
    return F

def rratio2str(Rratio):
    #%RRATIO2STR converts the numerical R-ratio into a string without characters
    #%which are incompatible with paths or filenames
    #%   e.g.:   R = 0.01 -> R = 'R001'
    #%           R = -1 -> R = 'Rm1'
    Rstr = 'R'
    
    if Rratio < 0:
        Rstr += 'm'
    
    ratioValStr = '{:g}'.format(np.abs(Rratio))
    if Rratio%1 > 0: #if Rratio is a decimal number
        ratioValStr = ratioValStr.replace('.','')
    
    return Rstr + ratioValStr


def evalNges(NgesString):
    try:
        tempRes = NgesString
        if tempRes.startswith('='):
            tempRes = tempRes[1:]
        if tempRes.endswith('+'):
            tempRes = tempRes[:-1]
    except:
        print('ERROR: Previous run results could not be evaluated!')
        tempRes = '-1'
    return eval(tempRes)


class testResults:
    cTR = None
    
    def __init__(self, name, desciption = '', components = [], path=''):
        self.name = name
        self.path = path
        self.desciption = desciption
        self.components = components
        self.SpecimensTested = np.array([]);
        
#    def addDisciption(self,desciption):
#        self.desciption = desciption
        
    def addStaticTestResults(self):
        self.sTR = self.readStaticTestData()
        
    def addCyclicTestResults(self,path=''):
        if path != '':
            self.path = path
        new_cTR = self.readCyclicTestData()
        #TODO: check if new_cTR are already in self.cTR and append new runs
        self.cTR = new_cTR 
        dictTable = dictList2DictTable(new_cTR.getRuns())
        self.SpecimensTested = np.unique(dictTable['SpNo'])
        self.cTR.AssyName = self.name
        
    def readStaticTestData(self):
        # --> Peter already did this implementation!!! -> no not double the work!
        sTR = staticTestResults()
        return sTR


    def readCyclicTestData(self): # Function to read test data
        AssyFileName = os.path.join(self.path, 'CollectedReadMeData_' + self.name + '.txt')
        cTR = cyclicTestResults()
        readTestRuns = []
        
        with open(AssyFileName, "r") as results:
            for i,line in enumerate(results):
    #            if i==0: #check if header of ReadMe-file is complete and correct
    #                #toDo
    #            elif i==1: #get the dimensions
    #                #toDo
                if i>1: #now the run data follows
                    parameters = line.split(';')
                    if parameters[9] == 'X':
                        Nges = 0
                    else:
                        Nges = int(evalNges(parameters[9]))
                        
                    run = oneCyclicTestRun(
                            int(parameters[0]),                                 #SpecimenNo
                            int(parameters[1]),                                 #RunNo
                            parameters[2],                                      #StartTime
                            parameters[3],                                      #EndTime
                            float(parameters[4]),                               #F_o
                            float(parameters[5]),                               #F_u
                            float(parameters[6]),                               #R
                            float(parameters[7]),                               #f
                            float(parameters[8]),                               #RT
                            Nges,                                               #Nges
                            parameters[10],                                     #FailureLoc
                            parameters[11].strip('\n')                          #Comments
                            )
                    readTestRuns.append(run)
                #end if
        cTR.testRuns = readTestRuns
        return cTR


if __name__ == '__main__':
#    AssyName = 'SDC'
    AssyName = '9A-I-TEST-S02'
    Assy = testResults(AssyName)
    
    Assy.addCyclicTestResults()
    
#    data = readCyclicTestData(AssyName)
##    data.name = '9A-I-TEST-S01'
#    
    fList = np.array(['same'])
    r = None
    Assy.cTR.defineValidResults(fList)
#    Assy.cTR.axisLimits = [1e4,1e6,30,55]
#    kd, d = Assy.cTR.calcLinearRegLine()
#    Assy.cTR.plotFNLine(ratio=r,failureList=fList)
#    
#    scatter = Assy.cTR.calcStatistics_4allData(ratio=0.01, failureList=fList)
#    data_pue = Assy.cTR.calcStatistics_4eachLL(ratio=0.01, failureList=fList)
#    Assy.cTR.plotStatistics_4eachLL(data_pue=data_pue)
#    plt.close('all')
#    
#    Assy.cTR.plotFNLine(ratio=r,failureList=fList,withScatter=True)
#    
    kd, data = Assy.cTR.calcLinearRegLine(ratio=r,failureList=fList)
#    data_FN = np.transpose(np.concatenate(([data['F_o']],[data['Nges']])))
#    
#    plt.close('all')
    
#    Assy.cTR.plotFNLine_Basquin(withScatter=True, ratio=0.01, failureList=['same'], withExcluded=True, saveFig=False, customSurvivalProb=0.95)

    Assy.cTR.plotFNLine_Basquin(withScatter=True,withExcluded=True, withSpNo = True)
#    
#    Assy.cTR.defineValidResults(['same'])
#    
#    Assy.cTR.plotFNLine_Basquin(withScatter=True,withExcluded=True)
#    Assy.cTR.plotFNLine(ratio=r,failureList=fList,withScatter=True)
#    Assy.cTR.plotFNLine_Basquin(failureList=fList, withScatter=True)
#    Assy.cTR.plotFNLine_Basquin(withScatter=True,saveFig=True)
#    Assy.cTR.plotFNLine_Basquin(withScatter=True, withExcluded=True, customSurvivalProb=0.99, saveFig=True)
#
#    data.defineValidResults([f])    
#    d = data.plotFNLine(ratio=r,failure=f)
#    print(Assy.cTR.calcScatter4LineParameters())
    
#    Assy.cTR.getRuns({})
    
#    Assy.cTR.saveFNLine_Basquin_parameter2file(F_kp=35)
    
    
