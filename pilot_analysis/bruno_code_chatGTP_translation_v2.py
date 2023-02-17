# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:02:58 2023

@author: uajt040
"""


import numpy as np
import scipy.stats as stats









#########################################################

def normInvChi(prior, data):
    postProb = {}
    postProb['nu'] = prior['nu'] + data['n']
    postProb['kappa'] = prior['kappa'] + data['n']
    postProb['mu'] = (prior['kappa']/postProb['kappa'])*prior['mu'] + (data['n']/postProb['kappa'])*data['mu']
    if data['n'] == 0:
        postProb['sig'] = prior['sig']
    else:
        postProb['sig'] = (prior['nu']*prior['sig'] + (data['n']-1)*data['sig'] + 
                         ((prior['kappa']*data['n'])/(postProb['kappa']))*(data['mu'] - prior['mu'])**2)/postProb['nu']
    return postProb




############################################
def posteriorPredictive(y, postProb):
    tvar = (1 + postProb["kappa"]) * postProb["sig"] / postProb["kappa"]
    sy = (y - postProb["mu"]) / np.sqrt(tvar)
    prob_y = stats.t.pdf(sy, postProb["nu"])
    return prob_y




####################################################
def computeContinue(utility, postProb0, x, ti):
    #postProb0.nu = ti-1
    postProb0['nu']
    utCont = np.zeros((len(x), 1))
    expData = {'n': 1, 'sig': 0}

    for xi in range(len(x)):
        expData['mu'] = x[xi]
        postProb = normInvChi(postProb0, expData)
        spx = posteriorPredictive(x, postProb)
        spx = (spx/np.sum(spx))

        utCont[xi] = np.dot(spx.T, utility)

    return utCont




##########################################################
def rnkBackWardInduction(sampleSeries, ts, priorProb, listLength, x, Cs, payoff_scheme):
    N = listLength
    Nx = len(x)

    if payoff_scheme == 1:  # if instruction are to maximize option value
        payoff = np.sort(sampleSeries, axis=-1, kind='quicksort')[::-1]
    else:  # if payoff scheme is to get some degree of reward for each of top three ranks
        payoff = [0.12, 0.08, 0.04]

    maxPayRank = len(payoff)
    payoff = np.concatenate([payoff, np.zeros(20)])

    data = {
        'n': ts+1,
        #'sig': np.var(sampleSeries[:ts]),
        #'mu': np.mean(sampleSeries[:ts])
        'sig': np.var(sampleSeries[:ts+1]),
        'mu': np.mean(sampleSeries[:ts+1])
    }

    utCont = np.zeros((Nx, 1))
    utility = np.zeros((Nx, N))
    
    if ts < 0:    #not sure what to do here
       ts = 0
  #  if ts == 0:
  #      ts = 1
   

#######I had to change this from chatGTP's version to get rnki
  #  rnkvl = np.sort(sampleSeries[:ts], axis=-1, kind='quicksort')[::-1]
 #########new version:
     #sampleSeries = list['vals'][:ts]
    #rnkvl, rnki = zip(*sorted(enumerate(sampleSeries), key=lambda x: x[1], reverse=True)) 
    rnki, rnkvl = zip(*sorted(enumerate(sampleSeries[:ts+1]), key=lambda x: x[1], reverse=True)) 
    #rnki = argsort(sampleSeries[:ts+1])
    rnki = [int(x) for x in rnki] #convert from tuple to list
##########

#    rnki = np.where(rnki == ts)[0][0]
    z = rnki.index(ts)  #gets the index where ts matches an element of rnki
    rnki = z


#I don't think this section is used for anything!
    ties = 0
    if len(np.unique(sampleSeries[:ts+1])) < ts+1:
        ties = 1

    mxv = ts
    if mxv > maxPayRank:
        mxv = maxPayRank

    rnkv = np.concatenate([np.full(1, np.inf), rnkvl[:mxv+1], -np.full(20, np.inf)])
    postProb = normInvChi(priorProb, data)
    px = posteriorPredictive(x, postProb)
    px = px / px.sum()

    Fpx = np.cumsum(px)
    cFpx = 1 - Fpx
    
    #Added this initialisation because Python below complains when they are undefined yet
    expectedStop = [0]*N
    expectedCont = [0]*N
    expectedUtility = [0]*N


#So ts and ti both iterate through options. I've been working with ts iterating from 
   # for ti in range(N, ts-1, -1):
    for ti in range(N, ts+1, -1):
        
        print(ts,ti)
        
        if ti == N:
            utCont = -np.full((Nx, 1), np.inf)
        elif ti == ts+1:
            #utCont = np.ones((Nx, 1)) * (px * utility[:, ti+1]).sum()
            utCont = np.ones((Nx, 1)) * (px * utility[:, ti]).sum()
        else:
#            utCont = computeContinue(utility[:, ti+1], postProb, x, ti)
            utCont = computeContinue(utility[:, ti], postProb, x, ti)


        # utility when rewarded for best 3, $5, $2, $1
        utStop = np.full((Nx, 1), np.nan)

        rd = N - ti  # remaining draws
        id = max(0, ti - ts - 1 - 1)  # intervening draws
        #id = max(0, ti - ts - 1)  # intervening draws
        td = rd + id
        ps = np.zeros((Nx, maxPayRank))

        for rk in range(maxPayRank):
            pf = np.prod(range(td, td-(rk), -1)) / np.math.factorial(rk)
            #pf = np.prod(range(td, td-rk-1, -1)) / np.math.factorial(rk)
            ps[:, rk] = pf * (Fpx ** (td-rk)) * ((cFpx) ** rk)

        
        for ri in range(1, maxPayRank + 1):
        #for ri in range(0, maxPayRank):
            z = np.where((x < rnkv[ri-1]) & (x >= rnkv[ri]))[0]   #Careful! This returns python indices so that will be one less than the matlab output on this line
            
            #So to make this work I had to not just adjust the first term in the 
            #product for zero-indexing but I also had to add the reshape method onto the end 
            #because it needs to be (50,1) not (50,) or it refuses to assign to the (50,1) indices of z in utstop
           # utStop[z] = np.sum(ps[z, 1:maxPayRank] * payoff[(ri-1):maxPayRank+(ri-1)], axis=1)
            utStop[z] = np.sum(ps[z, 0:maxPayRank] * payoff[(ri-1):maxPayRank+(ri-1)], axis=1).reshape(z.shape[0],1)
    
            
        if np.isnan(utStop).sum() > 0:
            print("Nan in utStop")
            
        if ti == ts:
            zi, zv = min(enumerate(np.abs(x - sampleSeries[ts])), key=lambda x: x[1])
            #zv, zi = min(enumerate(np.abs(x - sampleSeries[ts])), key=lambda x: x[1])
            if zi + 1 > len(utStop):
                zi = len(utStop) - 1
            utStop = utStop[zi+1] * np.ones((Nx, 1))
            
        utCont = utCont - Cs
        
        #utility[:, ti]      = np.maximum(utStop, utCont)
        #expectedUtility[ti] = np.dot(px, utility[:, ti])
        #expectedStop[ti]    = np.dot(px, utStop)
        #expectedCont[ti]    = np.dot(px, utCont)
        utility[:, ti-1]      = np.maximum(utStop, utCont).reshape((100,))
        expectedUtility[ti-1] = np.dot(px, utility[:, ti-1])
        expectedStop[ti-1]    = np.dot(px, utStop)[0]
        expectedCont[ti-1]    = np.dot(px, utCont)[0]

    return expectedStop, expectedCont








#################################################################
def analyzeSecretaryNick_python(prior, lst):
    sampleSeries = lst['vals']
    N = lst['length']
    Cs = lst['Cs']
    payoff_scheme = lst['payoff_scheme']
    
    sdevs = 8
    dx = 2 * sdevs * np.sqrt(prior['sig']) / 100
    #x = (prior['mu'] - sdevs * np.sqrt(prior['sig']) + dx, dx, prior['mu'] + sdevs * np.sqrt(prior['sig']) + dx)
    start = prior['mu'] - sdevs * np.sqrt(prior['sig']) + dx
    stop = prior['mu'] + sdevs * np.sqrt(prior['sig']) + dx
    x = start + np.arange(0, stop-start, dx)
    
    Nconsider = N
    
    difVal = [0] * Nconsider
    choiceCont = [0] * Nconsider
    choiceStop = [0] * Nconsider
    
    for ts in range(Nconsider):
        expectedStop, expectedCont = rnkBackWardInduction(sampleSeries, ts, prior, N, x, Cs, payoff_scheme)
        difVal[ts] = expectedCont[ts] - expectedStop[ts]
        choiceCont[ts] = expectedCont[ts]
        choiceStop[ts] = expectedStop[ts]
    
    return choiceCont, choiceStop, difVal









#########################################################
#def run_io():
    
#prior and list are dictionaries
prior = {}
prior['mu'] = 1.5863
prior['sig'] = 0.4673
prior['kappa'] = 2
prior['nu'] = 1

list = {}
list['vals'] = [1.6094, 1.6582, 0.8109, 1.7492, 0.8109, 1.7047, 0.8109, 2.1401, 1.3863, 1.7918, 0.8109, 2.2246]
list['length'] = 12
list['Cs'] = 0
list['payoff_scheme'] = 1

choiceCont, choiceStop, difVal = analyzeSecretaryNick_python(prior, list)

print(' ')