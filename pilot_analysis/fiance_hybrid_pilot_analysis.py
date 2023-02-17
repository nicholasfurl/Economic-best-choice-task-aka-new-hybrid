

import numpy as np
import scipy.stats as stats
import seaborn as sns
from matplotlib import pyplot as plt



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
    
    #This paramater appears to grow (for the posterior) with the number of options considered 
    #postProb0.nu = ti-1
    postProb0['nu'] = ti
    
    #initialise new verion of utCont to be 100 zeros tall
    utCont = np.zeros((len(x), 1))
    
    #was a struct in matlab
    expData = {'n': 1, 'sig': 0}
    
    #from 0 to 99
    for xi in range(len(x)):
        
        #This seems to test how much posterior would change if each of the samples from x (prior) were encountered
        expData['mu'] = x[xi]
        #then gets a new posterior prob based on attempted x
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

    #So matlab automatically computes sum-squares / N-1 but Python doesn't. 
    #Python needs a ddof=1 argument or it returns slightly different results.
    #Moreover, when ddof is used, Python returns nan when there is one observation
    #(presumably it divides by zero) while MATLAB returns 0. So we need all this
    #to fill in the sig key/field in data.
    if len(sampleSeries[:ts+1]) == 1:
        var_temp = 0
    else:
        var_temp=np.var(sampleSeries[:ts+1],ddof=1)
        

    data = {
        'n': ts+1,
        #'sig': np.var(sampleSeries[:ts]),
        #'mu': np.mean(sampleSeries[:ts])
        'sig': var_temp,
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

    #mxv = ts
    #if mxv > maxPayRank:
    #    mxv = maxPayRank
    
    if (ts >= maxPayRank):
        rnkv = np.concatenate([np.full(1, np.inf), rnkvl[:maxPayRank], -np.full(20, np.inf)])
    else:
        rnkv = np.concatenate([np.full(1, np.inf), rnkvl[:ts+1], -np.full(20, np.inf)])


        
    # if ts == 10:
    #     print('')

    #rnkv = np.concatenate([np.full(1, np.inf), rnkvl[:mxv+1], -np.full(20, np.inf)])
    #rnkv = np.concatenate([np.full(1, np.inf), rnkvl[:mxv], -np.full(20, np.inf)])
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
    for ti in range(N-1, ts-1, -1):
        
        #print(ts,ti)
        
        #If ti is last possible draw (start of backward induction for this current option ts)
        if ti == N-1: #11, or N-1 is the final index for 12 options so this is the FIRST ti loop of the back ind
        #if ti == N:
            utCont = -np.full((Nx, 1), np.inf)  #Agent can't continue if its the last option so force -Inf so any stop value is better
        
        #when ti comes back from the future and reaches the current option (end of backward induction)
        elif ti == ts:
            utCont = np.ones((Nx, 1)) * (px * utility[:, ti+1]).sum()
            
        #If ti is in between, working backward from N to current option ts
        else:
#            utCont = computeContinue(utility[:, ti+1], postProb, x, ti)
            utCont = computeContinue(utility[:, ti+1], postProb, x, ti)


        # just initialises utStop to be 100 nans
        utStop = np.full((Nx, 1), np.nan)

        #rd = N - ti
        rd = (N-1) - ti  # so how many draws between ti and the end? 0 if ti=11 (index for 12 option), 11 . 
        id = max(0, ti - ts - 1)  # intervening draws between ts and ti. If starting on first loop, last / 12th draw (ti=11) and first draw (ts=0) then id is 10
        td = rd + id #How far back from the future ti has searched already, plus how far ti still has to go to get to ts (current real decision option)
        ps = np.zeros((Nx, maxPayRank)) #inialised fresh for each ti, 100 by 12

        for rk in range(maxPayRank):
            pf = np.prod(range(td, td-(rk), -1)) / np.math.factorial(rk)
            #pf = np.prod(range(td, td-rk-1, -1)) / np.math.factorial(rk)
            ps[:, rk] = pf * (Fpx ** (td-rk)) * ((cFpx) ** rk)

        
        for ri in range(1, maxPayRank + 1 + 1):
        #for ri in range(0, maxPayRank):
            #z = np.where((x < rnkv[ri]) & (x >= rnkv[ri]))[0]   #Careful! This returns python indices so that will be one less than the matlab output on this line
            z = np.where((x < rnkv[ri-1]) & (x >= rnkv[ri]))[0]   #Careful! This returns python indices so that will be one less than the matlab output on this line
           
            #So to make this work I had to not just adjust the first term in the 
            #product for zero-indexing but I also had to add the reshape method onto the end 
            #because it needs to be (50,1) not (50,) or it refuses to assign to the (50,1) indices of z in utstop
           # utStop[z] = np.sum(ps[z, 1:maxPayRank] * payoff[(ri-1):maxPayRank+(ri-1)], axis=1)
            #utStop[z] = np.sum(ps[z, 0:maxPayRank] * payoff[1+(ri-1):maxPayRank+(ri-1)], axis=1).reshape(z.shape[0],1)
            utStop[z] = np.sum(ps[z, 0:maxPayRank] * payoff[ri-1:ri+maxPayRank-1], axis=1).reshape(z.shape[0],1)

            
        # if (ts == 11) and (ti == 11):
        #     print('')


        if np.isnan(utStop).sum() > 0:
            print("Nan in utStop")
            
        if ti == ts:
            #If we've already backward induced the rest of the options, 
            #then consider the difference between the current option (ts) and the samples from the prior
            #find their smallest absolute deviation zv and get its index zi
            zi, zv = min(enumerate(np.abs(x - sampleSeries[ts])), key=lambda x: x[1])
            #zv, zi = min(enumerate(np.abs(x - sampleSeries[ts])), key=lambda x: x[1])
            
            #Not really sure why this would be, 
            #so far as I've seen x and utStop have both always been 100
            #I guess if zi==100, then it could be a problem if we try to index zi+1 so set it to 99?
            #But this means that I should actually check if zi+1 is >= 100 (len(utStop==100))
            #if zi + 1 > len(utStop):
            if zi + 1 >= len(utStop):
                zi = len(utStop) - 1
            
            #If we've backward induced the sequence, then the current option (ts) gets all the same value, 
            #one more than the prior sample with the smallest absolute difference with the current option
            utStop = utStop[zi+1] * np.ones((Nx, 1))
            
        utCont = utCont - Cs
        
        #utility[:, ti]      = np.maximum(utStop, utCont)
        #expectedUtility[ti] = np.dot(px, utility[:, ti])
        #expectedStop[ti]    = np.dot(px, utStop)
        #expectedCont[ti]    = np.dot(px, utCont)
        #utility[:, ti]      = np.maximum(utStop, utCont).reshape((100,))
        utility[:, ti]      = np.maximum(utStop, utCont).reshape((-1,))
        expectedUtility[ti] = np.dot(px, utility[:, ti])
        expectedStop[ti]    = np.dot(px, utStop)[0]
        expectedCont[ti]    = np.dot(px, utCont)[0]
        
    # if (ts == 11) and (ti == 11):
    #     print('')

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
    
    if len(x) != 100:
        print('')
    
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







#########################################################################################
#Add path so can run from any directory and so Python can see any files I want to create
import sys
sys.path.append(r'C:\matlab_files\jspych\fiance_hybrid\pilot_analysis')

import pandas as pd
import numpy as np

#Read in data
raw_data = pd.read_csv(r'C:\matlab_files\jspych\fiance_hybrid\pilot_analysis\hybrid-seq-length-and-reward-scheme.csv')

#Phase 1
#get phase 1 data, just the rows and columns we want and discard the rest
ratings_data = raw_data[raw_data['name']=='rating_trial'][['PROLIFIC_PID', 'response','price']]
#for some reason it insists on reading in response as object, which gives false computations
ratings_data['response'] = ratings_data['response'].astype(float)
#One participant (used for pilot testing) didn't get an id from prolific
ratings_data['PROLIFIC_PID'] = ratings_data['PROLIFIC_PID'].fillna('pilotParticipant')
#average over the two ratings (returns series), restore cols and index 0 to n-1
ratings_data = ratings_data.groupby(['PROLIFIC_PID', 'price'])['response'].mean().reset_index()
#ensure data are sorted by price, so I can access price by index later if needed
ratings_data.sort_values(by=['PROLIFIC_PID', 'price'],ignore_index = True)

#Phase 2
#get phase 2 data, just the rows and columns we want and discard the rest 
sequence_data = raw_data[raw_data['name']=='response_prompt'][['PROLIFIC_PID', 'response','price','sequence','option','rank','num_options','num_seqs','reward_cond','array']]
#Fix the float issue with response again and other variables with incorrect types
sequence_data['response'] = sequence_data['response'].astype(float)
sequence_data['sequence'] = sequence_data['sequence'].astype(int)
sequence_data['num_options'] = sequence_data['num_options'].astype(int)
sequence_data['num_seqs'] = sequence_data['num_seqs'].astype(int)
#One participant (used for pilot testing) didn't get an id from prolific
sequence_data['PROLIFIC_PID'] = sequence_data['PROLIFIC_PID'].fillna('pilotParticipant')
#Get reduced dataset with only stop choices (columns option and rank now become the DVs)
stop_choices = sequence_data[sequence_data['response']==0].reset_index()
#ensure data are sorted by price, so I can access price by index later if needed
stop_choices.sort_values(by=['PROLIFIC_PID', 'sequence'],ignore_index = True)

#Models

#Make bigger dataframe to hold results for the two models
stop_choices_temp = stop_choices.copy()     #clone data from subs
stop_choices_temp[['response','option','rank']] = np.nan*len(stop_choices_temp) #blank out data that is human-specific
stop_choices_agents = pd.concat([stop_choices,stop_choices_temp,stop_choices_temp],axis=0).reset_index()
stop_choices_agents['Agent'] = ['participant']*len(stop_choices)+['modelObj']*len(stop_choices)+['modelSubj']*len(stop_choices)

#get full array of sequences of prices in float format
array_objVals = []
array_subVals = []	

#So we can save options and output them for verification with matlab
out_data = pd.DataFrame(columns=['pid', 'reward_cond','num_options','PriorMean','PriorVar','option01','option02','option03','option04','option05','option06','option07','option08','option09','option10','option11','option12'])
#out_data = pd.DataFrame()
	

it = 0		
#for participant in stop_choices['PROLIFIC_PID'].unique():
for p_count, participant in enumerate(stop_choices['PROLIFIC_PID'].unique()):

    
    #print(participant)
    
    #get data for this participant to process
    num_seqs = stop_choices[stop_choices['PROLIFIC_PID']== participant]['num_seqs'].iloc[0]
    num_options = stop_choices[stop_choices['PROLIFIC_PID']== participant]['num_options'].iloc[0]
    line = stop_choices[stop_choices['PROLIFIC_PID']== participant]['array'].iloc[0]
    thisP_ratings = ratings_data[ratings_data['PROLIFIC_PID']== participant]
    thisP_Rcondition = stop_choices[stop_choices['PROLIFIC_PID']== participant]['reward_cond'].iloc[0]

    
    #Process prices array
    line = line.replace('[','').replace(']','') #reduce to numbers and commas strings
    linesplit = line.split(',') #split into lists of str prices
    linefloat = np.array([float(linesplit[i]) for i in range(len(linesplit))])  #convert to floats
    
    #Get array of subjective values for these sequences
    price_to_rating = dict(zip(thisP_ratings['price'], thisP_ratings['response']))    #convert ratings dataframe to dict
    linefloat_ratings = np.array([price_to_rating[p] for p in linefloat])
    #linefloat_ratings = thisP_ratings.loc[ thisP_ratings['price'].isin(linefloat), 'response'].values    #get corresponding ratings
    
    
    
    reshapeObj = np.reshape(linefloat,(num_seqs,num_options))   #reshape raw prices into seqs and options
    reshapeSubj = np.reshape(linefloat_ratings,(num_seqs,num_options))   #reshape options
    array_objVals = array_objVals + [reshapeObj]   #accumulate objective value arrays over participants
    array_subVals = array_subVals + [reshapeSubj] 
    
    #Run models for each sequence
    for sequence in range(num_seqs):
       
        model_types = ['modelObj', 'modelSubj']
        for model_type in model_types:
            
            print('Running model participant: '+participant+' sequence: '+str(sequence)+' '+model_type)
            
            #Prepare model inputs
            if model_type == 'modelObj':
            
                #Standardise this participant's prices to be between zero and 100 with 100 the best, as with ratings
                old_min = 1;model_type
                old_max = thisP_ratings['price'].max()
                new_min = 1;
                new_max = 100;
                #prior distribution
                temp_ratings = thisP_ratings['price'].apply(lambda x:  (((new_max-new_min)*(x - old_min))/(old_max-old_min))+new_min)   #rescale to 1 to 100
                temp_ratings = -(temp_ratings - 50) + 50   #reflect around centre so lower prices are higher
                #this sequence
                temp_seq = (((new_max-new_min)*(reshapeObj[sequence] - old_min))/(old_max-old_min))+new_min
                temp_seq = -(temp_seq - 50) + 50
            
            else:
            
                temp_ratings = thisP_ratings['response']
                temp_seq = reshapeSubj[sequence]
                
                #Accumulate option subjective values for verification with matlab
                # create a new row with the values to save on this iteration
                row = pd.DataFrame([0] * len(out_data.columns)).T
                for j, val in enumerate(temp_seq.tolist()):
                    row[5+j] = val
                row[0] = p_count+1   
                row[1] = thisP_Rcondition
                row[2] = num_options
                row[3] = temp_ratings.mean()
                row[4] = temp_ratings.var()
                row.columns = ['pid', 'reward_cond','num_options','PriorMean','PriorVar','option01','option02','option03','option04','option05','option06','option07','option08','option09','option10','option11','option12']
                out_data = pd.concat([out_data,row],axis=0)
                #out_data = out_data.append(row,ignore_index=True)
                #out_data.iloc[it] = row
                #it = it+1
                
 
            print(temp_seq)        
 
            #prior and list are dictionaries
            prior = {}
            prior['mu'] = temp_ratings.mean()
            prior['sig'] = temp_ratings.var() #Note numpy.var divides by N by default but pandas divides by N-1 by default
            prior['kappa'] = 2
            prior['nu'] = 1
        
            list = {}
            list['vals'] = temp_seq.tolist()
            list['length'] = num_options
            list['Cs'] = 0
            list['payoff_scheme'] = thisP_Rcondition - 1  #in dataframe, 1 is 3-rank and 2 is continuous. In model code, 1 is continuous and anything else (0) is 3-rank
        
            choiceCont, choiceStop, difVal = analyzeSecretaryNick_python(prior, list)
            
            #Compute number of samples
            for i, num in enumerate(difVal):
                if num < 0:
                    break
            #Get ranks  
            ranklist = stats.rankdata(list['vals'])
            
            #Save samples and ranks
            stop_choices_agents.loc[(stop_choices_agents['PROLIFIC_PID']==participant) & (stop_choices_agents['sequence']==sequence+1) & (stop_choices_agents['Agent']==model_type),'option'] = i+1
            stop_choices_agents.loc[(stop_choices_agents['PROLIFIC_PID']==participant) & (stop_choices_agents['sequence']==sequence+1) & (stop_choices_agents['Agent']==model_type),'rank'] = ranklist[i]
            
            
out_data.to_csv('sequence_subjVals.csv')             
          
   
# 

# # map data to stripplot
# g.map(sns.stripplot, 'num_options', 'option', 'smoker', hue_order=['participant', 'modelObj','modelSubj'], order=[8, 12],
#       palette=sns.color_palette(), dodge=True, alpha=0.6, ec='k', linewidth=1)       
                                                                                                                  
#You need to collapse over sequence to ensure error bars are over participants
for_plots = stop_choices_agents.groupby(['PROLIFIC_PID','num_options','reward_cond','Agent']).mean().reset_index()
            
fig, axes = plt.subplots(1,2)      
   
sns.barplot(
    data=for_plots[for_plots['reward_cond']==1],
    x='Agent',
    y='option',
    hue='num_options',
    alpha=.15,
    edgecolor="black",
    errcolor="black",
    errwidth=1.5,
    capsize = 0.1,
    ax = axes[0],
    )

sns.stripplot(
    data=for_plots[for_plots['reward_cond']==1],
    x='Agent',
    y='option',
    hue='num_options',
    dodge=True,
    alpha=0.75,
    ax=axes[0],
    ec='k', 
    linewidth=1,
    )
#axes[0]._legend.remove()
#sns.legend([], [], frameon=False)
sns.move_legend(axes[0], "upper left", bbox_to_anchor=(-1, 1))
#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)


sns.barplot(
    data=for_plots[for_plots['reward_cond']==2],
    x='Agent',
    y='option',
    hue='num_options',
    alpha=.25,
    edgecolor="black",
    errcolor="black",
    errwidth=1.5,
    capsize = 0.1,
    ax = axes[1]
    )

sns.stripplot(
    data=for_plots[for_plots['reward_cond']==2],
    x='Agent',
    y='option',
    hue='reward_cond',
    dodge=True,
    alpha=0.5,
    ax=axes[1],
    ec='k', 
    linewidth=1,
    )

sns.move_legend(axes[1], "upper left", bbox_to_anchor=(1, 1))

#g = sns.catplot(
#     x="num_options", 
#     y="option", 
#     hue="Agent", 
#     row="reward_cond", 
#     data=for_plots, 
#     kind="bar", 
#     ci = "ci", 
#     edgecolor="black",
#     errcolor="black",
#     errwidth=1.5,
#     capsize = 0.1,
#     height=4, 
#     aspect=.7,
#     alpha=0.5)
