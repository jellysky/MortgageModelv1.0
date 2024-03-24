import pandas as pd
import numpy as np


def read_data(pmt_str,chars_str):

    dt_pmts = pd.read_csv(pmt_str,header=None,index_col=False)

    # assign headers
    headers = ['originator','originator_loan_id','loan_month','issue_date','outstanding_principal_bop','charge_off_indicator',
           'charge_off_amount','post_charge_off_recoveries','post_charge_off_recovery_fee','unscheduled_principal','coupon',
           'fico_origination','dti_ex_mortgage','original_principal','months_on_balance','grade','term','purpose','home_ind',
            'loan_status','never_late_indicator']
    dt_pmts.columns = headers
    print '\nReading in {originator} payments and borrower data...'.format(originator=dt_pmts['originator'][1])

    dt_pmts = dt_pmts.sort(['originator_loan_id','months_on_balance'])

    #reads lc and prosper characteristics csv and merges to pmts csv

    dt_chars = pd.read_csv(chars_str,index_col=False)
    charIndx = dt_chars.columns - dt_pmts.columns
    dt = pd.merge(dt_pmts,dt_chars[charIndx],how='outer',left_on='originator_loan_id',right_on='id',suffixes=['','_y'],copy=False)
    print '\nReading in {originator} characteristics data and merging...'.format(originator=dt_pmts['originator'][1])

    return dt,dt_chars,dt_pmts

def generate_regressors(dt):
    # Converts input data to classification columns and I tried to drop the columns containing 20% of the non-missing values

    cols = []
    dtOut = np.zeros(shape=(dt.shape[0],0))
    sparseCount = dt.count(axis=0)/dt.shape[0] # percentage of non-nans in columns

    topTolerance = .9
    bottomTolerance = .2
    freqTolerance = .2
    sparseTolerance = .2
    binTolerance = 10
    bins = xrange(0,int((1-sparseTolerance)*100),binTolerance)

    for i in xrange(2,dt.shape[1]-1):
        print i
        freqTable = dt.iloc[:,i].value_counts()/dt.iloc[:,i].count()

        if (freqTable.iloc[0] >= topTolerance) & (sparseCount[i] > sparseTolerance) & (freqTable.shape[0] > 1):
        # Case 0: If most frequent value is > 90% and column is not sparse, then (is frequent value, is missing)
            dtTemp = np.zeros(shape=(dt.shape[0],1))
            cols.append(dt.columns.values[i] + "_is" + str(dt.iloc[:, i].value_counts().index[0]))
            dtTemp[np.where(dt.iloc[:, i] == dt.iloc[:, i].value_counts().index[0]),0] = 1
            dtOut = np.concatenate((dtOut, dtTemp), axis=1)
            print "\nCase 0, for i = %d" %(i)

        elif (bottomTolerance <= freqTable.iloc[0] < topTolerance) & (sparseCount[i] > sparseTolerance) & (freqTable.shape[0] > 1):
        # Case 1: If most frequent value is in between tolerances and column is not sparse, then (is multiple vals above threshold, is missing)

            uniqCols = freqTable.index[(freqTable > freqTolerance).nonzero()[0]]
            if uniqCols.shape[0] == freqTable.shape[0]:
            # if uniqCols contains all values then drop last one
                uniqCols = uniqCols[0:uniqCols.shape[0]-1]

            dtTemp = np.zeros(shape=(dt.shape[0],uniqCols.shape[0]))
            # pulls values for columns

            for j,u in enumerate(uniqCols): # parse possible values
                cols.append(dt.columns.values[i] + "_is" + str(u))
                dtTemp[np.where(dt.iloc[:,i] == u),j] = 1
                #print '\ni = %d, j = %d' % (i, j)

            dtOut = np.concatenate((dtOut,dtTemp),axis=1)
            print "\nCase 1, for i = %d" % (i)

        elif (sparseCount[i] > sparseTolerance) & (freqTable.shape[0] > 1):
        # Case 2: Else numbers are numerical
            dtTemp = np.zeros(shape=(dt.shape[0],len(bins)))
            # divide column into deciles and exclude last 20%
            for j,u in enumerate(bins):
                print j,u
                dtTemp[np.where(np.logical_and(dt.iloc[:,i]>=np.percentile(dt.iloc[:,i],u),dt.iloc[:,i]<np.percentile(dt.iloc[:,i],u+binTolerance))),j] = 1
                cols.append(dt.columns.values[i] + "_bin" + str(u))

            dtOut = np.concatenate((dtOut,dtTemp),axis=1)
            print "\nCase 2, for i = %d" % (i)

        else:
        # Case 3: Values are either one value or numbers are too sparse - turns out only one column falls into this case, which was dropped
            print "\nCase 3, for i = %d" % (i)

        if (dt.iloc[:,i].isnull().sum() > 0):
        # Case 3: Add a column if some values are missing or if there is 1 value and blanks
            dtTemp = np.zeros(shape=(dt.shape[0],1))
            dtTemp[np.where(dt.iloc[:,i].isnull()),0] = 1
            cols.append(dt.columns.values[i] + "_isnull")
            dtOut = np.concatenate((dtOut,dtTemp),axis=1)
            #print "\nAdding col for missing vals, for i = %d" % (i)

    print "\nFinished processing input data..."
    #temp = pd.DataFrame(data=dtOut, index=dt.index, columns=cols)
    #temp.to_csv('temp.csv')
    return pd.DataFrame(data=dtOut,index=dt['id'],columns=cols)

def partition_data(dt_in,year,term):

    print('\nPartitioned data for year: %d and term: %d...' %(year,term))
    pmtIndx = np.where(np.logical_and(pd.DatetimeIndex(dt_in['IssuedDate']).year == year,dt_in['term'] == term))
    return dt_in.iloc[pmtIndx]

def translate_status(dtStatus):
    statusDict = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 3,
        5: 3,
        6: 3,
        7: 4,
        8: -1,
        9: 5,
    }
    return [statusDict.get(x,-1) for x in dtStatus]

def generate_transition_probs(dtRegr,dtResp,dtCharts,tuner):
    # tuner = 1
    # startState = 'C'

    startStates = ['C', 'D3', 'D6', 'D6+']
    exemptTransitions = ['CtoC', 'CtoD6', 'CtoD6+', 'D3toD3', 'D3toD6+', 'D6toC', 'D6toD6', 'D6toD', 'D6toP', 'D6+toC',
                         'D6+toD3', 'D6+toD6+', 'D6+toP']
    # dtCoefs = pd.DataFrame(np.zeros(shape=(dtRegr.shape[1],dtResp.shape[1]),dtype=float),index=dtRegr.columns.values,columns=dtResp.columns.values)
    modelList = [RandomForestClassifier for i in xrange(0, dtResp.shape[1])]

    for startState in startStates:

        i = 0  # selecting those columns that have the right starting state for comparison charts
        evalCols = np.zeros(shape=(len(dtResp.columns.values)), dtype=int)
        while (i < len(dtResp.columns.values)):
            if (dtResp.columns.values[i][0:dtResp.columns.values[i].index('t')] == startState):
                evalCols[i:i + const.maxStates()] = 1
                i = len(dtResp.columns.values) + 1
            else:
                i = i + 1

        # select only those rows where the starting state is the same as startState
        # so we sum the values across cols for startState cols and ensure they add to 1
        rIndx = np.where(dtResp[dtResp.columns.values[np.where(evalCols == 1)]].sum(axis=1) == 1)[0]
        # loop through each transition probability in that row of the matrix
        # colName = 'CtoD3'
        # colName = 'CtoD'
        for colName in dtResp.columns.values[evalCols == 1]:
            print startState, colName

            # don't have to calculate probs for transition states where endState == startState
            if not (colName in exemptTransitions):

                y = np.ravel(dtResp[colName].iloc[rIndx])
                X = dtRegr.iloc[rIndx]
                # print 'sum of y %f' %sum(y)

                # remove last states from rIndx
                rIndx = np.setdiff1d(rIndx, np.where(dtResp[colName] == 1)[0])

                # split the relevant data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
                modelList[np.where(colName == dtResp.columns.values)[0][0]] = calculate_model(X_train, y_train, tuner,
                                                                                              colName)

                # assign coefs to dtCoefs
                # dtCoefs[colName] = np.transpose(modelList[np.where(colName==dtResp.columns.values)[0][0]].coef_)
                # model = modelList[np.where(colName==dtResp.columns.values)[0][0]]
                analyze_model(modelList[np.where(colName == dtResp.columns.values)[0][0]], X_test, y_test, colName)

                # convert dtCharts to floats and generate predicted probs for X test only
                dtCharts[['probMod', 'y_act']] = dtCharts[['probMod', 'y_act']].astype(float)
                dtCharts['probMod'].iloc[X_test.index] = modelList[np.where(colName == dtResp.columns.values)[0][
                    0]].predict_proba(X_test)[:, 1]
                dtCharts['y_act'].iloc[X_test.index] = dtResp[colName].iloc[X_test.index]

                # plot X_test against y_test
                # plot_model_output(dtCharts.iloc[X_test.index],colName)
            else:
                print '\nNo need to calculate Pr(%s) ...' % colName
                # dtCoefs.to_csv('/media/koushik/Seagate/Files to Transfer to VM/' + 'dtCoefs.csv')
    return modelList, dtCharts.iloc[X_test.index]

def calculate_model(X_train,y_train,tuner,colName):

    # fits model on training set
    model = RandomForestClassifier(n_estimators=const.maxTrees(),max_depth = const.maxDepth(),max_features = 'auto',
                        bootstrap = True, oob_score = True, random_state = 531, min_samples_leaf = const.maxLeaf())
    #model = SGDClassifier(loss='log',penalty='l1',alpha='optimal',l1_ratio=1,n_iter=10,fit_intercept=False)
    model = model.fit(X_train, y_train)
    print '\nFitted model Pr(%s) ...' % colName
    return model

def plot_model_output(dtC,superTitle):

    #plt.switch_backend('Qt4Agg')
    plt.rcParams.update({'font.size':8})

    f, axArr = plt.subplots(4,4)
    f.suptitle(superTitle)
    terms = [36,60]
    fields = ['MOB','orig_fico','vintage','month','loan_amnt','coupon','purpose','emp_length']

    for i in xrange(0,len(terms)):
        for j in xrange(0,len(fields)):
            print i,j
            mobMod = dtC.groupby(['term',fields[j]])['probMod'].mean()
            mobAct = dtC.groupby(['term',fields[j]])['y_act'].sum()/dtC.groupby(['term',fields[j]])['y_act'].count()

            if j == 0:
                s0 = axArr[j//2,int(j%2)+2*i].plot(mobAct[terms[i]][0:terms[i]].index,mobAct[terms[i]][0:terms[i]],'-r')
                s1 = axArr[j//2,int(j%2)+2*i].plot(mobMod[terms[i]][0:terms[i]].index,mobMod[terms[i]][0:terms[i]],'-b')

                axArr[j//2,int(j%2)+2*i].legend((s0[0], s1[0]),('Act','Mod'))
                axArr[j//2,int(j%2)+2*i].set_title(fields[j] + ' term=' + str(terms[i]))

            else:
                ind = np.arange(len(mobAct[terms[i]].index))
                width = 0.4

                s0 = axArr[j//2,int(j%2)+2*i].bar(ind,mobAct[terms[i]],width,color='r')
                s1 = axArr[j//2,int(j%2)+2*i].bar(ind+width,mobMod[terms[i]],width,color='b')

                axArr[j//2,int(j%2)+2*i].set_title(fields[j] + ' term=' + str(terms[i]))
                axArr[j//2,int(j%2)+2*i].set_xticks(ind+width)
                axArr[j//2,int(j%2)+2*i].set_xticklabels(mobAct[terms[i]].index)

                # axArr[j//2,int(j%2)+2*i].legend((s0[0], s1[0]),('Act','Mod'))

            plt.show()
    #f.savefig('/media/koushik/Seagate/Files to Transfer to VM/' + superTitle + '.pdf',dpi=f.dpi)

def calculate_unconditional_probs(dtResp,modelList):

    # returns cumulative transition prob matrix
    uncondProbMat = np.ones(shape=(const.trMatCols(),const.trMatCols()),dtype=float)
    for i in xrange(0,const.trMatRows()):
        uncondProb = 1
        for j in xrange(0,const.trMatCols()):
            # if startState != endState and model exists
            if (i!=j) and (isinstance(modelList[j+i*const.trMatCols()],RandomForestClassifier)):
                uncondProbMat[i,j] = uncondProb
                uncondProb = uncondProb*len(np.where(dtResp.iloc[:,j+i*const.trMatCols()]==0)[0])/dtResp.shape[0]
    return uncondProbMat

def clean_data(dt_pmts):

    # Fix 1: Make End Bal = 0, when charge off occurs

    coIndx = np.where(np.logical_and(dt_pmts['COAMT'] > 0,dt_pmts['PBAL_END_PERIOD'] > 0))
    dt_pmts['PBAL_END_PERIOD'].iloc[coIndx] = 0
    print('\nFix 1 complete...')

    # Fix 2: Find where next opening balance is greater than previous opening balance for same loan

    pmtIndx = np.where(np.logical_and(dt_pmts['PBAL_BEG_PERIOD'].iloc[1:] > dt_pmts['PBAL_BEG_PERIOD'].iloc[:-1],
                                      dt_pmts['LOAN_ID'].iloc[1:] == dt_pmts['LOAN_ID'].iloc[:-1]))
    loanIndx = np.unique(dt_pmts['LOAN_ID'].iloc[pmtIndx])
    for l in loanIndx:
    #l = loanIndx[0]
        rIndx = np.where(dt_pmts['LOAN_ID'] == l)
        for r in range(0,len(rIndx[0])-1):
            if (dt_pmts['PBAL_BEG_PERIOD'].iloc[rIndx[0][-1]-r] > dt_pmts['PBAL_BEG_PERIOD'].iloc[rIndx[0][-1]-r-1]):
                dt_pmts['PBAL_BEG_PERIOD'].iloc[rIndx[0][-1]-r-1] = dt_pmts['PBAL_BEG_PERIOD'].iloc[rIndx[0][-1]-r]

    print('\nFix 2 complete...')

    # Fix 3: Find where closing balance is greater than opening balance

    pmtIndx = np.where(dt_pmts['PBAL_END_PERIOD'] > dt_pmts['PBAL_BEG_PERIOD'])
    dt_pmts['PBAL_END_PERIOD'].iloc[pmtIndx] = np.minimum(dt_pmts['PBAL_BEG_PERIOD'].iloc[pmtIndx] - dt_pmts['PRNCP_PAID'].iloc[pmtIndx] \
                                               - dt_pmts['COAMT'].iloc[pmtIndx],dt_pmts['PBAL_BEG_PERIOD'].iloc[pmtIndx])
    print('\nFix 3 complete...')

    # Fix 4: Find loans where next opening balance does not match previous closing balance

    pmtIndx = np.where(np.logical_and(dt_pmts['PBAL_BEG_PERIOD'].iloc[1:] != dt_pmts['PBAL_END_PERIOD'].iloc[:-1],
                                    dt_pmts['LOAN_ID'].iloc[1:] == dt_pmts['LOAN_ID'].iloc[:-1]))
    dt_pmts['PBAL_END_PERIOD'].iloc[pmtIndx[0]] = dt_pmts['PBAL_BEG_PERIOD'].iloc[pmtIndx[0]+1]
    print('\nFix 4 complete...')

    # Fix 5: Check to ensure PBAL BEG - PRIN PAID - COAMT = PBAL END, if not modify PRIN PAID

    pmtIndx = np.where(dt_pmts['PBAL_BEG_PERIOD']-dt_pmts['PRNCP_PAID']-dt_pmts['COAMT']-dt_pmts['PBAL_END_PERIOD'] != 0)
    dt_pmts['PRNCP_PAID'].iloc[pmtIndx] = dt_pmts['PBAL_BEG_PERIOD'].iloc[pmtIndx] - dt_pmts['COAMT'].iloc[pmtIndx] \
                                          - dt_pmts['PBAL_END_PERIOD']
    print('\nFix 5 complete...')

    # Fix 6: Check to ensure AMT PAID = INT PAID + FEE PAID, if not modify

    pmtIndx = np.where(dt_pmts['RECEIVED_AMT'] - dt_pmts['PRNCP_PAID'] - dt_pmts['INT_PAID'] - dt_pmts['FEE_PAID'] > 0)
    dt_pmts['RECEIVED_AMT'].iloc[pmtIndx] = dt_pmts['PRNCP_PAID'].iloc[pmtIndx] + dt_pmts['INT_PAID'].iloc[pmtIndx] \
                                            + dt_pmts['FEE_PAID'].iloc[pmtIndx]
    print('\nFix 6 complete...')

    # Fix 7: Find non-consecutive pmt months <not done>

    #month = pd.to_datetime(dt_pmts['MONTH'],format='%b%Y')
    #pmtIndx = np.where(np.logical_and(12 * month.iloc[:-1].dt.year + month.iloc[:-1].dt.month + 1 !=
                                      #12 * month.iloc[1:].dt.year + month.iloc[1:].dt.month,
                                      #dt_pmts['LOAN_ID'].iloc[1:] == dt_pmts['LOAN_ID'].iloc[:-1]))
    #dt_new = pd.DataFrame(data=np.zeros(shape=(len(pmtIndx[0]),dt_pmts.shape[1])))

    #if len(pmtIndx[0]) == 0:
        #print('\nFix 7 complete...')

    # Fix 8: Find multiple payments in one month

    # Fix 9: Ensure a zero payment exists when difference between MOB 1 and the issue date

    # Fix 10: Make sure all MOB's start at 1

    mobIndx = np.where(dt_pmts['MOB']==0)
    loanIndx = np.unique(dt_pmts['LOAN_ID'].iloc[mobIndx])
    mobIndx = np.where(np.in1d(dt_pmts['LOAN_ID'],loanIndx))
    dt_pmts['MOB'].iloc[mobIndx] = dt_pmts['MOB'].iloc[mobIndx]+1

    print('\nFix 10 is complete...')

    return dt_pmts

def main(argv = sys.argv):



tt = dt.iloc[-2000000:]
dtResp, matCount = generate_responses(tt)
dtRegr, dtCharts = generate_regressors(tt)
modelList, dtCharts = generate_transition_probs(dtRegr,dtResp,dtCharts,1)
uncondMat = calculate_unconditional_probs(dtResp,modelList)

ids = dt[np.logical_and(dt['months_on_balance']==30,dt['term']==36)].sample(10)
dtOrig = dt[np.logical_and(dt['months_on_balance']==0,np.in1d(dt['originator_loan_id'],ids['originator_loan_id']))]

dtRegrOrig = dtRegr.iloc[dtOrig.index]
outYield = calculate_par_yield(dtOrig,dtRegrOrig,uncondMat,modelList,100)
cc = calculate_output_curves(dt,dtRegr,10,36,uncondMat,modelList,100)

lIndx = np.where(np.in1d(dt['originator_loan_id'],outYield.index))

actC = calculate_actual_yield(dt.iloc[lIndx])
cc = pd.DataFrame.merge(outYield,actC,how='inner',left_index=True,right_index=True,suffixes=('_m','_a'))
cc.to_csv('cc.csv')

if __name__ == "__main__":
    sys.exit(main())