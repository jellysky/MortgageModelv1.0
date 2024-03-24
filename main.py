# __author__ = 'koushik'

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import sys

#from sklearn.linear_model import LogisticRegression
#import matplotlib.pyplot as plt

class const():
    @staticmethod
    def maxTerm():
        return 61
    @staticmethod
    def pi():
        return 3.14159
    @staticmethod
    def maxStates():
        return 6
    @staticmethod
    def ageKnots():
        return 6
    @staticmethod
    def trMatRows():
        return 4
    @staticmethod
    def trMatCols():
        return 6
    @staticmethod
    def maxTrees():
        return 30
    @staticmethod
    def maxDepth():
        return None
    @staticmethod
    def maxLeaf():
        return 1

def read_data(pmt_str,chars_str):
# reads in pmts and char files

    pmt_headers = ['seq_no','rep_date','end_act_upb','curr_delq_status','age','TTM','repurch_flag','mod_flag',
               'zero_bal_code','zero_bal_eff_date','curr_int_rate','curr_def_upb','due_date_last_paid_inst',
               'mi_recovs','net_sales_proc','non_mi_recovs','expenses','legal_costs','main_pres_costs','tax_ins',
               'misc_exp','act_loss_calc']

    dt_pmts = pd.read_table(pmt_str, header=None, sep='|', index_col=False)
    dt_pmts.columns = pmt_headers
    dt_pmts = dt_pmts.sort_values(by=['seq_no','age'])
    print('\nReading in mortgage payments data...')


    chars_headers = ['fico','first_pmt_date','first_time_homebuyer_flag','mat_date','msa','mort_ins_%','no_units',
                     'occ_status','orig_comb_ltv','orig_dti','orig_upb','orig_ltv','orig_int_rate','channel',
                     'prepay_penal_flag','prod_type','state','prop_type','postal_code','seq_no','loan_purpose',
                     'orig_loan_term','no_of_borrowers','seller_name','serv_name']
    dt_chars = pd.read_table(chars_str,header=None,sep='|',index_col=False)
    dt_chars.columns = chars_headers
    dt_chars = dt_chars.sort_values(by=['seq_no'])

    charIndx = dt_chars.columns.difference(dt_pmts.columns)
    dt = pd.merge(dt_pmts, dt_chars, how='inner', left_on='seq_no', right_on='seq_no', suffixes=['', '_y'], copy=False)

    print('\nReading in mortgage borrower data...')

    return dt

def insert_status(dt):
# states are defined as follows: 0-6: delq status, 7: foreclosure, 8: REO, 9: prepays, 10: default

    loan_status = np.zeros(shape=(dt.shape[0])) - 100 # -100 is a placeholder

    for i in range(0,7):
        #print(i)
        loan_status[np.where(dt['curr_delq_status'] == str(i))] = i

    loan_status[np.where(dt_pmts['zero_bal_code'] == 3)] = 7 # 7: foreclosure
    loan_status[np.where(dt_pmts['zero_bal_code'] == 9)] = 8 # 8: REO
    loan_status[np.where(np.logical_and(dt_pmts['zero_bal_code'] == 1,dt_pmts['TTM'] > 0))] = 9 # 9: prepays
    loan_status[np.where(loan_status == -100)] = 10 # 10: default
    dt['loan_status'] = loan_status
    print('\nFinished adding status column...')

    return dt

def clean_data(dt):

dt = dt_pmts
dt['curr_delq_status'] = dt['curr_delq_status'].astype(str) # change all delq statuses to strings, 'R' = REO
newColHeaders = ['beg_act_upb','prin_paid','int_paid','due_amt','rec_amt','co_amt','end_act_upb']
newCols = pd.DataFrame(data=np.zeros(shape=(dt.shape[0], len(newColHeaders))),
                                     index=dt.index,columns=newColHeaders)

begIndx = dt['seq_no'].iloc[:-1]==dt['seq_no'].iloc[1:]
newCols['beg_act_upb'].iloc[np.where(begIndx)[0]+1] = dt['end_act_upb'].iloc[np.where(begIndx)[0]].values # beg balance is prev end bal

begIndx = np.where(dt['seq_no'].iloc[:-1]!=dt['seq_no'].iloc[1:])[0]+1,
                         np.where(dt['curr_delq_status']!='0'))

tt = np.where(dt['curr_delq_status']!='0')
# Stopped here...figure out
newCols['beg_act_upb'].iloc[np.where(begIndx==False)[0]+1] = dt['end_act_upb'].iloc[np.where(begIndx==False)[0]+1] # first beg balance is first end bal

newCols['co_amt'].iloc[np.where(dt['zero_bal_code']>1)] = \
    newCols['beg_act_upb'].iloc[np.where(dt['zero_bal_code']>1)] # co's only for non prepay zero bal codes

newCols['prin_paid'] = newCols['beg_act_upb'] - dt['end_act_upb'] - newCols['co_amt']# prin paid
newCols['int_paid'] = newCols['beg_act_upb'] * dt['curr_int_rate']/1200 # int paid
newCols['due_amt'] = -np.pmt(dt['curr_int_rate']/1200,dt['orig_loan_term'],dt['orig_upb']) # amount due
newCols['rec_amt'] = newCols['prin_paid'] + newCols['int_paid'] # amount received
newCols['end_act_upb'] = dt['end_act_upb']

newCols.head(100000).to_csv('newCols.csv')
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

    # Fix 11: Make sure defaults only occur once
    pmtIndx = np.logical_and(np.in1d(dt_pmts['PERIOD_END_LSTAT'].iloc[:-1],('Default','Charged Off')),
                                     np.in1d(dt_pmts['PERIOD_END_LSTAT'].iloc[1:], ('Default', 'Charged Off')))
    dt_pmts.drop(dt_pmts.index[pmtIndx],inplace=True)

    return dt_pmts

def pre_processing(dt_pmts,dt_chars):


    dt['rep_date'] = pd.to_datetime(dt['rep_date'], format='%Y%m') # reformat rep_date and zero_bal_eff_date
    zIndx = np.where(dt['zero_bal_eff_date'].notnull())
    dt['zero_bal_eff_date'].iloc[zIndx] = pd.to_datetime(dt['zero_bal_eff_date'].iloc[zIndx])

    neverLate = np.ones(shape=(dt.shape[0],1)) # add neverLate indicator
    lateIndx = np.logical_and.reduce((dt['curr_delq_status'].iloc[:-1] == 0,
                                      dt['curr_delq_status'].iloc[1:] != 0,
                                     dt['seq_no'].iloc[:-1] == dt['seq_no'].iloc[1:]))
    lateLoans = np.unique(dt['seq_no'].iloc[lateIndx])

    for i,l in enumerate(lateLoans):
        startIndx = np.intersect1d(np.where(dt['seq_no'] == l)[0],lateIndx[0])[0]
        endIndx = np.where(dt['seq_no']==l)[0][-1]
        neverLate[startIndx:endIndx+1,0] = 0

    dt['never_late_indicator'] = neverLate

    print('\nProcessed data for LC payments and borrower data...')

    return dt

def create_transition_matrix(dt):

    states = dt['curr_delq_status'].unique()
    trMat = np.zeros(shape=(states.shape[0],states.shape[0]))

    for i,stState in enumerate(states):
        for j, edState in enumerate(states):

            print(i,j)
            trMat[i, j] = np.logical_and.reduce((dt['curr_delq_status'].iloc[:-1]==stState,
                               dt['curr_delq_status'].iloc[1:]==edState,
                               dt['seq_no'].iloc[:-1] == dt['seq_no'].iloc[1:])).sum()

    return trMat / np.tile(trMat.sum(axis=1),(states.shape[0],1)).transpose()



def generate_regressors(dt):
    # generates datatable of regressors
    dtRegr = np.zeros(shape=(dt.shape[0],88),dtype='float')

    # generates datatable of chart data
    dtCharts = pd.DataFrame(index=dt.index,columns=['term','MOB','orig_fico','vintage','month','loan_amnt','coupon','purpose','emp_length','probMod','y_act'])
    dtCharts[['probMod','y_act']] = 0 #important for plotting later on

    # Intercept
    # dtRegr[:,0] = 1

    lastCol = 0
    print lastCol

    # transform age into linear spline combinations
    for i in xrange(0,10):
        indx = np.where(np.logical_and(dt['MOB'] >= i*const.ageKnots(),dt['MOB'] < (i+1)*const.ageKnots()))
        dtRegr[indx,i] = np.maximum(dt['MOB'].iloc[indx] - i*const.ageKnots(),0)
        lastCol = lastCol+1
        print lastCol
    dtCharts['MOB'] = dt['MOB'] # for dtCharts

    # never late indicator
    dtRegr[np.where(dt['never_late_indicator']==1),lastCol] = 1
    lastCol = lastCol+1
    print lastCol

    # indicators for term
    dtRegr[np.where(dt['term']==60),lastCol] = 1
    lastCol = lastCol+1
    dtCharts['term'] = dt['term']
    print lastCol

    # indicators for term * age linear spline for term=36
    for j in xrange(0,10):
        dtRegr[:,lastCol] = np.multiply(np.absolute(dtRegr[:,11]-1),dtRegr[:,j])
        lastCol = lastCol+1
        print lastCol

    # indicators for term * age linear spline for term=60
    for j in xrange(0,10):
        dtRegr[:,lastCol] = np.multiply(dtRegr[:,11],dtRegr[:,j])
        lastCol = lastCol+1
        print lastCol

    # dti_ex_mortgage with nan's set to 0
    dtRegr[:,lastCol] = dt['dti']
    lastCol = lastCol+1
    print lastCol

    # fico ranges, exclude > 800
    ficoRanges = np.zeros(shape=(4,2),dtype='float')
    ficoRanges[:,0] = [0,675,700,750]
    ficoRanges[:,1] = [675,700,750,800]

    for i in xrange(0,len(ficoRanges)):
        dtRegr[np.where(np.logical_and(dt['orig_fico']<ficoRanges[i,1],dt['orig_fico']>=ficoRanges[i,0])),lastCol] = 1
        dtCharts['orig_fico'].iloc[np.where(np.logical_and(dt['orig_fico']<ficoRanges[i,1],dt['orig_fico']>=ficoRanges[i,0]))] = str(ficoRanges[i,0].astype(int)) + '-' + str(ficoRanges[i,1].astype(int))
        lastCol = lastCol+1
        print lastCol
    dtCharts['orig_fico'].iloc[np.where(pd.isnull(dtCharts['orig_fico']))] = '>800'

    # vintages
    year = np.array(map(float,dt['issue_d'].apply(lambda x: x[4:6])))+2000
    for i in xrange(2009,2016):
        dtRegr[np.where(year==i),lastCol] = 1
        dtCharts['vintage'].iloc[np.where(year==i)] = str(i)
        lastCol = lastCol+1
        print lastCol
    dtCharts['vintage'].iloc[np.where(pd.isnull(dtCharts['vintage']))] = '<2009'

    # month for seasonality effects
    month = np.array(map(float,dt['loan_month'].apply(lambda x: x[5:7])))
    for i in xrange(2,13):
        dtRegr[np.where(month==i),lastCol] = 1
        dtCharts['month'].iloc[np.where(month==i)] = str(i)
        lastCol = lastCol+1
        print lastCol
    dtCharts['month'].iloc[np.where(pd.isnull(dtCharts['month']))] = 1

    # loan size ranges
    loanSizes = np.zeros(shape=(6,2),dtype='float')
    loanSizes[:,0] = [0,5000,10000,15000,20000,25000]
    loanSizes[:,1] = loanSizes[:,0] + 5000

    for i in xrange(0,len(loanSizes)):
        dtRegr[np.where(np.logical_and(dt['loan_amnt']<loanSizes[i,1],dt['loan_amnt']>=loanSizes[i,0])),lastCol] = 1
        dtCharts['loan_amnt'].iloc[np.where(np.logical_and(dt['loan_amnt']<loanSizes[i,1],dt['loan_amnt']>=loanSizes[i,0]))] = str(loanSizes[i,0].astype(int)) + '-' + str(loanSizes[i,1].astype(int))
        lastCol = lastCol+1
        print lastCol
    dtCharts['loan_amnt'].iloc[np.where(pd.isnull(dtCharts['loan_amnt']))] = '>30000'

    # coupon ranges
    loanCoupons = np.zeros(shape=(6,2),dtype='float')
    loanCoupons[:,0] = [5,7.5,10,12.5,15,17.5]
    loanCoupons[:,1] = loanCoupons[:,0] + 2.5

    for i in xrange(0,len(loanCoupons)):
        dtRegr[np.where(np.logical_and(dt['coupon']<loanCoupons[i,1],dt['coupon']>=loanCoupons[i,0])),lastCol] = 1
        dtCharts['coupon'].iloc[np.where(np.logical_and(dt['coupon']<loanCoupons[i,1],dt['coupon']>=loanCoupons[i,0]))] = loanCoupons[i,0].astype(str) + '-' + loanCoupons[i,1].astype(str)
        lastCol = lastCol+1
        print lastCol
    dtCharts['coupon'].iloc[np.where(pd.isnull(dtCharts['coupon']))] = '>20.0'

    # loan purposes, dropped everything after major purchase
    loanPurposes = ['debt_consolidation','credit_card','home_improvement','major_purchase']
    for i in xrange(0,len(loanPurposes)):
        dtRegr[np.where(dt['purpose']==loanPurposes[i]),lastCol] = 1
        dtCharts['purpose'].iloc[np.where(dt['purpose']==loanPurposes[i])] = loanPurposes[i]
        lastCol = lastCol+1
        print lastCol
    dtCharts['purpose'].iloc[np.where(pd.isnull(dtCharts['purpose']))] = 'other'

    # employment length, dropped 9 years
    empLengths = ['< 1 year','1 year','2 years','3 years','4 years','5 years','6 years','7 years','8 years','10+ years','n/a']
    for i in xrange(0,len(empLengths)):
        dtRegr[np.where(dt['emp_length']==empLengths[i]),lastCol] = 1
        dtCharts['emp_length'].iloc[np.where(dt['emp_length']==empLengths[i])] = empLengths[i]
        lastCol = lastCol+1
        print lastCol
    dtCharts['emp_length'].iloc[np.where(pd.isnull(dtCharts['emp_length']))] = '9 years'

    # inquiries in last 6m, convert nan's to 0's
    dtRegr[:,lastCol] = dt['inq_last_6mths']
    lastCol = lastCol+1
    print lastCol

    # monthly gross income
    dtRegr[:,lastCol] = dt['annual_inc']/12
    lastCol = lastCol+1
    print lastCol

    # total outstanding accounts
    dtRegr[:,lastCol] = dt['total_acc']
    lastCol = lastCol+1
    print lastCol

    # revolving utilization
    dtRegr[:,lastCol] = dt['revolving_utilization']
    lastCol = lastCol+1
    print lastCol

    # delinquent accounts in last 2 years
    dtRegr[:,lastCol] = dt['delinq_2yrs']
    lastCol = lastCol+1
    print lastCol

    # total open credit lines
    dtRegr[:,lastCol] = dt['open_acc']
    #print lastCol

    headers = ['age_k0','age_k1','age_k2','age_k3','age_k4','age_k5','age_k6','age_k7','age_k8','age_k9',
        'nevlate','term60',
        'age_k0.term36','age_k1.term36','age_k2.term36','age_k3.term36','age_k4.term36','age_k5.term36',
        'age_k6.term36','age_k7.term36','age_k8.term36','age_k9.term36',
        'age_k0.term60','age_k1.term60','age_k2.term60','age_k3.term60','age_k4.term60','age_k5.term60',
        'age_k6.term60','age_k7.term60','age_k8.term60','age_k9.term60',
        'dti','fico675','fico700','fico750','fico800',
        'vint2009','vint2010','vint2011','vint2012','vint2013','vint2014','vint2015',
        'm2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12',
        'ls0','ls5','ls10','ls15','ls20','ls25','coup5','coup7.5','coup10','coup12.5','coup15','coup17.5',
        'lp_debt','lp_cc','lp_hi','lp_mp','el<1','el1','el2','el3','el4','el5','el6','el7','el8','el10+','elna',
        'inql6m','anninc','totacc','revutil','del2y','openacc']

    return pd.DataFrame(dtRegr,columns=headers), dtCharts

def generate_responses(dt):
    # generates datatable of transitions and transition counts
    # 0: current, 1: 1m late, 2: 2m late, 3: 2+m late, 4: default, 5: prepaid entirely

    matCount = np.zeros(shape=(const.maxStates()-2,const.maxStates()),dtype='float')
    dtResp = np.zeros(shape=(dt.shape[0]-1, (const.maxStates() - 2) * const.maxStates()), dtype='float')
    headers = ['CtoC', 'CtoD3', 'CtoD6', 'CtoD6+', 'CtoD', 'CtoP',
               'D3toC', 'D3toD3', 'D3toD6', 'D3toD6+', 'D3toD', 'D3toP',
               'D6toC', 'D6toD3', 'D6toD6', 'D6toD6+', 'D6toD', 'D6toP',
               'D6+toC', 'D6+toD3', 'D6+toD6', 'D6+toD6+', 'D6+toD', 'D6+toP', ]
    cols = ['C','D3','D6','D6+','D','P']


    for i in xrange(0,const.maxStates()-2):
        for j in xrange(0,const.maxStates()):
            dtResp[np.where(np.logical_and.reduce((np.array(dt['loan_status'].iloc[1:]==j),
               np.array(dt['loan_status'].iloc[:-1]==i),
               np.array(dt['LOAN_ID'].iloc[1:]==dt['LOAN_ID'].iloc[:-1])))),const.maxStates()*i+j] = 1

            matCount[i,j] = sum(dtResp[:,6*i+j])
            print('\nFinished calculating responses for transition state: %s ...' %headers[const.maxStates()*i+j])

    return pd.DataFrame(dtResp,columns=headers), pd.DataFrame(matCount,columns=cols,index=cols[:-2])

def analyze_model(model,X_test,y_test):

    # print model accuracy
    print '\nModel overall accuracy is: %.2f for startState: %s ...' \
          % (model.score(X_test,y_test),y_test.columns.values[0][0:y_test.columns.values[0].index('t')])

    # print OOB error rate
    print '\nModel OOB error rate is: %.2f for startState: %s ...' \
          % (1-model.oob_score_,y_test.columns.values[0][0:y_test.columns.values[0].index('t')])

    for i,c in enumerate(y_test.columns.values):

        #print(i,c)

        # print model vs actual score
        print '\nModel : Actual ones are %.0f : %.0f out of %.0f for Pr(%s) ...' \
              % (model.predict(X_test)[:,i].sum(),y_test[c].sum(),y_test.shape[0],c)

        # print AUC score
        if (y_test[c].sum() > 0):
            print '\nModel ROC AUC score is: %.2f for Pr(%s) ...' \
              % (metrics.roc_auc_score(y_test[c].values, model.predict_proba(X_test)[i][:,1]),c)
        else:
            print '\nTransition: %s has only one class present...' %(c)

def test_train_split(dt,term,test_size):

    #test_size = 0.33
    origIndx = np.where(np.logical_and(dt['MOB']==1,dt['term']==term))
    X_train, X_test, y_train, y_test = train_test_split(dt.iloc[origIndx], np.ones(shape=(len(origIndx[0]))), test_size=test_size, random_state=0)

    return np.where(np.in1d(dt['LOAN_ID'],dt['LOAN_ID'].iloc[X_train.index])),np.where(np.in1d(dt['LOAN_ID'],dt['LOAN_ID'].iloc[X_test.index]))

def generate_transition_probs(dtRegr,dtResp,dtCharts):

#tuner = 1
#startState = 'C'

    startStates = ['C','D3','D6','D6+']
    exemptTransitions = ['CtoC','CtoD6','CtoD6+','D3toD3','D3toD6+','D6toC','D6toD6','D6toD','D6toP','D6+toC',
                         'D6+toD3','D6+toD6+','D6+toP']
    modelList = [RandomForestClassifier for i in xrange(0,len(startStates))]

    for i,startState in enumerate(startStates):

        nonExemptIndx = np.where(np.in1d(dtResp.columns.values,exemptTransitions)==False)[0]
        startStateIndx = np.where([colName[0:colName.index('t')] == startState for colName in dtResp.columns.values])[0]
        #evalCols = np.intersect1d(nonExemptIndx,startStateIndx)

        # select only those rows where the starting state is the same as startState
        rIndx = np.where(dtResp[startStateIndx].sum(axis=1)==1)[0]

        y = dtResp[startStateIndx].iloc[rIndx]
        X = dtRegr.iloc[rIndx]

        # split the relevant data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

        model = RandomForestClassifier(n_estimators=const.maxTrees(), max_depth=const.maxDepth(), max_features='auto',
                               bootstrap=True, oob_score=True, random_state=531, min_samples_leaf=const.maxLeaf())
        # model = SGDClassifier(loss='log',penalty='l1',alpha='optimal',l1_ratio=1,n_iter=10,fit_intercept=False)
        modelList[i] = model.fit(X_train, y_train)

        analyze_model(modelList[i],X_test,y_test)

        # convert dtCharts to floats and generate predicted probs for X test only
        dtCharts[['probMod','y_act']] = dtCharts[['probMod','y_act']].astype(float)
        modelProbs = modelList[i].predict_proba(X_test)

        # gather test data and plot it on bar charts
        for j in xrange(0,const.maxStates()):
            stateIndx = np.in1d(X_test.index,np.where(dtResp[y_test.columns.values[j]]==1))
            dtCharts['probMod'].iloc[stateIndx] = 1-modelProbs[j][:,0][stateIndx]
            dtCharts['y_act'].iloc[stateIndx] = dtResp[y_test.columns.values[j]].iloc[stateIndx]
            #plot_model_output(dtCharts.iloc[stateIndx],y_test.columns.values[j])
            print('\nCalculated test probs for transition state: %s ...' %y_test.columns.values[j])

    return modelList, dtCharts.iloc[X_test.index]



def increment_regressor(X,mob):

    dtModRegr = X.copy(deep=True)
    lastCol = 0

    for i in xrange(0,10):
        if  (mob >= i*const.ageKnots()) and (mob < (i+1)*const.ageKnots()):
            dtModRegr.iloc[i] = max(mob - i*const.ageKnots(),0)
        else:
            dtModRegr.iloc[i] = 0
        lastCol = lastCol+1
        #print lastCol

    lastCol = np.where(np.in1d(X.index.values,'age_k0.term36'))[0][0]
    # indicators for term * age linear spline
    for j in xrange(0,10):
        dtModRegr.iloc[lastCol] = np.multiply(np.absolute(dtModRegr.iloc[11]-1),dtModRegr.iloc[j])
        lastCol = lastCol+1
        #print lastCol

    # indicators for term * age linear spline for term=60
    for j in xrange(0,10):
        dtModRegr.iloc[lastCol] = np.multiply(dtModRegr.iloc[11],dtModRegr.iloc[j])
        lastCol = lastCol+1
        #print lastCol

    # loan month
    # determines start month
    lastCol = np.where(np.in1d(X.index.values,'m2'))[0][0]
    if np.where(dtModRegr.iloc[lastCol:lastCol+11]==1)[0].sum()==0:
        month = 1
    else:
        month = np.where(dtModRegr.iloc[lastCol:lastCol+11]==1)[0][0]+2

    # determines new month based on mob
    dtModRegr.iloc[lastCol:lastCol+11] = 0
    if mob + month > 12:
        month = (mob + month) % 12
    else:
        month = mob + month

    # specifies column
    if month > 1:
        dtModRegr.iloc[lastCol+month-2] = 1

    return dtModRegr

def calculate_transition_matrix(X,modelList):
#X = dtRegrOrig.iloc[i]

    # trMat[0,:] = [.978,.985,.985,.985,.985,1]
    # trMat[1,:] = [.176,.324,.976,.977,.992,1]
    # trMat[2,:] = [.027,.055,.142,.945,.997,1]
    # trMat[3,:] = [.01,.013,.024,.56,998,1]

    # returns cumulative transition prob matrix
    trMat = np.zeros(shape=(const.trMatCols(),const.trMatCols()),dtype=float)
    for i in xrange(0,const.trMatRows()):
        for j in xrange(0,const.trMatCols()):
            if (i!=j):
                trMat[i,j] = 1-modelList[i].predict_proba(X.reshape(1,-1))[j][0][0]

        trMat[i,i] = 1 - trMat[i,:].sum()
    trMat[const.trMatCols()-2,const.trMatCols()-2] = 1 # for defs, end in defs
    trMat[const.trMatCols()-1,const.trMatCols()-2] = 1 # for prepays, move to def
    #print trMat

    return trMat.cumsum(axis=1)

def simulate_cashflows(coupon,term,loanAmt,xRegr,modelList,stState,nSims):

    np.set_printoptions(precision=4)
    cfMat = np.zeros(shape=(nSims,term),dtype=float)
    dtState = np.zeros(shape=(nSims,term),dtype=int)
    prinPaid = np.zeros(shape=(nSims,term),dtype=float) # cumulative
    prinPrepaid = np.zeros(shape=(nSims,term),dtype=float) # not cumulative
    prinDef = np.zeros(shape=(nSims,term),dtype=float) # not cumulative

    if stState == 0:
        cfMat[:,0] = np.pmt(coupon/1200,term,loanAmt,0,0) # at t=0 all loans are assumed to be current
        prinPaid[:,0] = np.ppmt(coupon/1200,0,term,loanAmt,0,0)

    for j in xrange(1,term): # j is the time index
        #print j
        trMat = calculate_transition_matrix(increment_regressor(xRegr,j),modelList)

        prinPaid[:,j] = prinPaid[:,j-1]
        stateVar = np.random.uniform(0,1,nSims).reshape(1,-1).transpose() # stateVar are the random uniforms
        dtState[:,j] = np.argmax(np.less(np.tile(stateVar,(1,trMat.shape[1])),trMat[dtState[:,j-1],:]),axis=1)

        prIndx = np.where(np.logical_and(dtState[:,j]==5,j<term-1)) # prepay index
        cfMat[prIndx,j] = np.ipmt(coupon/1200,j,term,loanAmt,0,0) - loanAmt - prinPaid[prIndx,j]
        prinPrepaid[prIndx,j] = -loanAmt - np.ppmt(coupon/1200,j,term,loanAmt,0,0) - prinPaid[prIndx,j]
        prinPaid[prIndx,j] = -loanAmt

        defIndx = np.where(np.logical_and(dtState[:,j]==4,dtState[:,j-1]!=4))
        prinDef[defIndx,j] = -loanAmt - prinPaid[defIndx,j]

        lsIndx = np.where(np.logical_and(dtState[:,j]<=dtState[:,j-1],dtState[:,j]<4)) # states are unequal and not prepay or default
        cfMat[lsIndx,j] = np.multiply(np.pmt(coupon/1200,term,loanAmt,0,0),dtState[lsIndx,j-1]-dtState[lsIndx,j]+1)
        prinPaid[lsIndx,j] += [np.ppmt(coupon/1200,np.arange(j-dtState[lsIndx[0][l],j-1]+dtState[lsIndx[0][l],j],j+1),term,loanAmt,0,0)[0] \
              for l in xrange(0,lsIndx[0].shape[0])]

        if (j==term-1): # terminal condition for current loans, make principal payback whole if current
            dtState[dtState[:,j]==5,j] = 0 # prepays on last date are basically currents
            cfMat[np.where(dtState[:,j]==0),j] = np.ipmt(coupon/1200,j,term,loanAmt,0,0) - loanAmt - prinPaid[np.where(dtState[:,j]==0),j-1]
            prinPaid[np.where(dtState[:,j]==0),j] = -loanAmt

    return cfMat,prinPrepaid,prinDef,prinPaid,dtState



def calculate_loan_price(dtLast,dtRegrLast,outYield,modelList,nSims):

#dtLast = dt.iloc[pmtIndx]
#dtRegrLast = dtRegr.iloc[pmtIndx]

    outPrice = np.zeros(shape=(dtLast.shape[0],3),dtype=float)
    outPrice[:,0] = dtLast['MOB']
    perfRates = np.zeros(shape=(dtLast['term'].iloc[0],5),dtype=float)

    for i in xrange(0,dtLast.shape[0]):   # i is the loan index

        if (dtLast['term'].iloc[i]>=dtLast['MOB'].iloc[i]):# and (lastStatus[i]<4):
            #print(i)

            cfMat,cfPrepaid,cfDefs,cfPaid,dtState = simulate_cashflows(dtLast['coupon'].iloc[i],dtLast['term'].iloc[i].astype(int)-dtLast['MOB'].iloc[i].astype(int)+1,
                 dtLast['PBAL_BEG_PERIOD'].iloc[i],dtRegrLast.iloc[i],modelList,dtLast['loan_status'].iloc[i],nSims)

            outPrice[i,1] = 100*np.npv(outYield['ModYld'].ix[dtLast['LOAN_ID'].iloc[i]],-cfMat.mean(axis=0))/dtLast['PBAL_BEG_PERIOD'].iloc[i]
            outPrice[i,2] = 100*([np.npv(outYield['ModYld'].ix[dtLast['LOAN_ID'].iloc[i]],-cfMat[row,:]) for row in xrange(0,nSims)]/dtLast['PBAL_BEG_PERIOD'].iloc[i]).mean()

            #calculate cpr and cdr
            perfRates[np.arange(dtLast['MOB'].iloc[i].astype(int)-1,dtLast['term'].iloc[i].astype(int)),0] += -dtLast['loan_amnt'].iloc[i] #- cfPaid.mean(axis=0)
            perfRates[np.arange(dtLast['MOB'].iloc[i].astype(int)-1,dtLast['term'].iloc[i].astype(int)),1] += cfPrepaid.mean(axis=0)

            #perfRates[np.arange(dtLast['MOB'].iloc[i].astype(int),dtLast['term'].iloc[i].astype(int)),3] += -dtLast['loan_amnt'].iloc[i] - cfPaid.mean(axis=0)
            perfRates[np.arange(dtLast['MOB'].iloc[i].astype(int)-1,dtLast['term'].iloc[i].astype(int)),2] += cfDefs.mean(axis=0)

        print '\nMod price: avg price is %.2f : %.2f for loan: %d @mob: %d' % (outPrice[i,1],outPrice[i,2],dtLast['LOAN_ID'].iloc[i],dtLast['MOB'].iloc[i])

    perfRates[:,3] = np.nan_to_num(np.divide(perfRates[:,1],perfRates[:,0]))
    perfRates[:,4] = np.nan_to_num(np.divide(perfRates[:,2],perfRates[:,0]))

    return pd.DataFrame(data=outPrice,index=dtLast['LOAN_ID'],columns=['MOB','ModPrice','AvgPrice']),pd.DataFrame(data=perfRates,index=np.arange(0,dtLast['term'].iloc[0]),columns=['BegBal','Prepays','Defs','CPR','CDR'])

def calculate_actual_curves(dtC):

#dtC = dt.iloc[pmtIndx]

    perfRates = np.zeros(shape=(dtC['term'].iloc[0],5),dtype=float)
    cfPrin = dtC.groupby(['MOB'])['loan_amnt'].sum()
    cfPrepaid = dtC.groupby(['MOB'])['unscheduled_principal'].sum()
    cfDefs = dtC.groupby(['MOB'])['COAMT'].sum()

    #perfRates[0:min(cfPrin.shape[0],dtC['term'].iloc[0]),0] = cfPrin.iloc[:dtC['term'].iloc[0]]
    perfRates[:,0] = cfPrin.iloc[0]
    perfRates[0:min(cfPrin.shape[0],dtC['term'].iloc[0]),1] = cfPrepaid.iloc[0:dtC['term'].iloc[0]]
    perfRates[0:min(cfPrin.shape[0],dtC['term'].iloc[0]),2] = cfDefs.iloc[0:dtC['term'].iloc[0]]
    perfRates[:,3] = np.nan_to_num(np.divide(perfRates[:,1],perfRates[:,0]))
    perfRates[:,4] = np.nan_to_num(np.divide(perfRates[:,2],perfRates[:,0]))

    return pd.DataFrame(data=perfRates,index=np.arange(0,dtC['term'].iloc[0]),columns=['BegBal','Prepays','Defs','CPR','CDR'])

def calculate_output_curves(dt,dtRegr,numLoans,term,modelList,nSims):

    # numLoans=10
    # term=36
    # nSims=100
    # dt = dt.iloc[testIndx]
    # dtRegr = dtRegr.iloc[testIndx]
    cumCurves = np.zeros(shape=(term,4))

    origIndx = np.where(np.in1d(dt.index,dt[np.logical_and(dt['MOB']==1,dt['term']==term)].sample(numLoans).index))
    pmtIndx = np.where(np.logical_and(np.in1d(dt['LOAN_ID'],dt['LOAN_ID'].iloc[origIndx]),dt['MOB']<=dt['term']))
    outYield = calculate_par_yield(dt.iloc[origIndx],dtRegr.iloc[origIndx],modelList,nSims)

    #pmtIndx = pmtIndx[0][np.where(dt.iloc[pmtIndx].duplicated(['LOAN_ID','MOB'])==False)]

    outPrice,modCurves = calculate_loan_price(dt.iloc[origIndx],dtRegr.iloc[origIndx],outYield,modelList,nSims)
    actCurves = calculate_actual_curves(dt.iloc[pmtIndx])
    compCurves = pd.merge(modCurves,actCurves,how='inner',left_index=True,right_index=True,suffixes=('_m','_a'))
    compCurves.to_csv('compCurves.csv')

    cumCurves[:,0:2] = modCurves.iloc[:,3:5].cumsum()
    cumCurves[:,2:4] = actCurves.iloc[:,3:5].cumsum()
    cumCurves = pd.DataFrame(cumCurves,columns=['CPR_mod','CDR_mod','CPR_act','CDR_act'])
    cumCurves.to_csv('cumCurves.csv')

    return compCurves,cumCurves,outPrice



def calculate_actual_yield(dtC):

    actYields = np.zeros(shape=(dtC['LOAN_ID'].unique().shape[0]),dtype=float)

    for i,loan in enumerate(dtC['LOAN_ID'].unique()):
        loanIndx = np.where(dtC['LOAN_ID']==loan)
        cfMat = np.zeros(shape=(len(loanIndx[0])+1),dtype=float)
        cfMat[0] = dtC['loan_amnt'].iloc[loanIndx].mean()

        cfMat[1:cfMat.shape[0]] = -dtC['RECEIVED_AMT'].iloc[loanIndx]
        #cfMat[1:cfMat.shape[0]] = -dt['unscheduled_principal'].iloc[loanIndx]+np.pmt(dtC['coupon'].iloc[loanIndx[0][0]]/1200,dtC['term'].iloc[loanIndx[0][0]],dtC['loan_amnt'].iloc[loanIndx[0][0]],0)
        actYields[i] = np.irr(cfMat)
        print('\nfor loan: %.0f the actual yield is: %.2f' % (loan,100*((1+actYields[i])**12-1)))

    return pd.DataFrame(data=actYields,index=dtC['LOAN_ID'].unique(),columns=['ActYld'])

def calculate_par_yield(dtOrig,dtRegrOrig,modelList,nSims):

    # outYields three yields are (1) coupon (2) IRR of avg CF's (3) avg yield over paths
    outYield = np.zeros(shape=(dtOrig.shape[0],3),dtype=float)

    for i in xrange(0,outYield.shape[0]):   # i is the loan index
        #print i
        cfMat = simulate_cashflows(dtOrig['coupon'].iloc[i],dtOrig['term'].iloc[i].astype(int),dtOrig['loan_amnt'].iloc[i],
                 dtRegrOrig.iloc[i],modelList,0,nSims)[0]
        cfMat = np.c_[-dtOrig['loan_amnt'].iloc[i]*np.ones(shape=(cfMat.shape[0])),-1*cfMat]

        outYield[i,0] = dtOrig['coupon'].iloc[i]
        outYield[i,1] = np.irr(cfMat.mean(axis=0))
        #y[i,1] = 100*((1+y[i,1])**12-1)
        outYield[i,2] = (100*((1+np.array([np.irr(cfMat[row,:]) for row in xrange(0,cfMat.shape[0])]))**12 - 1)).mean()
        print '\nFor loan: %.0f coupon : mod yield: avg yield is %.2f : %.2f : %.2f' % \
              (dtOrig['LOAN_ID'].iloc[i],outYield[i,0],100*((1+outYield[i,1])**12-1),outYield[i,2])

    return pd.DataFrame(data=outYield,index=dtOrig['LOAN_ID'],columns=['Coupon','ModYld','AvgYld'])

def calculate_output_yields(dt,dtRegr,numLoans,term,modelList,nSims):

    origIndx = np.where(np.in1d(dt.index,dt[np.logical_and(dt['MOB']==1,dt['term']==term)].sample(numLoans).index))
    pmtIndx = np.where(np.in1d(dt['LOAN_ID'],dt['LOAN_ID'].iloc[origIndx]))
    modYield = calculate_par_yield(dt.iloc[origIndx],dtRegr.iloc[origIndx],modelList,nSims)
    actYield = calculate_actual_yield(dt.iloc[pmtIndx])

    outYield = pd.merge(modYield,actYield,how='inner',left_index=True,right_index=True)
    outYield[['ModYld','ActYld']] = 100*((1+outYield[['ModYld','ActYld']])**12-1)
    outYield.to_csv('outYield.csv')
    return outYield


def main(argv = sys.argv):

    dt = read_data('sample_1999/sample_svcg_1999.txt','sample_1999/sample_orig_1999.txt')
    dt = insert_status(dt)

    dt = pre_processing(dt_pmts, dt_chars)
    dt_pmts_clean = clean_data(dt_pmts)
    dt_pmts_clean = insert_status(dt_pmts_clean)


    dtResp, matCount = generate_responses(dt)
    dtRegr, dtCharts = generate_regressors(dt)
    trainIndx, testIndx = test_train_split(dt,36,.33)
    modelList, dtCharts = generate_transition_probs(dtRegr.iloc[trainIndx],dtResp.iloc[trainIndx],dtCharts)

    del dt_pmts
    del dt_pmts_clean
    del dt_chars

    compCurves,cumCurves,outPrice = calculate_output_curves(dt.iloc[testIndx],dtRegr.iloc[testIndx],1000,36,modelList,100)
    outYield = calculate_output_yields(dt.iloc[testIndx],dtRegr.iloc[testIndx],1000,36,modelList,100)

if __name__ == "__main__":
    sys.exit(main())

