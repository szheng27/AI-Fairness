import pandas as pd
import numpy as np
#plot
'''
import matplotlib.pyplot as plt
import plotly
import copy
import seaborn as sns
import plotly.graph_objects as go
'''
#AI
'''
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.combine import SMOTEENN
'''
#fairness
import aif360.datasets as aif_datasets
import aif360.algorithms.preprocessing as aif_algorithms



'''
Goal:make AI fair
    by remove dependency due to catagorical in the form of 3 criteria
    

template function:given data, process the data by



Function:given data, process the the data by
    1)change catagorical into binary form?
    2)replace with {0,1}
    3)for each catagories, ensure independence for each numerical value
        a) check P(Y=1|Ci = 0)/P(Y=1|Ci = 0) <= eps where eps is a predetermined thershold value [0,1]
        b) if true, repair all numerical
            by small ratio and repeat? by max ratio 1?  can do analysis
        c)check P(Y=1|Ci = 0)/P(Y=1|Ci = 0) <= eps again(this might help what ratio repaired by?)
    4)drop all catagorical and return data as df with numerical data concate label

    dependent function
    -check P(Y=1|Ci = 0)/P(Y=1|Ci = 0) <= eps
    need Y,C, eps
    return bool

    dependent function
    -repair data
    need C,name,N,Y,ratio
    return [name,N_new]
'''
#its a good idea to rename all the protected attribute as
#A_1,A_2,...
#Y as label
#R as prediction
#to avoid introduce personal bias


#check P(Y=1|Ci = 0)/P(Y=1|Ci = 0) <= eps
#need Y,C, eps
#return bool

#df as y,c


def independence_check(df,eps,catagory_name,label_name):
    '''
    for (Y,R,A) in binary classification
    where 
        Y the label
        R the prediction(classification)
        A the protected attribute
    for disparity impact:   label_name = Y
    for independence:       label_name = R

    for seperation:         label_name = R, catagory_name = A where Y=0,1
    for sufficient:         label_name = Y, catagory_name = A where R=0,1
    '''
	#select c from df where df[c] == 0
    group_0 = df[df[catagory_name] == 0]
    y_in_group_0 = group_0[group_0[catagory_name] == 1]
    prob_0 = len(y_in_group_0)/len(group_0)

    group_1 = df[df[catagory_name] == 1]
    y_in_group_1 = group_0[group_0[label_name] == 1]

    prob_1 = len(y_in_group_1)/len(group_1)
    ratio = min([prob_0,prob_1])/max([prob_0,prob_1])
    independence = False
    if ratio < eps:
        independence = True
    return independence

#df as y,n,...,c,...
#unprivileged_group as list of value that the group is unprivileged
#catagories as list name catagorical names need to adjust around
def repair_data(df,repair_level,catagories,label_name):

    #need C,name,N,Y,ratio
    BLD =  aif_datasets.BinaryLabelDataset(\
df =df,label_names=[label_name],protected_attribute_names=catagories)

    DIR = aif_algorithms.DisparateImpactRemover(repair_level = repair_level)

    rp_BLD = DIR.fit_transform(BLD)

    rp_columns=np.append(np.array(df.columns.drop(label_name)),label_name)
    rp_df = pd.DataFrame(np.hstack([rp_BLD.features,rp_BLD.labels]),columns=rp_columns)
    return rp_df

def disparate_constraint(df,repair_level,A,Y):
    #Y independent of A
    return repair_data(df,repair_level,A,Y)
def independence_constraint(df,repair_level,A,R):
    #R independent of A
    return repair_data(df,repair_level,A,R)

def seperation_constraint(df,repair_level,A,R):
    #R independent of A given Y
    df_y0 = df.query("Y == 0")
    df_y1 = df.query("Y == 1")
    df_0 = repair_data(df_y0,repair_level,A,R)
    df_1 =repair_data(df_y1,repair_level,A,R)
    return pd.concat([df_0,df_1])
def sufficient_constraint(df,repair_level,A,Y):
    #R independent of A given Y
    df_r0 = df.query("R == 0")
    df_r1 = df.query("R == 1")
    df_0 =repair_data(df_r0,repair_level,A,Y)
    df_1 =repair_data(df_r1,repair_level,A,Y)
    return pd.concat([df_0,df_1])


