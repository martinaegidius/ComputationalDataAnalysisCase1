# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler 
import csv

#problem: contains many nans. before using pandas we need to know how to handle it.
def read_df(path=""):
    return pd.read_table(path+"case1Data.txt", delimiter=",")


def read_var_names(path=""):
    with open(path+'var_names.txt', newline='') as f:
        reader = csv.reader(f)
        var_names = list(reader)
    var_names = var_names[0]
    names = [x.replace(" ","").replace("_","") for x in var_names]
    return names


def update_plt_cfg():
    SMALL_SIZE = 8
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 16
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def split_data(df):
    dfNumeric = df.iloc[:, :96]
    dfCategorical = df.iloc[:, 96:]
    return dfNumeric, dfCategorical

def convertNumeric(df):
    #fix it so NaNs are not strings but rather np.NaN
    for col in df.columns:
        df[col]=pd.to_numeric(df[col], errors="coerce")
    return df

def missingValueInformation(df,plot=False):
    #checks the number of rows / cols with minimally one missing value
    #input:   numeric pandas df
    #returns: two pandas series objects for illustration.
    #           - the distribution of nans across variables
    #           - the distribution of nans across observations

    #obtain distribution as pandas series
    variable_nan_distribution = np.sum(np.isnan(df),axis=0)
    obs_nan_distribution = np.sum(np.isnan(df),axis=1)

    #check for minimally one value
    obs_min_one_missing = sum(df.apply(lambda x: sum(x.isnull().values), axis = 1)>0)
    var_min_one_missing = sum(df.apply(lambda x: sum(x.isnull().values), axis = 0)>0)
    print(f"At least one missing value occuring in {var_min_one_missing}/{df.shape[1]} variables")
    print(f"At least one missing value occuring in {obs_min_one_missing}/{df.shape[0]} observations")

    #mean number of missing values 
    print(f"Mean count missing per variable {np.mean(variable_nan_distribution)}")
    print(f"Mean count missing per observation {np.mean(obs_nan_distribution)}")
    

    #fix pandas series so it is easily plottable using seaborn
    variable_nan_distribution = variable_nan_distribution.rename_axis("variable").reset_index(name="count")
    obs_nan_distribution =  obs_nan_distribution.rename_axis("observation").reset_index(name="count")
    if(plot==True):
        missingValuePlots(variable_nan_distribution,obs_nan_distribution)

    return variable_nan_distribution, obs_nan_distribution


def missingValuePlots(var_df,obs_df):
    fig, ax = plt.subplots(2,1,figsize=(18,12))
    #fig.suptitle("number of missing values per variable")
    #plt.xticks(rotation=90)
    b1 = sns.barplot(data=var_df,x="variable",y="count",ax=ax[0])
    b2 = sns.barplot(data=obs_df,x="observation",y="count",ax=ax[1])
    for ax_ in ax:
        for tick in ax_.get_xticklabels():
            tick.set_rotation(90)
    ax[0].set_title("Missing values across variables",fontweight="bold",fontsize=16)
    ax[1].set_title("Missing values across observations",fontweight="bold",fontsize=16)
    
    fig.tight_layout()
    plt.show()

    fig.savefig('plots/missing_values.png')
    return None

def standardize_col(column):
    return (column - column.mean()) / column.std()

def standardize_df(df):
    scaler = StandardScaler() 
    df2 = scaler.fit_transform(df) 
    return df2

def printHighCorrelations(corrdf, k = 3):
    #prints and returns the three highest correlations per variable
    var_names = read_var_names()
    corrdf -= np.identity(corrdf.shape[0])*100.0 #kill diagonals just to be sure
    corrdf *= np.tri(*corrdf.shape) #kill upper triangular to 0.0  
    #get top three correlations
    completeMax = np.argmax(corrdf,axis=0)

    #print(corrdf)
    ind = np.argsort(corrdf,axis=0) #sort array along rows in ascending order, keep indices of argmax
    #print(ind)
    ind = ind[-k:,:] 
    #print(ind)
    for var in range(ind.shape[1]):
        print(f"{var_names[var]}")
        for maxInd in ind[:,var]:
            print(f"\tcorr with {var_names[maxInd]}: {corrdf[maxInd,var]}")
    return
    


def covariance_plot(df,title=None,covAnalysis=False):
    df_standardized = df.apply(standardize_col) 
    covs = df_standardized.cov().to_numpy()
    corrs = df_standardized.corr(numeric_only=True).to_numpy()
    fig, ax = plt.subplots(2,1,figsize=(10,10))
    covIm = ax[0].imshow(covs,cmap="viridis")
    #plt.colorbar(covIm,ax=ax[0])
    ax[0].set_title("Cov: " + title)
    plt.colorbar(covIm,fraction=0.046, pad=0.04)
    
    corrIm = ax[1].imshow(corrs,cmap="viridis")
    #cax = fig.add_axes([ax[1].get_position().x1-0.25,ax[1].get_position().y0,0.02,ax[0].get_position().y1-ax[1].get_position().y0])
    #fig.colorbar(covIm, cax=cax)
    plt.colorbar(corrIm,fraction=0.046, pad=0.04)
    ax[1].set_title("Corr: " + title)



    if(covAnalysis):
        printHighCorrelations(corrs)
    return 


#def standardize_data(df):


# %%
#update_plt_cfg()
#train_data = read_df()
#numeric_df, categorical_df = split_data(train_data)
#numeric_df = convertNumeric(numeric_df)
#var_missing_dist,obs_nan_dist = missingValueInformation(numeric_df,plot=True)

# %%
#norm_df = numeric_df.copy(deep=True)
#print(standardize_df(norm_df))
#covariance_plot(numeric_df,title="using pandas which handles NaN somehow",covAnalysis=True)

# %%
#printHighCorrelations(np.arange(0,16,dtype="float").reshape(4,4),k=2)

# %%
#test = np.arange(0,16,dtype="float").reshape(4,4)
#np.argpartition(test,-3)[-3:]

