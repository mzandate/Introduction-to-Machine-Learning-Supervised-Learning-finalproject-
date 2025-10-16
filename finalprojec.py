%matplotlib inline
import micropip
await micropip.install("pandas")
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
await micropip.install("seaborn")
import seaborn as sns



import statsmodels.formula.api as smf
import statsmodels.api as sm

# Define the column names from the dataset
columns = ['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure'
           ,'platelets','serum_creatinine','serum_sodium','sex','smoking','time','death_event']


df = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
plotting = df[['age', 'ejection_fraction', 'serum_creatinine', 'time']]
df.head()
df.info()


#models to see if age is a good predictor 
model = smf.ols(formula='diabetes ~ age', data=df).fit()
print(model.summary())
model = smf.ols(formula='high_blood_pressure ~ age ', data=df).fit()
print(model.summary())

# code below if finding out the best predictor for death_event
# and printing  out the best predictor and its r squared value
predictors = ['anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','sex','smoking','age ']

best_predictor=''
best_r_squared=0

for colums in predictors:

    model = smf.ols(formula=f'DEATH_EVENT ~ {colums}', data=df).fit()
    print(colums)
    print(model.rsquared)
    print("")
    
    if best_r_squared < model.rsquared:
        best_predictor = colums
        best_r_squared = model.rsquared
                 
print(best_predictor)
print(best_r_squared)

# buding modeles with teh best preditcotrs and seeing if add each one make the r squared value better
model = smf.ols(formula='DEATH_EVENT  ~ serum_creatinine', data=df).fit()
print(model.summary())

model = smf.ols(formula='DEATH_EVENT  ~ serum_creatinine + ejection_fraction', data=df).fit()
print(model.summary())

bestmodel = smf.ols(formula='DEATH_EVENT  ~ serum_creatinine + ejection_fraction + age ', data=df).fit()
print(bestmodel.summary() )


# finding the best degree for the polynomial regression and printing out the best degree and its r squared value
best_degree = 1
best_r_squared = 0
power = ''
n = 20

for i in range(n):
    print(i+1)
    power = power + f' + np.power(age,{i+1})+np.power(serum_creatinine,{i+1})+np.power(ejection_fraction,{i+1})'
    model = smf.ols(formula=f'DEATH_EVENT ~ {power}', data=df).fit()


    print(model.rsquared)
    print("")
    
    if best_r_squared < model.rsquared:
        best_degree = i +1
        best_r_squared = model.rsquared
        
    
print(best_degree)
print(best_r_squared)




best_degree = 1
best_r_squared = 0 
sound_degree = 1
n = 20
#doing normalization to see if it improves the r squared value 

df['serum_norm'] = df['serum_creatinine']/df['serum_creatinine'].mean()
deathpwr =''
# formula to save it for later when taking out points and making models 
formula = ''

for i in range(n):
    print(i+1)
    deathpwr = deathpwr + f' + np.power(serum_norm,{i+1})+ np.power(age,{i+1})+np.power(ejection_fraction,{i+1})'
    model = smf.ols(formula=f'DEATH_EVENT~ {deathpwr}', data=df).fit()
    

    print(model.rsquared)
    print("")
    
    if best_r_squared < model.rsquared:
        best_degree = i +1
        best_r_squared = model.rsquared
        bestmodel = model
        formula = deathpwr


print(best_degree)
print(best_r_squared)


#graphs to show the relationships between the variables in the best model
# and to show any outliers that may be affecting the model
plotting = df[['age', 'ejection_fraction', 'serum_creatinine', 'DEATH_EVENT',]]

print(bestmodel.summary())

plotting.describe(include='all')
sns.pairplot(plotting,diag_kind='kde')

sm.graphics.plot_leverage_resid2(bestmodel)

# taking out the points that are outliers and making a new model to see if it improves the r squared value
# the points taken out are 28 and 47
# these points were found using the leverage plot above
df.drop([28,47], inplace = True)
bestmodel = smf.ols(formula=f'DEATH_EVENT~ {formula}', data=df).fit()
print(bestmodel.summary())
sm.graphics.plot_leverage_resid2(bestmodel)


# taking out more points that are outliers and making a new model to see if it improves the r squared value

df.drop([8,86], inplace = True)
df.drop([131], inplace = True)
df.drop([228], inplace = True)
df.drop([186], inplace = True)
df.drop([10], inplace = True)

bestmodel = smf.ols(formula=f'DEATH_EVENT~ {formula}', data=df).fit()

print(bestmodel.summary())
sm.graphics.plot_leverage_resid2(bestmodel)