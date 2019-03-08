# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 22:18:02 2017

@author: Z
"""

from datetime import datetime, date, time, timedelta
import numpy as np
import pandas as pd
pd.set_option('display.width', 150)
pd.set_option('display.max_columns', 200)

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import xgboost as xgb
from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor
from scipy import stats

pwd = '/Users/Amber/Dropbox/[Homework]/bus/new/'
pwd = 'C:/Users/Z/Dropbox/Homework/bus/new/'

# Load cleaned data
df0 = pd.read_csv(pwd+'data9.csv')
df0['DateTime'] = pd.to_datetime(df0['DateTime'])

df1 = pd.read_csv(pwd+'dataNew-1.csv')
df1['DateTime'] = pd.to_datetime(df1['DateTime'])

df = pd.concat([df0.iloc[:,:-5], df1], axis=0)

df.set_index('DateTime', inplace=True)

# Generating historical features: progress/distance traveld from n minute ago to now.
nPastMinutes=10
DistNames = ['Dist'+str(i) for i in range(nPastMinutes+1)]

for i in range(1, nPastMinutes+1):
  df['Dist'+str(i)] = df.Dist0.shift(i,'min')
  df['Dist'+str(i)] = df['Dist'+str(i)] - df.Dist0

df.dropna(inplace = True)

df = df.loc[df['WaitTime'] < 40]

df['Day'] = df.apply(lambda x: 'Weekday' if x['Weekday']<5 else 'Weekend', axis=1)
df['TimeKey'] = pd.to_datetime(df.Time).dt.time
#%%
dfG = pd.read_csv(pwd+'GoogleTypicalTraffic.csv')
dfG.TimeKey = pd.to_datetime(dfG.TimeKey).dt.time
dfG.fillna(method='ffill', inplace=True)


df = df.merge(dfG, how='left', on=['Day','TimeKey'])
df[['Segment1', 'Segment2', 'Segment3']].fillna(method='bfill')

#convert unit of time to hours for regression
df['Time'] = pd.to_timedelta(df['Time'])/np.timedelta64(1,'h')




# remove days with WaitTime longer than 60 mins:
#df = df[df.merge(pd.DataFrame({'selDay': df.groupby('Date').WaitTime.max()<60}), left_on = 'Date', right_index=True).selDay]

trainInd = df.Date.isin(pd.Series(df.Date.unique()).sample(frac=0.7, random_state=3))

feats = ['Weekday', 'Time', 'Dist0', 'noDist0'] + DistNames[1:] #+ ['Segment1', 'Segment2', 'Segment3'] 
train = df.loc[trainInd]
test = df.loc[~trainInd]

#%%

############################# Gradient Boosted Trees #####################################

# XGB needs special data format, so convert to its matrix format
dtrain = xgb.DMatrix(train[feats].values, label=train['WaitTime'], feature_names = feats)
dtest = xgb.DMatrix(test[feats].values, label=test['WaitTime'], feature_names = feats)
# Things to watch during training.
watchlist = [(dtrain, 'train'), (dtest, 'test')]

# Function to calculate absolute error instead of RMSE
def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(y, yhat)

# Parameters for XGBoost
xgb_params = {
        'seed': 0,
        'colsample_bytree': 0.8,
        'silent': 1,
        'subsample': 0.55,
        'learning_rate': 0.01,
#        'gamma': 0.02,
        'objective': 'reg:linear',
        'max_depth': 10,
#        'min_child_weight': 2,
        'lambda': 1,
        'booster': 'gbtree',
    }

# Training.
gbdt = xgb.train(xgb_params, dtrain, 10000, watchlist,
#                             obj=logregobj,
                             feval=xg_eval_mae,
#                             maximize=False,
                             verbose_eval=100,
                             early_stopping_rounds=100)

xgbPred = gbdt.predict(dtest, ntree_limit=gbdt.best_ntree_limit)
stats.describe(np.absolute(test['WaitTime'] - xgbPred))
np.percentile(np.absolute(test['WaitTime'] - xgbPred), [10,25,50,75,90,95,99])




#zz=test.copy()
#zz['xgbPred'] = xgbPred
#zz['err'] = zz.WaitTime - zz.xgbPred
#zz.plot.scatter(x='xgbPred', y='WaitTime')

def plotActualVsPredDensity(modelName, modelPred):
  from matplotlib.colors import LogNorm
  plt.figure()
  plt.hist2d(modelPred, test.WaitTime, (120, 120), cmap=plt.cm.jet, norm=LogNorm())
  plt.colorbar().ax.set_ylabel('Number of Sample Points')
  plt.xlabel('Predicted Wait Time')
  plt.ylabel('Observed Wait Time')
  plt.title('Actual vs. Predicted Bus Wait Time Density Plot for ' + modelName)
  
plotActualVsPredDensity('GBT', xgbPred)

#%%

import tensorflow as tf

numBatch = 14
lstm_size = 20
numSteps = int(df.groupby('Date').count().WaitTime.median())
x = tf.placeholder(tf.float32, [numBatch, numSteps])
y = tf.placeholder(tf.float32, [numBatch, numSteps])
init_state = tf.zeros([numBatch, lstm_size])

lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
rnn_outputs, final_state = tf.nn.dynamic_rnn(lstm, x, initial_state=init_state)


losses = tf.losses.mean_squared_error(labels=y, predictions=rnn_outputs)
total_loss = tf.reduce_mean(losses)
learning_rate=0.1
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  training_losses = []
  for idx, epoch in enumerate()


"""
Placeholders
"""

x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')
init_state = tf.zeros([batch_size, state_size])

"""
Inputs
"""

rnn_inputs = tf.one_hot(x, num_classes)

"""
RNN
"""

cell = tf.contrib.rnn.BasicRNNCell(state_size)
rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

"""
Predictions, loss, training step
"""

with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, num_classes])
    b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
logits = tf.reshape(
            tf.matmul(tf.reshape(rnn_outputs, [-1, state_size]), W) + b,
            [batch_size, num_steps, num_classes])
predictions = tf.nn.softmax(logits)

losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)







# Generating historical features: progress/distance traveld from n minute ago to now.
nPastMinutes=15
DistNames = ['Dist'+str(i) for i in range(nPastMinutes+1)]

for i in range(1, nPastMinutes+1):
  df['Dist'+str(i)] = df.Dist0.shift(i,'min')
  df['Dist'+str(i)] = df['Dist'+str(i)] - df.Dist0

df.dropna(inplace = True)

# Most recent Distn > 0.
# First replace 0 with nan, then back fill nan's so that the first column is the first Distn>0
# then fill na with 0 for those with no movement at all.
df['LastMove'] = (df[DistNames[1:]].clip_lower(0).replace(0, np.nan).bfill(1).iloc[:, 0]).fillna(0)
# Time elapsed with no movement. i.e. the number of 0 or negatives after Dist1....
df['JamTime'] = (df[DistNames[1:]].clip_lower(0) == 0).sum(axis=1)+1
# The most up to date speed info.
df['JamSpeed'] = df['LastMove'] / df['JamTime']

# Data from 11-18 to 12-10 are missing, which produec huge WaitTime. Here first filter out really long times.
df = df.loc[df['WaitTime'] < 60]
df.shape

WaitTimeCap = 40
# Judgementally filter out WaitTime > 40
df = df.loc[df['WaitTime'] < WaitTimeCap]
#df.WaitTime = df.WaitTime.clip_upper(WaitTimeCap)

# Exclude points where there's no GPS data for more than 15 min.
df = df[df.noDist0 <= 0+nPastMinutes]

df = df[ ~ (((df.Dist0>0.005)&(df.Dist0<0.01)&(df.WaitTime>15)) | ((df.Dist0>0.01)&(df.Dist0<0.1)&(df.WaitTime>20)))]
df = df[~ ((df.JamSpeed == 0) & (df.Dist0/df.WaitTime > 0.015) & (df.Dist0>0.25))]

# Plot the distribution of WaitTime
plt.figure()
df.WaitTime.hist(bins=40)

#convert unit of time to hours for regression
df['Time'] = pd.to_timedelta(df['Time'])/np.timedelta64(1,'h')

# Import and merge public holidays
publicHolidays = pd.read_csv(pwd+'publicHolidays.csv', header=0)
#publicHolidays = pd.to_datetime(publicHolidays['Holidays'], format='%Y-%m-%d')
df['isHoliday'] = df.Date.isin(publicHolidays.Holidays)

# List of featuers
#DistNames[:-6]
featsAll = ['Weekday', 'isHoliday', 'Time'] + DistNames + ['noDist0', 'LastArrival', 'JamTime', 'JamSpeed'] + ['TemperatureC', 'WeatherCond', 'WeatherCond30min', 'WeatherCond60min']
target = ['WaitTime']
featsRemoved = DistNames[-6:] + ['JamTime', 'JamSpeed'] + ['TemperatureC', 'WeatherCond', 'WeatherCond30min', 'WeatherCond60min']
feats = [f for f in featsAll if f not in featsRemoved]

# Copy data, and then scale continuous variables for easier training.
dat = df[feats + target].copy()
#dat[lastArrivalTimes] = (-dat[lastArrivalTimes]).clip(upper=60)

train = dat.iloc[0:int(0.7*len(dat))]
test = dat.iloc[int(0.7*len(dat))+1:]

#train = dat.loc[df.Date.isin(pd.Series(df.Date.unique()).sample(frac=0.7, random_state=3))]
#test = dat.loc[~dat.index.isin(train.index)]

############################# Gradient Boosted Trees #####################################

# XGB needs special data format, so convert to its matrix format
dtrain = xgb.DMatrix(train[feats].values, label=train['WaitTime'], feature_names = feats)
dtest = xgb.DMatrix(test[feats].values, label=test['WaitTime'], feature_names = feats)
# Things to watch during training.
watchlist = [(dtrain, 'train'), (dtest, 'test')]

# Function to calculate absolute error instead of RMSE
def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(y, yhat)

# Parameters for XGBoost
xgb_params = {
        'seed': 0,
        'colsample_bytree': 0.8,
        'silent': 1,
        'subsample': 0.55,
        'learning_rate': 0.01,
#        'gamma': 0.02,
        'objective': 'reg:linear',
        'max_depth': 10,
#        'min_child_weight': 2,
        'lambda': 1,
        'booster': 'gbtree',
    }

# Training.
gbdt = xgb.train(xgb_params, dtrain, 10000, watchlist,
#                             obj=logregobj,
                             feval=xg_eval_mae,
#                             maximize=False,
                             verbose_eval=100,
                             early_stopping_rounds=100)

xgbPred = gbdt.predict(dtest, ntree_limit=gbdt.best_ntree_limit)
stats.describe(np.absolute(test['WaitTime'] - xgbPred))
np.percentile(np.absolute(test['WaitTime'] - xgbPred), [10,25,50,75,90,95,99])




#zz=test.copy()
#zz['xgbPred'] = xgbPred
#zz['err'] = zz.WaitTime - zz.xgbPred
#zz.plot.scatter(x='xgbPred', y='WaitTime')

def plotActualVsPredDensity(modelName, modelPred):
  from matplotlib.colors import LogNorm
  plt.figure()
  plt.hist2d(modelPred, test.WaitTime, (120, 120), cmap=plt.cm.jet, norm=LogNorm())
  plt.colorbar().ax.set_ylabel('Number of Sample Points')
  plt.xlabel('Predicted Wait Time')
  plt.ylabel('Observed Wait Time')
  plt.title('Actual vs. Predicted Bus Wait Time Density Plot for ' + modelName)
  
plotActualVsPredDensity('GBT', xgbPred)
#
#xgb.plot_importance(gbdt, importance_type='gain', xlabel='Gain', title='Feature Importance - Gain')
#xgb.plot_importance(gbdt, importance_type='weight', xlabel='Frequency', title='Feature Importance - Frequency')
#xgb.plot_importance(gbdt, importance_type='cover', xlabel='Cover', title='Feature Importance - Cover')


########################### Random Forest #####################################

# 1000 trees, each leaf must have at least 3 rows
rf = RandomForestRegressor(n_estimators=1000, max_depth= 12, max_features=0.9, min_samples_leaf=3, random_state=1, oob_score=True, verbose=1, n_jobs=-1)
# Train.
rf.fit(train[feats], train[target[0]])
# Predictions
rfPred = rf.predict(test[feats])

stats.describe(np.absolute(test['WaitTime'] - rfPred))
np.percentile(np.absolute(test['WaitTime'] - rfPred), [10,25,50,75,90,95,99])
# Plot distribution of errors.

plotActualVsPredDensity('RF', rfPred)

rfImp = pd.DataFrame({'feature':feats,'importance':np.round(rf.feature_importances_,3)})
rfImp = rfImp.sort_values('importance',ascending=True).set_index('feature')
print(rfImp)
rfImp.plot.barh(title='Random Forest Feature Importance', legend=False).set_xlabel('Normalized Importance')

#from sklearn.model_selection import GridSearchCV
#rfTune = RandomForestRegressor(n_estimators=1000, oob_score=True, n_jobs=-1)
#rfParamGrid = {'min_samples_leaf':[2,3,4,5,10]}
#rfCVs = GridSearchCV(rfTune, rfParamGrid, cv=5, verbose=2)
#
#rfCVs.fit(train[feats], train[target[0]])


#################################################################

from sklearn import svm

svr = svm.SVR(verbose=True)

svr.fit(train[feats], train[target[0]])

svrPred = svr.predict(test[feats])

stats.describe(np.absolute(test['WaitTime'] - svrPred))
np.percentile(np.absolute(test['WaitTime'] - svrPred), [10,25,50,75,90,95,99])

plotActualVsPredDensity('SVR', svrPred)

############################## NN ##################################

import tensorflow as tf

catFeats = ['Weekday']#, 'WeatherConditions',
#         'WeatherConditions30minPrior', 'WeatherConditions60minPrior']#, 'WeatherConditions90minPrior', 'WeatherConditions120minPrior']

nFeats = len(feats) + 6
initMax = 5

nnL = [nFeats, 20, 4, 1]

#with tf.device('/gpu:0'):
X = tf.placeholder('float', [None, nFeats])
Y = tf.placeholder('float', [None,1])

W1 = tf.Variable(tf.random_uniform([nnL[0], nnL[1]], minval=-initMax, maxval=initMax))
b1 = tf.Variable(tf.random_uniform([nnL[1]], minval=-initMax, maxval=initMax))

W2 = tf.Variable(tf.random_uniform([nnL[1], nnL[2]], minval=-initMax, maxval=initMax))
b2 = tf.Variable(tf.random_uniform([nnL[2]], minval=-initMax, maxval=initMax))

W3 = tf.Variable(tf.random_uniform([nnL[2], nnL[3]], minval=-initMax, maxval=initMax))
b3 = tf.Variable(tf.random_uniform([nnL[3]], minval=-initMax, maxval=initMax))

Layer1 = tf.nn.sigmoid(tf.add(b1, tf.matmul(X, W1)))
Layer2 = tf.nn.sigmoid(tf.add(b2, tf.matmul(Layer1, W2)))
Y_ = tf.add(b3, tf.matmul(Layer2, W3))
#Y_ = tf.matmul(Layer2, W3)

cost = tf.reduce_mean(tf.pow(Y - Y_, 2))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

accuracy = tf.reduce_mean(tf.abs(Y-Y_))

init = tf.global_variables_initializer()

training_epochs = 2000
display_step_freq = 20
batch_size = 10000
mae_curr_min = 1e10
iStop, nStop = 0, 10

sess = tf.Session()
sess.run(init)

# Training cycle
for epoch in range(training_epochs):
  avg_cost = 0.
  total_batch = int(len(train)/batch_size)
  # Loop over all batches
  for i in range(total_batch):
    train_batch = train.sample(n=batch_size)
    batch_x = pd.get_dummies(train_batch[feats], columns = catFeats)
    batch_y = train_batch[['WaitTime']]
    # Run optimization op (backprop) and cost op (to get loss value)
    _, c = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y})
    # Compute average loss
    avg_cost += c / total_batch
  # Display logs per epoch step
  if (epoch + 1) % display_step_freq == 0:
    mae_train = sess.run(accuracy, feed_dict={X: pd.get_dummies(train[feats], columns = catFeats), Y: train[['WaitTime']]})
    mae_test  = sess.run(accuracy, feed_dict={X: pd.get_dummies(test[feats], columns = catFeats), Y: test[['WaitTime']]})
    print("Epoch:", '%04d' % (epoch+1), "cost =", "{:.4f}".format(avg_cost), "mae_train =", "{:.4f}".format(mae_train), "mae_test =", "{:.4f}".format(mae_test))
    
    logb1, logW1, logb2, logW2, logb3, logW3 = sess.run((b1,W1,b2,W2,b3,W3))
    if (mae_curr_min < mae_test):
      iStop += 1
    else:
      mae_curr_min = mae_test
      iStop = 0
    
    if (iStop > nStop): break
print("Optimization Finished!")

#  # Test model
#  correct_prediction = tf.equal(tf.argmax(Y_, 1), tf.argmax(Y, 1))
#  # Calculate accuracy
#  accuracy = tf.reduce_mean(Y-Y_)
print("Final Accuracy:", sess.run(accuracy, feed_dict={X: pd.get_dummies(test[feats], columns = catFeats), Y: test[['WaitTime']]}))
nnPred = pd.Series(sess.run(Y_, feed_dict={X: pd.get_dummies(test[feats], columns = catFeats), Y: test[['WaitTime']]}).reshape(-1), index=test.index)

stats.describe(np.absolute(test['WaitTime'] - nnPred))
np.percentile(np.absolute(test['WaitTime'] - nnPred), [10,25,50,75,90,95,99])

#W1 = tf.Variable(logW1)
#b1 = tf.Variable(logb1)
#
#W2 = tf.Variable(logW2)
#b2 = tf.Variable(logb2)
#
#W3 = tf.Variable(logW3)
#b3 = tf.Variable(logb3)

baggedPred = (rfPred + xgbPred)/2
np.percentile(np.absolute(test['WaitTime'] - baggedPred), [10,25,50,75,90,95,99])

####################################################

from sklearn.linear_model import LinearRegression
# A linear model as reference/benchmark.

lr = LinearRegression()
lr.fit(train[feats], train[target[0]])
lrPred = lr.predict(test[feats])

stats.describe(np.absolute(test['WaitTime'] - lrPred))
np.percentile(np.absolute(test['WaitTime'] - lrPred), [10,25,50,75,90,95,99])
####################################################

from sklearn.neural_network import MLPRegressor
# All neural networks trained with regularization parameter alpha=0.0001, might need to be higher for Deep NN

mlpBig = MLPRegressor(hidden_layer_sizes=(100,10), activation='logistic', solver='adam', alpha=0.001, tol=0.00001, max_iter=2000, random_state=1, early_stopping=False, verbose=True)
mlpBig.fit(train[feats], train[target[0]])
mlpBigPred = mlpBig.predict(test[feats])

stats.describe(np.absolute(test['WaitTime'] - mlpBigPred))
np.percentile(np.absolute(test['WaitTime'] - mlpBigPred), [10,25,50,75,90,95,99])

mlpSmall = MLPRegressor(hidden_layer_sizes=(20,5), activation='logistic', solver='adam', alpha=0.0001, tol=0.00001, max_iter=2000, random_state=1, early_stopping=False, verbose=True)
mlpSmall.fit(train[feats], train[target[0]])
mlpSmallPred = mlpSmall.predict(test[feats])

stats.describe(np.absolute(test['WaitTime'] - mlpSmallPred))
np.percentile(np.absolute(test['WaitTime'] - mlpSmallPred), [10,25,50,75,90,95,99])

#mlpDeep = MLPRegressor(hidden_layer_sizes=(4,4,4,4,4,4,4), activation='relu', solver='adam', alpha=0.0001, tol=0.0000001, max_iter=2000, random_state=2, early_stopping=False, verbose=True)
mlpDeep = MLPRegressor(hidden_layer_sizes=(20,20,10,10,10,5), activation='relu', solver='adam', alpha=0.001, tol=0.000001, max_iter=2000, random_state=2, early_stopping=False, verbose=True)
mlpDeep.fit(train[feats], train[target[0]])
mlpDeepPred = mlpDeep.predict(test[feats])

stats.describe(np.absolute(test['WaitTime'] - mlpDeepPred))
np.percentile(np.absolute(test['WaitTime'] - mlpDeepPred), [10,25,50,75,90,95,99])


plotActualVsPredDensity('Small NN', mlpSmallPred)
plotActualVsPredDensity('Wide NN', mlpBigPred)
plotActualVsPredDensity('Deep NN', mlpDeepPred)
#####################################################

# Function to export several models' results in a table
def summarizeModels(dictTestPred, testActual):
  modelSummary = pd.DataFrame(index=range(len(dictTestPred)), columns=['Model', 'RMSE', 'MAE', '25thPerc', '50thPerc', '75thPerc', '90thPerc', '95thPerc', '99thPerc'])
  
  i=0
  for k,pred in dictTestPred.items():
    modelSummary.ix[i,'Model'] = k
    err=testActual['WaitTime'] - pred
    modelSummary.ix[i,'RMSE']=np.round(np.sqrt((err*err).mean()),4)
    modelSummary.ix[i,'MAE'] = np.round(np.abs(err).mean(),4)
    modelSummary.ix[i,['25thPerc', '50thPerc', '75thPerc', '90thPerc', '95thPerc', '99thPerc']] = np.round(np.percentile(np.abs(err), [25,50,75,90,95,99]),4)
  #  modelSummary.append([k, rmse, mae]+percentiles)
    i+=1
  return(modelSummary)

# Export results in a table
modelsPreds = {}
modelsPreds['GBT'] = xgbPred
modelsPreds['RF'] = rfPred
modelsPreds['LR'] = lrPred
modelsPreds['Wide NN'] = mlpBigPred
modelsPreds['Small NN'] = mlpSmallPred
modelsPreds['Deep NN'] = mlpDeepPred

summModels = summarizeModels(modelsPreds,test)
print(summModels)

# A summary of performance tested on TRAFFIC JAM ONLY:
# This helps to see if the model performs well during traffic congestion.
modelsPredsJam = {k : v[test.Dist3<0.005] for (k,v) in modelsPreds.items()}
summModelsJam = summarizeModels(modelsPredsJam,test[test.Dist3<0.005])
print(summModelsJam.sort_values('MAE'))

# Function to plot Area Under Curve (AUC) plot, or Gain curve.
# This is used to measure the "sorting power" of models, i.e. if the model can differenciate a long wait time from a short wait time.
# The larger the area between the curve and the diagonal line, the better.
def plotGains(dictPreds, actual):
  plt.figure()
#  ax = fig.add_subplot()
  for k, pred in dictPreds.items():
    # first sort by predicted values
    dfGain = pd.DataFrame({'actual': actual, k: pred}).sort_values(k).reset_index()
    # then cumulatively sum the ACTUAL values, normalize to 100% by dividing by sum.
    dfGain[k+' Gain'] = dfGain.actual.cumsum()/dfGain.actual.sum()
    dfGain['PercentageOfSample'] = 100* (1+dfGain.index)/len(dfGain)
    plt.plot(dfGain['PercentageOfSample'], dfGain[k+' Gain'], label=k)
#    dfGain.plot.line(x='PercentageOfSample', y=k)
  
  dfGain = pd.DataFrame({'actual': actual}).sort_values('actual').reset_index()
  dfGain['Max Gain'] = dfGain.actual.cumsum()/dfGain.actual.sum()
  dfGain['PercentageOfSample'] = 100* (1+dfGain.index)/len(dfGain)
  plt.plot(dfGain['PercentageOfSample'], dfGain['Max Gain'], label='Best Possible Model', color='g', linestyle=':')
  
  plt.xlabel('% of Sample')
  plt.ylabel('Cumulative Actual Gain')
  plt.title('Gains Curves for Models Fitted')
  plt.plot([0,100], [0,1],linestyle='-.', color='k', label='Average Value Model')
  plt.legend(loc=2)
  
  plt.show()
    
plotGains(modelsPreds, test.WaitTime)
