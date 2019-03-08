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

pwd = 'C:/Users/Z/Dropbox/Homework/bus/new/'
#pwd = '/Users/Amber/Dropbox/[Homework]/bus/new/'
dfAll = pd.read_csv(pwd+'RapidData_Arrival_201705.csv')
dfAll['DateTime'] = pd.to_datetime(dfAll['DateTime'])

#dfStops=dfAll.groupby(by=['SourceBusStop']).mean()

# Removing duplicate rows by grouping consecutive rows with exactly the same GPS coordinates, i.e. Latitude and Longitude
df1 = dfAll.groupby(by=['DestinationBusStop', 'DateTime', 'Latitude', 'Longitude']).first().reset_index()

# There are still duplicate rows with GPS coordinates very close together, i.e. not really moving. Remove thoese by bus stop name.
df2 = df1.groupby(by=['DestinationBusStop', 'DateTime', 'SourceBusStop']).first().reset_index()

# Raw data is not sorted, so sort by time here. Also remove rows before 8-26 for incomplete days.
#df2 = df2.ix[df2.DateTime>datetime(2016,8,26)].sort_values('DateTime')
df2 = df2.ix[df2.DateTime>datetime(2017,2,26)].sort_values('DateTime')
#df.to_csv('C:\\Users\\Z\\Dropbox\\Homework\\bus\\new\\RapidData_AllClean.csv')

# Separate 2 directions into 2 data frames.
dfA = df2.ix[df2.DestinationBusStop == 'Hub Gertak Sanggul'].copy()
dfB = df2.ix[df2.DestinationBusStop == 'Depo Sg. Nibong'].copy()

# Load file containing bus route coordinates.
dfRouteS = pd.read_csv(pwd+'BusStopCoordinates-SouthBound.txt')
dfRouteS = dfRouteS[['Latitude', 'Longitude']]
dfRouteS['Progress'] = dfRouteS.index/len(dfRouteS)

#naRow = pd.DataFrame({'Latitude':np.nan, 'Longitude':np.nan, 'Progress':np.nan})

# Function calculating the progress completed by current bus coordinates, given the route map, and buse's lat and long.
def routeCompletion(routeMap, lat, lon):
  dist = (routeMap.Latitude-lat).abs() + (routeMap.Longitude-lon).abs()
  return routeMap.Progress[dist.idxmin()]


# Select Test Stop: Queensbay Mall, but use GPS coordinates instead since bus stop name not reliable.
#selectedStops = ['Queensbay Mall']
#selectedStopLoc = {'Latitude':5.33234, 'Longitude':100.3075}  # Queensbay Mall's coordinates
selectedStopLoc = {'Latitude':5.32564, 'Longitude':100.2876}  # Sunshine Square's coordinates



# Route completion at Sunshine Square
selectedStopCompletion = routeCompletion(dfRouteS, selectedStopLoc['Latitude'], selectedStopLoc['Longitude'])

## Calculate route completion for all rows southbound and write data to file.
#dfA['RouteCompleted'] = np.nan
#
#for i,row in dfA.iterrows():
#  if(i%10000 == 0): print(i)
#  dfA.ix[i,'RouteCompleted'] = routeCompletion(dfRouteS, row.Latitude, row.Longitude)
#dfA.to_csv('C:/Users/Z/Dropbox/Homework/bus/new/SorthboundAllStops2-201705.csv', index=False)

# Read the produced file from commented sectoin above with route progress.
dfA = pd.read_csv(pwd+'SorthboundAllStops2-201705.csv')
dfA['DateTime'] = pd.to_datetime(dfA['DateTime'])
# Select only data before selected stop 
dfA = dfA[dfA.RouteCompleted <= selectedStopCompletion]
dfA['Dist0'] = selectedStopCompletion - dfA.RouteCompleted
dfDist = dfA.Dist0.groupby(dfA.DateTime).min()

#dfA.set_index(dfA.DateTime, inplace=True)

## Use GPS coordinates to filter all rows that's approaching 
##dfTrueArrival = dfA.ix[((selectedStopLoc['Latitude'] - dfA.Latitude).abs() + (selectedStopLoc['Longitude'] - dfA.Longitude).abs() < 0.005) & (selectedStopLoc['Latitude'] < dfA.Latitude)]
#dfTrueArrival = dfA.ix[(selectedStopCompletion>=dfA.RouteCompleted) & (selectedStopCompletion<dfA.RouteCompleted+0.005)]
#len(dfTrueArrival)
## Remove entries that are too close in time, i.e. duplicates when traffic is slow
#dfTrueArrival = dfTrueArrival.ix[dfTrueArrival.shift(-1).DateTime - dfTrueArrival.DateTime > timedelta(minutes=5)]
#len(dfTrueArrival)


startDate = date(2017,2,26)
endDate = date(2017,7,4)
startTime = time(hour=6, minute=0)
endTime = time(hour=23, minute=0)

# Helper function giving a range of dates
def daterange(start_date, end_date):
  for i in range(1 + int ((end_date - start_date).days)):
    yield start_date + timedelta(i)
    # 'yield' is like 'return', but it returns a "generator" that's like an iterator but can only be used once.

# Make a timeindex from start date to end date.
timeIndex = pd.DatetimeIndex([])
for di in daterange(startDate, endDate-timedelta(days=1)):
  timeIndex = timeIndex.append(pd.date_range(datetime.combine(di,startTime), datetime.combine(di,endTime), freq='min'))     


# Make a new dataframe with above time index, and columns of features to be extracted

#df = pd.DataFrame(index=timeIndex, columns = ['Date','Weekday','Time']+DistNames+['WaitTime']).
df = pd.DataFrame(index=timeIndex, columns = ['Date','Weekday','Time']) #, 'Dist0']) #, 'WaitTime'])
dt = pd.Series(df.index,index=timeIndex).dt
df.Date = dt.date
df.Weekday = dt.weekday.astype('category')
df.Time = dt.time

df = df.merge(pd.DataFrame(dfDist), how='left', right_index=True, left_index=True)


# feature for duration of no current gps data.
df['noDist0'] = pd.isnull(df.Dist0)
df['noDist0Block'] = (df['noDist0'] != df['noDist0'].shift(1)).astype(int).cumsum()
df['noDist0'] = df.groupby('noDist0Block').cumsum().noDist0

# fill missing current data with most recent values.
df.ix[df.Time==time(6,0,0), 'Dist0'] = selectedStopCompletion
ffillTemp = df.Dist0.fillna(method = 'ffill')
bfillTemp = df.Dist0.fillna(method = 'bfill')
#ffillTemp[(bfillTemp - ffillTemp) > 0.01] = bfillTemp[(bfillTemp - ffillTemp) > 0.01]
ffillTemp[(bfillTemp - ffillTemp) > 0.01] = selectedStopCompletion

plt.figure()
plt1=df.loc[datetime(2017,3,8,6,0,0):datetime(2017,3,8,23,0,0)].Dist0.plot(title='Dist0 on 2017-01-18 before Filling Missing Values')
plt1.set_xlabel('Time')
plt1.set_ylabel('Dist0')

df.Dist0 = ffillTemp
plt.figure()
plt2=df.loc[datetime(2017,3,8,6,0,0):datetime(2017,3,8,23,0,0)].Dist0.plot(title='Dist0 on 2017-01-18 after Filling Missing Values')
plt2.set_xlabel('Time')
plt2.set_ylabel('Dist0')

plt.figure()
df[df.noDist0<=60].noDist0.plot.hist(bins=60, title='Distribution of Time of Missing Dist0 Capped at 1 Hour').set_xlabel('noDist0')

# measuring true arrival
df['DistNext']=df.Dist0.shift(-1,'min')
df['Arriving'] = (df.Dist0<0.005) & (df.DistNext > df.Dist0 + 0.005)

# Label each bus according to arrival
df['BusID'] = ((df.Arriving.shift(1)) | (df.Date!=df.Date.shift(1))).astype(int).cumsum()
revBusID = df.BusID.reindex(index=df.BusID.index[::-1])
df['WaitTime'] = revBusID.groupby(revBusID).cumcount().iloc[::-1]


# Time elapsed since last arrival
#df['BusID'] = (df.WaitTime.shift(1,'min')-df.WaitTime < -1).astype(int).cumsum()
df['LastArrival'] = df.groupby('BusID').cumcount()


# Inport weather data
dfWeather = pd.read_csv(pwd+'weather/CombinedWeather_20160825-20170213.csv')
dfWeather['DateTime'] = pd.to_datetime(dfWeather['DateTime'])
dfWeather['WeatherConditions30minPrior'] = dfWeather['WeatherConditions'].shift(1)
dfWeather['WeatherConditions60minPrior'] = dfWeather['WeatherConditions'].shift(2)
#dfWeather['WeatherConditions90minPrior'] = dfWeather['WeatherConditions'].shift(3)
#dfWeather['WeatherConditions120minPrior'] = dfWeather['WeatherConditions'].shift(4)

# Helper function to change a single column name of dataframe
def setColname(d, colIndex, newname):
  colnames = d.columns.tolist()
  colnames[colIndex] = newname
  d.columns = colnames
  return d

# Merge weather data.
df = setColname(df.reset_index(), 0, 'DateTime')
df = pd.merge_asof(df, dfWeather, on='DateTime')

# Write to file, this is the data used for model training.
#df.to_csv(pwd + 'data9.csv', index = False)
