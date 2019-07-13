#Python program to predict the finish placement of the player in PUBG(Game) with machine learning.
#Author - Hitesh Thakur

#Importing required libraries 
import pandas as pd;
import numpy as np;
import seaborn as sns;
import matplotlib.pyplot as plt;
from copy import deepcopy;
import pickle;
from sklearn.model_selection import train_test_split;
from sklearn.preprocessing import StandardScaler;
from sklearn.metrics import accuracy_score,mean_absolute_error;
from sklearn.ensemble import RandomForestRegressor;

#Reading data from .csv file using pandas
pubg_data=pd.read_csv('train_V2.csv');

#Dropping features that are irrelevant with the finish placement [Target values]
pubg_data.drop(columns=['Id','groupId', 'matchId'], axis='columns', inplace=True);


#KNOWING THE DATA FOR PREPROCESSING

#For feature : assists
print("Assits:\n");
print("Max and Min values:",pubg_data['assists'].max(),pubg_data['assists'].min())
print("Total null values:",pubg_data['assists'].isnull().sum());
plt.hist(pubg_data['assists'],bins=22,facecolor='blue');
plt.show();
print("Outliers:",pubg_data['assists'][pubg_data['assists']>5].count()); #Outliers

##For feature : boosts
print("Boosts:\n");
print("Max and Min values:",pubg_data['boosts'].max(),pubg_data['boosts'].min())
print("Total null values:",pubg_data['boosts'].isnull().sum());
plt.hist(pubg_data['boosts'],bins=33,facecolor='blue');
plt.show();
print("Outliers:",pubg_data['boosts'][pubg_data['boosts']>10].count()); #Outliers

##For feature : damageDealt
print("Damage Dealt:\n");
print("Max and Min values:",pubg_data['damageDealt'].max(),pubg_data['damageDealt'].min())
print("Total null values:",pubg_data['damageDealt'].isnull().sum());
plt.hist(pubg_data['damageDealt'],bins=50,facecolor='blue');
plt.show();
print("Outliers:",pubg_data['damageDealt'][pubg_data['damageDealt']>1500].count()); #Outliers

##For feature : DBNOs
print("Knouckouts:\n");
print("Max and Min values:",pubg_data['DBNOs'].max(),pubg_data['DBNOs'].min())
print("Total null values:",pubg_data['DBNOs'].isnull().sum());
plt.hist(pubg_data['DBNOs'],bins=53,facecolor='blue');
plt.show();
print("Outliers:",pubg_data['DBNOs'][pubg_data['DBNOs']>10].count()); #Outliers

##For feature : headshotKills
print("Headshot Kills:\n");
print("Max and Min values:",pubg_data['headshotKills'].max(),pubg_data['headshotKills'].min())
print("Total null values:",pubg_data['headshotKills'].isnull().sum());
plt.hist(pubg_data['headshotKills'],bins=64,facecolor='blue');
plt.show();
print("Outliers:",pubg_data['headshotKills'][pubg_data['headshotKills']>6].count()); #Outliers

##For feature : heals
print("Heals:\n");
print("Max and Min values:",pubg_data['heals'].max(),pubg_data['heals'].min())
print("Total null values:",pubg_data['heals'].isnull().sum());
plt.hist(pubg_data['heals'],bins=80,facecolor='blue');
plt.show();
print("Outliers:",pubg_data['heals'][pubg_data['heals']>22].count()); #Outliers

##For feature : 'killPlace'
print("Kill Place:\n");
print("Max and Min values:",pubg_data['killPlace'].max(),pubg_data['killPlace'].min())
print("Total null values:",pubg_data['killPlace'].isnull().sum());
plt.hist(pubg_data['killPlace'],bins=10,facecolor='blue');
plt.show();
print("Outliers:",pubg_data['killPlace'][pubg_data['killPlace']>101].count()); #Outliers-No need

##For feature : 'killPoints'
print("Kill Points:\n");
print("Max and Min values:",pubg_data['killPoints'].max(),pubg_data['killPoints'].min())
print("Total null values:",pubg_data['killPoints'].isnull().sum());
plt.hist(pubg_data['killPoints'],bins=20,facecolor='blue');
plt.show();
print("Outliers:",pubg_data['killPoints'][pubg_data['killPoints']>2170].count()); #Outliers- No need

##For feature : 'kills'
print("Kills:\n");
print("Max and Min values:",pubg_data['kills'].max(),pubg_data['kills'].min())
print("Total null values:",pubg_data['kills'].isnull().sum());
plt.hist(pubg_data['kills'],bins=72,facecolor='blue');
plt.show();
print("Outliers:",pubg_data['kills'][pubg_data['kills']>12].count()); #Outliers

##For feature : 'killStreaks'
print("Kill Streaks:\n");
print("Max and Min values:",pubg_data['killStreaks'].max(),pubg_data['killStreaks'].min())
print("Total null values:",pubg_data['killStreaks'].isnull().sum());
plt.hist(pubg_data['killStreaks'],bins=20,facecolor='blue');
plt.show();
print("Outliers:",pubg_data['killStreaks'][pubg_data['killStreaks']>5].count()); #Outliers

##For feature : 'longestKill'
print("Longest Kills:\n");
print("Max and Min values:",pubg_data['longestKill'].max(),pubg_data['longestKill'].min())
print("Total null values:",pubg_data['longestKill'].isnull().sum());
plt.hist(pubg_data['longestKill'],bins=10,facecolor='blue');
plt.show();
print("Outliers:",pubg_data['longestKill'][pubg_data['longestKill']>450].count()); #Outliers

##For feature : 'matchDuration'
print("Match Duration:\n");
print("Max and Min values:",pubg_data['matchDuration'].max(),pubg_data['matchDuration'].min())
print("Total null values:",pubg_data['matchDuration'].isnull().sum());
plt.hist(pubg_data['matchDuration'],bins=100,facecolor='blue');
plt.show();
print("Outliers:",pubg_data['matchDuration'][pubg_data['matchDuration']<800].count()); #Outliers

##For feature : 'maxPlace'
print("Max Place:\n");
print("Max and Min values:",pubg_data['maxPlace'].max(),pubg_data['maxPlace'].min())
print("Total null values:",pubg_data['maxPlace'].isnull().sum());
plt.hist(pubg_data['maxPlace'],bins=100,facecolor='blue');
plt.show();
print("Outliers:",pubg_data['maxPlace'][pubg_data['maxPlace']<5].count()); #Outliers - No need

##For feature : 'numGroups'
print("NumGroups:\n");
print("Max and Min values:",pubg_data['numGroups'].max(),pubg_data['numGroups'].min())
print("Total null values:",pubg_data['numGroups'].isnull().sum());
plt.hist(pubg_data['numGroups'],bins=10,facecolor='blue');
plt.show();
print("Outliers:",pubg_data['numGroups'][pubg_data['numGroups']<5].count()); #Outliers

##For feature : 'rankPoints'
print("Rank Points:\n");
print("Max and Min values:",pubg_data['rankPoints'].max(),pubg_data['rankPoints'].min())
print("Total null values:",pubg_data['rankPoints'].isnull().sum());
plt.hist(pubg_data['rankPoints'],bins=50,facecolor='blue');
plt.show();

##For feature : 'revives'
print("Revives:\n");
print("Max and Min values:",pubg_data['revives'].max(),pubg_data['revives'].min())
print("Total null values:",pubg_data['revives'].isnull().sum());
plt.hist(pubg_data['revives'],bins=39,facecolor='blue');
plt.show();
print("Outliers:",pubg_data['revives'][pubg_data['revives']>4].count()); #Outliers

##For feature : 'rideDistance'
print("Ride Distance:\n");
print("Max and Min values:",pubg_data['rideDistance'].max(),pubg_data['rideDistance'].min())
print("Total null values:",pubg_data['rideDistance'].isnull().sum());
plt.hist(pubg_data['rideDistance'],bins=50,facecolor='blue');
plt.show();
print("Outliers:",pubg_data['rideDistance'][pubg_data['rideDistance']>12000].count()); #Outliers

##For feature : 'roadKills'
print("Road Kills:\n");
print("Max and Min values:",pubg_data['roadKills'].max(),pubg_data['roadKills'].min())
print("Total null values:",pubg_data['roadKills'].isnull().sum());
plt.hist(pubg_data['roadKills'],bins=18,facecolor='blue');
plt.show();
print("Outliers:",pubg_data['roadKills'][pubg_data['roadKills']>2].count()); #Outliers

##For feature : 'swimDistance'
print("Swim Distance:\n");
print("Max and Min values:",pubg_data['swimDistance'].max(),pubg_data['swimDistance'].min())
print("Total null values:",pubg_data['swimDistance'].isnull().sum());
plt.hist(pubg_data['swimDistance'],bins=10,facecolor='blue');
plt.show();
print("Outliers:",pubg_data['swimDistance'][pubg_data['swimDistance']>500].count()); #Outliers

##For feature : 'teamKills'
print("Team Kills:\n");
print("Max and Min values:",pubg_data['teamKills'].max(),pubg_data['teamKills'].min())
print("Total null values:",pubg_data['teamKills'].isnull().sum());
plt.hist(pubg_data['teamKills'],bins=12,facecolor='blue');
plt.show();
print("Outliers:",pubg_data['teamKills'][pubg_data['teamKills']>2].count()); #Outliers

##For feature : 'vehicleDestroys'
print("Vehicle Destroyed:\n");
print("Max and Min values:",pubg_data['vehicleDestroys'].max(),pubg_data['vehicleDestroys'].min())
print("Total null values:",pubg_data['vehicleDestroys'].isnull().sum());
plt.hist(pubg_data['vehicleDestroys'],bins=5,facecolor='blue');
plt.show();
print("Outliers:",pubg_data['vehicleDestroys'][pubg_data['vehicleDestroys']>2].count()); #Outliers

##For feature : 'walkDistance'
print("Walk Distance:\n");
print("Max and Min values:",pubg_data['walkDistance'].max(),pubg_data['walkDistance'].min())
print("Total null values:",pubg_data['walkDistance'].isnull().sum());
plt.hist(pubg_data['walkDistance'],bins=20,facecolor='blue');
plt.show();
print("Outliers:",pubg_data['walkDistance'][pubg_data['walkDistance']>7000].count()); #Outliers

##For feature : 'weaponsAcquired'
print("Weapons Acquired:\n");
print("Max and Min values:",pubg_data['weaponsAcquired'].max(),pubg_data['weaponsAcquired'].min())
print("Total null values:",pubg_data['weaponsAcquired'].isnull().sum());
plt.hist(pubg_data['weaponsAcquired'],bins=10,facecolor='blue');
plt.show();
print("Outliers:",pubg_data['weaponsAcquired'][pubg_data['weaponsAcquired']>20].count()); #Outliers

#Performing preprocessing on the data
pubg_data_filtered=deepcopy(pubg_data[(pubg_data['assists']<5) & (pubg_data['boosts']<10) & (pubg_data['damageDealt']<1500) & (pubg_data['DBNOs']<10) & (pubg_data['headshotKills']<6) & (pubg_data['heals']<22)
       & (pubg_data['kills']<12) & (pubg_data['killStreaks']<4) & (pubg_data['longestKill']<450)
       & (pubg_data['matchDuration']>750) & (pubg_data['revives']<4) & (pubg_data['rideDistance']<12000)
       & (pubg_data['roadKills']<2) & (pubg_data['swimDistance']<500) & (pubg_data['teamKills']<2)
       & (pubg_data['vehicleDestroys']<2) & (pubg_data['walkDistance']<7000) & (pubg_data['weaponsAcquired']<20)]);

#Resetting the index
pubg_data_filtered.reset_index(inplace=True);


##For feature 'rankPoints' and 'winPoints': COMBINING BOTH FEATURES TOGETHER
##NOTE:- TIME TAKING PROCESS
##ALREADY PERFORMED IT AND SAVED TO "save_model_pickle.sav" USING PICKLE. IF INTERESTED COMMENT LINES FOLLOWED BY '####' AND UNCOMMENT LINES FOLLOWED BY '##'
for i in range(len(pubg_data_filtered['rankPoints'])):                                            ####
    if pubg_data_filtered['rankPoints'][i]==-1 or pubg_data_filtered['rankPoints'][i]==0:         ####
        pubg_data_filtered['rankPoints'][i]=pubg_data_filtered['winPoints'][i];                   ####
    print(i);                                                                                     ####
print(pubg_data_filtered['rankPoints']);                                                          ####

# FH_pk=open("save_model_pickle.sav",'wb');                                                       ##
# pickle.dump(pubg_data_filtered,FH_pk);                                                          ##
# pickle_file=open("save_model_pickle.sav",'rb');                                                 ##
# pubg_data_filtered=pickle.load(pickle_file);                                                    ##

pubg_data_filterd.drop(columns=['winPoints'],inplace=True)                                        

#Plotting correlation heatmap
correlation=pubg_data_filtered.corr()                                                             
plt.figure(figsize=(16,12));
sns.heatmap(correlation,annot=True,cmap=plt.cm.Reds);
plt.show();
correlation_winPlace=correlation['winPlacePerc'][abs(correlation['winPlacePerc'])>=0.5];
print("Correlation greater than 0.5:",correlation_winPlace);

#Converting non numeric features into numeric using get_dummies
Final_PUBG=pd.DataFrame();
dummy_variable=pd.get_dummies(pubg_data_filtered['matchType'],sparse=True);
dummy_variable.drop(dummy_variable.columns[0],axis='columns',inplace=True);

Final_PUBG = pd.concat([pubg_data_filtered['walkDistance'],pubg_data_filtered['killPlace'], dummy_variable], axis='columns');
target=pubg_data_filtered['winPlacePerc'].round(2);

#splitting data into two sets : Training and Testing
X_train,X_test,Y_train,Y_test=train_test_split(Final_PUBG,target,test_size=0.33,random_state=0);
STD=StandardScaler();
X_train=STD.fit_transform(X_train);
X_test=STD.transform(X_test);

#Trainning Model
bp = {'criterion': 'mse', 'max_depth': 10, 'min_samples_leaf': 3, 'min_samples_split': 3, 'n_estimators': 60};
forest = RandomForestRegressor(criterion=bp['criterion'],
                              min_samples_leaf=bp['min_samples_leaf'],
                              min_samples_split=bp['min_samples_split'],
                              max_depth=bp['max_depth'],
                              n_estimators=bp['n_estimators'],verbose=3,n_jobs=2);
forest.fit(X_train, Y_train);
Y_pred=forest.predict(X_test)

# Explained variance score: 1 is perfect prediction
print('Score: %.2f' % forest.score(X_test, Y_test));
print(mean_absolute_error(Y_test,Y_pred));
