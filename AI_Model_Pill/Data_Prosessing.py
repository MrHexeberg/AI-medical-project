#what do we need to do to the data
#what can be smart to do
# 1 . normalice tablet hardness and hight
# 2 clean up the pressforce
# 3 do tomting about the partical size data
from xml.sax.handler import feature_namespace_prefixes, feature_namespaces

import pandas as pd

from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

#getting the data and setting up what is featuresr and targetrs
FilePath = 'https://raw.githubusercontent.com/MrHexeberg/AI-medical-project/refs/heads/main/Tablet%20examination%20-%20Munka1.csv'
FeatureColumns =['Press Force (kg)','Motor Speed (tablets/min)','Particle Size (μm)']
TargetColumns = ['Tablet Hardness (N)','Tablet Height (mm)','Tablet Friability (%)']

Input_data = pd.read_csv(FilePath, header=1, skiprows=0)
Input_data = Input_data.drop(columns="Measurement")

print(Input_data)

print("we are doing stuff with the data...")
#spliting the data in to target and feature
data_out_feature = Input_data.drop(columns= TargetColumns )
data_out_target = Input_data.drop(columns = FeatureColumns)
#minmax scaling the feature data
scaler = preprocessing.MinMaxScaler()
data_out_feature = scaler.fit_transform(data_out_feature)
data_out_feature = pd.DataFrame(data_out_feature, columns = FeatureColumns)
#bit of printing to see if anyting is wronge
print(data_out_feature)
print(data_out_target)

#testing out a model on the data and the scaling. fist spliting it up
data_out_feature_train, data_out_feature_test, data_out_target_train, data_out_target_test = train_test_split(data_out_feature, data_out_target, test_size=0.2, random_state=42)
#valling the model and puting in the data
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(data_out_feature_train, data_out_target_train)

test_pred = model.predict(data_out_feature_test)

#evaluate the model using the MEA medod
mae_target1 = mean_absolute_error(data_out_target_test.iloc[:, 0], test_pred[:, 0])
mae_target2 = mean_absolute_error(data_out_target_test.iloc[:, 1], test_pred[:, 1])
mae_target3 = mean_absolute_error(data_out_target_test.iloc[:, 2], test_pred[:, 2])

print("MAE Tablet Hardness (N):", mae_target1,"N")
print("MAE Tablet Height (mm):", mae_target2,"mm")
print("MAE Tablet Friability (%):", mae_target3*100 , " %")

# pringing the data set out to a csv file for later user
Output_Data = pd.concat([data_out_feature,data_out_target], axis = 1)
Output_Data.to_csv("MinMaxScaler.csv", index=False)




# Plotting Tablet Hardness (N)
sample = 'Tablet Hardness (N)'
test_sample = data_out_target_test[sample]
plt.figure(figsize=(10,6))
plt.scatter(test_sample,test_pred[:, 0])
plt.plot([test_sample.min(), test_sample.max()],
         [test_sample.min(), test_sample.max()], 'r--', lw=2)

# Extension of the line ideal line
plt.plot([test_sample.min()-2, test_sample.max()+2],
         [test_sample.min()-2, test_sample.max()+2], 'r--', lw=2)

plt.xlabel(f'Actual {sample}')
plt.ylabel(f'Predicted {sample}')
plt.title(f'Actual vs Predicted {sample}')


# Making custom grid for the plot
xy = np.arange(52, 102, 2)
plt.xticks(xy)
plt.yticks(xy)
plt.grid()

plt.show()

# Tablet height
sample = 'Tablet Height (mm)'
test_sample =data_out_target_test[sample]
plt.figure(figsize=(10,6))
plt.scatter(test_sample, test_pred[:, 1])
plt.plot([test_sample.min(), test_sample.max()],
         [test_sample.min(), test_sample.max()], 'r--', lw=2, label='y_test minmax')
plt.xlabel(f'Actual {sample}')
plt.ylabel(f'Predicted {sample}')
plt.title(f'Actual vs Predicted {sample}')
plt.legend()

# Making custom grid for the plot
xy = np.arange(4.4, 6.2, 0.1)
plt.xticks(xy)
plt.yticks(xy)
plt.grid()
plt.show()

# Plotting Tablet Friability (%)
sample = 'Tablet Friability (%)'
test_sample = data_out_target_test[sample]
plt.scatter(test_sample, test_pred[:, 2])
plt.plot([test_sample.min(), test_sample.max()],
         [test_sample.min(), test_sample.max()], 'r--', lw=2)
plt.xlabel(f'Actual {sample}')
plt.ylabel(f'Predicted {sample}')
plt.title(f'Actual vs Predicted {sample}')

# Making custom grid for the plot
xy = np.arange(0.1, 0.5, 0.05)
plt.xticks(xy)
plt.yticks(xy)
plt.grid()
plt.show()