import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


column_names = ['complexAge', 'totalRooms', 'totalBedrooms', 'complexInhabitants', 'apartmentsNr', 'medianCompexValue']
feature_cols = ['complexAge', 'totalRooms', 'totalBedrooms', 'complexInhabitants', 'apartmentsNr']
data = pd.read_csv('https://raw.githubusercontent.com/kiseli98/ai_lab3/main/apartmentComplexData%20-%20Copy.csv', names=column_names)


train_dataset = data.sample(frac=0.9, random_state=0)
test_dataset = data.drop(train_dataset.index)

# train_dataset = data.sample(n = 100)
# test_dataset = data.drop(train_dataset.index)
# test_dataset = test_dataset.sample(n = 20)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

# Separate the target value (label) from the features.
# Model will predict label
train_labels = train_features.pop('medianCompexValue')
test_labels = test_features.pop('medianCompexValue')

lr_model = linear_model.LinearRegression(normalize=True)

# Train the model using the training sets
lr_model.fit(train_features, train_labels)

# Make predictions using the testing set
prediction = lr_model.predict(test_features)

print('Coefficients: \n', lr_model.coef_)
print('Mean squared error: %.2f'
      % mean_squared_error(test_labels, prediction))
print('Coefficient of determination: %.2f'
      % r2_score(test_labels, prediction))


a = plt.axes(aspect='equal')
plt.scatter(test_labels, prediction)
plt.xlabel('True Values [medianCompexValue]')
plt.ylabel('Predictions [medianCompexValue]')
lims = [0, 500000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


#predict for custom variables
user_input = [[21,916,194,451,178]]   # should be 63300 - original value
prediction =  lr_model.predict(user_input).flatten()
print('Predicted value: ',prediction)

dirname = os.path.dirname(__file__)
model_name = os.path.join(dirname, 'lr_model.pkl')

# Saving model to disk
pickle.dump(lr_model, open(model_name,'wb'))