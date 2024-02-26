# Project 2: 30 Days of Data Sets Training

## Introduction
Welcome to my second machine learning project! In this project, I have worked on training a machine learning model using a dataset and implemented a simple slider using Jupyter widgets for data exploration.

## Dataset
I used the Delaney solubility dataset with descriptors, which is available [here](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv). The dataset contains various molecular descriptors along with the solubility values.

```python
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv")

![11](https://github.com/Gill-tech/machine-learning/assets/101551262/820676f9-8629-46d5-91d9-4de799c0c07d)

```

## Data Preparation
I started by preparing the data for training the model. Here are the initial steps:

```python
# Extracting the target variable
y = df["logS"]

# Extracting features
x = df.drop('logS', axis=1)
```

## Data Splitting
Next, I split the data into training and testing sets using the `train_test_split` function from scikit-learn:
![10](https://github.com/Gill-tech/machine-learning/assets/101551262/e93b83e1-ab1d-45dd-bb5e-2c5aceee715a)


```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
```

## Model Building
For this project, I chose to use a simple linear regression model. Here's how I built and trained the model:

```python
from sklearn.linear_model import LinearRegression

# Initializing the linear regression model
lr = LinearRegression()

# Training the model
lr.fit(x_train, y_train)
```

## Applying the Model
I applied the trained model to make predictions on both the training and testing sets:

```python
# Making predictions on the training and testing sets
y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)
```
![12](https://github.com/Gill-tech/machine-learning/assets/101551262/0e69b794-3bf2-4435-bb2b-4725f32a379d)


## Conclusion
This project is a part of my 30 days of data sets training, where I am exploring different datasets and building machine learning models. The next steps involve further analysis, model evaluation, and potentially improving the model's performance.

Feel free to explore the Jupyter notebook for more details and insights. If you have any suggestions or questions, please let me know!
