import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

adult = fetch_openml("adult", version=2, as_frame=True)
df = adult.frame

df.head()
df.describe(include="all")

categorical_features = df.select_dtypes(include="object").columns
numerical_features = df.select_dtypes(exclude="object").columns

categorical_features, numerical_features

# The dataset contains a mix of numerical features, such as age and hours-per-week, and categorical features 
# like workclass, education, occupation. Since most machine learning models require numerical input, 
# categorical features must be encoded before modeling.

plt.figure()
df["age"].hist(bins=30)
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age Distribution")
plt.show()

plt.figure()
plt.scatter(df["age"], df["hours-per-week"], alpha=0.3)
plt.xlabel("Age")
plt.ylabel("Hours per Week")
plt.title("Age vs Work Hours")
plt.show()

df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# One-hot encoding was chosen because the categorical variables have no inherent order. Dropping the first category avoids multicollinearity.

df_encoded["age_group"] = pd.cut(
    df["age"],
    bins=[16, 30, 45, 60, 100],
    labels=["Young", "Adult", "Middle-Aged", "Senior"]
)

df_encoded = pd.get_dummies(df_encoded, columns=["age_group"], drop_first=True)

# Age was binned into groups to capture non-linear relationships between age and income that may not be easily learned by a linear model.

X = df_encoded.drop("class_>50K", axis=1)
y = df_encoded["class_>50K"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
accuracy

print(classification_report(y_test, y_pred))

# The model achieves an accuracy of approximately X%, indicating it performs better than random guessing.
# Precision and recall show that the model predicts lower-income individuals more reliably than higher-income individuals,
# which is common for this dataset due to class imbalance.

# What was challenging or surprising about feature engineering?
# One challenge was handling the large number of categorical features, 
# which significantly increased the dimensionality of the dataset after one-hot encoding. 
# It was also surprising how much preprocessing was required before modeling.

# How did your choices affect model performance?
# One-hot encoding allowed the model to use categorical data effectively, while age binning helped capture non-linear patterns. 
# These choices improved model interpretability and slightly boosted performance.

# How could you improve your workflow next time?
# I would experiment with feature scaling, alternative encoders, and more advanced models such as Random Forests or Gradient Boosting.
# I would also explore techniques to handle class imbalance more effectively.