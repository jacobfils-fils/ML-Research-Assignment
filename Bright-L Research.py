# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC **
# MAGIC C1. Explore and clean the data**
# MAGIC - Before cleaning, inspect the number of rows and columns, column types, summary statistics, missing values, and duplicate rows. These checks help identify data-quality problems before modelling.

# COMMAND ----------

# MAGIC %md
# MAGIC Clean the data
# MAGIC The sample includes one duplicate row and a few missing numeric values. We remove exact duplicate rows, then fill missing numeric values with the median. The median is a reasonable practice choice when a numeric column may contain outliers; in a real project, explain and justify the method you choose.
# MAGIC
# MAGIC We also remove student_id from the model features because it is an identifier, not a meaningful measure of student performance

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = {
    "study_hours": [2, 4, 1, 6, 8, 3, 7, 5, 9, 2, 4, 10, 6, 8, 1, 7, 5, 3, 9, 4],
    "attendance":  [55, 65, 50, 75, 90, 60, 85, 70, 95, 58, 68, 98, 80, 88, 52, 82, 72, None, 92, 64],
    "course": ["Python", "SQL", "Python", "SQL", "Python", "SQL", "Python", "SQL",
               "Python", "SQL", "Python", "SQL", "Python", "SQL", "Python", "SQL",
               "Python", None, "Python", "SQL"],
    "passed": [0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0]
}
df = pd.DataFrame(data)
df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # intentional duplicate
display(df)
print("Rows and columns:", df.shape)
print("\nColumn types:\n", df.dtypes)
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicates:", df.duplicated().sum())

# COMMAND ----------

df = df.drop_duplicates().reset_index(drop=True)
print("Rows after removing duplicates:", len(df))
print("Missing values remaining:\n", df.isna().sum())

# COMMAND ----------

df.info()


# COMMAND ----------

df.describe()

# COMMAND ----------

df.isnull().sum()

# COMMAND ----------

df.duplicated()

# COMMAND ----------

df.drop_duplicates()

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

X = df[["study_hours", "attendance", "course"]]
y = df["passed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

numeric_steps = Pipeline([
    ("fill_missing", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])
category_steps = Pipeline([
    ("fill_missing", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore"))
])
prepare = ColumnTransformer([
    ("numbers", numeric_steps, ["study_hours", "attendance"]),
    ("categories", category_steps, ["course"])
])

classifier = Pipeline([
    ("prepare", prepare),
    ("model", LogisticRegression(max_iter=1000))
])
classifier.fit(X_train, y_train)
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Missing values in original training features:\n", X_train.isna().sum())
print("Test predictions:", classifier.predict(X_test).tolist())
print("Actual test answers:", y_test.tolist())

# COMMAND ----------

plt.figure(figsize=(6, 2.5))
plt.boxplot(df["study_hours"].dropna(), vert=False)
plt.title("Study hours — boxplot")
plt.xlabel("Hours")
plt.show()

q1 = df["study_hours"].quantile(0.25)
q3 = df["study_hours"].quantile(0.75)
iqr = q3 - q1
print("IQR lower bound:", q1 - 1.5 * iqr)
print("IQR upper bound:", q3 + 1.5 * iqr)

# COMMAND ----------

example = df.copy()
example["high_attendance"] = (example["attendance"] >= 80).astype(int)
display(example[["attendance", "high_attendance"]].head(10))
print("Correlation between numeric columns:")
display(example.select_dtypes(include="number").corr().round(2))

# COMMAND ----------

# MAGIC %md
# MAGIC C2. Handling missing values
# MAGIC Common choices include:
# MAGIC - dropna(): remove rows or columns with missing values. This can lose valuable data, especially in a small dataset.
# MAGIC - fillna(): fill missing values with a chosen value such as the median, mean, mode, or a meaningful constant.
# MAGIC - SimpleImputer: a scikit-learn transformer that learns an imputation value from training data and applies it consistently to later data.
# MAGIC Use the mean when a numeric variable is reasonably symmetric and not strongly affected by outliers. Use the median when outliers or skewness may distort the mean. Use the mode for a common categorical value when that is appropriate. Do not fill values blindly: the method should make sense for the data and question.

# COMMAND ----------

from sklearn.tree import DecisionTreeClassifier

for depth in [1, 2, 5, None]:
    tree = Pipeline([
        ("prepare", prepare),
        ("model", DecisionTreeClassifier(max_depth=depth, random_state=42))
    ])
    tree.fit(X_train, y_train)
    print(f"Depth {str(depth):>4} | training: {tree.score(X_train, y_train):.2f} | test: {tree.score(X_test, y_test):.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC Underfitting and overfitting
# MAGIC A shallow tree may miss patterns (underfit). A deep tree can memorise training examples (overfit). Compare training accuracy and test accuracy; this tiny dataset makes both scores unstable.

# COMMAND ----------

from sklearn.cluster import KMeans

# Unsupervised classroom illustration; this section does not evaluate on a holdout.
cluster_input = df[["study_hours", "attendance"]].copy()
cluster_input = cluster_input.fillna(cluster_input.median())
cluster_scaled = StandardScaler().fit_transform(cluster_input)

inertias = []
for k in [1, 2, 3, 4, 5]:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(cluster_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(5, 3))
plt.plot([1, 2, 3, 4, 5], inertias, marker="o")
plt.title("Elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.show()

km = KMeans(n_clusters=2, n_init=10, random_state=42)
clusters = km.fit_predict(cluster_scaled)
result = cluster_input.copy()
result["cluster"] = clusters
display(result.head(10))

# COMMAND ----------

# Example of pandas options (run individually if you want to compare)
example_missing = pd.DataFrame({"score": [60, 70, np.nan, 90]})

print("Original:")
display(example_missing)

print("Drop rows with missing values:")
display(example_missing.dropna())

print("Fill with the mean:")
display(example_missing.fillna({"score": example_missing["score"].mean()}))

print("Fill with the median:")
display(example_missing.fillna({"score": example_missing["score"].median()}))

# COMMAND ----------

# MAGIC %md
# MAGIC C3. Encode categorical data
# MAGIC Most scikit-learn estimators expect numeric input features, so text categories often need encoding.
# MAGIC - Label encoding: LabelEncoder converts target labels such as Yes and No into integers. It is generally intended for the target (y), not unordered input columns.
# MAGIC - Ordinal encoding: OrdinalEncoder converts categories with a meaningful order, such as Low < Medium < High, into ordered numbers.
# MAGIC - One-hot encoding: OneHotEncoder creates a separate indicator column for each category. It is often suitable for nominal (unordered) input features such as province or learning method.
# MAGIC Nominal categories have no natural ranking. Ordinal categories do have a meaningful order. Using arbitrary integer codes for nominal categories can accidentally suggest a ranking to some models.
# MAGIC
# MAGIC One-hot encoding can create many columns when a feature has hundreds or thousands of unique categories. This increases memory use and can make the model more complex; options include grouping rare categories, using carefully selected encoders, or choosing a model that handles categorical features appropriately.

# COMMAND ----------

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, ConfusionMatrixDisplay)

predictions = classifier.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, predictions), 2))
print("Precision:", round(precision_score(y_test, predictions, zero_division=0), 2))
print("Recall:", round(recall_score(y_test, predictions, zero_division=0), 2))
print("F1 score:", round(f1_score(y_test, predictions, zero_division=0), 2))
ConfusionMatrixDisplay.from_predictions(y_test, predictions)
plt.title("Confusion matrix — learner pass prediction")
plt.show()

print("Fraud example: predicting 'not fraud' for 99 non-fraud + 1 fraud transaction")
print("Accuracy = 99%, but fraud recall = 0%")

# COMMAND ----------

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder

# LabelEncoder example for a target label
target_encoder = LabelEncoder()
target_example = pd.Series(["Yes", "No", "Yes", "No"])
print("Target labels:", target_example.tolist())
print("Encoded target:", target_encoder.fit_transform(target_example))
print("Class mapping order:", target_encoder.classes_)

# OrdinalEncoder example for a genuinely ordered input feature
ordinal_encoder = OrdinalEncoder(
    categories=[["Low", "Medium", "High"]]
)
satisfaction_encoded = ordinal_encoder.fit_transform(df[["satisfaction"]])
print("\nFirst five satisfaction values and encodings:")
display(pd.DataFrame({
    "satisfaction": df["satisfaction"].head().to_list(),
    "encoded": satisfaction_encoded[:5, 0]
}))

# OneHotEncoder example for an unordered feature
onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
province_encoded = onehot_encoder.fit_transform(df[["province"]])
province_encoded_df = pd.DataFrame(
    province_encoded,
    columns=onehot_encoder.get_feature_names_out(["province"]),
    index=df.index
)
print("\nOne-hot encoded province columns:")
display(province_encoded_df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC C4. Scale numeric features
# MAGIC Columns can have very different ranges. For example, salary might be in thousands while age is measured in tens. For distance-based or gradient-based methods, a large-scale feature can dominate learning.
# MAGIC - Standardisation (StandardScaler) transforms values to have approximately mean 0 and standard deviation 1.
# MAGIC - Min–max scaling (MinMaxScaler) maps values to a chosen range, commonly 0 to 1.
# MAGIC - Robust scaling (RobustScaler) uses the median and interquartile range, making it less influenced by extreme values.
# MAGIC Scaling is often important for K-nearest neighbours (KNN), support vector machines (SVM), and logistic regression. Decision trees and random forests generally do not require feature scaling because their splits depend on feature thresholds rather than distances or feature magnitudes.

# COMMAND ----------

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

numeric_demo = df[["age", "study_hours"]].copy()

standard_scaler = StandardScaler()
standard_values = standard_scaler.fit_transform(numeric_demo)

minmax_scaler = MinMaxScaler()
minmax_values = minmax_scaler.fit_transform(numeric_demo)

robust_scaler = RobustScaler()
robust_values = robust_scaler.fit_transform(numeric_demo)

print("Original numeric features:")
display(numeric_demo.head())

print("StandardScaler result:")
display(pd.DataFrame(standard_values, columns=["age_standard", "study_hours_standard"]).head())

print("MinMaxScaler result:")
display(pd.DataFrame(minmax_values, columns=["age_minmax", "study_hours_minmax"]).head())

print("RobustScaler result:")
display(pd.DataFrame(robust_values, columns=["age_robust", "study_hours_robust"]).head())

# COMMAND ----------

# MAGIC %md
# MAGIC C5. Split into training and test sets without data leakage
# MAGIC The training set is used to fit the model; the test set is held back to estimate how well the model works on unseen data. An 80/20 split is a common starting point.
# MAGIC - random_state makes the random split reproducible.
# MAGIC - stratify=y helps preserve class proportions for classification when the classes have enough examples.
# MAGIC - fit() learns values or parameters from data.
# MAGIC - transform() applies the learned transformation.
# MAGIC - fit_transform() learns and applies the transformation to the same data.
# MAGIC Data leakage occurs when information that should be unavailable during training influences the training process. For example, fitting a scaler on the full dataset before splitting lets the test set influence the scaling parameters. Split first, fit preprocessing on the training set only, and then transform the test set using those same fitted objects.

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Define target and features. student_id is excluded because it is an identifier.
X = df[["age", "study_hours", "province", "learning_method", "satisfaction"]].copy()
y = df["passed"].map({"No": 0, "Yes": 1})

# 80/20 split. Stratify preserves the target proportions when feasible.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Training rows:", X_train.shape[0])
print("Test rows:", X_test.shape[0])
print("\nTraining target proportions:")
display(y_train.value_counts(normalize=True).rename("proportion").to_frame())
print("Test target proportions:")
display(y_test.value_counts(normalize=True).rename("proportion").to_frame())

numeric_features = ["age", "study_hours"]
nominal_features = ["province", "learning_method"]
ordinal_features = ["satisfaction"]

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

nominal_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

ordinal_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ordinal", OrdinalEncoder(
        categories=[["Low", "Medium", "High"]],
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ))
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_pipeline, numeric_features),
    ("nominal", nominal_pipeline, nominal_features),
    ("ordinal", ordinal_pipeline, ordinal_features)
])

# Pipeline fits imputation, encoding, scaling and model using X_train only.
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Predicted test labels:", predictions)
print("Actual test labels:", y_test.to_numpy())
print("Test accuracy:", round(accuracy_score(y_test, predictions), 3))
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

# COMMAND ----------

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

hours = pd.DataFrame({"hours": [1, 2, 3, 4, 5, 6]})
marks = [35, 42, 46, 57, 62, 70]
reg = LinearRegression().fit(hours, marks)
estimated = reg.predict(hours)
print("MAE:", round(mean_absolute_error(marks, estimated), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(marks, estimated)), 2))
print("R²:", round(r2_score(marks, estimated), 2))