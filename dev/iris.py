#
# Step 1: feature engineering
#

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn2pmml.decoration import ContinuousDomain

pipe = Pipeline(steps=[('continuous_domain', ContinuousDomain()), ('maxab', MaxAbsScaler()), ('pca', PCA(n_components=3)), ('logistic', LogisticRegressionCV())])

import pandas
import sklearn_pandas

iris = load_iris()

iris_df = pandas.concat((pandas.DataFrame(iris.data[:, :], columns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]), pandas.DataFrame(iris.target, columns = ["Species"])), axis = 1)

iris_mapper = sklearn_pandas.DataFrameMapper([
    (["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"], [x[1] for x in pipe.steps[:-1]]),
    ("Species", None)
])

# iris_mapper = sklearn_pandas.DataFrameMapper([
#     (["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"], [ContinuousDomain(), MaxAbsScaler(), PCA(n_components=3)]),
#     ("Species", None)
# ])

print iris_mapper

iris = iris_mapper.fit_transform(iris_df)

#
# Step 2: training a logistic regression model
#

iris_X = iris[:, 0:3]
iris_y = iris[:, 3]

iris_classifier = LogisticRegressionCV()
# iris_classifier = pipe.steps[-1][1]

iris_classifier.fit(iris_X, iris_y)

#
# Step 3: conversion to PMML
#

from sklearn2pmml import sklearn2pmml

sklearn2pmml(iris_classifier, iris_mapper, "LogisticRegressionIris.pmml", with_repr = True)