# %%
import sys
!{sys.executable} -m pip install nbformat

# %%
import pandas as pd
import numpy as np
import dalex as dx
import matplotlib.pyplot as plt

# import shap -- not installed yet
import joblib

from dalex._explainer.yhat import yhat_proba_default
from sklearn.model_selection import train_test_split

# %%
# %%
# load models and data
clfNB = joblib.load('models/clfNB.pkl')
clfGB = joblib.load('models/clfGB.pkl')
clfGB_reduced = joblib.load('models/clfGB_final.pkl')
compVars = pd.read_pickle('models/compVars.pkl').tolist()
dat_ml = pd.read_pickle('dat_ml.pkl')
y_pred = dat_ml.before1980
X_pred = dat_ml.drop(['yrbuilt', 'before1980'], axis = 1)
X_pred_reduced = dat_ml.filter(compVars, axis = 1)

X_train, X_test, y_train, y_test = train_test_split(
    X_pred, y_pred, test_size = .34, random_state = 76) 
# may not be the most efficient way to do this
X_train_reduced, X_test_reduced, y_train, y_test = train_test_split(
    X_pred_reduced, y_pred, test_size = .34, random_state = 76)  
# %%

# Using DALEX
# Partial-dependence profiles
# Accumulated-local Profiles
# Create explainer objects and show variable importance chart

expReduced = dx.Explainer(clfGB_reduced, X_test_reduced, y_test)
explanationReduced = expReduced.model_parts()
explanationReduced.plot(max_vars=15)

# %%
# show model performance
mpReduced = expReduced.model_performance(model_type = 'classification')
print(mpReduced.result)
mpReduced.plot(geom="roc")

# %%
# Explain variables
# y axis shows probably of house being a 1 which is pre1980 houses
pdp_num_red = expReduced.model_profile(type = 'partial', label="pdp", variables = compVars)
ale_num_red = expReduced.model_profile(type = 'accumulated', label="ale", variables = compVars)
pdp_num_red.plot(ale_num_red)

# %%
# Explain observation
# shapley values
# shows how each variable will affect a specific observations value of being a pre1980 house 
sh = expReduced.predict_parts(X_test_reduced.iloc[0,:], type='shap', label="first observation")

sh.plot(max_vars=12)
# %%

# using shapley

# %%
# Build shap explainer
explainerShap = shap.Explainer(clfGB_reduced)
shap_values = explainerShap(X_test_reduced)

# %%
# Show variable importance based on shap values
shap.plots.bar(shap_values)

# %%
# https://medium.com/dataman-in-ai/the-shap-with-more-elegant-charts-bc3e73fa1c0c
shap.plots.beeswarm(shap_values)

# %%
# comparable to the bar plot
shap.plots.beeswarm(shap_values.abs, color="shap_red")

# %%
# combine the above charts
shap.plots.heatmap(shap_values[0:1000],  max_display=13) 
# each line is a person, how much the shapley value is influenced on mode, red positive, blue negative,