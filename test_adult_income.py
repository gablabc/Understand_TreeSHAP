# %%
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import shap
from treeshap import custom_treeshap

# The company has private access to a dataset
X,y = shap.datasets.adult()
X.columns = ["Age", "Workclass", "EducationNum", "MaritalStatus", "Occupation",
             "Relationship", "Race", "Sex", "CapitalGain", "CapitalLoss",
             "HoursPerWeek", "Country"]
# Fit the model
model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10, min_samples_leaf=50)
model.fit(X, y)


# %%
# We let the background and foreground be
# the dataset conditioned on sex=male and female

background = X[X["Sex"]==1].iloc[np.arange(1000)]
foreground = X[X["Sex"]==0].iloc[np.arange(1001)]
gap = model.predict_proba(foreground)[:, 1].mean() - \
      model.predict_proba(background)[:, 1].mean()
print(gap)

# %%
# Our custom treeshap implementation
custom_shap = custom_treeshap(model, foreground, background)
# Do SHAP values sum-up to the gap?
print(np.mean(np.sum(custom_shap, 1)))

# %%
# Run TreeSHAP as normal and compare
from shap.explainers import Tree
from shap.maskers import Independent
masker=Independent(background, max_samples=background.shape[0])
explainer = Tree(model, data=masker)
shap_values = explainer(foreground)
true_shap = shap_values.values[:, :, 1]

# %%
print(np.isclose(true_shap, custom_shap).all())
