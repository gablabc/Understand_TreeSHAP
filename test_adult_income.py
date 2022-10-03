# %%
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd

# Our custom treeshap implementation
from treeshap import custom_treeshap, custom_taylor_treeshap

# Get the Adult-Income dataset
X, y = shap.datasets.adult()
X.columns = ["Age", "Workclass", "EducationNum", "MaritalStatus", "Occupation",
             "Relationship", "Race", "Sex", "CapitalGain", "CapitalLoss",
             "HoursPerWeek", "Country"]
# Fit the model
model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10, min_samples_leaf=50)
model.fit(X, y)


# %%
# We let the background and foreground be
# the dataset conditioned on sex=male and female
# We subsample the first 100 instances to make computations tracklable

background = X[X["Sex"]==1].iloc[np.arange(100)]
foreground = X[X["Sex"]==0].iloc[np.arange(100)]
gap = model.predict_proba(foreground)[:, 1].mean() - \
      model.predict_proba(background)[:, 1].mean()
print(f"P(y=1|female) - P(y=1|male) = {gap:.4f}")

# %% [markdown]
# ### SHAP attributions
# We observe a difference in average outcome between genders. The SHAP feature
# attribution can be used to measure which features contribute the most to the
# model disparity.
# %%
# Run TreeSHAP as normal
from shap.explainers import Tree
from shap.maskers import Independent
masker=Independent(background, max_samples=background.shape[0])
explainer = Tree(model, data=masker)
shap_values = explainer(foreground)
true_shap = shap_values.values[:, :, 1]

# %%
# Our custom treeshap implementation
custom_shap = custom_treeshap(model, foreground, background)
# Make sure we output the same result
print("Is our implementation correct ? ", np.isclose(true_shap, custom_shap).all())

# %% [markdown]
# The SHAP values respect the following property $\sum_{i=1}^d\phi_i = \mathbb{P}(y=1|\text{female}) - \mathbb{P}(y=1|\text{male})$.
# We verify this property emipirically
# %%

global_shap_values = custom_shap.mean(0)
assert np.isclose(global_shap_values.sum(), gap)

# %%

# Sort the features
sorted_features_idx = np.argsort(global_shap_values)

# Plot results
df = pd.DataFrame(global_shap_values[sorted_features_idx],
                  index=[X.columns[i] for i in sorted_features_idx])
df.plot.barh(capsize=4)
plt.plot([0, 0], plt.gca().get_ylim(), "k-")
plt.xlabel('Shap value')
plt.show()

# %% [markdown]
# We see that Relationship is the feature that drives disparities 
# the most for this model.
# %% [markdown]
# ### SHAP interactions
# We observe a difference in average outcome between genders. The SHAP feature
# attribution can be used to measure which features contribute the most to the
# model disparity.
# %%

# ## Tabular data with independent (Shapley value) masking
# build an Exact explainer and explain the model predictions on the given dataset
explainer = shap.explainers.Exact(model.predict_proba, masker=masker)
exact_shap_values = explainer(foreground, interactions=2).values
# Reshape into (N, d, d)
exact_shap_values = exact_shap_values.reshape((-1, 12, 12, 2))[...,1]

# %%
# Our implementation
Phi = custom_taylor_treeshap(model, foreground, background)
print("Is our implementation correct?", np.isclose(exact_shap_values, Phi).all())

# %%

global_Phi = Phi.mean(0)
global_Phi[np.abs(global_Phi) < 2e-3] = 0

from matplotlib.colors import TwoSlopeNorm
fig, ax = plt.subplots(figsize=(10,10))
im = ax.imshow(global_Phi, cmap='seismic', norm=TwoSlopeNorm(0))

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(12))
ax.set_xticklabels(X.columns)
ax.set_yticks(np.arange(12))
ax.set_yticklabels(X.columns)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(12):
    for j in range(12):
        text = ax.text(j, i, f"{global_Phi[i, j]:.3f}",
                       ha="center", va="center", color="w")

ax.set_title("Shapley-Taylor Global indices")
fig.tight_layout()
plt.show()
