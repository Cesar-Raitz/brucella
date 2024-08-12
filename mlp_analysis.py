"""
	This module develops a Multi-layer Perceptron model to fit the data
	obtained from fiber optic sensors developed at the Mackgraphe Lab. /
	Mackenzie - São Paulo. This study was developed in colaboration with
	Supernano / UFRJ - Rio de Janeiro.

	Author: Cesar Raitz Junior
	Creation: Feb-10-2024
	Licence: MIT

	Format: UTF-8 w/tabs
"""
__version__ = "1.00.1"

#%%
# Load the Dataset
#===============================================================================
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
import pandas

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
import functools

# HELPER FUNCTIONS
def _if_present(cols: list, df: DataFrame) -> list:
	return [c for c in cols if c in df.columns]

def if_present(cols: list) -> list:
	return functools.partial(_if_present, cols)

def simplify_feature_names(features: list) -> list:
	return [f.split('__')[-1] for f in features]

#%%
# Load the Dataset
#===============================================================================

def load_dataset(fname):
	if fname.endswith("csv"):
		df = pandas.read_csv(fname, index_col=0)
	else:
		df = pandas.read_excel(fname)

	target = df["class"].map({"positive":1, "negative":0})
	df.drop(columns=["title", "warnings"], inplace=True)
	
	# Categorize the measurement time
	bins = list(range(0, 101, 20)) + [np.inf]
	df["time_cat"] = pandas.cut(df["time"], bins=bins,
									  labels=range(1, len(bins)))
	cat_counts = df["time_cat"].value_counts()
	
	print("Time Categories:")
	for i in range(1, len(bins)):
		print(f"   {i}: [{bins[i-1]:3}, {bins[i]:3}) min"
		  		f" - {cat_counts[i]} instances")
	return df, target, bins


# Load the features extract from Jéssica's experiments in 2023
X_data, y_data, time_bins = load_dataset("data_jessica_2.csv")

# Feature groups
fts_hsv = ["mean_h", "mean_s", "mean_v"]
fts_histo = ["h1", "h2", "h3", "h4", "h5"]
fts_blobs = ["blobs_log", "blobs_dog", "blobs_doh"]

cc = y_data.value_counts()
print(len(X_data), "instances in this dataset")
print(cc[1], "Positive,", cc[0], "Negative")
X_data.head(5)

#%%
nice_names = {a: b for a, b in zip(fts_hsv + fts_histo + fts_blobs,
	["Mean Hue", "Mean Saturation", "Mean Value", "H1", "H2", "H3", "H4",
	"H5", "Blob count (LoG)", "Blob count (DoG)", "Blob count (DoH)"])}
nice_names["time"] = "Time [min]"
# Convert a list of feature names to nice names
nfts = lambda fts: [nice_names[n] for n in fts]

#%%
# Split the population in train/test sets keeping the time category proportions
#===============================================================================
from sklearn.model_selection import train_test_split

# We create a combined feature to stratify on both time and target class
X_data["strat_cat"] = X_data["time_cat"].astype(int)-1 + y_data*6

strat_series = X_data["time_cat"]

X_train, X_test, y_train, y_test = train_test_split(
		X_data, y_data, test_size=0.2, random_state=42,
		stratify=strat_series)

#%% Inspect the test/data proportions per time with a histogram
rel_counts = lambda s: s.value_counts().sort_index() / len(s)
test_counts = rel_counts(X_test["time_cat"])
total_counts = rel_counts(X_data["time_cat"])

_, ax = plt.subplots()
assert isinstance(ax, plt.Axes)
xticks = np.arange(1, 7)
w = 0.38
x = xticks + w/2
ax.bar(x, test_counts, width=w, edgecolor="black", label="Test set")
ax.bar(x+w, total_counts, width=w, edgecolor="black", label="All data")

ax.set_xticks(xticks)
ax.set_xticklabels(time_bins[:-1])
# yticks = np.arange(0, 0.181, 0.03)
yticks = np.arange(0, 0.21, 0.05)
ax.set_yticks(yticks)
ax.set_yticklabels([f'{100*n:.0f}%' for n in yticks])
ax.set_xlabel("Time Range (min)")
# ax.set_ylabel("Sample Frequency")
ax.legend();

#%% Inspect the test/data proportions per time with a histogram
rel_counts = lambda s: s.value_counts().sort_index() / len(s)
test_counts = rel_counts(X_test["time_cat"])
total_counts = rel_counts(X_data["time_cat"])

_, ax = plt.subplots()
assert isinstance(ax, plt.Axes)
xticks = np.arange(1, 7)
w = 0.38
x = xticks
ax.bar(x-w/2, test_counts, width=w, edgecolor="black", label="Test set")
ax.bar(x+w/2, total_counts, width=w, edgecolor="black", label="All data")

ax.set_xticks(xticks)
xlabels = [f"{n1}-{n2}" for n1, n2 in zip(time_bins, time_bins[1:])]
ax.set_xticklabels(xlabels)

ax.set_ylim(0, 0.21)
# yticks = np.arange(0, 0.26, 0.05)
# ax.set_yticks(yticks)
yticks = ax.get_yticks()
ylabels = [f'{100*n:.0f}%' for n in yticks]
ax.set_yticklabels(ylabels)
ax.set_xlabel("Time Interval (min)")
# ax.set_ylabel("Sample Frequency")
ax.legend(loc=9)
plt.plot()

#%%
# Create a Multi-Layer Perceptron Cassifier model
#===============================================================================
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer


ct = make_column_transformer(
	(StandardScaler(), if_present(fts_hsv)),
	(StandardScaler(), if_present(fts_blobs)),
	("passthrough", if_present(fts_histo)))

mlp = MLPClassifier(activation="relu",
						  hidden_layer_sizes=(1,),
						  #learning_rate="constant",
						  solver="lbfgs",
						  max_iter=2000)

pipeline = make_pipeline(ct, mlp)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
correct = np.count_nonzero(y_pred == y_test)
print("Features:", simplify_feature_names(pipeline[0].get_feature_names_out()))
print(f"{correct} ({correct/len(y_test)*100:.2f}%) correct predictions")
print("Total layers =", mlp.n_layers_)
pipeline


#%%
# It is difficult to evaluate the contribution of each feature to the model.
# Even for a single perceptron the weights change at each run.
coefficients = pandas.Series(
	data=abs(pipeline[-1].coefs_[0][:,0]),
	index=fts_hsv + fts_blobs + fts_histo
)
coefficients.sort_values(ascending=False)

#%%
from sklearn.metrics import confusion_matrix, roc_auc_score, brier_score_loss

def my_scorer(clf, X, y):
	"""A function to calculate several scores at once. If used for calculating
		scores in cross validation methods (i.e. `scoring=my_scorer`), the
		evaluation score must be specified with the `refit` parameter
		(e.g. `refit=F1`).

		Returns
		-------
		A dict with the following keys: Accuracy, Precision, Recall, AUC, Loss,
		F1, TN, FP, FN, TP, where AUC is the Area-Under-Curve and Loss is the
		Brier Loss function.
	"""
	y_pred = clf.predict(X)
	TN, FP, FN, TP = confusion_matrix(y, y_pred).ravel()
	accuracy = (TN+TP) / len(y_pred)
	precision = TP / (TP+FP)
	recall = TP / (TP+FN)
	f1 = 2*precision*recall / (precision+recall)
	scores = {
		"Accuracy": accuracy, "Precision": precision, "Recall": recall,
		"Loss": brier_score_loss(y, y_pred), "F1": f1,
		"TN": TN, "FP": FP, "FN": FN, "TP": TP}
	try:
		scores["AUC"] = roc_auc_score(y, y_pred)
	except ValueError:
		pass
	return scores

#%% Test on the current MLP pipeline
DataFrame([
	 my_scorer(pipeline, X_train, y_train),
	 my_scorer(pipeline, X_test, y_test)
], index=["Train", "Test"])

#%%
def scores_from_gridsearch(gs: GridSearchCV, index=-1,
									train_d=2, valid_d=2) -> DataFrame:
	"""Produce a pandas.DataFrame for Train/Test scores on separate rows.
	"""
	results = gs.cv_results_
	if index < 0: index = gs.best_index_
	# ve_to_str = lambda val, err: f"{val:.2f} ±{err:.2f}"

	def scores_for(set_name, decimals):
		fmt = f"{{:.{decimals}f}}"
		fmt = fmt + '±' + fmt
		
		the_dict = {}
		prefix = "mean_" + set_name + "_"
		for key, array in results.items():
			if key.startswith(prefix):
				new_key = key[len(prefix):]
				mean = array[index]
				std = results["std_"+key[5:]][index]
				the_dict[new_key] = fmt.format(mean, std)
		
		return the_dict
	
	return DataFrame([
		scores_for("train", train_d), scores_for("test", valid_d)],
		index=["Training", "Validation"])


# scores_from_gridsearch(gs, train_d=2)

#%% time
# Use GridSearchCV to find the best hyper-parametes
#===============================================================================
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import get_scorer
from util import beep
import time


def mlp_gridsearch(X_train, y_train, X_test, y_test, param_grid,
						 n_splits=50, scoring=my_scorer, refit="Recall",
						 features="all", random_state=None) -> tuple:
	"""Build GridSearchCV and run.
	"""
	X = X_train[features] if isinstance(features, list) else X_train
	pipeline[0].fit(X, y_train)
	
	cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.15,
									    random_state=random_state)
	
	grid_search = GridSearchCV(pipeline, param_grid, scoring=scoring,
										refit=refit, return_train_score=True,
										cv=cv, n_jobs=8, verbose=4)
	
	st = time.time()
	grid_search.fit(X, y_train)
	tt = time.time() - st
	print(f"The search time is {tt/60:.0f} min {tt%60:.0f} seconds")
	print(f"The best {refit} is {grid_search.best_score_:.2f}")
	print("Best model parameters:")
	for prm, val in grid_search.best_params_.items():
		print(f"  {prm}: {val}")
	features_used = pipeline[0].get_feature_names_out()
	print("Features:", simplify_feature_names(features_used))

	# Summarize scores for train/validation sets
	df_scores = scores_from_gridsearch(grid_search)

	# Add scores for the Test set
	if isinstance(scoring, str):
		score = get_scorer(scoring)(grid_search, X_test, y_test)
		new_dict = {scoring: score}
	if callable(scoring):
		new_dict = {key: f"{value:.2f}" for key, value in \
			scoring(grid_search, X_test, y_test).items()}
	df_scores.loc["Test"] = new_dict
	return grid_search, df_scores


param_grid = {
	"mlpclassifier__hidden_layer_sizes":
		# [(1,)],
		# [(1,), (10, 5), (10,), (100,), (100,10), (10,5,1)],
		[(10,),  (10, 10),   (10, 10, 10),
		 (50,),  (50, 50),   (50, 50, 50),
		 (100,), (100, 100), (100, 100, 100)],
	"mlpclassifier__activation": ["logistic", "relu"],
	"mlpclassifier__alpha": [1e-4, 5e-4, 1e-3],
	"mlpclassifier__solver": ["lbfgs"],
	"mlpclassifier__max_iter": [500],
}

gs, scores = mlp_gridsearch(
	X_train, y_train, X_test, y_test,
	param_grid, n_splits=100, features="all")
beep()
scores

#%%
for k in gs.cv_results_:
	if "mean" in k:
		print(k)

print(len(gs.cv_results_["mean_fit_time"]))

#%%
# Save/load the grid search for the MLP model.
#===============================================================================
import joblib

if False:
	# Save the grid_search
	joblib.dump(gs, "models/gs_240617_b.joblib")

else:
	# Reload the grid_search and use the best estimator to predict the outcomes
	# for the Test set. Need my_scorer(), if_present(), and X_test to be defined.
	saved_gs: GridSearchCV = joblib.load("models/gs_240617.joblib")
	best_clf: Pipeline = saved_gs.best_estimator_
	results = my_scorer(best_clf, X_test, y_test)
	for metric, score in results.items():
		if metric[-1] not in 'NP':
			score = f"{score*100:4.1f} %"
		print(f"{metric:>10} = {score}")
	print(scores_from_gridsearch(saved_gs))
	print("Best model parameters:")
	for prm, val in saved_gs.best_params_.items():
		print(f"  {prm}: {val}")

#%%
# Lets see which samples are incorrectly classified
y_pred = best_clf.predict(X_test)
X_test[y_pred != y_test]


#%% Load data from Wanderson's experiments
X2, y2, _ = load_dataset("data_wanderson.csv")

cc = y2.value_counts()
print(len(X2), "instances in this dataset")
print(cc[1], "Positive,", cc[0], "Negative")
X2.head(5)

#%%
for dilution, X in X2.groupby("dilution"):
	
	y = y2[X.index]
	res = my_scorer(best_clf, X, y)
	print("Results for dilution", dilution)
	print("  #Samples:", len(X))
	print("  Accuracy:", res["Accuracy"])
	print("    Recall:", res["Recall"])

# results = my_scorer(best_clf, X2, y2)
# for metric, score in results.items():
# 	if metric[-1] not in 'NP':
# 		score = f"{score*100:4.1f} %".replace("100.0", "100.")
# 	print(f"{metric:>10} = {score}")

#%% Where are the wrong predictions?
y2_pred = best_clf.predict(X2)
X2_false = X2[y2_pred != y2]


#%%
# Look for the best threshold parameters in blob counting
#===============================================================================
import os
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from skimage.feature import blob_dog, blob_log, blob_doh

def multi_blob_count(img_path: str,
							thrs: np.ndarray,
							sigmas=(5, 5, 7),
							add_constrast=False,
							) -> np.ndarray:
	"""Utility function to generate a matrix of blob counts using
	   three blob detection functions: blob_log, blob_dog, and blob_doh,
		and threshold arrays for each one (with the same ordering!)
	"""
	img = imread(img_path)
	if add_constrast: img = rescale_intensity(img)
	gray = rgb2gray(img)
	bc_matrix = []
	fns = (blob_log, blob_dog, blob_doh)
	for f, ts, ms in zip(fns, thrs, sigmas):
		m = [len(f(gray, max_sigma=ms, threshold=t)) for t in ts]
		bc_matrix.append(m)

	return np.array(bc_matrix)


row = X_train.iloc[0]
path = os.path.join("output", row["folder"], row["name"])

num_points = 16
x = [np.linspace(0.001, 0.14, num_points),     # LoG scale
	  np.linspace(0.001, 0.14, num_points),     # DoG scale
	  np.linspace(0.0001, 0.0045, num_points)]  # DoH scale
bcm = multi_blob_count(path, add_constrast=True, thrs=x)
print(bcm)

_, ax = plt.subplots(figsize=(6, 3))
ax.plot(x[0], bcm[0], '.-')
ax.plot(x[1], bcm[1], '.-')
ax.plot(x[2], bcm[2], '.-')
plt.show()

#%%
# Count all blobs in the training set
from util import beep, progress_bar, Curve

def df_blob_count(X: DataFrame, thrs: list, sigmas=(5, 5, 7),
						add_constrast=False) -> dict:
	"""Utility function that count blobs for several images in a Pandas
		DataFrame. Don't forget the order of thresholds (ts) lists:
		LoG, DoG, DoH.
	"""
	bcmm = []
	k, kk = 0, len(X)
	progress_bar(k, kk)
	for _, row in X.iterrows():
		path = os.path.join("output", row["folder"], row["name"])
		bcm = multi_blob_count(path, thrs=thrs, sigmas=sigmas,
									  add_constrast=add_constrast)
		bcmm.append(bcm)
		k += 1
		progress_bar(k, kk)
	print(f"\n-> {len(X)} instances for target={t}")

	bcd = {}
	bcmm = np.array(bcmm)
	stds = bcmm.std(axis=0)
	means = bcmm.mean(axis=0)
	labels = ["LoG", "DoG", "DoH"]
	for x, y, e, l in zip(thrs, means, stds, labels):
		bcd[l] = dict(x=x, y=y, e=e)
	return bcd

# X = X_train.iloc[:30]
# y = y_train.iloc[:30]
X = X_train
y = y_train
num_points = 20
x_thr = [np.linspace(0.1, 0.3, num_points), # LoG scale
			np.linspace(0.05, 0.25, num_points), # DoG scale
	      np.linspace(0.0005, 0.012, num_points)] # DoH scale

bc_curves = [[0, 0] for _ in range(3)]
tnames = ["Negative", "Positive"]
colors = ["darkorange", "deepskyblue"]
for t, tn, c in zip([0, 1], tnames, colors):
	# Calculate the blob counts' mean and error for one target
	bcd = df_blob_count(X[y==t], thrs=x_thr, sigmas=(3, 3, 4.24),
							  add_constrast=False)
	# Create Curve's from data dictionaries to plot them later
	for i, key in enumerate(bcd):
		bc_curves[i][t] = Curve(bcd[key], color=c, label=tn)
	beep()

#%%
# Plot all blob's threshold curves
def plot_blob_curves(curves: list[Curve], ax: plt.Axes):
	for c in curves: c.plot_error(ax)
	for c in curves: c.plot_curve(ax)
	ax.set_xlim(curves[0]._x[[0,-1]])
	ax.set_yticks(range(0, 31, 10))
	ax.legend()


titles = ["Laplacian of Gaussian",
			 "Difference of Gaussian",
			 "Determinant of Hessian"]

if False:
	# Plot Log, Dog, and Doh results
	_, axs = plt.subplots(3, 1, figsize=(6, 8))
	for curves, title, ax in zip(bc_curves, titles, axs):
		assert isinstance(ax, plt.Axes)
		plot_blob_curves(curves, ax)
		ax.set_title(title)
		ax.set_ylim(0, 30)
		ax.grid()
	axs[1].set_ylabel("Average Blob Count")
	axs[2].set_xlabel("Threshold")

else:
	# Plot the Log and Doh results only (to fit in the presentation)
	_, axs = plt.subplots(2, 1, figsize=(6, 5))
	for curves, title, ax in zip(bc_curves[::2], titles[::2], axs):
		assert isinstance(ax, plt.Axes)
		plot_blob_curves(curves, ax)
		ax.set_title(title)
		ax.set_ylim(0, 30)
		ax.grid()
		ax.set_ylabel("Average Blob Count")
	axs[1].set_xlabel("Threshold")
	
plt.tight_layout()

#%%
# Blob counts for negative and positive images seem to intersect no matter what.
# So we'll look for some minimum intersection.
from util import overlap

_, axs = plt.subplots(3, 1, figsize=(4, 6))
for curves, title, ax in zip(bc_curves, titles, axs):
	yn, en = curves[0]._y, curves[0]._e
	yp, ep = curves[1]._y, curves[1]._e
	ovr = np.array([
		overlap(max(a, 0), b, max(c, 0), d)/(d-c)
			for a, b, c, d in zip(yn-en, yn+en, yp-ep, yp+ep)
	])
	
	# imin = np.where(ovr < 0.10)[0][0]
	# print(ovr[imin])

	yticks = np.arange(0, 0.21, 0.05)
	
	ax.grid()
	ax.set_yticks(yticks)
	ax.set_yticklabels([f'{int(100*n)}%' for n in yticks])
	ax.set_ylim(yticks[0], yticks[-1])
	ax.plot(curves[0]._x, ovr)
	ax.set_title(title)

axs[2].set_xlabel("Threshold")
axs[1].set_ylabel("Positive-Negative Blob Counts Overlap")
plt.tight_layout()

#%%
# Calculate the Point-Biserial Correlation

from pandas import DataFrame, Series
from sklearn.compose import make_column_selector

def PBC(data: DataFrame, target: Series, return_dfs=False) -> Series:
	"""Calculate the Point-Biserial Correlation on numerical columns of a
		DataFrame."""
	cols = make_column_selector(dtype_include=np.number)(data)
	neg_data = data.loc[target==0, cols]
	pos_data = data.loc[target==1, cols]
	M0 = neg_data.mean()
	M1 = pos_data.mean()
	n0 = len(neg_data)
	n1 = len(pos_data)

	pbc: Series = (M1 - M0) / data[cols].std() * np.sqrt(n0*n1/(n0+n1)**2)
	pbc = pbc.sort_values(key=np.abs, ascending=False)
	if return_dfs: return pbc, neg_data, pos_data
	return pbc

pbc = PBC(X_data, y_data)

#%%
# Plot the values of some features as a function of time

_, neg_df, pos_df = PBC(X_data, y_data, return_dfs=True)
fig, axs = plt.subplots(3, 1, figsize=(6,10), sharex="col")

feats = ["mean_v", "blobs_dog", "blobs_log"]

for feature, ylabel, ax in zip(feats, nfts(feats), axs):

	neg_df.plot.scatter("time", feature, c="blue", ax=ax)
	pos_df.plot.scatter("time", feature, c="red", ax=ax)
	ax.set_ylabel(ylabel)

ax.set_xlim([0, 125])
ax.set_xlabel("Time (min)")
plt.tight_layout()
plt.show()

#%%
#

def features_std(input_df, features) -> DataFrame:

	dt = 5
	bins = np.arange(2.5, 125, dt)
	labs = range(0, len(bins)-1)
	time_cat = pandas.cut(input_df["time"], bins=bins, labels=labs)

	std_data = []
	for i in labs:
		bin_data = input_df[time_cat==i]
		std_data.append(bin_data[features].std())

	df = DataFrame(std_data)
	df['time'] = (np.array(labs)+1)*dt
	return df

neg_std = features_std(neg_df, feats)
pos_std = features_std(pos_df, feats)

#%%
fig, axs = plt.subplots(3, 1, figsize=(5,8), sharex="col")

for feature, ylabel, ax in zip(feats, nfts(feats), axs):
	assert isinstance(ax, plt.Axes)
	# pos_std.plot.scatter("time", feature, c="red", ax=ax, label="Positive")
	# neg_std.plot.scatter("time", feature, c="blue", ax=ax, label="Negative")

	ax.plot(neg_std["time"], neg_std[feature]**2, "s-", label="Negative", ms=5)
	ax.plot(pos_std["time"], pos_std[feature]**2, ".-", label="Positive", ms=10)
	ax.set_ylabel(ylabel)
	ax.legend()

# ax.set_xlim([0, 125])
axs[0].set_title("Variance")
ax.set_xlabel("Time (min)")
plt.tight_layout()
plt.show()

 #%%============================================================================
 #  PREPARE THE DATA TO PLOT SCATTER MATRICES
 #==============================================================================


# Make a single time copy of the dataset with target class
# to be used with the pairplot
df_data = X_data.rename(nice_names, axis=1)
df_data["Sample"] = y_data.map({0:"Negative", 1:"Positive"})
df_data.sort_values("Sample", ascending=False, inplace=True)

#%%
# Plot the scatter matrix using Seaborn
import seaborn as sns

def scatter_matrix(features, legend_pos=(0.8, 0.85), kind="scatter",
						 alpha=None):
	global nice_names, df_data
	palette = {"Negative": "dodgerblue",
				  "Positive": "orange"}
	kws = {"alpha": alpha} if alpha else {}

	features = [nice_names[n] for n in features]

	g = sns.pairplot(df_data, hue="Sample", vars=features,
						  kind=kind, markers=['o', 's'], plot_kws=kws,
						  corner=True)
	if legend_pos is not None:
		g.legend.set_bbox_to_anchor(legend_pos)
		g.legend.set_frame_on(True)

# scatter_matrix(fts_blobs)
# scatter_matrix(fts_histo)
scatter_matrix(["mean_v", "h4", "blobs_dog"])

#%%
from sklearn.neighbors import KernelDensity
from pandas import DataFrame, Series

def plot_kde(X: Series, hue: Series, hues: list, ax: plt.Axes):
	if isinstance(X, Series):
		X = X.values
		if isinstance(hue, Series):
			hue = hue.values
		else:
			raise ValueError
	
	a, b = min(X), max(X)
	d = (b - a) * 0.1
	xd = np.linspace(a-d, b+d, 100)
	
	# Calculate the density distributions with a Gaussian kernel
	distros = []
	for h in hues:
		# x = data.loc[data[hue]==h, [column]].values
		x = X[hue==h]
		kde = KernelDensity(bandwidth=x.std()*0.35)
		ylog = kde.fit(x[:, None]).score_samples(xd[:, None])
		distros.append(np.exp(ylog))
	
	# Plot the distributions normalized 1
	m = max(max(dist) for dist in distros)
	if isinstance(hues, dict):
		colors = hues.values()
	else:
		colors = [None] * len(hues)
	for yd, color in zip(distros, colors):
		yd /= m
		ax.fill_between(xd, yd, color=color, alpha=0.2)
		ax.plot(xd, yd, color=color)


def my_scatter_matrix(data, hue, x_vars, y_vars, hues=None, dph=0.4, kws=None):
	if hues is None: hues = data[hue].unique()
	cols, rows = len(x_vars), len(y_vars)
	fs = np.array([cols, dph+rows]) * 4
	_, axs = plt.subplots(rows+1, cols, sharey="row", sharex="col",
								 figsize=fs, height_ratios=[dph] + [1]*rows)

	# PLOT THE UNIVARIATE DISTRIBUTIONS
	for column, ax in zip(x_vars, axs[0]):
		plot_kde(data[column], data[hue], hues, ax=ax)
		ax.get_yaxis().set_visible(False)

	# PLOT THE BIVARIATE SCATTER PLOTS
	for j, column in enumerate(x_vars):
		for i, row in enumerate(y_vars):
			ax = axs[i+1][j]
			for k, h in enumerate(hues):
				x = data.loc[data[hue]==h, column]
				y = data.loc[data[hue]==h, row]
				args = {}
				if isinstance(kws, dict):
					args = {key: val[k] for key, val in kws.items()}
				if isinstance(hues, dict): args["color"] = hues[h]
				ax.plot(x, y, lw=0, label=h, **args)
			ax.grid("on")

	for j, column in enumerate(x_vars): axs[-1][j].set_xlabel(column, fontsize=16)
	for i, row in enumerate(y_vars): axs[i+1][0].set_ylabel(row, fontsize=16)
	axs[-1][-1].legend(prop={"size": 14})
	plt.subplots_adjust(hspace=0.1, wspace=0.1)


options = dict(marker='os')
hues = {"Positive": "orange", "Negative": "dodgerblue"}
my_scatter_matrix(df_data, "Sample", hues=hues,
						y_vars=nfts(["mean_v", "mean_s"]),
						x_vars=["H4", "H5", "H1", "Blob count (LoG)"],
						kws=options)

#%%

my_scatter_matrix(df_data, "Sample", hues=hues,
						x_vars=nfts(fts_histo),
						y_vars=nfts(fts_hsv),
						kws=options)

#%%
my_scatter_matrix(df_data, "Sample", hues=hues,
						x_vars=nfts(fts_blobs)+["time"],
						y_vars=nfts(fts_hsv),
						kws=options)
