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

def load_dataset(fname):
	if fname.endswith("csv"):
		df = pandas.read_csv(fname, index_col=0)
	else:
		df = pandas.read_excel(fname)

	target = df["class"].map({"positive":1, "negative":0})
	df.drop(columns=["class", "title", "warnings"],
			  inplace=True)
	
	# Categorize the measurement time
	bins = list(range(0, 101, 20)) + [np.inf]
	df["time"] = pandas.cut(df["time"], bins=bins, labels=range(1, len(bins)))
	cat_counts = df["time"].value_counts()
	print("Time Categories:")
	for i in range(1, len(bins)):
		print(f"   {i}: [{bins[i-1]:3}, {bins[i]:3}) min"
		  		f" - {cat_counts[i]} instances")
	return df, target, bins


X_data, y_data, time_bins = load_dataset("data_jessica.csv")

cc = y_data.value_counts()
print(len(X_data), "instances in this dataset")
print(cc[1], "Positive,", cc[0], "Negative")
X_data.head(5)

#%%
X_data.info()

#%%
# Split the population in train/test sets keeping the time category proportions
#===============================================================================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
		X_data, y_data, test_size=0.2, random_state=42,
		stratify=X_data["time"])

# Inspect the test/data proportions per time with a histogram
rel_counts = lambda s: s.value_counts().sort_index() / len(s)
test_counts = rel_counts(X_test['time'])
total_counts = rel_counts(X_data['time'])

_, ax = plt.subplots()
assert isinstance(ax, plt.Axes)
xticks = np.arange(1, 7)
w = 0.36
x = xticks + w/2
ax.bar(x, test_counts, width=w, edgecolor="black", label="Test set")
ax.bar(x+w, total_counts, width=w, edgecolor="black", label="All data")

ax.set_xticks(xticks)
ax.set_xticklabels(time_bins[:-1])
# yticks = np.arange(0, 0.181, 0.03)
yticks = np.arange(0, 0.21, 0.05)
ax.set_yticks(yticks)
ax.set_yticklabels([f'{100*n:.0f}%' for n in yticks])
ax.set_xlabel("Time (min)")
ax.set_title("Sample Frequency")
ax.legend();


#%%
# Create a Multi-Layer Perceptron classifier model
#===============================================================================
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


mlp_clf = MLPClassifier(activation="relu", alpha=0.0001, hidden_layer_sizes=(10),
								learning_rate="constant", solver="lbfgs", max_iter=2000)
drop_cats = make_column_transformer(("drop", ["time", "name", "folder", "date"]),
												remainder="passthrough")
pipeline = make_pipeline(drop_cats, StandardScaler(), mlp_clf)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
correct = np.count_nonzero(y_pred == y_test)
print(f"{correct} ({correct/len(y_test)*100:.2f}%) correct predictions")
print("Total layers =", mlp_clf.n_layers_)

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
	TN, FP, FN, TP = confusion_matrix(y_pred, y).ravel()
	accuracy = (TN+TP) / len(y_pred)
	precision = TP / (TP+FP)
	recall = TP / (TP+FN)
	f1 = 2*precision*recall / (precision+recall)
	return {"Accuracy": accuracy, "Precision": precision, "Recall": recall,
			  "AUC": roc_auc_score(y_pred, y), "Loss": brier_score_loss(y_pred, y),
			  "F1": f1, "TN": TN, "FP": FP, "FN": FN, "TP": TP}

DataFrame([
	 my_scorer(pipeline, X_train, y_train),
	 my_scorer(pipeline, X_test, y_test)
], index=["Train", "Test"])

#%%
from sklearn.model_selection import GridSearchCV

def scores_from_grid(gs: GridSearchCV, index=-1) -> DataFrame:
	"""Produce a pandas.DataFrame for Train/Test scores on separate rows.
	"""
	results = gs.cv_results_
	if index < 0: index = gs.best_index_
	ve_to_str = lambda val, err: f"{val:.2f} ±{err:.2f}"

	def scores_for(set_name):
		the_dict = {}
		prefix = "mean_" + set_name + "_"
		for key, array in results.items():
			if key.startswith(prefix):
				new_key = key[len(prefix):]
				mean = array[index]
				std = results["std_"+key[5:]][index]
				the_dict[new_key] = ve_to_str(mean, std)
		
		return the_dict
	
	return DataFrame([scores_for("train"), scores_for("test")],
							index=["Training", "Validation"])

#%%
# Use GridSearchCV to find the best hyper-parametes
#===============================================================================
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedShuffleSplit

# Define strategies for splitting and searching
cv = StratifiedShuffleSplit(n_splits=50, test_size=0.15, random_state=0)
param_grid = {
	"mlpclassifier__hidden_layer_sizes": [(1,), (5,), (10,), (100,)],
	"mlpclassifier__learning_rate_init": [0.001, 0.005]
}

# Scorers used for model evaluation
scoring = my_scorer
# scoring = dict(Accuracy="accuracy", Precision="precision",
# 					Recall="recall", AUC="roc_auc",
# 					Loss="neg_brier_score", F1="f1")
refit = "Recall"

grid_search = GridSearchCV(pipeline, param_grid, scoring=scoring,
									refit=refit, return_train_score=True,
									cv=cv, n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best {refit} is {grid_search.best_score_:.2f}")
print("Best model parameters:")
for prm, val in grid_search.best_params_.items():
	print(f"  {prm}: {val}")

# Summarize scores for train/validation sets
df_scores = scores_from_grid(grid_search)
# df_scores

# Add scores for the test group (only for the final model)
if False:
	if isinstance(scoring, dict):
		new_dict = {}
		for name, scorer in scoring.items():
			if isinstance(scorer, str):
				scorer = get_scorer(scorer)
			value = scorer(grid_search, X_test, y_test)
			new_dict[name] = value
		df_scores.loc["Test"] = new_dict

	elif callable(scoring):
		new_dict = {key: f"{value:.2f}" for key, value in \
			scoring(grid_search, X_test, y_test).items()}
		df_scores.loc["Test"] = new_dict

df_scores

#%%
# Look for the best threshold parameters in blob counting
#===============================================================================
import os
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from skimage.feature import blob_dog, blob_log, blob_doh

def multi_blob_count(img_path: str,
							tlog: np.ndarray,
							tdog: np.ndarray=None,
							tdoh: np.ndarray=None,
							add_constrast=False,
							max_sigma=5) -> np.ndarray:
	"""Utility function to generate a matrix of blob counts using
	   three blob detection functions: blob_log, blob_dog, and blob_doh,
		and threshold arrays for each one (with the same ordering!)
	"""
	img = imread(img_path)
	if add_constrast: img = rescale_intensity(img)
	gray = rgb2gray(img)
	if isinstance(tlog, list):
		tlog, tdog, tdoh = tlog  # unpack scales
	assert len(tlog) == len(tdog) == len(tdoh)
	
	bc_matrix = []
	for f, ts in zip((blob_log, blob_dog, blob_doh), (tlog, tdog, tdoh)):
		m = [len(f(gray, max_sigma=max_sigma, threshold=t)) for t in ts]
		bc_matrix.append(m)

	return np.array(bc_matrix)


row = X_train.iloc[0]
path = os.path.join("output", row["folder"], row["name"])

num_points = 16
x = [np.linspace(0.001, 0.14, num_points),     # LoG scale
	  np.linspace(0.001, 0.14, num_points),     # DoG scale
	  np.linspace(0.0001, 0.0045, num_points)]  # DoH scale
bcm = multi_blob_count(path, add_constrast=True, tlog=x)
print(bcm)

plt.plot(x[0], bcm[0], '.-')
plt.plot(x[1], bcm[1], '.-')
plt.plot(x[2], bcm[2], '.-')
plt.show()

#%%
# Count all blobs in the training set
from util import beep, progress_bar, Curve

def df_blob_count(X: DataFrame, ts: list, add_constrast=False) -> dict:
	"""Utility function that count blobs for several images in a Pandas
		DataFrame. Don't forget the order of thresholds (ts) lists:
		LoG, DoG, DoH.
	"""
	bcmm = []
	k, kk = 0, len(X)
	progress_bar(k, kk)
	for _, row in X.iterrows():
		path = os.path.join("output", row["folder"], row["name"])
		bcm = multi_blob_count(path, tlog=ts, add_constrast=add_constrast)
		bcmm.append(bcm)
		k += 1
		progress_bar(k, kk)
	print(f"\n-> {len(X)} instances for target={t}")

	bcd = {}
	bcmm = np.array(bcmm)
	means = bcmm.mean(axis=0)
	stds = bcmm.std(axis=0)
	labels = ["LoG", "DoG", "DoH"]
	for x, y, e, l in zip(ts, means, stds, labels):
		bcd[l] = dict(x=x, y=y, e=e)
	return bcd

# X = X_train.iloc[:30]
# y = y_train.iloc[:30]
X = X_train
y = y_train
num_points = 20
x_thr = [np.linspace(0.1, 0.3, num_points), # LoG scale
			np.linspace(0.05, 0.25, num_points), # DoG scale
	      np.linspace(0.0005, 0.02, num_points)] # DoH scale

bc_curves = [[0, 0] for _ in range(3)]
tnames = ["Negative", "Positive"]
colors = ["darkorange", "deepskyblue"]
for t, tn, c in zip([0, 1], tnames, colors):
	# Calculate the blob counts' mean and error for one target
	bcd = df_blob_count(X[y==t], ts=x_thr, add_constrast=True)
	# Create Curve's from data dictionaries to plot them later
	for i, key in enumerate(bcd):
		bc_curves[i][t] = Curve(bcd[key], color=c, label=tn)
	beep()

#%%
# Plot all curves

def plot_blob_curves(curves: list[Curve], ax: plt.Axes):
	for c in curves: c.plot_error(ax)
	for c in curves: c.plot_curve(ax)
	ax.set_xlim(curves[0]._x[[0,-1]])
	ax.set_yticks(range(0, 31, 10))
	ax.legend()


titles = ["Laplacian of Gaussian",
			 "Difference of Gaussian",
			 "Determinant of Hessian"]

_, axs = plt.subplots(3, 1, figsize=(6, 8))
for curves, title, ax in zip(bc_curves, titles, axs):
	assert isinstance(ax, plt.Axes)
	plot_blob_curves(curves, ax)
	ax.set_title(title)
	ax.set_ylim(0, 30)
	ax.grid()

axs[1].set_ylabel("Average Blob Count")
axs[2].set_xlabel("Threshold")
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
