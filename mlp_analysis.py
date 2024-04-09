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
		and threshold arrays for each one.
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
plt.show()
plt.plot(x[2], bcm[2], '.-')
plt.show()

#%%
# Count all blobs in the training set
from util import beep

def count_blobs_for_class(X, ts: list, add_constrast=False):
	"""Utility function that count blobs for several images in a Pandas
		DataFrame. Don't forget about the thresholds (ts) lists' order:
		LoG, DoG, DoH.
	"""
	bcmm = []
	k, kk = 0, 0
	print(end='0')
	for _, row in X.iterrows():
		path = os.path.join("output", row["folder"], row["name"])
		bcm = multi_blob_count(path, tlog=ts, add_constrast=add_constrast)
		bcmm.append(bcm)
		k += 1
		if k == 10:
			k, kk = 0, kk+1
			print(end=str(kk))

	bcmm = np.array(bcmm)
	print(f" -> {len(X)} instances for target={c}")
	return bcmm.mean(axis=0), bcmm.std(axis=0)


X = X_train #.iloc[:30]
y = y_train #.iloc[:30]
num_points = 16
x_thr = [np.linspace(0.1, 0.2, num_points), # LoG scale
			np.linspace(0.05, 0.15, num_points), # DoG scale
	      np.linspace(0.2e-3, 3e-3, num_points)] # DoH scale
# x_thr[1] = x_thr[0]

bcm_me = []
for c in [0, 1]:
	# Calculate the blob counts' mean and error for target class c
	bcme = count_blobs_for_class(X[y==c], ts=x_thr, add_constrast=True)
	bcm_me.append(bcme)
	print(bcme)
	beep()

# neg_cnt, neg_err = bcm_me[0]
# pos_cnt, pos_err = bcm_me[1]

#%%
def plot_werror(xyel: list):
	"""Plot one or more x, y, error, label sets.
	"""
	colors = ["dodgerblue", "darkorange"]
	for (x, y, e, l), c in zip(xyel, colors):
		y1, y2 = y-e, y+e
		plt.fill_between(x, y1, y2, color=c, alpha=0.25)
		plt.plot(x, y1, color=c, alpha=0.4)
		plt.plot(x, y2, color=c, alpha=0.4)
		
	for (x, y, e, l), c in zip(xyel, colors):
		plt.plot(x, y, color=c, label=l)
	
	# plt.xscale("log")
	plt.xlabel("Threshold")
	plt.ylabel("Average Blob Count")
	plt.legend()

titles = ["Laplacian of Gaussian",
			 "Difference of Gaussian",
			 "Determinant of Hessian"]
labels = ["LoG", "DoG", "DoH"]
for x, yn, en, yp, ep, title, l in zip(x_thr,
														 bcm_me[0][0], bcm_me[0][1],
														 bcm_me[1][0], bcm_me[1][1],
														 titles, labels):
	plot_werror([
		(x, yp, ep, l+" Positive"),
		(x, yn, en, l+" Negative"),
		])
	plt.xlim(x[0], x[-1])
	plt.title(title)
	plt.show()
# beep()

#%%
# Blob counts for negative and positive images seem to intersect no matter what.
# So we'll look for the least intersection.
# from util import overlap

intersection_curves = []
for x, yn, en, yp, ep, title, l in zip(x_thr,
														 bcm_me[0][0], bcm_me[0][1],
														 bcm_me[1][0], bcm_me[1][1],
														 titles, labels):
	intersection_curves.append([
		overlap(a0, a1, b0, b1) for a0, a1, b0, b1 in
			zip(yn-en, yn+en, yp-ep, yp+ep)
		])

#%%
plt.plot(x_thr[0], intersection_curves[0], "ro-", label="LoG", ms=5)
plt.plot(x_thr[1], intersection_curves[1], "gs-", label="DoG", ms=5)

plt.legend()
plt.xlabel("Threshold")
plt.ylabel("Positive-Negative Counts Overlap")
plt.title("Laplacian/Difference of Gaussian")
plt.show()

# Plot the DoH curve alone since it has a very different scale
plt.plot(x_thr[2], intersection_curves[2], "b^-")
plt.xlabel("Threshold")
plt.ylabel("Positive-Negative Counts Overlap")
plt.title("Determinant of Hessian")
# %%
""