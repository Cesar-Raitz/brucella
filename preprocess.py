"""
	This module includes image preprocessing tools for optics based PoC
	(Point-of-Care) biosensors. Developed in colaboration with
	Supernano / UFRJ - Rio de Janeiro and
	Mackgraphe / Mackenzie - São Paulo.

	Author: Cesar Raitz Junior
	Creation: Feb-10-2024
	Licence: MIT

	Format: UTF-8 w/tabs
"""
__version__ = "1.00"

#%%
import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import skimage

if __name__ == "__main__":
	# Selected images to test cropping and feature extraction
	test_folder = "test_images"
	test_names = [
		"02-03-23_positive2_90min.jpg",
		"03-07-23_negative1_18min.jpg",
		"15-06-23_negative1_82min.jpg",
		"30-06-23_positive2_13min.jpg",
		"03-07-23_negative1_15min.jpg",
		"03-07-23_negative1_12min.jpg",
		"12_Positivo_gota_diluição 1 para 10_11.3 min.bmp",
		"4_Negativo_gota_diluição 1 para 25__10 min.bmp",
		"3_Positivo_gota_diluição 1 para 50_7.3 min.bmp",
		"3_Negativo_gota_diluição 1 para 100_7.3 min.bmp",
	]

# Default blob detection parameters
LOG_SIGMA = 3
DOG_SIGMA = 3
DOH_SIGMA = 4.3

LOG_THRES = 0.2
DOG_THRES = 0.15
DOH_THRES = 0.007

#%%
def read_image(path: str|np.ndarray) -> np.ndarray:
	"""If `path` is a string, read the image at this location and return the
		image data. Return `path` itself if it's a NumPy array. Raise
		`ValueError` otherwise.
	"""
	if isinstance(path, str):
		img = skimage.io.imread(path)
	elif isinstance(path, np.ndarray):
		img = path
	else:
		raise ValueError()
	return img


#%% Image cropping function
def __find_center(image: np.ndarray, warnings: set=None, ax: plt.Axes=None) \
	-> tuple[float, float]:
	""""""
	if warnings is None:
		warnings = set()
	profile = image.sum(axis=0)
	assert len(profile.shape) == 1

	# Find the abscissas where the distribution crosses from
	# one side to the other of half its amplitude
	half = (profile.max() + profile.min()) / 2
	x = np.where(np.diff(np.sign(profile-half)))[0]
	
	# Find the peak closest to the image's center (for edgy cases)
	x0 = len(profile) // 3
	center = profile[x0: 2*x0].argmax() + x0

	if len(x) > 2:
		# More than two crossings could mean image noise (like a flare artifact)
		warnings.add("flare")
		try:
			a = x[x < center][-1]  # last x below the center
			b = x[x > center][0]   # first x above the center
			x = np.array([a, b])
		except IndexError:
			x = []                 # let the next else deal with this
	
	if len(x) == 2:
		# Interpolate to find the closest abscissas
		tx = x + (half - profile[x]) / (profile[x+1] - profile[x])
		center = sum(tx) / 2
		width = tx[1] - tx[0]
	else:
		# Unable to find the distribution's half!
		warnings.add("no_fwhm")
		tx = [0, len(profile)]
		width = 30.0

	if ax is not None:
		ax.plot(profile)
		ax.plot(tx, [half]*2)
		if len(x) == 2:
			ax.plot(x, profile[x], 'r.')
			ax.plot(x+1, profile[x+1], 'b.')
	
	return center, width


def crop_image(image_path: str|np.ndarray,
					fiber_length=0, fiber_diameter=0,
					pre_rotate: np.ndarray=None,
					rescale: float=None,
					correct_angle=True, show=False) -> dict:
	"""Open an image and crops the part containing the optical fiber.

		This function assumes that the wire is vertically aligned within the image
		and that red is the main channel for finding the optical fiber's center 
		position.

		Parameters
		----------
		  * `image_path` - Location of the image file or the image data array.
		  * `fiber_length` - Default is the image's height (maximum length).
		  * `fiber_diameter` - Default is 1.5&times;FWHM of the pixel intensity distribution along the transverse direction.
		  * `pre_rotate` - Rotate the image before angle correction if two fiber endpoints are given (in px). Then the rotation angle and centerpoint are calculated, and the image is aligned vertically for cropping.
		  * `rescale` - A scaling factor to resize the image after rotation.
		  * `correct_angle` - Default is True, to detect the fibers' angle and rotate the image to align the fiber to the vertical direction (small angles only).
		  * `show` - Shows the original and cropped images along with the cropping info, for evaluation.

		Returns
		-------
		A dictionary with the following keys:
		  * `name` - Name of the image file
		  * `folder` - Folder name (last level)
		  * `title` - Folder name + file name
		  * `cropped` - The cropped image
		  * `warnings` - Possible image problems
	"""
	# CREATE A DICTIONARY FOR THE EXTRACTED DATA
	the_dict = dict(name="", folder="", title="")
	if isinstance(image_path, str):
		l = image_path.replace('/','\\').split('\\')
		the_dict["name"] = l[-1]
		if len(l) >= 2:
			the_dict["folder"] = l[-2]
			the_dict["title"] = '\\'.join(l[-2:])
		else:
			the_dict["title"] = l[-1]
	
	img = read_image(image_path)
	warnings = set()
	
	# Prepare a figure to plot the profiles
	if show:
		_, axs = plt.subplots(3 if correct_angle else 0, 1, figsize=(5, 6))
		axs[0].set_title(the_dict["title"])
	else:
		axs = [None]*3
	
	# PRE-ROTATE THE IMAGE
	if isinstance(pre_rotate, list):
		pre_rotate = np.array(pre_rotate)

	if isinstance(pre_rotate, np.ndarray):
		delta = pre_rotate[1] - pre_rotate[0]
		rot_center = (pre_rotate[0] + pre_rotate[1])/2
		rot_angle = np.rad2deg(np.arctan(delta[1]/delta[0]))
		img = skimage.transform.rotate(img, -rot_angle, center=rot_center)
	
	if isinstance(rescale, float):
		img = skimage.transform.rescale(img,
											 (rescale, rescale, 1),
											 anti_aliasing=False)

	# FIND THE FIBER'S CENTER POSITION
	red = img[:, :, 0]
	height = img.shape[0]
	xc, width = __find_center(red, warnings, axs[0])
	if fiber_length <= 0: fiber_length = height
	y0 = (height - fiber_length)//2
	y1 = y0 + fiber_length
	yc = (y0 + y1)/2
	
	if correct_angle:
		# Calculate the angle from the tips' abscissas
		x = (xc + width * np.array([-4.0, 4.0])).astype(int)
		precut = red[y0:y1, x[0]:x[1]]
		if show:
			axs[1].set_title("Upper Tip")
			axs[2].set_title("Lower Tip")
		xa, wa = __find_center(precut[:31],  warnings, axs[1])  # yA = y0+15
		xb, wb = __find_center(precut[-31:], warnings, axs[2])  # yB = y1-15
		
		xc = (xa + xb)/2 + x[0]      # average distribution center
		width = round((wa + wb)/2)   # average distribution width
		angle = np.rad2deg(np.arctan((xb-xa) / (fiber_length-30)))
		rotated = skimage.transform.rotate(img, -angle, center=(xc, yc))
	else:
		rotated = img
	
	# Show the distributions' figure
	if show:
		plt.tight_layout()
		plt.show()

	# CROP THE FIBER WIRE
	if fiber_diameter <= 0:
		fiber_diameter = int(width*1.5)
	x0 = int(xc - fiber_diameter/2)
	x1 = x0 + fiber_diameter
	cropped = np.rot90(rotated[y0:y1, x0:x1], 3)
	if (cropped >= 0.996).any():
		warnings.add("overflow")

	# Stringify the warnings
	warnings = ', '.join(warnings)

	# SHOW EACH STEP FOR TESTING
	if show:
		_, axs = plt.subplots(2, 1, figsize=(6, 5.5), height_ratios=[10, 1])
		axs[0].imshow(rotated)
		axs[0].plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], 'y')
		axs[0].set_title(the_dict["title"])
		
		# Display information text
		text = f"{fiber_length}x{fiber_diameter}\n{angle:.2f}°"
		text += '\n' + str(warnings)
		text += f"\nx∈[{x0}, {x1})"
		axs[0].text(0.1, 0.7, text, size=14, color='y',
						transform=axs[0].transAxes)
		
		with_constrast = skimage.exposure.rescale_intensity(cropped)
		axs[1].imshow(with_constrast)
		axs[1].set_title("Cropped Fiber")
		axs[1].set_axis_off()
		plt.tight_layout()
		plt.show()
	
	the_dict["cropped"] = cropped
	the_dict["warnings"] = warnings
	return the_dict


if __name__ == "__main__":
	def test_crop_image(i: int, **kwargs) -> dict:
		"""Helper function to crop an image.
		"""
		global test_folder, test_names
		assert 0 <= i < len(test_names)
		# Crop a single image and return its data in a dictionary
		img_path = os.path.join(test_folder, test_names[i])
		return crop_image(img_path, fiber_length=700, show=True,
								**kwargs)
	

	# d = test_crop_image(5)
	# d = test_crop_image(6, pre_rotate=[(598, 1178), (1509, 191)], rescale=0.5)
	d = test_crop_image(7, pre_rotate=[(280, 1066), (1176, 140)], rescale=0.5)
	# d = test_crop_image(8, pre_rotate=[(614, 1336), (1524, 380)], rescale=0.5)
	# d = test_crop_image(9, pre_rotate=[(590, 1308), (1554, 296)], rescale=0.5)
	print(d.keys())


#%%
def get_hsv_means(image_path: dict|str|np.ndarray, bins=5, show=False) -> tuple:
	"""
		Return a tuple with mean values for the HSV channels.
		
		If `image_path` is a dictionary, use the RGB array of the 'cropped' key, also add the keys 'mean_h', 'mean_s', and 'mean_v' for the mean values.

		Parameters
		----------
		  * `image_path` - Location of the image file or the image data or the
		                   dictionary containing the cropped image.
		  * `bins` - Number of bins to generate the histogram, the counts will be
		             divided by the number of pixels in the image, and added to
						 the dictionary (if there's one) under the 'hX' keys.
		  * `show` - Wether to show an image summary for the procedure.
		
		Returns
		-------
			A tuple with the mean value of each channel (HSV).
	"""
	if isinstance(image_path, dict):
		the_dict = image_path
		img = the_dict["cropped"]
	else:
		img = read_image(image_path)
		the_dict = None

	# Calculate the mean value for the HSV channels
	hsv_img = skimage.color.rgb2hsv(img)
	images = [hsv_img[:,:,i] for i in range(3)]
	hsv_mean = tuple(np.mean(img) for img in images)
	
	if bins > 0:
		# Generate a histogram for the Value channel
		counts, hist_bins = np.histogram(images[2], bins)
		# Normalize counts because the cropped images may be of different sizes
		h, w, _ = img.shape
		counts = counts / (w*h)

	# Save these info into the dictionary
	if the_dict is not None:
		the_dict.update({
			"mean_" + channel: value \
				for channel, value in zip("hsv", hsv_mean)
		})
		if bins > 0:
			the_dict.update({
				f"h{i+1}": value for i, value in enumerate(counts)
			})
	
	if show:
		# Plot each channel's image
		if bins <= 0:
			_, axs = plt.subplots(3, 1, figsize=(6, 2))
		else:
			_, axs = plt.subplots(4, 1, figsize=(6, 5), height_ratios=[1,1,1,5])
		
		ch_names = ["Hue", "Saturation", "Value"]
		for ax, img, ch_name, m in zip(axs, images, ch_names, hsv_mean):
			assert isinstance(ax, plt.Axes)
			ax.imshow(img, cmap='gray')
			ax.set_title(f"{ch_name} ({m:.2f})")
			ax.set_axis_off()
		
		if the_dict is not None:
			plt.suptitle(the_dict["title"])
		
		# If there's an histogram, plot it on the fourth axes
		if bins > 0:
			axs[3].hist(hist_bins[:-1], hist_bins, weights=counts, edgecolor="black")
			axs[3].set_position([0.125, 0.11, 0.77, 0.4])
			axs[3].set_title("Value Histogram")
		
		if bins == 0: plt.subplots_adjust(hspace=0, bottom=0, top=0.8)
		plt.show()

	return hsv_mean


if __name__ == "__main__":
	# d = crop_image(os.path.join(test_folder, test_names[0]),
	# 				   fiber_length=700, fiber_diameter=36)
	print(get_hsv_means(d, bins=5, show=True))
	print(d.keys())


#%%
from skimage.color import rgb2gray
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.morphology import area_closing, area_opening


def count_blobs(image_path: dict|str|np.ndarray,
					 log_sigma=LOG_SIGMA, log_thres=LOG_THRES,
					 dog_sigma=DOG_SIGMA, dog_thres=DOG_THRES,
					 doh_sigma=DOH_SIGMA, doh_thres=DOH_THRES,
					 return_blobs=False,
					 show=False) \
					 -> tuple | list:
	"""Count the number of blobs using LoG, DoG, and DoH techniques.

		Written with the help of Bianca Tieppo.

		Parameters
		----------
		  * `image_path` - The path to read the image file or a dictionary
					containing the RGB image array within the key "cropped".

		  * `return_blobs` - True if the desired return value is the three arrays
		  			containing the blob coordinates (x, y, r) for each technique.
		
		Returns
		-------
		  A tuple containing the blob count for each technique OR a list
		  containing the blob coordinates for each technique.
	"""
	if isinstance(image_path, dict):
		the_dict = image_path
		img = the_dict["cropped"]
	else:
		img = read_image(image_path)
		the_dict = None

	# Convert the original image to grayscale and bitmap images
	gray = rgb2gray(img)
	# Apply opening followed by closing filters
	# (removes small bright and dark spots less than 5)
	# img_morph = area_closing(area_opening(gray > 0.4, 5), 5)
	img_morph = gray

	# Detect blobs using the three methods
	blobs_log = blob_log(img_morph, max_sigma=log_sigma, threshold=log_thres)
	blobs_dog = blob_dog(img_morph, max_sigma=dog_sigma, threshold=dog_thres)
	blobs_doh = blob_doh(img_morph, max_sigma=doh_sigma, threshold=doh_thres)
	all_blobs = [blobs_log, blobs_dog, blobs_doh]
	# Compute the radii
	blobs_log[:, 2] *= np.sqrt(2)
	blobs_dog[:, 2] *= np.sqrt(2)
	# Count the number of blobs for each method
	num_blobs = tuple(len(b) for b in all_blobs)
	
	if show:
		_, axs = plt.subplots(5, 1, figsize=(6, 2), sharex=True)
		yt, xt = gray.shape
		yt, xt = yt/2, xt+5

		# Plot bitmap and morphed images
		axs[0].imshow(img)
		axs[1].imshow(img_morph, cmap="gray")
		axs[0].text(xt, yt, "Original", va="center")
		axs[1].text(xt, yt, "Filtered", va="center")

		# Plot the blobs found for each method
		colors = ["yellow", "lime", "magenta"]
		titles = ["LoG", "DoG", "DoH"]

		for ax, blobs, blob_count, color, title in \
			zip(axs[2:], all_blobs, num_blobs, colors, titles):
			assert isinstance(ax, plt.Axes)
			
			# Draw the image along with the blobs
			ax.imshow(gray, cmap="gray")
			for blob in blobs:
				y, x, r = blob
				c = plt.Circle((x, y), r, color=color, lw=1.2, fill=False)
				ax.add_patch(c)
			
			t = f"{title} ({blob_count} blobs)"
			ax.text(xt, yt, t, va="center")

		for ax in axs: ax.set_axis_off()

		if the_dict is not None:
			plt.suptitle(the_dict["title"])

		plt.show()

	# Add the blob counts to the dictionary
	if the_dict is not None:
		names = ["blobs_log", "blobs_dog", "blobs_doh"]
		the_dict.update({
			name: num for name, num in zip(names, num_blobs)
			})
	
	if return_blobs:
		return all_blobs
	return num_blobs


if __name__ == "__main__":
	img_path = os.path.join(test_folder, test_names[0])
	d = crop_image(img_path, fiber_length=700, fiber_diameter=36)
	
	# num_blobs = count_blobs(d, show=True,
	# 		log_sigma=3, dog_sigma=3, doh_sigma=5,
	# 		log_thres=0.2, dog_thres=0.15, doh_thres=0.007)
	num_blobs = count_blobs(d, show=True)
	print(num_blobs)


#%%
import re
import os
from pandas import DataFrame


def __is_image(file_name: str) -> bool:
	extensions = [".bmp", ".jpg"]
	for ext in extensions:
		if file_name.endswith(ext):
			return True
	return False


__re_min = re.compile(r"(\d+)\s?min")
__re_date = re.compile(r"\d{1,2}-\d{1,2}-\d{1,2}")
__re_dilution = re.compile(r"(\d+)\spara\s(\d+)")

def __info_from_path(path: str) -> dict:
	path = path.lower()
	for cls in ["positive", "negative", "NA"]:   # find the class (pos/neg)
		if cls in path: break
	r1 = __re_min.search(path)           # find the measurement time
	r2 = __re_date.search(path)          # find the measurement date
	r3 = __re_dilution.search(path)      # find the solution's dilution
	time = int(r1.group(1)) if r1 else 0
	date = r2.group() if r2 else "NA"
	dilution = ':'.join(r3.groups()) if r3 else "NA"
	return {"class": cls, "time": time, "date": date, "dilution": dilution}
	

def process_folder(search_folder: str,
						 bins=5, output_folder="output", as_frame=True,
						 **kwargs) \
						-> DataFrame | list[dict]:
	"""
		Read and process all images within a folder and subfolders. Return a list
		with dictionaries containing extracted information for each image to apply
		Machine Learning algorithms.

		Parameters
		----------
		  * `folder` - Base folder where to search for images.
		  * `fiber_length` - Length to be cropped within the image.
		  * `fiber_diameter` - Diameter to be cropped within the image.
		  * `bins` - Number of bins to generate the histogram for the Value
		             channel.
		  * `output_folder` - The folder where the cropped images will be saved
		                      ("output" by default). Already existing images will
									 be skipped. If empty string, the input images will
									 be processed but the resulting cropped images won't
									 be saved.
	"""
	info_list = []
	for base, subdirs, files in os.walk(search_folder):
		if len(files) == 0: continue
		_, folder_name = os.path.split(base)
		out_folder = os.path.join(output_folder, folder_name)
		print("•", folder_name, end=" - .")

		# PROCESS ALL IMAGES IN THIS SUBFOLDER
		skipped = 0
		for j, img_name in enumerate(filter(__is_image, files)):
			if output_folder:
				# Prevent reprocessing images
				out_path = os.path.join(out_folder, img_name)
				if os.path.exists(out_path):
					skipped += 1
					continue

			print("\b", end="▀▌▄▐"[j%4])
			# Crop the image and extract information
			path = os.path.join(base, img_name)
			info = crop_image(path, **kwargs)
			_ = get_hsv_means(info, bins)
			_ = count_blobs(info)
			
			info.update(__info_from_path(path))

			# Save the cropped image
			if output_folder:
				if not os.path.isdir(out_folder): os.mkdir(out_folder)
				cropped = skimage.util.img_as_ubyte(info["cropped"])
				skimage.io.imsave(out_path, cropped, quality=95,
							 			check_contrast=False)
				# Todo: Check if the quality keyword fails for .PNG files
			
			if output_folder:
				del info["cropped"]
			info_list.append(info)
		end_text = f"\b{j+1} images"
		if skipped: end_text += f" ({skipped} skipped)"
		print(end_text)
	
	print("Done!")
	if as_frame: return DataFrame(info_list)
	return info_list


if __name__ == "__main__":
	f = "test_images"
	df = process_folder(f, fiber_diameter=50, output_folder="")
 
	# folders = [
	# 	("dilution_1_10",  [(598, 1178), (1509, 191)]),
	# 	("dilution_1_25",  [(280, 1066), (1176, 140)]),
	# 	("dilution_1_50",  [(614, 1336), (1524, 380)]),
	# 	("dilution_1_100", [(590, 1308), (1554, 296)]),
	# ]
	# f, pr = folders[0]
	# df = process_folder(os.path.join("data_wanderson", f),
	# 						fiber_length=700, fiber_diameter=36,
	# 						pre_rotate=pr, rescale=0.5)

	print(df.head())


#%%
def folders_summary(cropped_df: DataFrame, summary_folder="summaries",
						  output_folder="output", plot_blobs="log", dpi=100):
	"""
		Generate a summary of cropped images for each folder.

		Parameters
		----------
		  * `cropped_df` - A DataFrame containing a "folder" column,
		  			and a "title" column with the file names (without extension)
					or a "cropped" column with the images as NumPy arrays.

		  * `summary_folder` - Where to save the summary images.

		  * `output_folder` - Folder where the cropped images files will be read
					if the DataFrame does not have the "cropped" column. It is
					actually an "input" folder.

		  * `plot_blobs` - A string indicating which detection algorithm to
		  			use for plotting the blobs ("log", "dog" or "doh").
					If empty, no blobs will be plotted.

		  * `dpi` - The resolution to save the summary images.
	"""
	assert isinstance(cropped_df, DataFrame)
	grouped = cropped_df.groupby("folder")
	for folder, group in grouped:
		if summary_folder:
			name = folder.replace(' ', '_') + "_summary.jpg"
			out_path = os.path.join(summary_folder, name)
			
			if os.path.exists(out_path):
				print("○", out_path, "already exists!")
				continue

		# GENERATE AN IMAGE STACK AND PLOT
		sorted_df = group.sort_values("time")
		#TODO Can we sort by class first?
		
		num = len(sorted_df)
		_, axs = plt.subplots(num, 1, figsize=(6, num*0.3))
		col_x = (1.03, 1.12, 1.24)

		for (_, row), ax in zip(sorted_df.iterrows(), axs):
			# assert isinstance(ax, plt.Axes)
			if "cropped" in row:
				img = row["cropped"]
			else:
				path = os.path.join(output_folder, row["title"])
				img = skimage.io.imread(path) / 255.0
			
			ax.imshow(img)

			if plot_blobs:
				# The only problem here is that the parameters used here for blob
	 			# detection are default. They may be different from the parameters
	  			# used during the main processing.
				all_blobs = count_blobs(img, return_blobs=True)
				types = dict(log=0, dog=1, doh=2)
				if plot_blobs in types:
					for y, x, r in all_blobs[types[plot_blobs]]:
						c = plt.Circle((x, y), r, color="cyan", lw=1.2, fill=False)
						ax.add_patch(c)
			
			# Print the target class, measurement time and number of blobs
			num_blobs = row["blobs_log"] + row["blobs_dog"] + row["blobs_doh"]
			legend = [row["class"][0].upper(), row["time"], num_blobs]
			for x, text in zip(col_x, legend):
				ax.text(x, 0.5, text, va="center", ha="center",
				        transform=ax.transAxes)
			ax.axis("off")

		# Draw the column titles
		columns = ["C", "min", "blobs"]
		for x, text in zip(col_x, columns):
			first_ax = axs[0]
			first_ax.text(x, 1.15, text, weight="bold", ha="center",
					      transform=first_ax.transAxes)
		first_ax.set_title(f"Summary of /{folder}")

		if summary_folder:
			if not os.path.isdir(summary_folder): os.mkdir(summary_folder)
			plt.savefig(out_path, bbox_inches="tight", pad_inches=0.1, dpi=dpi)
			print("•", out_path, "saved")
		
		plt.show()


if __name__ == "__main__":
	# Generate a summary for all folders in the current df and no save
	folders_summary(df, "")
	
	# Generate summaries AND SAVE
	# folders_summary(df, "summaries")

#%%
# MAIN
#===============================================================================
if __name__ == "__main__":
	def process_all_data() -> DataFrame:
		"""Process images from experiments performed by Jéssica.
		"""
		folder = "data_jessica"
		df = process_folder(folder, fiber_length=700, fiber_diameter=36)
		# folders_summary(df, "summaries")
		df.to_excel("data_jessica.xlsx")
		df.to_csv("data_jessica.csv")
		print("Done!")
		return df
	
	# df_all = process_all_data()   # go grab a cup of coffee...

#%%
if __name__ == "__main__":
	# Read data obtained from experiments performed by Jéssica
	df_all = pandas.read_csv("data_jessica.csv", index_col=0)


#%%
# Generate summaries for processed data
if __name__ == "__main__":
	# Generate summaries without saving
	summary_no_save = lambda folder, pb: \
		folders_summary(
			df_all[df_all["folder"] == folder],
			summary_folder="", plot_blobs=pb
		)
	
	# summary_no_save("01-03-23_positive1", "log")
	# summary_no_save("30-06-23_positive2", "log")
	# summary_no_save("15-06-23_negative1", "log")

	folders_summary(df_all, "summaries_2")   # go grab a cup of coffee...


#%%
# Count blobs again with different parameters and save
if __name__ == "__main__":
	new_count = []
	for title in df_all["title"]:
		path = os.path.join("output", title)
		new_count.append(count_blobs(path))
	
	new_count = np.array(new_count).T
	df_all["blobs_log"] = new_count[0]
	df_all["blobs_dog"] = new_count[1]
	df_all["blobs_doh"] = new_count[2]

	# Save again
	csv_name = "data_jessica_2.csv"
	df_all.to_csv(csv_name)
	print(csv_name, "saved!")

#%% Process images from  Wanderson's tests
if __name__ == "__main__":
	folders = [
		("dilution_1_10",  [(598, 1178), (1509, 191)]),
		("dilution_1_25",  [(280, 1066), (1176, 140)]),
		("dilution_1_50",  [(614, 1336), (1524, 380)]),
		("dilution_1_100", [(590, 1308), (1554, 296)]),
	]

	info_list = []
	for f, pr in folders:
		info_list.extend(
			process_folder(os.path.join("data_wanderson", f), as_frame=False,
						fiber_length=700, fiber_diameter=36,
						pre_rotate=pr, rescale=0.5)
		)
	df = DataFrame(info_list)
	df.to_csv("data_wanderson.csv")

#%%
folders_summary(df, "summaries_2")