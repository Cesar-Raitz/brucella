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
import matplotlib.pyplot as plt
import numpy as np
import skimage

if __name__ == "__main__":
	test_folder = "test_images"
	test_names = [
		"02-03-23_positive2_90min.jpg",
		"03-07-23_negative1_18min.jpg",
		"15-06-23_negative1_82min.jpg",
		"30-06-23_positive2_13min.jpg",
		"03-07-23_negative1_15min.jpg",
		"03-07-23_negative1_12min.jpg"
	]

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
					correct_angle=True, show=False) -> dict:
	"""Open an image and crops the part containing the optical fiber.

		This function assumes that the wire is vertically aligned within the image
		and that red is the main channel for finding the optical fiber's center 
		position.

		Parameters
		----------
		  * image_path - Location of the image file or the image data.
		  * fiber_length - Default is the image's height (maximum length)
		  * fiber_diameter - Default is 0, where the diameter will be 1.5&times;FWHM of the pixel intensity distribution along the transverse direction
		  * correct_angle - Default is True, to detect the fibers' angle and rotate the image to align the fiber to the vertical direction (small angles only)
		  * show - Shows the original and cropped images along with the cropping informations

		Returns
		-------
		A dictionary with the following keys:
		  * name - Name of the image file
		  * folder - Folder name (last level)
		  * title - Folder name + file name
		  * cropped - The cropped image
		  * warnings - Possible image problems
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
	def test_crop_image(i: int=None) -> dict:
		global test_folder, test_names
		if isinstance(i, int) and 0 <= i < len(test_names):
			img_path = os.path.join(test_folder, test_names[i])
			return crop_image(img_path, fiber_length=700, show=True)
		else:
			for name in test_names:
				img_path = os.path.join(test_folder, name)
				d = crop_image(test_folder + name, show=True)
			return d

	d = test_crop_image(0)
	print(d.keys())


#%%
def get_hsv_means(image_path: dict|str|np.ndarray, bins=5, show=False) -> tuple:
	"""
		Return a tuple with mean values for the HSV channels.
		
		If `image_path` is a dictionary, use the RGB array of the 'cropped' key, also add the keys 'mean_h', 'mean_s', and 'mean_v' for the mean values.

		Parameters
		----------
		  * image_path - Location of the image file or the image data or the dictionary containing the cropped image.
		  * bins - Number of bins to generate the histogram, the counts will be divided by the number of pixels in the image, and added to the dictionary (if there's one) under the 'hX' keys.
		  * show - Wether to show an image summary for the procedure.
		
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
					 log_sigma=5, log_thres=0.15,
					 dog_sigma=5, dog_thres=0.10,
					 doh_sigma=5, doh_thres=0.05,
					 show=False) -> tuple:
	"""Count the blob number using LoG, DoG, and DoH techniques.

		In the case `image_path` is a dictionary, the image in the 'cropped' key
		is used for analysis.
		
		Written with the help of Bianca Tieppo.
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
	# img_morph = area_closing(area_opening(gray < 0.4, 5), 5)
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
	
	return num_blobs


if __name__ == "__main__":
	d = crop_image(os.path.join(test_folder, test_names[0]),
						fiber_length=700, fiber_diameter=36)
	print( count_blobs(d, show=True) )


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

def __info_from_path(path: str) -> dict:
	path = path.lower()
	for cls in ["positive", "negative", "NA"]:   # find the class (pos/neg)
		if cls in path: break
	r1 = __re_min.search(path)                    # find the measurement time
	r2 = __re_date.search(path)                   # find the measurement date
	time = int(r1.group(1)) if r1 else 0
	date = r2.group() if r2 else "NA"
	return {"class": cls, "time": time, "date": date}
	

def process_folder(search_folder: str, fiber_length=0, fiber_diameter=0, bins=5,
						 output_folder="output", as_frame=True) \
						-> DataFrame | list[dict]:
	"""
		Read and process all images within a folder and subfolders. Return a list
		with dictionaries containing extracted information for each image to apply
		Machine Learning algorithms.

		Parameters
		----------
		  * folder - Base folder where to search for images.
		  * fiber_length - Length to be cropped within the image.
		  * fiber_diameter - Diameter to be cropped within the image.
		  * bins - Number of bins to generate the histogram for the Value channel.
		  * output_folder - The folder where to save the cropped images ("output"
			 by default). Already existing images will be skipped. If empty, the
			 input images will be processed but the resulting cropped images won't
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
			info = crop_image(path, fiber_length, fiber_diameter)
			_ = get_hsv_means(info, bins)
			_ = count_blobs(info)
			
			info.update(__info_from_path(path))

			# Save the cropped image
			if output_folder:
				if not os.path.isdir(out_folder): os.mkdir(out_folder)
				cropped = skimage.util.img_as_ubyte(info["cropped"])
				skimage.io.imsave(out_path, cropped, quality=95, check_contrast=False)
				# Todo: Check if the quality keyword fails for .PNG files
			
			if not output_folder:
				del info["cropped"]
			info_list.append(info)
		end_text = f"\b{j+1} images"
		if skipped: end_text += f" ({skipped} skipped)"
		print(end_text)
	
	print("Done!")
	if as_frame: return DataFrame(info_list)
	return info_list


if __name__ == "__main__":
	folder = "test_images"
	df = process_folder(folder, fiber_diameter=50, output_folder="")
	df.head()  # won't show


#%%
def __stack_images(img_list: list, vpad=10) -> tuple[np.ndarray, list]:
	max_width = max([i.shape[1] for i in img_list])
	white_stripe = np.ones((vpad, max_width, 3), dtype='float64')
	new_list = []   # list of aligned images to be stacked
	pos_list = []   # list of images' vertical position
	y = 0           # current image's top position

	for j, img in enumerate(img_list):
		h, w, _ = img.shape
		if w < max_width:
			# Pad the current image to align horizontally
			hpad = np.ones((h, max_width-w, 3), dtype='float64')
			img = np.hstack([hpad, img])
		
		# Add the padded image to the list
		new_list.append(img)
		new_list.append(white_stripe)
		pos_list.append(y + h//2)
		y += h + vpad

	summary_img = np.vstack(new_list[:-1])
	return summary_img, pos_list


def folders_summary(cropped_df: DataFrame, summary_folder="summaries",
						  output_folder="output", dpi=100):
	"""
		Generate a summary of cropped images for each folder.
		
		If the cropped column is not available in the DataFrame, this funtion
		will read the images from the `output_folder` ("output" by default).
		The summary images are saved in the `summary_folder`
		("summaries" by default).
	"""
	assert isinstance(cropped_df, DataFrame)
	grouped = cropped_df.groupby("folder")
	for folder, group in grouped:
		if summary_folder:
			out_path = os.path.join(summary_folder,
				folder.replace(' ', '_') + "_summary.jpg")
			
			if os.path.exists(out_path):
				print("○", out_path, "already exists!")
				continue

		# GENERATE AN IMAGE STACK AND PLOT
		sorted = group.sort_values("time")
		if "cropped" in cropped_df:
			img_list = sorted["cropped"]
		else:
			img_list = []
			for title in sorted["title"]:
				img_list.append(skimage.io.imread(
					os.path.join(output_folder, title)
				) / 255.0)
		summary_img, pos_list = __stack_images(img_list)
		# fig = plt.figure(figsize=(6, len(img_list)*0.2))
		plt.imshow(summary_img)

		# Add legends for class and measurement time
		_, w, _ = summary_img.shape
		for y, (index, row) in zip(pos_list, sorted.iterrows()):
			total = sum(row[k] for k in ["blobs_log", "blobs_dog", "blobs_doh"])
			plt.text(w+5,   y, row["class"][0].upper(), va="center")
			plt.text(w+120, y, str(row["time"]), va="center", ha="right")
			plt.text(w+230, y, str(total), va="center", ha="right")
		
		plt.text(w+120, -3, "min", va="bottom", ha="right", weight="bold")
		plt.text(w+230, -3, "blobs", va="bottom", ha="right", weight="bold")
		plt.title(f"Summary of /{folder}")
		plt.tight_layout()
		plt.axis("off")
		if summary_folder:
			if not os.path.isdir(summary_folder): os.mkdir(summary_folder)
			plt.savefig(out_path, bbox_inches="tight", pad_inches=0.1, dpi=dpi)
			print("•", out_path, "saved")
		plt.show()


if __name__ == "__main__":
	# folders_summary(df[df["folder"] == "01-03-23_positive1"], "")  # no saving
	# folders_summary(df[df["folder"] == "30-06-23_positive2"], "")  # no saving
	# folders_summary(df[df["folder"] == "15-06-23_negative1"], "")  # no saving
	folders_summary(df, "")  # don't save, just show
	# folders_summary(df, "summaries")

#%% MAIN
if __name__ == "__main__":
	def process_all_data():
		"Process images generated by Jéssica."
		folder = "data_jessica"
		df = process_folder(folder, fiber_length=700, fiber_diameter=36)
		# folders_summary(df, "summaries")
		df.to_excel("data_jessica.xlsx")
		df.to_csv("data_jessica.csv")
		print("Done!")
	
	# process_all_data()
 