import numpy
from scipy import stats, signal
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from matplotlib.pyplot import figure as Figure


def make_delays(matrix, delays = range(5), fill = 0):
	"""
	Makes a version of the a matrix with delays
	:param matrix:
	:param delays:
	:param fill:
	:return:
	"""
	if (matrix.ndim == 1):
		matrix = matrix[...,None]
	out = numpy.ones((matrix.shape[0], matrix.shape[1] * len(delays)),
					 dtype = matrix.dtype) * fill

	for i in range(len(delays)):
		delay = delays[i]
		start = i * matrix.shape[1]
		end = start + matrix.shape[1]

		if delay < 0:
			out[:delay, start:end] = matrix[-delay:, :]
		elif delay == 0:
			out[:, start:end] = matrix
		else:
			out[delay:, start:end] = matrix[:-delay, :]

	return out


def estimate_devonvolved_response(features, responses, delays, **kwargs):
	"""
	Uses voxelwise modelling to estimate the brain activity had it not been passed
	through the hemodynamic response function.
	:param features: 	[TR x voxel] brain activity
	:param responses: 	[TR x features] feature space used to estimate brain activity
	:param delays:		number of delays to use in VM
	:param kwargs: 		parameters to RidgeCV
	:return: [TR x voxels] estimated brain activity
	"""
	ridge = RidgeCV(**kwargs)
	ridge.fit(make_delays(features, delays), responses)
	mean_weights = ridge.coef_.reshape(delays, features.shape[1], -1).mean(0)
	return stats.zscore(numpy.dot(features, mean_weights))


def extract_state_space_positions(conditions, responses, num_PCs = 24):
	"""
	Find the low-dimensional state space corresponding to the conditions
	:param conditions:	[TR x dimensions] state variables
	:param responses:	[TR x voxels] brain activity
	:param num_PCs:		int, number of PCs to use for denoising
	:return: [TR x dimensions] projects of brain activity into the task-related subspace
	"""
	parameters = numpy.ones([conditions.shape[0], conditions.shape[1] + 1])
	parameters[:, :-1] = conditions
	covar = numpy.dot(parameters.transpose(), parameters)
	coeffs = numpy.dot(numpy.linalg.pinv(covar), numpy.dot(parameters.transpose(), responses))
	pca = PCA(num_PCs)
	pca.fit(responses)
	coeffs_denoised = numpy.dot(pca.components_.transpose(),
								numpy.dot(pca.components_,
										  coeffs.transpose()))
	# note QR decomposition can cause sign flips
	Q, _ = numpy.linalg.qr(coeffs_denoised, 'full')
	state_directions = Q[:, :conditions.shape[1]]
	positions = numpy.dot(responses, state_directions)
	return stats.zscore(positions)


def subsample_passive_data(data, selection):
	"""
	Subsamples the parts of the passive data that had the same stimuli as the attentive condition
	:param data: 		deconvoled passive data
	:param selection: 	binary indicator for each second that shows which section is used
	:return: subsampled data 
	"""
	upsampled = signal.resample(data, 7200)
	subsampled = upsampled[selection[0, 0]:selection[0, 1], :]
	for i in range(1, selection.shape[0]):
		subsampled = numpy.vstack((subsampled, upsampled[selection[i, 0]:selection[i, 1], :]))
	downsampled = signal.resample(subsampled, 900)
	return downsampled


def JSD(mean1, cov1, mean2, cov2, samples = 1000):
	"""
	Empirical calculation of jensen-shannon divergence between two MVNs
	:param mean1: 	mean of first MVN
	:param cov1: 	covariance of first MVN
	:param mean2: 	mean of second MVN
	:param cov2: 	covariance of second MVN
	:param samples: number of samples to take
	:return: JSD value
	"""
	samples1 = stats.multivariate_normal.rvs(mean1, cov1, samples)
	samples2 = stats.multivariate_normal.rvs(mean2, cov2, samples)

	pdf1 = lambda x: stats.multivariate_normal.pdf(x, mean1, cov1, allow_singular = True)
	pdf2 = lambda x: stats.multivariate_normal.pdf(x, mean2, cov2, allow_singular = True)
	pdfMean = lambda x: 0.5 * (pdf1(x) + pdf2(x))

	KLD1Mean = lambda x: (1.0 / samples) * numpy.sum(numpy.log2(pdf1(x) / pdfMean(x)))
	KLD2Mean = lambda x: (1.0 / samples) * numpy.sum(numpy.log2(pdf2(x) / pdfMean(x)))

	divergence = 0.5 * (KLD1Mean(samples1) + KLD2Mean(samples2))
	return divergence


def stimulus_contains_class(stimulus, is_in_class):
	"""
	At each TR, does the stimulus contain this class of objects?
	:param stimulus: 	features
	:param is_in_class: indicator of whether each feature is in the desired class
	:return: boolean presence of the class at each TR
	"""
	hasLabel = numpy.zeros([stimulus.shape[0]], dtype = bool)
	for i in range(stimulus.shape[0]):
		for j in range(stimulus.shape[1]):
			if stimulus[i, j] > 0 and is_in_class[j]:
				hasLabel[i] = True
	return hasLabel


def sort_by_condition(values, conditions, n_conditions = None):
	"""
	Separates the set of points in state space by the actual trial condition

	:param values:     		set of points in [state dim][time]
	:param conditions:  	conditions, unique int per condition, assumes [0, n_conditions]
	:param n_conditions:    number of unique conditions, if None is max(conditions) + 1
	"""

	if n_conditions == -1:
		n_conditions = int(numpy.max(conditions) + 1)
	conditions = conditions.astype(int)
	sorted_values = []

	if len(values.shape) > 1:
		for i in range(n_conditions):
			nPoints = numpy.sum(conditions == i);
			trial = numpy.zeros([values.shape[0], nPoints])
			sorted_values.append(trial)
		counts = numpy.zeros([n_conditions], dtype = int)
		for i in range(values.shape[1]):
			condition = conditions[i]
			sorted_values[condition][:, counts[condition]] = values[:, i]
			counts[condition] += 1
	else:
		for i in range(n_conditions):
			nPoints = numpy.sum(conditions == i)
			trial = numpy.zeros([nPoints])
			sorted_values.append(trial)
		counts = numpy.zeros([n_conditions], dtype = int)
		for i in range(values.shape[0]):
			condition = conditions[i]
			sorted_values[condition][counts[condition]] = values[i]
			counts[condition] += 1

	return sorted_values


def plot_state_space_projection(values, conditions, figure_size = (5, 5), alpha = 0.5):
	"""
	Plots things in 2D for a single attentional condition

	:param values:		[state dimension][time]
	:param conditions:	conditions points by actual condition? vect of ints, one per 4 conds
	:param colormap:	conditions to use per actual condition
	:param figure_size:	matplotlib figure size, useful when embedding into ipython notebook
	:param alpha:		alpha of the individual dots
	:return: reference to figure object
	"""
	figure = Figure(figsize = figure_size)
	axes = figure.add_subplot(111, aspect = 'equal')

	values_by_condition = sort_by_condition(values, conditions, 4)

	centroids = numpy.zeros([4, values.shape[0]])
	for i in range(4):
		centroids[i, :] = numpy.mean(values_by_condition[i], 1)

	# QR decomposition can flip signs on axes
	# check and flip back if needed
	x_sign = 1
	y_sign = 1
	if centroids[3, 0] < centroids[0, 0]:
		x_sign = -1
	if centroids[3, 1] < centroids[0, 1]:
		y_sign = -1

	axes.scatter(x_sign * values_by_condition[0][0, :], y_sign * values_by_condition[0][1, :], alpha = alpha,
			     marker = 'o', c = '#ff5100', edgecolor = 'none', lw = 0)
	axes.scatter(x_sign * values_by_condition[1][0, :], y_sign * values_by_condition[1][1, :], alpha = alpha,
			     marker = 'o', c = '#91d90d', edgecolor = 'none', lw = 0)
	axes.scatter(x_sign * values_by_condition[2][0, :], y_sign * values_by_condition[2][1, :], alpha = alpha,
			     marker = 'o', c = '#0084ff', edgecolor = 'none', lw = 0)
	axes.scatter(x_sign * values_by_condition[3][0, :], y_sign * values_by_condition[3][1, :], alpha = alpha,
			     marker = 'o', c = '#a200ff', edgecolor = 'none', lw = 0)

	for i in range(4):
		colors = {0: '#ff5100', 1: '#91d90d', 2: '#0084ff', 3: '#a200ff'}

		axes.scatter(x_sign * centroids[i, 0], y_sign * centroids[i, 1], alpha = 1,
				     c = "#000000", marker = '+', s = 700, linewidth = 8)
		axes.scatter(x_sign * centroids[i, 0], y_sign * centroids[i, 1], alpha = 1,
				     c = colors[i], marker = '+', s = 500, linewidth = 4)

	axes.set_xlabel('Human presence')
	axes.set_ylabel('Vehicle presence')

	axes.title.set_fontsize(16)
	axes.xaxis.label.set_fontsize(20)
	axes.yaxis.label.set_fontsize(20)
	axes.set_xlim([-4, 4])
	axes.set_ylim([-4, 4])
	for item in (axes.get_xticklabels() + axes.get_yticklabels()):
		item.set_fontsize(12)

	return figure