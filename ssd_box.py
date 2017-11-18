from __future__ import division
import numpy as np
idx = 1
def iou(boxes1, boxes2, coords='centroids'):
	
	# Estimate the intersection over union of two bounding boxes
	### Arguments ###
	# boxes1, boxes2: 1-D array with 4 elements. ['xmin', 'xmax', 'ymin', 'ymax'] or ['xcenter', 'ycenter', 'width', 'height']
	### Output ###
	# 1-D array with size of number of boxes
	
	if len(boxes1.shape) > 2 or len(boxes2.shape) > 2:
		raise ValueError('all box shape should be less than 2, but get {}, {}'.format(box1.shape, box2.shape))

	if len(boxes1.shape) == 1:
		boxes1 = np.expand_dims(boxes1, axis = 0)

	if len(boxes2.shape) == 1:
		boxes2 = np.expand_dims(boxes2, axis = 0)

	if not boxes1.shape[1] == boxes2.shape[1]:
		raise ValueError('all boxes should contain 4 values but get {}, {}'.format(boxes1.shape[1], boxes2.shape[1]))

	if coords == 'centroids':
		boxes1 = convert_coordinates(boxes1, start_index=0, conversion='centroids2minmax')
		boxes2 = convert_coordinates(boxes2, start_index=0, conversion='centroids2minmax')
	elif coords != 'minmax':
		return ValueError('only support centroids or minmax but get'.format(coords))
	intersection = np.maximum(0, np.minimum(boxes1[:,1], boxes2[:,1]) - np.maximum(boxes1[:,0], boxes2[:, 0]))
	intersection *= np.maximum(0, np.minimum(boxes1[:,3], boxes2[:,3]) - np.maximum(boxes1[:,2], boxes2[:, 2]))
	union = (boxes1[:, 1] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 2])
	union += (boxes2[:, 1] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 2]) - intersection
	return intersection / union

def convert_coordinates(tensor, start_index, conversion='centroids2minmax'):
	
	# convert coordinate from center to minmax or from minmax to center.
	### Arguments ###
	# tensor: A N-D array, where tensor.shape[0] contains the number of boxes
	# start_index: start index in the last dimension of the tensor for conversion. tensor[..., start_index:start_index+4] should contain the coordinates of each box.	
	### Output ###
	# A N-D array with converted coordinate

	ind = start_index
	output = np.copy(tensor).astype(np.float)

	if conversion == 'centroids2minmax':
		output[..., ind] = tensor[..., ind] - tensor[..., ind + 2] / 2.0
		output[..., ind + 1] = tensor[..., ind] + tensor[..., ind + 2] / 2.0
		output[..., ind + 2] = tensor[..., ind + 1] - tensor[..., ind + 3] / 2.0
		output[..., ind + 3] = tensor[..., ind + 1] + tensor[..., ind + 3] / 2.0
	elif conversion == 'minmax2centroids':
		output[..., ind] = (tensor[..., ind] + tensor[..., ind + 1]) / 2.0
		output[..., ind + 1] = (tensor[..., ind + 2] + tensor[..., ind + 3]) / 2.0
		output[..., ind + 2] = tensor[..., ind + 1] - tensor[..., ind]
		output[..., ind + 3] = tensor[..., ind + 3] - tensor[..., ind + 2]
	else:
		raise ValueError("conversion can only be centroids2minmax or minmax2centroids, but got {}".format(conversion))
	return output

def _greedy_nms(prediction, iou_threshold=0.45, coords='minmax'):
	
	# nom-maximum suppresion.
	### Arguments ###
	# prediction: A N-D array with shape of (nboxes, 6). The first two elements of the last dimension is the class confidence, and the last four elements are the coordiante of the each boxe
	# iou_threshold: The minimum intersection over union of the maximum box with remaining boxes. Each box with larger than iou_threshold would be left for next iteration
	### Output ###
	# similart to prediction, but with less boxes
	boxes_left = np.copy(prediction)
	maxima = []
	while boxes_left.shape[0] > 0:
		maximum_index = np.argmax(boxes_left[:,  0])
		maximum_box = np.copy(boxes_left[maximum_index])
		if boxes_left.shape[0] == 0:
			break
		similarities = iou(boxes_left[:, 1:], maximum_box[1:], coords = coords)
		boxes_left = boxes_left[similarities <= iou_threshold]
	return np.array(maxima) 


def decode_y(y_pred,
             confidence_thresh=0.01,
             iou_threshold=0.45,
             top_k=200,
             input_coords='centroids',
             normalize_coords=False,
             img_height=None,
             img_width=None):

	# decode bounding boxes. For more details, please refer to the original paper.	
	### Arguments ###
	# y_pred: a tensor with shape (nboxes, 14). The first two elements in the last dimension are class confidence.
	# The last 12 elements are shift coordinates, default coordinate and variances.
	# confidence_thresh: [0, 1], the minimum confidence for each positive boxes.
	# iou_threshold: [0, 1]. Used for _greedy_nms.
	# top_k: an integer, the number of bounding boxes left in the end.
	# input_coords: a string, "centroids" or "minmax"
	# normalize_coords: True or False. Whether the coordinates in y_pred are normalized
	# img_height, img_width: the height and width of the original image
	### Output ###
	# decoded prediction. A array with shape (nboxes, 5), the las dimension contains (confidence of foreground) + 4 coordinates
	if normalize_coords and (img_height == None or img_width == None):
		raise ValueError('needs image height and image width to reconstruct the coordinates, but get {}, {}'.format(img_height, img_width))
	#print "======================================================================="
	y_pred_decoded_raw = np.copy(y_pred[...,:-8])
	if input_coords == 'centroids':
		y_pred_decoded_raw[..., [-2, -1]] = np.exp(y_pred_decoded_raw[...,[-2, -1]] * y_pred[..., [-2, -1]])
		y_pred_decoded_raw[..., [-2, -1]] *= y_pred[..., [-6, -5]]
		y_pred_decoded_raw[..., [-4, -3]] *= y_pred[..., [-4, -3]] * y_pred[..., [-6, -5]]
		y_pred_decoded_raw[..., [-4, -3]] += y_pred[...,[-8,-7]]
		y_pred_decoded_raw = convert_coordinates(y_pred_decoded_raw, start_index = -4, conversion = 'centroids2minmax')
	elif input_coords == 'minmax':
		y_pred_decoded_raw[..., -4:] *= y_pred[..., -4:]
		y_pred_decoded_raw[..., [-4, -3]] *= np.expand_dims(y_pred[..., -7] - y_pred[..., -8], axis = -1)
		y_pred_decoded_raw[..., [-2, -1]] *= np.expand_dims(y_pred[..., -5] - y_pred[..., -6], axis = -1)
		y_pred_decoded_raw[..., -4:] += y_pred[..., -8:-4]
	else:
		raise ValueError("input coords should be centroids or minmax, but get {}".format(input_coords))
	if normalize_coords:
		y_pred_decoded_raw[..., [-4, -3]] *= img_width
		y_pred_decoded_raw[..., [-2, -1]] *= img_height

	y_pred_decoded = []
	for batch_item in y_pred_decoded_raw:
		single_class = batch_item[:, 1:]
		threshold_met = single_class[single_class[:, 0] > confidence_thresh]
		if threshold_met.shape[0] > 0:
			max_idx = np.argmax(threshold_met[:, 0]) 
			pred = threshold_met[max_idx]
		else:
			pred = np.array([0, 0.0, 0, 0.0, 0.0])
		y_pred_decoded.append(pred)
	return y_pred_decoded

class SSDBoxEncoder:

	def __init__(
		self,
		img_height,
		img_width,
		n_classes,
		predictor_sizes,
		scales=None,
		aspect_ratios=[0.5, 0.1, 2.0],
		two_boxes_for_ar1=True,
		limit_boxes=True,
		variances=[1.0, 1.0, 1.0, 1.0],
		pos_iou_threshold=0.5,
		neg_iou_threshold=0.3,
		coords='centroids',
		normalize_coords=False):

		predictor_sizes = np.array(predictor_sizes)
		if len(predictor_sizes.shape) == 1:
			predictor_sizes = np.expand_dims(predictor_sizes, axix = 0)

		if scales is None:
			raise ValueError('scales need to be specified.')

		if scales:
			if len(scales) != len(predictor_sizes) + 1:
				raise ValueError('scales must be None or length of predictor size + 1')
			scales = np.array(scales)

			if np.any(scales <= 0):
				raise ValueError('All scales must be larger than zero')

		else:
			if not 0 < min_scale <= max_scale:
				raise ValueError('it must be 0 < min_cale <= max_scale')


		
		if not aspect_ratios:
			raise ValueError("aspect_ratios cannot be None")
		if np.any(aspect_ratios <= 0):
			raise ValueError("all aspect ratios must be greater than zero")

		if len(variances) != 4:
			raise ValueError("4 variance values must be pased")

		variances = np.array(variances)

		if np.any(variances <= 0):
			raise ValueError("all variances must be > 0")

		if neg_iou_threshold > pos_iou_threshold:
			raise ValueError('neg iou threshold must be smaller than pos iou threshold')

		if not (coords == 'minmax' or coords == 'centroids'):
			raise ValueError('coords must be minmax or centroids')

		self.img_height = img_height
		self.img_width = img_width
		self.n_classes = n_classes
		self.predictor_sizes = predictor_sizes
		self.scales = scales
		self.aspect_ratios = aspect_ratios
		self.two_boxes_for_ar1 = two_boxes_for_ar1
		self.limit_boxes = limit_boxes
		self.variances = variances
		self.pos_iou_threshold = pos_iou_threshold
		self.neg_iou_threshold = neg_iou_threshold
		self.coords = coords
		self.normalize_coords = normalize_coords

		
		if (1 in aspect_ratios) & two_boxes_for_ar1:
			self.n_boxes = len(aspect_ratios) + 1
		else:
			self.n_boxes = len(aspect_ratios)

	def generate_anchor_boxes(
		self,
		batch_size,
		feature_map_size,
		aspect_ratios,
		this_scale,
		next_scale,
		diagnostics=False):

		aspect_ratios = np.sort(aspect_ratios)
		size = min(self.img_height, self.img_width)
		wh_list = []
		n_boxes = len(aspect_ratios)
		for ar in aspect_ratios:
			if (ar == 1) & self.two_boxes_for_ar1:
				w = this_scale * size * np.sqrt(ar)
				h = this_scale * size / np.sqrt(ar)
				wh_list.append((w, h))
				w = np.sqrt(this_scale * next_scale) * size * np.sqrt(ar)
				h = np.sqrt(this_scale * next_scale) * size / np.sqrt(ar)
				wh_list.append((w, h))
				n_boxes += 1
			else:
				w = this_scale * size * np.sqrt(ar)
				h = this_scale * size / np.sqrt(ar)
				wh_list.append((w,h))
		wh_list = np.array(wh_list)
		cell_height = self.img_height / feature_map_size[0]
		cell_width = self.img_width / feature_map_size[1]
		cx = np.linspace(cell_width * 0.5, self.img_width - cell_width * 0.5, feature_map_size[1])
		cy = np.linspace(cell_height * 0.5, self.img_height - cell_height * 0.5, feature_map_size[0])
		cx_grid, cy_grid = np.meshgrid(cx, cy)
		cx_grid = np.expand_dims(cx_grid, -1)
		cy_grid = np.expand_dims(cy_grid, -1)

		boxes_tensor = np.zeros((feature_map_size[0], feature_map_size[1], n_boxes, 4))
		boxes_tensor[...,0] = np.tile(cx_grid, (1, 1, n_boxes))
		boxes_tensor[...,1] = np.tile(cy_grid, (1, 1, n_boxes))
		boxes_tensor[...,2] = wh_list[:,0]
		boxes_tensor[...,3] = wh_list[:,1]
		boxes_tensor = convert_coordinates(boxes_tensor, start_index = 0, conversion = 'centroids2minmax')
		if self.limit_boxes:
			x_coords = boxes_tensor[:, :, :, [0, 1]]
			x_coords[x_coords >= self.img_width] = self.img_width - 1
			x_coords[x_coords < 0] = 0
			boxes_tensor[:, :, :,[0, 1]] = x_coords
			y_coords = boxes_tensor[:, :, :, [2, 3]]
			y_coords[y_coords >= self.img_height] = self.img_height - 1
			y_coords[y_coords < 0] = 0
			boxes_tensor[:, :, :, [2, 3]] = y_coords

		if self.normalize_coords:
			boxes_tensor[..., :2] /= self.img_width
			boxes_tensor[..., 2:] /= self.img_height
		if self.coords == 'centroids':
			boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='minmax2centroids')
		boxes_tensor = np.expand_dims(boxes_tensor, axis = 0)
		boxes_tensor = np.tile(boxes_tensor, (batch_size, 1, 1, 1, 1))
		boxes_tensor = np.reshape(boxes_tensor, (batch_size, -1, 4))
		if diagnostics:
			return boxes_tensor, wh_list, (int(cell_height), int(cell_width))
		else:
			return boxes_tensor

	def generate_encode_template(self, batch_size, diagnostics= False):

		if self.scales is None:
			self.scales = np.linspace(self.min_scale, self.max_scale, len(self.predictor_sizes) + 1)

		boxes_tensor = []
		if diagnostics:
			wh_list = []
			cell_sizes = []
			for i in range(len(self.predictor_sizes)):
				boxes, wh, cells = self.generate_anchor_boxes(
					batch_size=batch_size,
					feature_map_size=self.predictor_sizes[i],
					aspect_ratios=self.aspect_ratios,
					this_scale=self.scales[i],
					next_scale=self.scales[i+1],
					diagnostics=True)
				boxes_tensor.append(boxes)
				wh_list.append(wh)
				cell_sizes.append(cells)
		else:
			for i in range(len(self.predictor_sizes)):
				boxes_tensor.append(self.generate_anchor_boxes(
					batch_size=batch_size,
					feature_map_size=self.predictor_sizes[i],
					aspect_ratios=self.aspect_ratios,
					this_scale=self.scales[i],
					next_scale=self.scales[i+1],
					diagnostics=False))

		boxes_tensor = np.concatenate(boxes_tensor, axis=1)
		classes_tensor = np.zeros((batch_size, boxes_tensor.shape[1], self.n_classes))
		variances_tensor = np.zeros_like(boxes_tensor)
		variances_tensor += self.variances
		y_encode_template = np.concatenate((classes_tensor, boxes_tensor, boxes_tensor, variances_tensor), axis=2)
		if diagnostics:
			return y_encode_template, wh_list, cell_sizes
		else:
			return y_encode_template

	def encode_y(self, ground_truth_labels):

		y_encode_template = self.generate_encode_template(batch_size=len(ground_truth_labels), diagnostics=False)
		y_encoded = np.copy(y_encode_template)
		class_vector = np.eye(self.n_classes)
                for i in range(y_encode_template.shape[0]):
			available_boxes = np.ones((y_encode_template.shape[1]))
			negative_boxes = np.ones((y_encode_template.shape[1]))
			for true_box in ground_truth_labels[i]:
				true_box = true_box.astype(np.float)
				if abs(true_box[2] - true_box[1] < 5) or abs(true_box[4] - true_box[3] < 5):
					continue
				if self.normalize_coords:
					true_box[1:3] /= self.img_width
					true_box[3:5] /= self.img_height
				if self.coords == 'centroids':
					true_box = convert_coordinates(true_box, start_index = 1, conversion = 'minmax2centroids')
				similarities = iou(y_encode_template[i,:,-12:-8], true_box[1:], coords=self.coords)
				negative_boxes[similarities >= self.neg_iou_threshold] = 0
				similarities *= available_boxes
				available_and_thresh_met = np.copy(similarities)
				available_and_thresh_met[available_and_thresh_met < self.pos_iou_threshold] = 0
				assign_indices = np.nonzero(available_and_thresh_met)[0]
				flag = True
				if len(assign_indices) > 0:
					y_encoded[i, assign_indices, :-8] = np.concatenate((class_vector[int(true_box[0])], true_box[1:]), axis = 0)
					available_boxes[assign_indices] = 0
				else:
					best_match_index = np.argmax(similarities)
					y_encoded[i, best_match_index, :-8] = np.concatenate((class_vector[int(true_box[0])], true_box[1:]), axis=0)
					available_boxes[best_match_index] = 0
					negative_boxes[best_match_index] = 0
			background_class_indices = np.nonzero(negative_boxes)[0]
			y_encoded[i, background_class_indices, 0] = 1
		if self.coords == 'centroids':
			y_encoded[:,:,[-12, -11]] -= y_encode_template[:, :, [-12, -11]]
			y_encoded[:,:,[-12, -11]] /= y_encode_template[:, :, [-10, -9]] * y_encode_template[:,:,[-4, -3]]
			y_encoded[:,:,[-10, -9]] /= y_encode_template[:, :, [-10, -9]]
			y_encoded[:,:,[-10, -9]] = np.log(y_encoded[:, :, [-10, -9]]) / y_encode_template[:,:,[-2, -1]]
		else:
			y_encoded[:, :, -12:-8] -= y_encode_template[:, :, -12:-8]
			y_encoded[:, :, [-12, -11]] /= np.expand_dims(y_encode_template[:, :, -11] - y_encode_template[:, :, -12], axis = -1)
			y_encoded[:, :, [-10, -9]] /= np.expand_dims(y_encode_template[:, :, -9] - y_encode_template[:, :, -10], axis = -1)
			y_encoded[:, :, -12:-8] /= y_encode_template[:, :, -4:]
		return y_encoded
