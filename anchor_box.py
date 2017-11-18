from __future__ import division
import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
import numpy as np

from ssd_box import convert_coordinates
idx =1 
class AnchorBoxes(Layer):
	def __init__(
		self,
		img_height,
		img_width,
		this_scale,
		next_scale,
		aspect_ratios=[0.5, 1.0, 2.0],
		two_boxes_for_ar1=True,
		limit_boxes=True,
		variances=[1.0, 1.0, 1.0, 1.0],
		coords='centroids',
		normalize_coords=False,
		**kwargs):

		if len(variances) != 4:
			raise ValueError('Length of variance must be four, but get {}'.format(len(variances)))
		variances = np.array(variances)
		if np.any(variances <= 0):
			raise ValueError('variances must be larger than 0')

		self.img_height = img_height
		self.img_width = img_width
		self.this_scale = this_scale
		self.next_scale = next_scale
		self.aspect_ratios = aspect_ratios 
		self.two_boxes_for_ar1 = two_boxes_for_ar1
		self.limit_boxes = limit_boxes
		self.variances = variances
		self.coords = coords
		self.normalize_coords = normalize_coords

		if (1 in aspect_ratios) and two_boxes_for_ar1:
			self.n_boxes = len(aspect_ratios) + 1
		else:
			self.n_boxes = len(aspect_ratios)
		super(AnchorBoxes, self).__init__(**kwargs)

	def build(self, input_shape):
		self.input_spec = InputSpec(shape = input_shape)
		super(AnchorBoxes, self).build(input_shape)

	def call(self, x, mask=None):
		self.aspect_ratios = np.sort(self.aspect_ratios)
		size = min(self.img_width, self.img_height)
		wh_list = []
		for ar in self.aspect_ratios:
			if (ar==1) & self.two_boxes_for_ar1:
				w = self.this_scale * size  * np.sqrt(ar)
				h = self.this_scale * size  / np.sqrt(ar)
				wh_list.append((w, h))
				w = np.sqrt(self.this_scale * self.next_scale) * size * np.sqrt(ar)
				h = np.sqrt(self.this_scale * self.next_scale) * size / np.sqrt(ar)
				wh_list.append((w, h))
			else:
				w = self.this_scale * size * np.sqrt(ar)
				h = self.this_scale * size / np.sqrt(ar)
				wh_list.append((w, h))
		wh_list = np.array(wh_list)


		batch_size, feature_map_height, feature_map_width, feature_map_channels = x._keras_shape
		cell_height = self.img_height / feature_map_height
		cell_width = self.img_width / feature_map_width
		cx = np.linspace(cell_width * 0.5, self.img_width - cell_width * 0.5, feature_map_width)
		cy = np.linspace(cell_height * 0.5, self.img_height - cell_height * 0.5, feature_map_height)
		cx_grid, cy_grid = np.meshgrid(cx, cy)
		cx_grid = np.expand_dims(cx_grid, -1)
		cy_grid = np.expand_dims(cy_grid, -1)

		boxes_tensor = np.zeros((feature_map_height, feature_map_width, self.n_boxes, 4))

		boxes_tensor[..., 0] = np.tile(cx_grid, (1, 1, self.n_boxes))
		boxes_tensor[..., 1] = np.tile(cy_grid, (1, 1, self.n_boxes))
		boxes_tensor[..., 2] = wh_list[:, 0]
		boxes_tensor[..., 3] = wh_list[:, 1]
		boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2minmax')

		if self.limit_boxes:
			x_coords = boxes_tensor[:, :, :, [0, 1]]
			x_coords[x_coords > self.img_width] = self.img_width - 1
			x_coords[x_coords < 0] = 0
			boxes_tensor[:, :, :, [0, 1]] = x_coords
			y_coords = boxes_tensor[:, :, :, [2, 3]]
			y_coords[y_coords > self.img_height] = self.img_height - 1
			y_coords[y_coords < 0] = 0
			boxes_tensor[:, :, :, [2, 3]] = y_coords

		if self.normalize_coords:
			boxes_tensor[:, :, :, :2] /= self.img_width
			boxes_tensor[:, :, :, 2:] /= self.img_height

		if self.coords == 'centroids':
			boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='minmax2centroids')

		variances_tensor = np.zeros_like(boxes_tensor)
		variances_tensor += self.variances
		boxes_tensor = np.concatenate((boxes_tensor, variances_tensor), axis=-1)
		dim = boxes_tensor.shape
		test_boxes = boxes_tensor.reshape(-1, 8)
		boxes_tensor = np.expand_dims(boxes_tensor, axis=0)
		boxes_tensor = K.tile(K.constant(boxes_tensor, dtype='float32'), (K.shape(x)[0], 1, 1, 1, 1))
		

		return boxes_tensor

	def compute_output_shape(self, input_shape):
		batch_size, feature_map_height, feature_map_width, feature_map_channels = input_shape
		return (batch_size, feature_map_height, feature_map_width, self.n_boxes, 8)
