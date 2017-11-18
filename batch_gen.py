import numpy as np
import cv2, random
from sklearn.utils import shuffle
from copy import deepcopy
import csv
import os
import io
import cv2

def _blur(image, mean, var):
    noised_img = np.copy(image).astype(np.float)
    rows, cols, ch = image.shape
    noise = np.random.normal(loc=mean, scale=var, size=(rows, cols, ch))
    noised_img += noise
    return noised_img


def _translate(image, horizontal = (0, 40), vertical = (0, 10)):
    rows, cols, ch = image.shape
    x = np.random.randint(horizontal[0], horizontal[1] + 1)
    y = np.random.randint(vertical[0], vertical[1] + 1)
    x_shift = random.choice([-x, x])
    y_shift = random.choice([-y, y])
    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    return (cv2.warpAffine(image, M, (cols, rows)), x_shift, y_shift)


def _flip(image, orientation = 'horizontal'):
    if orientation == 'horizontal':
        return cv2.flip(image, 1)
    else:
        return cv2.flip(image, 0)


def _scale(image, min = 0.9, max = 1.1):
    rows, cols, ch = image.shape
    scale = np.random.uniform(min, max)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 0, scale)
    return (cv2.warpAffine(image, M, (cols, rows)), M, scale)


def _brightness(image, min = 0.5, max = 2.0):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_br = np.random.uniform(min, max)
    mask = hsv[:, :, 2] * random_br > 255
    v_channel = np.where(mask, 255, hsv[:, :, 2] * random_br)
    hsv[:, :, 2] = v_channel
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def histogram_eq(image):
    image1 = np.copy(image)
    image1[:, :, 0] = cv2.equalizeHist(image1[:, :, 0])
    image1[:, :, 1] = cv2.equalizeHist(image1[:, :, 1])
    image1[:, :, 2] = cv2.equalizeHist(image1[:, :, 2])
    return image1


class BatchGenerator:

    def __init__(self, images_path, box_output_format = ['class_id', 'xmin','xmax', 'ymin','ymax']):

        self.images_path = images_path
        self.box_output_format = box_output_format
        self.labels_path = None
        self.input_format = None
        self.filenames = []
        self.labels = []

    def parse_csv(self, labels_path = None, input_format = None, ret = False):
        if labels_path is not None:
            self.labels_path = labels_path
        if input_format is not None:
            self.input_format = input_format
        if self.labels_path is None or self.input_format is None:
            raise ValueError('labels_path and/or input_format have not been set yet')
        self.filenames = []
        self.labels = []
        data = []
        with io.open(self.labels_path, newline='') as csvfile:
            csvread = csv.reader(csvfile, delimiter=',')
            k = 0
            for i in csvread:
                if k == 0:
                    k += 1
                    continue
                else:
                    obj = []
                    obj.append(os.path.join(self.images_path, i[self.input_format.index('image_name')].strip()))
                    obj.append(1)
                    for item in self.box_output_format:
                        if not item == 'class_id':
                            obj.append(float(i[self.input_format.index(item)].strip()))

                data.append(obj)

        data = sorted(data)
        current_file = ''
        current_labels = []
        for idx, i in enumerate(data):
            if current_file == '':
                current_file = i[0]
                current_labels.append(i[1:])
                if len(data) == 1:
                    self.labels.append(np.stack(current_labels, axis=0))
                    self.filenames.append(current_file)
            elif i[0] == current_file:
                current_labels.append(i[1:])
                if idx == len(data) - 1:
                    self.labels.append(np.stack(current_labels, axis=0))
                    self.filenames.append(current_file)
            else:
                self.labels.append(np.stack(current_labels, axis=0))
                self.filenames.append(current_file)
                current_labels = []
                current_file = i[0]
                current_labels.append(i[1:])
                if idx == len(data) - 1:
                    self.labels.append(np.stack(current_labels, axis=0))
                    self.filenames.append(current_file)
	for i in range(len(self.labels)):
		if self.labels[i].shape[0] == 0:
			raise Exception
        if ret:
            return (self.filenames, self.labels)

    def generate(self, batch_size = 32, train = True, ssd_box_encoder = None, equalize = False, brightness = False, flip = False, translate = False, noise = False, scale = (0.75, 1.2, 0.5), limit_boxes = True, include_thresh = 0.3, diagnostics = False):
        xmin = self.box_output_format.index('xmin')
        xmax = self.box_output_format.index('xmax')
        ymin = self.box_output_format.index('ymin')
        ymax = self.box_output_format.index('ymax')
	
        epochs = 0 
        current = 0

        while True:
            batch_X, batch_y = [], []
            if current >= len(self.filenames):
                if train == False:
                    break
                self.filenames, self.labels = shuffle(self.filenames, self.labels)
                current = 0
		epochs += 1

            this_filenames = []

            while len(batch_X) < batch_size and current < len(self.filenames):
                if self.labels[current][:, 2] - self.labels[current][:, 1] > 5 and self.labels[current][:, 4] - self.labels[current][:, 3] > 5 or not train:
                    img = cv2.imread(self.filenames[current])
		    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.astype(np.float32)
                    batch_X.append(np.copy(img))
                    batch_y.append(np.copy(self.labels[current]))
                    this_filenames.append(self.filenames[current])
            	current += 1

            ## If the training batch is bad, we resample the training data
            if len(batch_y) == 0:
                continue

            if diagnostics:
                original_images = np.copy(batch_X)
                original_labels = np.copy(batch_y)

            for i in range(len(batch_X)):
                img_height, img_width, ch = batch_X[i].shape
                batch_y[i] = np.array(batch_y[i])
                if equalize:
                    batch_X[i] = histogram_eq(batch_X[i])
                if brightness:
                    p = np.random.uniform(0, 1)
                    if p >= 1 - brightness[2]:
                        batch_X[i] = _brightness(batch_X[i], min=brightness[0], max=brightness[1])
                if noise:
                    p = np.random.uniform(0, 1)
                    if p >= 1 - noise[2]:
                        batch_X[i] = _blur(batch_X[i], noise[0], noise[1])
                if flip:
                    p = np.random.uniform(0, 1)
                    if p >= 1 - flip:
                        batch_X[i] = _flip(batch_X[i])
                        batch_y[i][:, [xmin, xmax]] = img_width - batch_y[i][:, [xmax, xmin]]
                if translate:
                    p = np.random.uniform(0, 1)
                    if p >= 1 - translate[2]:
                        batch_X[i], xshift, yshift = _translate(batch_X[i], translate[0], translate[1])
                        batch_y[i][:, [xmin, xmax]] += xshift
                        batch_y[i][:, [ymin, ymax]] += yshift
                    if limit_boxes:
                        before_limiting = deepcopy(batch_y[i])
                        x_coords = batch_y[i][:, [xmin, xmax]]
                        x_coords[x_coords >= img_width] = img_width - 1
                        x_coords[x_coords < 0] = 0
                        batch_y[i][:, [xmin, xmax]] = x_coords
                        y_coords = batch_y[i][:, [ymin, ymax]]
                        y_coords[y_coords >= img_height] = img_height - 1
                        y_coords[y_coords < 0] = 0
                        batch_y[i][:, [ymin, ymax]] = y_coords
                        before_area = (before_limiting[:, xmax] - before_limiting[:, xmin]) * (before_limiting[:, ymax] - before_limiting[:, ymin])
                        after_area = (batch_y[i][:, xmax] - batch_y[i][:, xmin]) * (batch_y[i][:, ymax] - batch_y[i][:, ymin])
                        if include_thresh == 0:
                            batch_y[i] = batch_y[after_area > 0]
                        else:
                            batch_y[i] = batch_y[i][after_area >= include_thresh * before_area]
                if scale:
                    p = np.random.uniform(0, 1)
                    if p >= 1 - scale[2]:
                        batch_X[i], M, scale_factor = _scale(batch_X[i], scale[0], scale[1])
                        toplefts = np.array([batch_y[i][:, xmin], batch_y[i][:, ymin], np.ones(batch_y[i].shape[0])])
                        bottomrights = np.array([batch_y[i][:, xmax], batch_y[i][:, ymax], np.ones(batch_y[i].shape[0])])
                        new_toplefts = np.dot(M, toplefts).T
                        new_bottomrights = np.dot(M, bottomrights).T
                        batch_y[i][:, [xmin, ymin]] = new_toplefts.astype(np.int)
                        batch_y[i][:, [xmax, ymax]] = new_bottomrights.astype(np.int)
                        if limit_boxes and scale_factor > 1:
                            before_limiting = deepcopy(batch_y[i])
                            x_coords = batch_y[i][:, [xmin, xmax]]
                            x_coords[x_coords >= img_width] = img_width - 1
                            x_coords[x_coords < 0] = 0
                            batch_y[i][:, [xmin, xmax]] = x_coords
                            y_coords = batch_y[i][:, [ymin, ymax]]
                            y_coords[y_coords >= img_height] = img_height - 1
                            y_coords[y_coords < 0] = 0
                            batch_y[i][:, [ymin, ymax]] = y_coords
                            before_area = (before_limiting[:, xmax] - before_limiting[:, xmin]) * (before_limiting[:, ymax] - before_limiting[:, ymin])
                            after_area = (batch_y[i][:, xmax] - batch_y[i][:, xmin]) * (batch_y[i][:, ymax] - batch_y[i][:, ymin])
                            if include_thresh == 0:
                                batch_y[i] = batch_y[i][after_area > 0]
                            else:
                                batch_y[i] = batch_y[i][after_area >= include_thresh * before_area]
                                
            if train:
		        if ssd_box_encoder == None:
			        raise Exception("training requires ssd_box, but get None") 
                	y_true = ssd_box_encoder.encode_y(batch_y)

            if train:
                if diagnostics:
                    yield (np.array(batch_X), y_true, batch_y, epochs, this_filenames, original_images, original_labels)
                else:
                    yield (np.array(batch_X, dtype=np.float32), y_true)
            else:
                yield (np.array(batch_X), batch_y, this_filenames)

    def get_n_samples(self):
        return len(self.filenames)

    def get_filenames_labels(self):
        return (self.filenames, self.labels)
