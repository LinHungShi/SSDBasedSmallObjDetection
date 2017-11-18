from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from context_ssd import context_ssd
from ssd_loss import SSDLoss
from ssd_box import SSDBoxEncoder, decode_y
from batch_gen import BatchGenerator
import os
from utils import splitDataToCsv
import pandas as pd
import sys
from matplotlib import pyplot as plt
import cv2
import tensorflow as tf
from keras.layers import Input
from keras.models import Model
from tabulate import tabulate
from ssd_box import iou

#===============Hyperparameters===============
seed = 12345
np.random.seed(seed)
img_height = 720 
img_width = 1280 
img_channels = 3 
n_classes = 2
scales= [0.03, 0.05]
aspect_ratios = [0.5, 1.0, 2.0]
two_boxes_for_ar1 = True
limit_boxes = True 
variances = [1, 1, 1, 1]
batch_size = 2 
epochs = 50 
coords = 'centroids'
normalize_coords = False 
ratio = 0.1
lr = 0.001
images_path = 'images_subset'
labels = os.path.join(images_path, 'bbox_labels.csv')
train_labels = os.path.join(images_path, 'bbox_labels_train.csv')
val_labels = os.path.join(images_path, 'bbox_labels_val.csv')
pred_save_path = os.path.join(images_path, "prediction.csv")
input_format = ['image_name', 'xmin', 'xmax', 'ymin', 'ymax']
output_format = ['class_id', 'xmin', 'xmax', 'ymin', 'ymax']
pos_iou_threshold = 0.6
neg_iou_threshold = 0.2
brightness=(0.5, 2, 0.3)
flip=0.3
#translate=((0, 30), (0, 30), 0.3)
noise = (0, 1, 0.5)
scale = None
translate = None
include_thresh = 0.4
neg_pos_ratio  = 3 
n_neg_min = 0
alpha = 0.1
confidence_thresh = 0.4
iou_threshold = 0.45
top_k = 1
decode_freq = 5

#===============End HyperParameters===============

def buildModel(pre_weight=False):
        model, predictor_sizes = context_ssd(image_size=(img_height, img_width, img_channels),
                                         n_classes=n_classes,
                                         scales=scales,
                                         aspect_ratios=aspect_ratios,
                                         two_boxes_for_ar1=two_boxes_for_ar1,
                                         limit_boxes=limit_boxes,
                                         variances=variances,
                                         coords=coords,
                                         normalize_coords=normalize_coords)
        rmsprop = RMSprop(lr)
        ssd_loss = SSDLoss(neg_pos_ratio=neg_pos_ratio, n_neg_min=n_neg_min, alpha=alpha)
        if pre_weight:
                model.load_weights(pre_weight)
        model.compile(optimizer=rmsprop, loss=ssd_loss.compute_loss )
        ssd_box_encoder = SSDBoxEncoder(img_height=img_height,
                                       img_width=img_width,
                                       n_classes=n_classes,
                                       predictor_sizes=predictor_sizes,
                                       scales=scales,
                                       aspect_ratios=aspect_ratios,
                                       two_boxes_for_ar1=two_boxes_for_ar1,
                                       limit_boxes=limit_boxes,
                                       variances=variances,
                                       pos_iou_threshold=pos_iou_threshold,
                                       neg_iou_threshold=neg_iou_threshold,
                                       coords=coords,
                                       normalize_coords=normalize_coords)
	return model, ssd_box_encoder

def predict(model, X, ssd_bx, decode=True):
	y_pred = model.predict(X)
	if decode:
        	y_pred_decoded = decode_y(y_pred,
					confidence_thresh=confidence_thresh,
                                	iou_threshold=iou_threshold,
                                	top_k=top_k,
                                	input_coords=coords,
                                	normalize_coords=normalize_coords,
                                	img_height=img_height,
                                	img_width=img_width)
	return y_pred_decoded

def drawBoxFromCSV(csv, image_format, pred_format, gt_format, save_dir, diagnostic):
	df = pd.read_csv(csv)
	img_names = df[image_format]
	if pred_format:
		preds = np.array(df[pred_format]).astype(np.uint16)
	if gt_format:
		gts = np.array(df[gt_format]).astype(np.uint16)
	for img_name, gt, pred in zip(img_names, gts, preds):
		img = cv2.imread(img_name)
		lxmin, lxmax, lymin, lymax = pred.T
		gxmin, gxmax, gymin, gymax = gt.T
        	cv2.rectangle(img, (lxmin, lymin), (lxmax, lymax),(0, 0, 255), 2)
        	cv2.rectangle(img, (gxmin, gymin), (gxmax, gymax),(0, 255, 0), 2) 
		img_path = img_name.split('/')[-1]
		cv2.imwrite(os.path.join(save_dir, img_path), img)
		if diagnostic:
			plt.imshow(img)
			plt.show()

def printResult(filenames, batch_y, y_pred):
	try:
		headers = ["file", "confidence", "gxmin", "lxmin", "gxmax", "xmax", "gymin", "ymin", "gymax", "ymax"]
		table = [[filename, l[0], g[0, 1], l[1], g[0, 2], l[2], g[0, 3], l[3], g[0, 4], l[4]] for filename, g, l in zip(filenames, batch_y, y_pred)]
		print tabulate(table, headers, tablefmt='fancy_grid')
	except:
		print "Some error occurs in printResult"
		print "filenames:", filenames
		print "batch_y: ", batch_y
		print "y_pred: ", y_pred
		raise Exception
def train(val_labels, train_labels):
	model, predictor_sizes = context_ssd(image_size=(img_height, img_width, img_channels),
        	                         n_classes=n_classes,
                                	 scales=scales,
                                 	 aspect_ratios=aspect_ratios,
                                 	 two_boxes_for_ar1=two_boxes_for_ar1,
                                 	 limit_boxes=limit_boxes,
                                 	 variances=variances,
                                 	 coords=coords,
                                 	 normalize_coords=normalize_coords)
	rmsprop = RMSProp(lr)
	ssd_loss = SSDLoss(neg_pos_ratio=neg_pos_ratio, n_neg_min=n_neg_min, alpha=alpha)
	if pre_weight:
		model.load_weights(pre_weight) 
	model.compile(optimizer=rmsprop, loss=ssd_loss.compute_loss )
	ssd_box_encoder = SSDBoxEncoder(img_height=img_height,
 	                               img_width=img_width,
 	                               n_classes=n_classes,
 	                               predictor_sizes=predictor_sizes,
 	                               scales=scales,
 	                               aspect_ratios=aspect_ratios,
 	                               two_boxes_for_ar1=two_boxes_for_ar1,
 	                               limit_boxes=limit_boxes,
 	                               variances=variances,
 	                               pos_iou_threshold=pos_iou_threshold,
 	                               neg_iou_threshold=neg_iou_threshold,
 	                               coords=coords,
 	                               normalize_coords=normalize_coords)
	train_dataset = BatchGenerator(images_path=images_path,
	                               box_output_format=output_format)
	val_dataset = BatchGenerator(images_path=images_path,
					box_output_format=output_format)
	splitDataToCsv(labels, ratio, val_labels, train_labels)
	train_dataset.parse_csv(labels_path=train_labels,
	                        input_format=input_format,
	                        ret=False)
	val_dataset.parse_csv(labels_path=val_labels,
				input_format=input_format,
				ret=False)
	val_samples = val_dataset.get_n_samples()
	train_generator = train_dataset.generate(batch_size=batch_size,
	                                         train=True,
	                                         ssd_box_encoder=ssd_box_encoder,
	                                         equalize=False,
	                                         brightness=False,
	                                         flip=False,
						 noise=False,
	                                         translate=False,
	                                         limit_boxes=True,
						 scale=scale,
	                                         include_thresh=include_thresh,
	                                         diagnostics=False)
	val_generator = val_dataset.generate(batch_size= val_samples,
						train=True,
						ssd_box_encoder=ssd_box_encoder,
						equalize=False,
						brightness=False,
						flip=False,
						translate=False,
						limit_boxes=True,
						include_thresh=include_thresh,
						diagnostics=False)
	n_train_samples = train_dataset.get_n_samples()
	n_val_samples = val_dataset.get_n_samples()
	history = model.fit_generator(generator = train_generator,
	                              steps_per_epoch = ceil(n_train_samples/batch_size),
	                              epochs = epochs,
	                              callbacks = [ModelCheckpoint('weights/context_ssd_weights_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
	                                                           monitor='val_loss',

	                                                           verbose=1,
	                                                           save_best_only=True,
	                                                           save_weights_only=True,
	                                                           mode='auto',
	                                                           period=1)],
	                              validation_data = val_generator,
	                              validation_steps = ceil(n_val_samples/batch_size))



def evaluate(weight, images_path, labels_csv, save_path, draw_box, drawed_box_save_path, diagnostic):
	model, predictor_sizes = context_ssd(image_size=(img_height, img_width, img_channels),
                                  n_classes=n_classes,
                                  scales=scales,
                                  aspect_ratios=aspect_ratios,
                                  two_boxes_for_ar1=two_boxes_for_ar1,
                                  limit_boxes=limit_boxes,
                                  variances=variances,
                                  coords=coords,
                                  normalize_coords=normalize_coords)
	model.load_weights(weight)
	ssd_box_encoder = SSDBoxEncoder(img_height=img_height,
                                img_width=img_width,
                                n_classes=n_classes,
                                predictor_sizes=predictor_sizes,
                                scales=scales,
                                aspect_ratios=aspect_ratios,
                                two_boxes_for_ar1=two_boxes_for_ar1,
                                limit_boxes=limit_boxes,
                                variances=variances,
                                pos_iou_threshold=pos_iou_threshold,
                                neg_iou_threshold=neg_iou_threshold,
                                coords=coords,
                                normalize_coords=normalize_coords)
	predict_dataset = BatchGenerator(images_path=images_path, box_output_format=output_format)
	predict_dataset.parse_csv(labels_path=labels_csv,
               	        	input_format=input_format,
                       		ret=False)
	predict_generator = predict_dataset.generate(batch_size=batch_size,
						train=False,
                                        	ssd_box_encoder=ssd_box_encoder,
                                        	equalize=False,
                                        	brightness=False,
                                        	flip=False,
                                        	translate=False,
                                        	limit_boxes=True,
                                        	scale=False,
                                        	include_thresh=include_thresh,
                                        	diagnostics= False)
	predictions = []
	img_names = []
 	ground_truths = []	
	try:
		while True:
			X, batch_y, filenames = next(predict_generator)  
			y_pred = model.predict(X)
			y_pred_decoded = decode_y(y_pred,
                          confidence_thresh=confidence_thresh,
                          iou_threshold=iou_threshold,
                          top_k=top_k,
                          input_coords=coords,
                          normalize_coords=normalize_coords,
                          img_height=img_height,
                          img_width=img_width)	
			if diagnostic:
				printResult(filenames, batch_y, y_pred_decoded)
			predictions.append(y_pred_decoded)
			img_names.append(filenames)
			ground_truths.append(batch_y)
	except StopIteration:
		"Prediction Over"
	
	print "Writing result to csv file..."
	flat_image_files = reduce(lambda x, y: x + y, img_names)
	flat_label_files = reduce(lambda x, y: x + y, predictions)
	flat_label_files = np.stack(flat_label_files, axis=0)
	ground_truth_decode = reduce(lambda x, y: x + y, ground_truths)
	ground_truth_decode = np.concatenate(ground_truth_decode, axis=0)
	pred_format = ['pxmin','pxmax', 'pymin', 'pymax']
	gt_format = ['gxmin', 'gxmax', 'gymin', 'gymax']
	df = pd.DataFrame(columns = ['image_name', 'conf'] + pred_format + gt_format)
	df['image_name'] = flat_image_files
	df[['conf'] + pred_format] = flat_label_files 
	df[gt_format] = ground_truth_decode[..., 1:5]
	df.to_csv(save_path, header = True, index = False)	
	
	print "evaluate..."
	predicted = np.array(df[pred_format])
	gt = np.array(df[gt_format])
	scores = iou(predicted, gt, coords="minmax")
	num_samples = predicted.shape[0]
	average_iou = np.sum(scores) / num_samples
	print "average_iou: ", average_iou
	if draw_box:
		print "draw box..."
		if drawed_box_save_path == None:
			raise Exception("to draw box, drawed_box_save_path must be provided, but get None")
		drawBoxFromCSV(save_path, "image_name", pred_format,gt_format, drawed_box_save_path, False)
	print "Done!!"

def trainOnBatch(labels_file, images_path, weights, diagnostic=True):
	
	## Define context_aware model
	model, predictor_sizes = context_ssd(image_size=(img_height, img_width, img_channels),
                                  n_classes=n_classes,
                                  scales=scales,
                                  aspect_ratios=aspect_ratios,
                                  two_boxes_for_ar1=two_boxes_for_ar1,
                                  limit_boxes=limit_boxes,
                                  variances=variances,
                                  coords=coords,
                                  normalize_coords=normalize_coords)
	
	ssd_loss = SSDLoss(neg_pos_ratio=neg_pos_ratio, n_neg_min=n_neg_min, alpha=alpha)
	if weights:
		print "load weight ", pre_weight
		model.load_weights(pre_weight) 
	
	'''
	 Compile ssd model
	 Metrics:
	 	compute_PossLoss compute the cross entropy for positive default boxes
		 compute_NegLoss compute the top k cross entropy for negative default boxes
		 compute LocLoss compute the smooth loss for postive default boxes
	  	         maxPred the higheset class confidence
			 minPred teh lowest class confidence
	'''
         
	rmsprop = RMSprop(lr)
	model.compile(optimizer=rmsprop, loss=ssd_loss.compute_loss, metrics=[ssd_loss.compute_PosLoss, ssd_loss.compute_NegLoss, ssd_loss.compute_LocLoss, ssd_loss.maxPred, ssd_loss.minPred, ssd_loss.numPos])
	
	# Define default boxes. This class store all default boxes, which will be used for predicting the shift of each default box
	ssd_box_encoder = SSDBoxEncoder(img_height=img_height,
                                img_width=img_width,
                                n_classes=n_classes,
                                predictor_sizes=predictor_sizes,
                                scales=scales,
                                aspect_ratios=aspect_ratios,
                                two_boxes_for_ar1=two_boxes_for_ar1,
                                limit_boxes=limit_boxes,
                                variances=variances,
                                pos_iou_threshold=pos_iou_threshold,
                                neg_iou_threshold=neg_iou_threshold,
                                coords=coords,
                                normalize_coords=normalize_coords)
	
	# Define our dataset. All preprocessing will be done here
	train_dataset = BatchGenerator(images_path=images_path, box_output_format=output_format)
	train_dataset.parse_csv(labels_path=labels_file,
                        	input_format=input_format,
                        	ret=False)
	train_generator = train_dataset.generate(batch_size=batch_size,
						train=True,
                                        	ssd_box_encoder=ssd_box_encoder,
                                        	equalize=False,
                                        	brightness=brightness,
                                        	flip=flip,
                                        	translate=translate,
                                        	limit_boxes=True,
                                        	scale=False,
                                        	include_thresh=include_thresh,
                                        	diagnostics=True)

	ep = 0
	
	# epochs that decode predictions and print out result
	eps_for_decode = range(1, epochs, decode_freq)

	while ep <= epochs:
		
		# X is training images with shape (batch, image_height, image_width)
		# y_true is the ground truth for default boxes with shape (batch, nboxes, 14) where the last dimension contains background, foreground, gx, gy, gw, gh, dx, dy, dw, dh, variance1, variance2, variance3 and variance4
		# batch_y is a list of length batch_size where each element is the grouth true bounding box with shape (1, 5) where the last dimension contains 1, xmin, xmax, ymin, ymax
		# ep_tmp is the current epoch
		# filenames is a list of length batch_size containing image names
		# original images is 4D array with shape (batch, image_height, image_width, channels) containing images before preprocessing
		# original labels is a 3D array with shape (batch_size, num_objects, 5) where the last dimension contains ground truth (1) and four numbers for the location of bounding box for original images

		X, y_true, batch_y, ep_tmp, filenames, original_images, original_labels = next(train_generator)
		losses = model.train_on_batch(X, y_true)

		if diagnostic:
			print "filename: ", filenames
			print "epochs: ", ep_tmp
			print "losses:{}\t PosLoss:{}\tNegLoss:{}\tLocLoss:{}\tmaxPred:{}\tminPred:{}\nnumPos:{}".format(*losses)

		if ep != ep_tmp:
			print "save_model..."
			model.save_weights('weights/context_ssd_loss_ep{}_{}.hdf5'.format(ep, losses[0]))
			ep = ep_tmp

                if ep in eps_for_decode and diagnostic:
                        y_pred = model.predict(X)
			if y_pred.shape[0] == 0 or len(batch_y) == 0 or len(filenames) == 0:
				print "y_pred shape == 0, y_pred = {}".format(y_pred)
				print "batch_y shape == 0, batch_y = {}".format(batch_y)
				print "filenames = {}".format(filenames)
				raise Exception
                        y_pred_decoded = decode_y(y_pred,
                                confidence_thresh=confidence_thresh,
                                iou_threshold=iou_threshold,
                                top_k=top_k,
                                input_coords=coords,
                                normalize_coords=normalize_coords,
                                img_height=img_height,
                                img_width=img_width)
			printResult(filenames, batch_y, y_pred_decoded)	
	print "Training Done!!!!!"

if __name__ == "__main__":
	mode = int(sys.argv[1])
	if len(sys.argv) > 2:
		pre_weight = sys.argv[2]
	else:
		pre_weight = None
	
	if len(sys.argv) > 3:
		if int(sys.argv[3]) == 1:
			splitDataToCsv(labels, ratio, val_labels, train_labels)		
	if mode == 0:
		train(val_labels, train_labels)
	elif mode == 1: 
		evaluate(pre_weight, images_path, train_labels, save_path= pred_save_path, draw_box=True, drawed_box_save_path="drawed_images", diagnostic=True)
	else: 
		trainOnBatch(train_labels, images_path, pre_weight, diagnostic=True)

