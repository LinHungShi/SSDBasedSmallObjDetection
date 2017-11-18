import tensorflow as tf
idx =1 
class SSDLoss:

	def __init__(self, neg_pos_ratio=1, n_neg_min=0, alpha=1.0):
		self.neg_pos_ratio = tf.constant(neg_pos_ratio)
		self.n_neg_min = tf.constant(n_neg_min)
		self.alpha = alpha

	def smooth_L1_loss(self, y_true, y_pred):

		absolute_loss = tf.abs(y_true - y_pred)
		square_loss = 0.5 * (y_true - y_pred)**2
		l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
		return tf.reduce_sum(l1_loss, axis = -1)
	
	def square_loss(self, y_true, y_pred):
		return tf.reduce_sum((y_true - y_pred)**2)

	def l1_loss(self, y_true, y_pred):
		return tf.reduce_sum(tf.abs(y_true -  y_pred))

	### Classification error
	def log_loss(self, y_true, y_pred):
		y_pred = tf.maximum(y_pred, 1e-15)
		log_loss = -tf.reduce_sum(y_true * tf.log(y_pred), axis = -1)
		return log_loss

   	def compute_loss(self, y_true, y_pred):
        	batch_size = tf.shape(y_pred)[0]
        	n_boxes = tf.shape(y_pred)[1]

        	classification_loss = tf.to_float(self.log_loss(y_true[:,:,:-12], y_pred[:,:,:-12])) 
        	localization_loss = tf.to_float(self.smooth_L1_loss(y_true[:,:,-12:-8], y_pred[:,:,-12:-8]))
        	negatives = tf.to_float(y_true[:,:,0]) 
        	positives = tf.to_float(y_true[:,:,1])
		
	        n_positive = tf.reduce_sum(positives)
	
	        pos_class_loss = tf.reduce_sum(classification_loss * positives, axis=-1)
	
	        neg_class_loss_all = classification_loss * negatives 
		n_neg_losses = tf.count_nonzero(neg_class_loss_all, dtype=tf.int32) 
	        n_negative_keep = tf.minimum(tf.maximum(self.neg_pos_ratio * tf.to_int32(n_positive), self.n_neg_min), n_neg_losses)
	
	        def f1():
	        	return tf.constant(0.0)

       		def f2():

            		neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1])
            		values, indices = tf.nn.top_k(neg_class_loss_all_1D, n_negative_keep, False)
            		negatives_keep = tf.scatter_nd(tf.expand_dims(indices, axis=1), updates=tf.ones_like(indices, dtype=tf.int32), shape=tf.shape(neg_class_loss_all_1D)) 
            		negatives_keep = tf.to_float(tf.reshape(negatives_keep, [batch_size, n_boxes])) 
            		neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep) 
            		return neg_class_loss

        	neg_class_loss = tf.cond(tf.equal(n_neg_losses, tf.constant(0)), f1, f2)
		class_loss = pos_class_loss + neg_class_loss 
        	loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1) 
        	total_loss = (class_loss +  self.alpha * loc_loss) / tf.maximum(1.0, n_positive)
        	return total_loss 

### Used for deugging
        def compute_PosLoss(self, y_true, y_pred):
		class_error = self.log_loss(y_true[...,:-12], y_pred[...,:-12])
		positive_boxes = y_true[..., 1]
                return  tf.reduce_sum(class_error * positive_boxes)

        def compute_NegLoss(self, y_true, y_pred):
		batch_size = tf.shape(y_pred)[0]
                n_boxes = tf.shape(y_pred)[1]
                classification_loss = self.log_loss(y_true[:,:, :-12], y_pred[:, :, :-12])
                positives = tf.to_float(y_true[:,:,1]) 
                n_positive = tf.reduce_sum(positives)
                negatives = y_true[..., 0]
                neg_class_loss_all = classification_loss * negatives
                n_neg_losses = tf.count_nonzero(neg_class_loss_all, dtype=tf.int32)
                n_negative_keep = tf.minimum(tf.maximum(self.neg_pos_ratio * tf.to_int32(n_positive), self.n_neg_min), n_neg_losses)

                def f1():
                        return tf.zeros([batch_size])

                def f2():
                        neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1])
                        values, indices = tf.nn.top_k(neg_class_loss_all_1D, n_negative_keep, False)
                        negatives_keep = tf.scatter_nd(tf.expand_dims(indices, axis=1), updates=tf.ones_like(indices, dtype=tf.int32), shape=tf.shape(neg_class_loss_all_1D))
                        negatives_keep = tf.to_float(tf.reshape(negatives_keep, [batch_size, n_boxes]))
                        neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep, axis=-1)
                        return neg_class_loss

                neg_class_loss = tf.cond(tf.equal(n_neg_losses, tf.constant(0)), f1, f2)
                return tf.reduce_sum(neg_class_loss) 
	
	def compute_LocLoss(self, y_true, y_pred):

		loc_loss = tf.to_float(self.smooth_L1_loss(y_true[:,:,-12:-8], y_pred[:,:,-12:-8]))
                positives = tf.to_float(y_true[..., 1])
		loc_loss = tf.reduce_sum(loc_loss * positives)
		return loc_loss 
	
	def maxPred(self, y_true, y_pred):
		return tf.reduce_max(y_pred[..., 1] * y_true[..., 1])
	
	def minPred(self, y_true, y_pred):
		return tf.reduce_min(y_pred[..., 1] * y_true[..., 1])
	
	def numPos(self, y_true, y_pred):
		return tf.reduce_sum(y_true[..., 1])	
