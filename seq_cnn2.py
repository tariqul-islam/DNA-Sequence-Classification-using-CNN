import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

def seq_pre_processor(x,sess,vocabulary_length,region_size):
    #max_len = x.shape[1]
    #num_regions = max_len-region_size+1
    #N = vocabulary_length*region_size
    #x_out = np.zeros((N,num_regions))
    
    p = tf.one_hot(x,vocabulary_length)
    p = sess.run([p])
    p = p[0]
    
    x_out = p.reshape(x.shape[0],-1,1,1)
    
    return x_out
        
    
class seqCNN(object):
    """
    This CNN is an atempt to recreate the CNN described in this paper:
    Effective Use of Word Order for Text Categorization with Convolutional Neural Networks
    Rie Johnson, Tong Zhang
    """
    
    def __init__(self,num_classes, num_filters, num_pooled, vocabulary_length, region_size, max_sentence_length, filter_lengths=3, l2_reg_lambda=0.0):
        
        
        #input layers and params
        filter_length = vocabulary_length*region_size
        sentence_length = max_sentence_length*vocabulary_length
        
        self.x_input = tf.placeholder(tf.float32, [None, sentence_length, 1, 1], name="x_input")
        self.y_input = tf.placeholder(tf.float32, [None, num_classes], name="y_input")
        
        
        self.dropout_param = tf.placeholder(tf.float32, name="dropout_param")
        
        cnn_filter_shape = [filter_length, 1, 1, num_filters[0]]
        W_CN = tf.Variable(tf.truncated_normal(cnn_filter_shape, stddev=0.1), name="W_CN")
        b_CN = tf.Variable(tf.truncated_normal([num_filters[0]],stddev=0.1),name="b_CN")
        
        cnn_filter2_shape = [filter_lengths, 1, num_filters[0], num_filters[1]]
        W_CN2 = tf.Variable(tf.truncated_normal(cnn_filter2_shape, stddev=0.1), name="W_CN2")
        b_CN2 = tf.Variable(tf.truncated_normal([num_filters[1]],stddev=0.1),name="W_CN2")
        
        #conv-relu-pool layer
        conv1 = tf.nn.conv2d(
                        self.x_input,
                        W_CN,
                        strides=[1, vocabulary_length, 1, 1],
                        padding="VALID",
                        name="conv"
                        )
        relu1 = tf.nn.relu(
                        tf.nn.bias_add(conv1, b_CN),
                        name="relu"
                        )
        
        conv2 = tf.nn.conv2d(
                        relu1,
                        W_CN2,
                        strides=[1, 1, 1, 1],
                        padding="SAME",
                        name="conv2"
                        )
                        
        relu2 = tf.nn.relu(
                        tf.nn.bias_add(conv2, b_CN2),
                        name="relu2"
                        )
                        
        pool_stride = [1,int((max_sentence_length-region_size+1)/num_pooled),1,1]
        pool = tf.nn.avg_pool(
                        relu2,
                        ksize = pool_stride,
                        strides = pool_stride,
                        padding="VALID",
                        name="pool"
                        )
        
        
        #dropout
        drop = tf.nn.dropout(
                        pool,
                        self.dropout_param
                        )
        
        #response normalization
        normalized = tf.nn.local_response_normalization(drop)                
        
        #feature extraction and flatting for future 
        self.pool_flat = tf.reshape(normalized, [-1, num_pooled*num_filters[1]])                
                        
        #affine layer
        affine_filter_shape = [num_pooled*num_filters[1], num_classes]
        W_AF = tf.Variable(
                        tf.truncated_normal(affine_filter_shape, stddev=0.1),
                        name="W_AF"
                        )
        b_AF = tf.Variable(
                        tf.truncated_normal([num_classes], stddev=0.1),
                        name="b_AF"
                        )
        scores = tf.matmul(self.pool_flat,W_AF)+b_AF
        self.predictions = tf.argmax(scores, 1, name="predictions")
        
        #losses
        losses = tf.nn.softmax_cross_entropy_with_logits(scores, self.y_input)
        self.loss = tf.reduce_mean(losses) + l2_reg_lambda * ( tf.nn.l2_loss(W_CN) + tf.nn.l2_loss(b_CN) + tf.nn.l2_loss(W_AF) + tf.nn.l2_loss(b_AF) )
        
        #predictions
        correct_predictions = tf.equal(self.predictions, tf.argmax(self.y_input, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
