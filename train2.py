#! /usr/bin/env python

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
import data_helpers as dhrt

import os
import time
import datetime

from seq_cnn2 import seqCNN
#from seq_cnn_old import seq_pre_processor

#load data
x_rt, y_rt = dhrt.load_data_and_labels('h3.pos','h3.neg')
lens = [len(x.split(" ")) for x in x_rt];
max_document_length = max(lens)

if max_document_length%2 != 0:
    max_document_length=max_document_length+1

print "Max Document Length = ", max_document_length
print "Number of Samples =", len(y_rt)
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x_rt_proc = np.array(list(vocab_processor.fit_transform(x_rt)))
l_x_rt = len(y_rt)
vocab_size = len(vocab_processor.vocabulary_)

print "Vocab Size: ", vocab_size


#np.random.seed(10) #for 'reproducible research' B-)
#^comment this line out to evaluate the model more thoroughly
#in case this line is commented out, standard cross validation is assumed
#so, a separate test dataset is necessary -
#in order to ensure the model(s) have been trained properly
#this code does not save the model in a file
#somene has to implement
#^NOTES or TO DO

shuffled_rt = np.random.permutation(range(l_x_rt))
x_rt_shuffled = x_rt_proc[shuffled_rt]
y_rt_shuffled = y_rt[shuffled_rt]

rt_split_size = 3000
x_rt_train = x_rt_shuffled[:-rt_split_size]
x_rt_val = x_rt_shuffled[-rt_split_size:]
y_rt_train = y_rt_shuffled[:-rt_split_size]
y_rt_val = y_rt_shuffled[-rt_split_size:]

rt_train_length = len(y_rt_train)
rt_val_length = len(y_rt_val)


num_classes = 2
num_filters = [16, 8]

region_size = 51 #can be considered as filter size but not really

#this value has to be selected based on max_document_length and region_size
#here I ensured that max_docu length is even and region size in odd
#so division by 2 is possible
num_pooled = (max_document_length-region_size+1)/2 

print "Pool Size: ", num_pooled

batch_size = 10
num_epochs = 20

evaluate_every = 200

dropout_param = 0.5

out_dir = "out"

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = seqCNN(
                num_classes,
                num_filters,
                num_pooled,
                vocab_size,
                region_size,
                max_document_length)
                
        
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars,
                                             global_step=global_step)
        
        pre_x = tf.placeholder(tf.int32,
                                [None, max_document_length] 
                                )                                     
        pre_processor = tf.one_hot(pre_x,vocab_size)
                                             
        print "Initialiing Saver: "
        #checkpoint saver
        saver = tf.train.Saver(tf.all_variables())
        print "Done Initializing Saver"
        
        # Write vocabulary
        vocab_processor.save(os.path.join("out/vocab"))
        
        sess.run(tf.initialize_all_variables())
        
        
        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.x_input: x_batch,
              cnn.y_input: y_batch,
              cnn.dropout_param: dropout_param
            }
            _, step, loss, accuracy = sess.run(
                [train_op, global_step, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            #train_summary_writer.add_summary(summaries, step)
            
        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.x_input: x_batch,
              cnn.y_input: y_batch,
              cnn.dropout_param: 1.0
            }
            step, loss, accuracy = sess.run(
                [global_step, cnn.loss, cnn.accuracy],
                feed_dict)
            
            #if writer:
            #    writer.add_summary(summaries, step)
            return step, loss, accuracy
        
            
        #Training Loop for RT Database
        rt_nit = int(rt_train_length/batch_size*num_epochs)
        print "Total Number of Iterations for RT Data: ", rt_nit
        rt_npe = int(rt_train_length/batch_size)
        print "Number of Iterations per Epoch: ", rt_npe
        evaluate_every = rt_npe
            
        for batch in range(rt_nit):
            choices = np.random.choice(rt_train_length,batch_size)
            x_batch = x_rt_train[choices]
            y_batch = y_rt_train[choices]
            pre_xo = sess.run([pre_processor],feed_dict={pre_x:x_batch})
            x_batch = pre_xo[0].reshape(x_batch.shape[0],-1,1,1)
            
            if batch%rt_npe == 0:
                shuffle_indices = np.random.permutation(
                        range(rt_train_length))
                x_rt_train = x_rt_train[shuffle_indices]
                y_rt_train = y_rt_train[shuffle_indices]
            
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % evaluate_every == 0:
                print("\nEvaluation:")
                loss = []
                accuracy = []
                val_batch_size = 1
                for i in range(int(rt_val_length/val_batch_size)):
                    x_val_batch = x_rt_val[i*val_batch_size:(i+1)*val_batch_size]
                    pre_xo = sess.run([pre_processor],feed_dict={pre_x:x_val_batch})
                    x_val_batch = pre_xo[0].reshape(x_val_batch.shape[0],-1,1,1)
                    
                    step, loss_, accuracy_ = dev_step(
                            x_val_batch,
                            y_rt_val[i*val_batch_size:(i+1)*val_batch_size])
                    loss.append(loss_)
                    accuracy.append(accuracy_)
                
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, np.mean(loss), np.mean(accuracy)))
                print("")
                saver.save(sess, "out/model")
                
        final_rt_loss = np.mean(loss)
        final_rt_accuracy = np.mean(accuracy)
        
        
        print ""
        print ""
        print "+++++++++++++++++++++++"
        print "+++++++++++++++++++++++"
        print "RT ACCURACY: ", final_rt_accuracy
        print "+++++++++++++++++++++++"
        print "+++++++++++++++++++++++"
        print ""
        print ""
