# Solution is available in the other "solution.py" tab
import tensorflow as tf


def run():
    output = None
    logit_data = [2.0, 1.0, 0.1]
    logits = tf.placeholder(tf.float32)
    
    # TODO: Calculate the softmax of the logits
    softmax = tf.nn.softmax(logits)     
    
    with tf.Session() as sess:
        # TODO: Feed in the logit data
        output = sess.run(softmax, feed_dict={logits: logit_data})
        #print(output)
    return output

#THIS DOES NOT WORKD need the run def !!
# #def run():
# output = None
# logit_data = [2.0, 1.0, 0.1]
# logits = tf.placeholder(tf.float32)

# # TODO: Calculate the softmax of the logits
# softmax = tf.nn.softmax(logits)     

# with tf.Session() as sess:
    
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     # TODO: Feed in the logit data
#     output = sess.run(softmax, feed_dict={logits: logit_data})
# print(output)
#     #return output