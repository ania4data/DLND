import tensorflow as tf

#def run():
        
softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)

# TODO: Print cross entropy from session

with tf.Session() as sess:
    
    cross_entropy = -tf.reduce_sum(tf.multiply(one_hot,tf.log(softmax)))   #for vector multiply
    
    output=sess.run(cross_entropy,feed_dict={one_hot:one_hot_data,softmax:softmax_data})
    print(output)
#    return output