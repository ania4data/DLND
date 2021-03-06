# Solution is available in the other "solution.py" tab
import tensorflow as tf

output = None
hidden_layer_weights = [
    [0.1, 0.2, 0.4],
    [0.4, 0.6, 0.6],
    [0.5, 0.9, 0.1],
    [0.8, 0.2, 0.8]]
out_weights = [
    [0.1, 0.6],
    [0.2, 0.1],
    [0.7, 0.9]]

# Weights and biases
weights = [
    tf.Variable(hidden_layer_weights),
    tf.Variable(out_weights)]
biases = [
    tf.Variable(tf.zeros(3)),
    tf.Variable(tf.zeros(2))]

# Input
features = tf.Variable([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0], [11.0, 12.0, 13.0, 14.0]])


# TODO: Create Model
hidden_out1=tf.add(tf.matmul(features,weights[0]),biases[0])
hidden_out1=tf.nn.relu(hidden_out1)
output=tf.add(tf.matmul(hidden_out1,weights[1]),biases[1])
#output=tf.nn.softmax(output)

# TODO: Print session results

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer()) #even without random weight
    print(sess.run(output))