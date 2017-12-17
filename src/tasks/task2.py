def task2():
    import tensorflow as tf

    from hops import tensorboard
    from hops import hdfs
    from tensorflow.examples.tutorials.mnist import input_data
    
    fashion_mnist = input_data.read_data_sets('data/fashion', one_hot=True, source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
    
    # Helpers
    def weight_var(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_var(shape, value=0.1):
        initial = tf.constant(value, shape=shape)
        return tf.Variable(initial)
    
    def bias_var_z(shape):
        return tf.Variable(tf.zeros(shape))
    
    def layer(tensor, in_dim, out_dim, name, activation=tf.nn.sigmoid):
        weights = weight_var([in_dim, out_dim])
        biases = bias_var_z([out_dim])
        pre = tf.matmul(tensor, weights) + biases
        post = activation(pre)
        tf.summary.histogram('activations', post)
        return post
    
    # Hardcoded params
    num_ch = 1
    num_classes = 10
    image_height = image_width = 28
    layer_widths = [200, 100, 60, 30, 10]
    
    # 1. Define variables and placeholders
    X = tf.placeholder(tf.float32, shape=[None, image_height, image_width, num_ch])
    Y_ = tf.placeholder(tf.float32, shape=[None, 10])

    XX = tf.reshape(X, [-1, image_height * image_width])
    
    HSig1 = layer(XX,    784,             layer_widths[0], 'sigmoid-1', tf.nn.sigmoid)
    HSig2 = layer(HSig1, layer_widths[0], layer_widths[1], 'sigmoid-2', tf.nn.sigmoid)
    HSig3 = layer(HSig2, layer_widths[1], layer_widths[2], 'sigmoid-3', tf.nn.sigmoid)
    HSig4 = layer(HSig3, layer_widths[2], layer_widths[3], 'sigmoid-4', tf.nn.sigmoid)
    Y     = layer(HSig4, layer_widths[3], layer_widths[4], 'identity',  tf.identity)

#     W1 = tf.Variable(tf.truncated_normal([784, 200], stddev=0.1))
#     W2 = tf.Variable(tf.truncated_normal([200, 100], stddev=0.1))
#     W3 = tf.Variable(tf.truncated_normal([100, 60 ], stddev=0.1))
#     W4 = tf.Variable(tf.truncated_normal([60,  30 ], stddev=0.1))
#     W5 = tf.Variable(tf.truncated_normal([30,  10 ], stddev=0.1))

#     B1 = tf.Variable(tf.zeros([200]))
#     B2 = tf.Variable(tf.zeros([100]))
#     B3 = tf.Variable(tf.zeros([60 ]))
#     B4 = tf.Variable(tf.zeros([30 ]))
#     B5 = tf.Variable(tf.zeros([10 ]))

#     #Define the model
#     XX = tf.reshape(X, [-1, 784])   
#     Y1 = tf.nn.sigmoid(tf.matmul(XX, W1) + B1)
#     Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)
#     Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + B3)
#     Y4 = tf.nn.sigmoid(tf.matmul(Y3, W4) + B4)
#     Ylogits = tf.matmul(Y4, W5) + B5
#     Y = tf.nn.softmax(Ylogits)
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y, labels=Y_))
    
    tf.summary.scalar('cross_entropy', cross_entropy)
        
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    
    with tf.name_scope('train'):
        with tf.name_scope('gradient_descent'):
            train_step_gd = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        with tf.name_scope('adam_optimizer'):
            train_step_adam = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)

    
    # Define accuracy
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction =  tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("cost", cross_entropy)
    tf.summary.scalar("accuracy", accuracy)
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    
    
    logdir = tensorboard.logdir()
    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(logdir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(logdir + '/test')
        
    def epochs(train, test, train_step, num_epochs=100, batch_size=100):
        sess.run(init)
        
        accuracies = []
        losses = []

        for epoch in range(10000):
            for it in range(100):
                batch_xs, batch_ys = train.next_batch(batch_size)
                feed_dict =  {XX: batch_xs, Y_: batch_ys}
                _, summary = sess.run([train_step, summary_op], feed_dict=feed_dict)
                train_writer.add_summary(summary, epoch * 100 + it)

            # Compute accuracy and loss every 100 rounds
            feed_dict = {XX: test.images, Y_: test.labels}
            summary, acc = sess.run([summary_op, accuracy], feed_dict=feed_dict)
            loss = sess.run(cross_entropy, feed_dict=feed_dict)

            accuracies.append(acc)
            losses.append(loss)
            test_writer.add_summary(summary, epoch)
        return (accuracies, losses)

    acc, loss = epochs(fashion_mnist.train, fashion_mnist.test, train_step_gd)

    train_writer.close()
    test_writer.close()
    writer.close()
    
    print("Accuracy: {}".format(acc))
    print("Loss: {}".format(loss))
    

