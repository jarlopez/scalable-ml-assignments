def task1():
    import tensorflow as tf
    from hops import tensorboard
    from hops import hdfs
    
    from tensorflow.examples.tutorials.mnist import input_data
    
    fashion_mnist = input_data.read_data_sets('input/data', one_hot=True)
    
    # Hardcoded params
    num_ch = 1
    num_classes = 10
    image_height = image_width = 28
    
    # 1. Define variables and placeholders
    X = tf.placeholder(tf.float32, shape=[None, image_height, image_width, num_ch])
    Y_ = tf.placeholder(tf.float32, shape=[None, 10])
    
    W = tf.Variable(tf.zeros([image_height * image_width, num_classes]))
    b = tf.Variable(tf.zeros([num_classes]))
    
    # Flatten images into a single vector
    XX = tf.reshape(X, [-1, image_height * image_width])
    
    # 2. Define the model Y = softmax(XX*W + b)
    Y = tf.nn.softmax(tf.matmul(XX, W) + b )
    
    # 3a. Define the loss function (neg. log loss) and 3b.compute its mean over the trained batches
    #cross_entropy_mean = tf.reduce_mean(-tf.reduce_sum(Y_ * tf.log(Y), reduction_indices=[1]))
    cross_entropy_mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y, labels=Y_))
    
    # 4. Accuracy definition
    correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # Define optimizers used
    
    train_step_gd = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy_mean)
    train_step_adam = tf.train.AdamOptimizer(0.005).minimize(cross_entropy_mean)

    
    # Define accuracy
    prediction_correctness = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction_correctness, tf.float32))
    
    init =  tf.global_variables_initializer()
    
    def epochs(train, test, train_step, num_epochs=100, batch_size=100):
        sess = tf.Session()
        sess.run(init)
        
        accuracies = []
        losses = []
        for epoch in range(num_epochs):
            
            for it in range(100):
                batch_xs, batch_ys = train.next_batch(batch_size)
                feed_dict =  {XX: batch_xs, Y_: batch_ys}
                sess.run(train_step, feed_dict)

            # Compute accuracy and loss every 100 rounds
            feed_dict = {XX: test.images, Y_: test.labels}
            acc = sess.run(accuracy, feed_dict)
            loss = sess.run(cross_entropy_mean, feed_dict)
            
            accuracies.append(acc)
            losses.append(loss)
        return (accuracies, losses)


    acc, loss = epochs(fashion_mnist.train, fashion_mnist.test, train_step_gd)
    
    print("Loss: {}".format(loss))
    print("Accuracy: {}".format(acc))
    