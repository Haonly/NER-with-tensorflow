
import os
import tensorflow as tf

class BaseModel(object):

    def __init__(self, config):

        self.config = config
        self.logger = config.logger
        self.sess   = None
        self.saver  = None

    def reinitialize_weights(self, scope_name):
        variables = tf.contrib.framework.get_variables(scope_name)
        init = tf.variables_initializer(variables)
        self.sess.run(init)

    def add_train_op(self, lr_method, lr, loss):

        with tf.variable_scope("train_step"):
            
            optimizer=tf.train.AdamOptimizer(lr)
            self.train_op=optimizer.minimize(loss)

    def initialize_session(self): ## session open
        self.logger.info("Initializing tf session")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def restore_session(self, dir_model):
        self.logger.info("Reloading the latest trained model...")
        self.saver.restore(self.sess, dir_model)

    def save_session(self):
        if not os.path.exists(self.config.dir_model):
            os.makedirs(self.config.dir_model)
        self.saver.save(self.sess, self.config.dir_model)

    def close_session(self): ## session close
        self.sess.close()

    def train(self, train, dev):
  
        best_score = 0
        nepoch_no_imprv = 0 # for early stopping

        for epoch in range(self.config.nepochs):
            self.logger.info("Epoch {:} out of {:}".format(epoch + 1,self.config.nepochs))

            score = self.run_epoch(train, dev, epoch)
            self.config.lr *= self.config.lr_decay # decay learning rate

            # early stopping and saving best parameters
            if score >= best_score:
                nepoch_no_imprv = 0
                self.save_session()
                best_score = score
                self.logger.info("- new best score!")
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                    self.logger.info("- early stopping {} epochs without "\
                            "improvement".format(nepoch_no_imprv))
                    break

    def evaluate(self, test):

        self.logger.info("Testing model over test set")
        metrics = self.run_evaluate(test)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)
