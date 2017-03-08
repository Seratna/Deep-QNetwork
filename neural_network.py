"""

"""
import tensorflow as tf

from models import Model, FullyConnected


class NeuralNetwork(object):
    """

    """
    def __init__(self, neuron_list: list, scope):
        with tf.variable_scope(scope) as scope:
            # placeholder of input
            self.x = tf.placeholder(tf.float32, shape=[None, neuron_list[0]])

            # build hidden layers
            self.layers = []
            x = self.x
            for i, num_neurons in enumerate(neuron_list[1:], start=1):
                activation = tf.nn.relu if i < (len(neuron_list)-1) else None  # NO activation for the last layer

                # create a fully connected layer
                fully_connected_layer = FullyConnected(num_neurons, activation=activation, scope='layer_{}'.format(i))
                self.layers.append(fully_connected_layer)
                x = fully_connected_layer(x)

            self.output = x

            self.y = tf.placeholder(tf.float32, shape=[None, neuron_list[-1]])  # training labels
            self.loss = tf.nn.l2_loss(self.y - self.output)
            self.training = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
            # self.training = tf.train.RMSPropOptimizer(0.1).minimize(self.loss)


    def train(self, x, y, session: tf.Session):
        _, loss = session.run([self.training, self.loss], feed_dict={self.x: x, self.y: y})
        return loss

    def copy_network(self, network, session: tf.Session):
        """

        :param network:
        :type network: NeuralNetwork
        :return:
        """
        assert len(self.layers) == len(network.layers)
        for self_layer, source_layer in zip(self.layers, network.layers):  # type: FullyConnected
            """:type : FullyConnected"""
            session.run(tf.assign(self_layer.weights, source_layer.weights))
            session.run(tf.assign(self_layer.bias, source_layer.bias))


def main():
    pass


if __name__ == '__main__':
    main()