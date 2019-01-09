"""network2.py
~~~~~~~~~~~~~~
An improved version of network.py, implementing the stochastic
gradient descent learning algorithm for a feedforward neural network.
Improvements include the addition of the cross-entropy cost function,
regularization, and better initialization of network weights.  Note
that I have focused on making the code simple, easily readable, and
easily modifiable.  It is not optimized, and omits many desirable
features.
Note by Pedro:  I modified the code based on the paper
                An overview of gradient descent optimization algorithms
                by Sebastian Ruder
                In this code I'm implementing the momentum-based and
                the adagrad algorithms with success, but the adadelta
                is giving me errors
"""

#### Libraries
# Standard library
import json
import random
import sys

# Third-party libraries
import numpy as np


#### Define the quadratic and cross-entropy cost functions

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.
        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).
        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.
        """
        return (a-y)


#### Main Network class
class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost=cost

    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.
        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.
        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.
        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison.  It
        will usually be better to use the default weight initializer
        instead.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def velocity_initializer(self):
        self.vel_w = [np.zeros(w.shape) for w in self.weights]
        self.vel_b = [np.zeros(b.shape) for b in self.biases]
    
    def g_initializer(self, nabla_w, nabla_b):
        self.Gtii_w = np.diag(np.diag(np.square(nabla_w)))
        self.Gtii_b = np.diag(np.diag(np.square(nabla_b)))
    
    def e_grad_initializer(self, nabla_w, nabla_b, mu):
        self.e_grad_w = (1 - mu)*np.square(nabla_w)
        self.e_grad_b = (1 - mu)*np.square(nabla_b)

    def e_delta_initializer(self, delta_w, delta_b, mu):
        self.e_delta_w = (1 - mu)*np.square(delta_w)
        self.e_delta_b = (1 - mu)*np.square(delta_b)

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, no_improvement_in, mini_batch_size,
            eta=0,\
            validation_data=None,
            n_epochs=0,
            lmbda=0.0,
            mu=0.0,
            fudge_factor=0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            lrs=False,
            stop=False,
            nag=False,
            adagrad=False,
            adadelta=False):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.
        """
        if (validation_data == None) and lrs:
            raise ValueError('For LRS please insert validation data')

        # Initializing velocities
        self.velocity_initializer()

        evaluation_data = list(evaluation_data)
        training_data = list(training_data)
        
        validation_data = list(validation_data)
        validation_accuracy = []

        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        max_accuracy = 0
        max_accuracy_vd = 0
        i = 0
        epochs = 0
        previous_eta = eta

        while True:
            epochs += 1
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                # Initializing the mini_batch matrix with the first
                # column vector. After acquiring the data this will be
                # deleted
                mini_batch_input = mini_batch[0][0]
                mini_batch_output = mini_batch[0][1]
                # doing the rest
                for x, y in mini_batch:
                    mini_batch_input = np.append(mini_batch_input, x, axis = 1)
                    mini_batch_output = np.append(mini_batch_output, y, axis = 1)
                # Deleting the first column
                mini_batch_input = np.delete(mini_batch_input, 0, 1)
                mini_batch_output = np.delete(mini_batch_output, 0, 1)
                # Uptading parameters following the NAG algorithm
                if nag:
                    self.weights = [w - mu*vw
                                    for w,vw in zip(self.weights, self.vel_w)]
                    self.biases = [b - mu*vb
                                   for b,vb in zip(self.biases, self.vel_b)]
                # Updating the parameters through backpropagation
                if (epochs == 1) and adagrad:
                    self.update_mini_batch(mini_batch_size, mini_batch_input, \
                                       mini_batch_output, lmbda, mu, n,\
                                       adagrad, adadelta, eta=eta, \
                                       fudge_factor=fudge_factor, init=True)
                elif (epochs == 1) and adadelta:
                    self.update_mini_batch(mini_batch_size, mini_batch_input, \
                                       mini_batch_output, lmbda, mu, n,\
                                       adagrad, adadelta, fudge_factor=fudge_factor,\
                                       init=True)
                else:
                    self.update_mini_batch(mini_batch_size, mini_batch_input, \
                                        mini_batch_output, lmbda, mu, n,\
                                        adagrad, adadelta, eta=eta)

            if monitor_training_accuracy or monitor_evaluation_cost or\
               monitor_training_cost or monitor_evaluation_accuracy:
                print ("\nEpoch %s training complete" % epochs)
            if monitor_training_cost:
                cost = self.total_cost(training_data)
                training_cost.append(cost)
                print ("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print ("Accuracy on training data: {} / {}".format(\
                    accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print ("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print ("Accuracy on evaluation data: {} / {}".format(\
                    self.accuracy(evaluation_data), n_data))
                if accuracy > max_accuracy:
                    max_accuracy = accuracy

            if lrs:
                accuracy = self.accuracy(validation_data)
                validation_accuracy.append(accuracy)
                #print('Accuracy on validation data: {} / {}'\
                #      .format(self.accuracy(validation_data), len(validation_data)))
                if accuracy > max_accuracy_vd:
                    i = 0
                    max_accuracy_vd = accuracy
                else:
                    i += 1
                
                if (i == no_improvement_in):
                    if ((previous_eta/eta) == 128) and not stop:
                        break
                    eta /= 2
                    #print('Dividing eta by 2: {}'.format(eta))
                    i = 0
                    max_accuracy_vd = accuracy
                
            if (epochs == n_epochs) and stop:
                break
        if monitor_evaluation_accuracy:
            print ("\nThe maximum accuracy on evaluation data was {} after {} epochs\n"\
                    .format(max_accuracy, epochs))

        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

    def update_mini_batch(self, mini_batch_size, mini_batch_input, mini_batch_output,\
                             lmbda, mu, n, adagrad, adadelta, eta=0, fudge_factor=0,\
                             init=False):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, `` is the regularization parameter, and
        ``n`` is the total size of the training data set.
        """
        nabla_b, nabla_w = self.backprop(mini_batch_input, mini_batch_output)
        
        if adagrad:
            if init:
                # Initializing Gtii matrix
                self.g_initializer(nabla_w, nabla_b)
            else:
                self.Gtii_w += np.diag(np.diag(np.square(nabla_w)))
                self.Gtii_b += np.diag(np.diag(np.square(nabla_b)))
            # Updating the gradient
            nabla_w = np.divide(nabla_w, np.sqrt(self.Gtii_w + fudge_factor))
            nabla_b = np.divide(nabla_b, np.sqrt(self.Gtii_b + fudge_factor))

        if adadelta:
            # Taking the average value of the gradient (This part is working)
            nabla_w = np.divide(nabla_w, mini_batch_size)
            nabla_b = np.divide(nabla_b, mini_batch_size)
            
            if init:
                #Initializing the matrix
                self.e_grad_initializer(nabla_w, nabla_b, mu)

                rms_grad_w = np.sqrt(self.e_grad_w + fudge_factor)
                rms_grad_b = np.sqrt(self.e_grad_b + fudge_factor)

                # Initializing deltas ~(with eta=1 multiplying)~
                delta_w = -np.divide(nabla_w*np.sqrt(fudge_factor), rms_grad_w)
                delta_b = -np.divide(nabla_b*np.sqrt(fudge_factor), rms_grad_b)

                self.e_delta_initializer(delta_w, delta_b, mu)
            else:
                self.e_grad_w = mu*self.e_grad_w + (1 - mu)*np.square(nabla_w)
                self.e_grad_b = mu*self.e_grad_b + (1 - mu)*np.square(nabla_b)

                rms_grad_w = np.sqrt(self.e_grad_w + fudge_factor)
                rms_grad_b = np.sqrt(self.e_grad_b + fudge_factor)

                rms_prev_delta_w = np.sqrt(self.e_delta_w + fudge_factor)
                rms_prev_delta_b = np.sqrt(self.e_delta_b + fudge_factor)

                # Now, calculating the actual deltas
                delta_w = -nabla_w*np.divide(rms_prev_delta_w, rms_grad_w)
                delta_b = -nabla_b*np.divide(rms_prev_delta_b, rms_grad_b)

                #delta_w = [dw*rms_pdw
                #           for dw, rms_pdw in zip(delta_w, rms_prev_delta_w)]
                #delta_b = [db*rms_pdb
                #           for db, rms_pdb in zip(delta_b, rms_prev_delta_b)]
                #delta_w = -np.matmul(np.divide(rms_prev_delta_w, rms_grad_w), nabla_w)
                #delta_b = -np.matmul(np.divide(rms_prev_delta_b, rms_grad_b), nabla_b)

            # Updating the parameters
            self.weights = [(1-eta*(lmbda/n))*w + dw
                        for w, dw in zip(self.weights, delta_w)]
            self.biases = [b + db
                        for b, db in zip(self.biases, delta_b)]

            if not init:
                # Update the e_deltas for the next iteration
                self.e_delta_w = mu*self.e_delta_w + (1 - mu)*np.square(delta_w)
                self.e_delta_b = mu*self.e_delta_b + (1 - mu)*np.square(delta_b)

        else:
            self.vel_w = [mu*vw + eta*(nw/mini_batch_size)
                        for vw, nw in zip(self.vel_w, nabla_w)]
            self.vel_b = [mu*vb + eta*(nb/mini_batch_size)
                        for vb, nb in zip(self.vel_b, nabla_b)]

            self.weights = [(1-eta*(lmbda/n))*w - vw
                            for w, vw in zip(self.weights, self.vel_w)]
            self.biases = [b - vb
                        for b, vb in zip(self.biases, self.vel_b)]


    def backprop(self, mini_batch_input, mini_batch_output):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = mini_batch_input
        activations = [mini_batch_input] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], mini_batch_output)

        nabla_b[-1] = delta[-1].sum()
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp

            nabla_b[-l] = delta[-l].sum()
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.
        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.
        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### Miscellaneous functions
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
return sigmoid(z)*(1-sigmoid(z))
