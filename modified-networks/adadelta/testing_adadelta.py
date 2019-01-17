import mnist_loader
import network2_adadelta_only

net = network2_adadelta_only.Network([784, 100, 10], cost=network2_adadelta_only.CrossEntropyCost)

training_data, validation_data, test_data = \
mnist_loader.load_data_wrapper()
net.large_weight_initializer()

# Respectively the number of epochs and the fudge_factor
# for easy test
epochs = 20
fudge_factor = 1e-3

print("Learning with Adadelta+LRS (E={:.0e})...".format(fudge_factor))

net.SGD(training_data, epochs, 10,\
        evaluation_data=test_data, lmbda=5.0, mu=0.4, fudge_factor=fudge_factor,
        monitor_evaluation_cost=True, monitor_evaluation_accuracy=True)
