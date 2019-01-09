import mnist_loader
import network2_with_adadelta

net = network2_with_adadelta.Network([784, 100, 10], cost=network2_with_adadelta.CrossEntropyCost)

training_data, validation_data, test_data = \
mnist_loader.load_data_wrapper()
net.large_weight_initializer()

print("Learning with Adadelta+LRS (E=1e-8)...")

evaluation_cost, evaluation_accuracy,_,_ = net.SGD(training_data, 1, 10,\
    validation_data=validation_data, n_epochs=21, lmbda=5.0, mu=0.4,\
    fudge_factor=1e-8, evaluation_data=test_data, monitor_evaluation_cost=True,\
    monitor_evaluation_accuracy=True, lrs=True, stop=True, adadelta=True)
