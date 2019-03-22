import matplotlib.pyplot as plt

def plot(num_graphs, epochs, accuracy, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x_axis = [x+1 for x in range(epochs)]
    plt.xticks(x_axis)
    for i in range(num_graphs):
        ax.plot(x_axis, accuracy[i], color=labels[i][1],\
                label=labels[i][0])
    plt.legend(loc="lower right")
    ax.grid(True)
    ax.set_xlabel('Epochs')
    ax.set_title('Accuracy (%) on the test data')
    plt.show()
