import matplotlib.pyplot as plt

def plot(num_graphs, epochs, accuracy):
    
    fig, ax = plt.subplots(1, 1)
    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.suptitle('Accuracy (%) on test data')

    x_axis = [x+1 for x in range(epochs)]
    plt.xticks(x_axis)
    ax.plot(x_axis, accuracy)
    ax.grid(True)
    
    plt.show()
