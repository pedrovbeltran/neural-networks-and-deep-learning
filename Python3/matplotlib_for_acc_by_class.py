import matplotlib.pyplot as plt

def plot(num_graphs, epochs, accuracy):
    
    fig, axes = plt.subplots(2, 5, sharex=True, sharey=True, figsize=(10,5))
    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.suptitle('Accuracy (%) by class')

    x_axis = [x+1 for x in range(epochs)]
    plt.xticks(x_axis)
    for i, row in enumerate(axes):
        for j, cell in enumerate(row):
            cell.plot(x_axis, accuracy[j + i*5])
            cell.set_title(str(j + i*5))
            #cell.grid(True)
    
    plt.show()
