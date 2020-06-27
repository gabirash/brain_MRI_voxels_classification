##################### Brain MRI Voxels Classification Project #####################
import os
import random
import cv2
from model import model
import numpy as np
import matplotlib.pyplot as plt
import json

#####################  Dataset Creation #####################
def create_dataset(train_or_val):
    # create lables (0 - neg, 1 - pos)
    data_classes = ["neg", "pos"]
    # load images
    data_arr = []
    datadir = os.path.dirname(os.path.abspath(__file__))
    if train_or_val == "training":
        path = os.path.join(datadir, "training")
    else:
        path = os.path.join(datadir, "validation")
    for data_class in data_classes:
        label_num = data_classes.index(data_class)
        for img in os.listdir(path):
            if data_class in img:
                img = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                # normalize images
                img = img/255.0
                data_arr.append([img, label_num])
    random.shuffle(data_arr)
    x = []
    y = []
    for img, label in data_arr:
        x.append(img.reshape(-1))
        y.append(label)
    x = np.array(x)
    y = np.array(y)
    y = y.reshape(y.shape[0],1)
    # # Normalize the data: subtract the mean image
    # mean_image = np.mean(x, axis=0)
    # x -= mean_image

    return x, y

###################### Optimaze hyper-parameters ######################
# generate random values in a given range
def generate_random_hyperparams(lr_min, lr_max, reg_min, reg_max, h_min, h_max, b_min, b_max):
    lr = 10**np.random.uniform(lr_min,lr_max)
    reg = 10**np.random.uniform(reg_min,reg_max)
    hidden = np.random.randint(h_min, h_max)
    batch_size = np.random.randint(b_min, b_max)
    return lr, reg, hidden, batch_size

# get random hyperparameters given arrays of potential values
def random_search_hyperparams(lr_values, reg_values, h_values, b_values):
    lr = lr_values[np.random.randint(0,len(lr_values))]
    reg = reg_values[np.random.randint(0,len(reg_values))]
    hidden = h_values[np.random.randint(0,len(h_values))]
    batch_size = b_values[np.random.randint(0,len(b_values))]
    return lr, reg, hidden, batch_size

def print_results_sorted(results):
    def takeacc(elm):
        return elm[0]

    results.sort(key=takeacc)
    for r in results: print(r)

def make_json(trained_dict, path_to_save):
       """
       make json file with trained parameters.
       W1: numpy arrays of shape (1024, nn_h_dim)
       W2: numpy arrays of shape (nn_h_dim, 1)
       b1: numpy arrays of shape (1, nn_h_dim)
       b2: numpy arrays of shape (1, 1)
       id1: id1 - int
       id2: id2 - int
       activation1: one of only: 'sigmoid', 'tanh', 'ReLU', 'final_act' - str
       activation2: 'sigmoid' - str
        number of neirons in hidden layer - int
       :param nn_h_dim: trained_dict = {'weights': (W1, W2),
                                       'biases': (b1, b2),
                                       'nn_hdim': nn_h_dim,
                                       'activation_1': activation1,
                                       'activation_2': activation2,
                                       'IDs': (id1, id2)}
       """
       file_path = os.path.join(path_to_save, 'trained_dict_{}_{}.json'.format(
           trained_dict.get('IDs')[0], trained_dict.get('IDs')[1])
                                )
       with open(file_path, 'w') as f:
           json.dump(trained_dict, f, indent=4)

#####################  Main  #####################
if __name__ == '__main__':
    #     x_train: (512, 1024) y_train: (512, )
    x_train, y_train = create_dataset("training")
    x_val, y_val = create_dataset("validation")
    print("training data: x_train {}, y_train {}".format(x_train.shape, y_train.shape))
    print("validation data: x_val {}, y_val {}".format(x_val.shape, y_val.shape))

    # const hyper parameters"
    np.random.seed(1)
    num_iter = 5000
    image_vector_size = 1024
    output_size = 1
    std = np.sqrt(2.0 / 1024)  # according to cs231n course weight std init.
    verbose = True
    # hyper parameters optimization:
    hidden_size = 53
    batch_size = 117
    lr = 0.038446182351894766
    reg = 1.8653823506313295e-06
    # (0.9807692307692307, 1.0, {'lr': 0.038446182351894766, 'reg': 1.8653823506313295e-06, 'hidden': 53, 'batch': 117})

    search_parameters = False
    if search_parameters: # random search for parameters values
        best_val=0
        results = []
        params = ['lr', 'reg', 'hidden', 'batch']
        for i in range(20):
            # Use generate_random function with 1000 iterations
            lr, reg, hidden_size, batch_size = generate_random_hyperparams(-5, -1, -5, 0, 1, 100, 1, 256) #1000 iters
            lr, reg, hidden_size, batch_size = generate_random_hyperparams(-3, -1, -6, -4, 8, 92, 32, 256) #1000 iters
            lr, reg, hidden_size, batch_size = generate_random_hyperparams(-2, -1, -6, -5, 40, 60, 100, 230)  # 1000 iters

            # According to the previous results, reduce the exploration by selecting set of fixed ranges
            # use this ranges in the random search function to explore random combinations
            lr, reg, hidden_size, batch_size = random_search_hyperparams(   [0.038446182351894766, 0.038446182351894766, 0.031557371805925306],
                                                                            [1.8653823506313295e-06, 2.47040676891957e-06, 4.2812448183270965e-06],
                                                                            [50, 53, 55], [117,200,180])

            net = model(image_vector_size, hidden_size, output_size, std=std)
            stats = net.train(x=x_train, y=y_train, x_val=x_val, y_val=y_val, learning_rate=lr,
                              reg=reg, num_iters=num_iter,
                              batch_size=batch_size, verbose=verbose)

            # Predict on the training set
            train_accuracy = net.get_accuracy(x_train,y_train)

            # Predict on the validation set
            val_accuracy = net.get_accuracy(x_val,y_val)

            # Save best values
            if val_accuracy > best_val:
                best_val = val_accuracy
                best_net = net
                best_stats = stats

            # save results
            results.append((val_accuracy, train_accuracy, dict(zip(params,[lr, reg, hidden_size, batch_size]))))
            # Print results
            print('lr %e reg %e hid %d batch %d train accuracy: %f val accuracy: %f' % (
                lr, reg, hidden_size, batch_size, train_accuracy, val_accuracy))
        print('best validation accuracy achieved: %f' % best_val)

        # sort and print results
        print_results_sorted(results)

    else: #use default parameters
        net = model(image_vector_size, hidden_size, output_size, std=std)
        stats, params_arr = net.train(x=x_train, y=y_train, x_val=x_val, y_val=y_val, learning_rate=lr,
                          reg=reg, num_iters=num_iter,
                          batch_size=batch_size, verbose=verbose)
        # Predict on the training set
        train_accuracy = net.get_accuracy(x_train, y_train)

        # Predict on the validation set
        val_accuracy = net.get_accuracy(x_val, y_val)

    if verbose:
        # Plot the loss function and train / validation accuracies
        plt.subplot(2, 1, 1)
        plt.plot(stats['loss_history'], label='train')
        plt.plot(stats['val_loss_history'], label='val')
        plt.title('Loss versus iteration')
        plt.xlabel('Iteration')
        plt.legend()
        plt.ylabel('Loss')

        plt.subplot(2, 1, 2)
        plt.plot(stats['train_acc_history'], label='train')
        plt.plot(stats['val_acc_history'], label='val')
        plt.title('Accuracy versus epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Classification accuracy')
        plt.legend()
        plt.show()
    print("finished training")

    W1 = params_arr['W1'].tolist()
    W2 = params_arr['W2'].tolist()
    b1 = params_arr['b1'].tolist()
    b2 = params_arr['b2'].tolist()

    trained_dict = {'weights': (W1, W2),
                    'biases': (b1, b2),
                    'nn_hdim': 53,
                    'activation_1': 'ReLU',
                    'activation_2': 'sigmoid',
                    'IDs': (204219273, 312178999)}

    path_to_save = os.path.dirname(os.path.abspath(__file__))

    make_json(trained_dict, path_to_save)



