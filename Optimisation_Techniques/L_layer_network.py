# this code will contain a running implementation of L layer network as well as optimisation technique like mini batch as well as
# momentum
import numpy as np
import math
import matplotlib.pyplot as plt

from Optimisation_Techniques.deep_neural_utility import initialize_parameters_deep, L_model_forward, compute_cost, \
    L_model_backward, update_parameters, load_data, predict


def randomised_mini_batch(X,y,batch_size=64,seed=0):
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []
    permutation_list = list(np.random.permutation(m))
    shuffle_X = X[:,permutation_list]
    shuffle_y = y[:, permutation_list]
    num_complete_batches = math.floor(m/batch_size)
    for i in range(num_complete_batches):
        mini_batch_X = shuffle_X[:,i*batch_size:(i+1)* batch_size]
        mini_batch_y = shuffle_y[:, i * batch_size:(i + 1) * batch_size]
        mini_batch = (mini_batch_X,mini_batch_y)
        mini_batches.append(mini_batch)
    if(m % batch_size!=0):
        mini_batch_X = shuffle_X[:, num_complete_batches * batch_size:]
        mini_batch_y = shuffle_y[:, num_complete_batches * batch_size:]
        mini_batch = (mini_batch_X, mini_batch_y)
        mini_batches.append(mini_batch)
    return mini_batches

def model(X,y,layer_dims,optimiser="gd",learning_rate=0.007,mini_batch_size=64,number_of_epochs=1000,print_cost=True):
    L = len(layer_dims)
    param = initialize_parameters_deep(layer_dims)

    costs = []
    seed = 10
    for i in range(number_of_epochs):
        seed = seed+1
        mini_batches = randomised_mini_batch(X,y,mini_batch_size,seed)
        for mini_batch in mini_batches:
            (batch_X, batch_y) = mini_batch
            a, cache = L_model_forward(batch_X,param)
            cost = compute_cost(a,batch_y)
            grads = L_model_backward(a, batch_y, cache)
            if optimiser=="gd":
                param = update_parameters(param,grads,learning_rate)
                # Print the cost every 1000 epoch
        if print_cost and i % 100 == 0:
            print("Cost after epoch %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()
    return param


train_x_orig, train_y_orig, test_x_orig, test_y_orig,classes = load_data()
train_X = train_x_orig.reshape(train_x_orig.shape[0],-1).T
train_y = train_y_orig.reshape(1,-1)
test_X = test_x_orig.reshape(test_x_orig.shape[0],-1).T
test_y = test_y_orig.reshape(1,-1)
train_X = train_X / 255
test_X = test_X / 255

layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_y, layers_dims, optimiser="gd")

# Predict train accuracy
predictions = predict(train_X, train_y, parameters)

# Predict test accuracy
predictions = predict(test_X, test_y, parameters)
