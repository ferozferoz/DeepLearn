import numpy as np
import h5py

class NNMOdel():

    def _load_dataset(self,training_path, test_path):

        train_dataset = h5py.File(training_path, "r")
        test_dataset = h5py.File(test_path, "r")
        train_set_x_orig = np.array(train_dataset["train_set_x"])  # your train set features
        train_set_y_orig = np.array(train_dataset["train_set_y"])  # your train set labels
        test_set_x_orig = np.array(test_dataset["test_set_x"])  # your test set features
        test_set_y_orig = np.array(test_dataset["test_set_y"])  # your test set labels
        return train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig

    def _reshape(self,train_x_orig,train_y_orig,test_x_orig,test_y_orig):

        train_x = train_x_orig.reshape(train_x_orig.shape[0], -1).T
        train_y = train_y_orig.reshape(1, -1)
        test_x = test_x_orig.reshape(test_x_orig.shape[0], -1).T
        test_y = test_y_orig.reshape(1, -1)
        train_x = train_x / 255
        test_x = test_x / 255
        return train_x,train_y,test_x,test_y

    def _initialize_with_zero(self,dim):

        #w = np.zeros((1,dim))
        w = np.random.randn(1,dim) * 0.01
        b = 0
        assert (w.shape == (1,dim))
        assert (isinstance(b,float) or isinstance(b,int))
        return w,b

    def _sigmoid(self,z):

        s = 1/(1+np.exp(-z))
        return s

    def _propagate(self,w,b,X,Y):

        m = X.shape[1]
        A = self._sigmoid(np.dot(w,X)+b)
        cost = -(1/m)*np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))
        dw = (1/m) * np.dot((A-Y),X.T)
        db = (1/m) * np.sum(A-Y)
        assert (dw.shape == w.shape)
        assert(db.dtype == float)
        grads = {"dw":dw,"db":db}
        return grads,cost

    def _optimize(self,w,b,X,Y,num_iterations,learning_rate):

        costs = []
        for i in range(num_iterations):
            grads,cost = self._propagate(w,b,X,Y)
            dw=grads["dw"]
            db=grads["db"]
            w = w - learning_rate*dw
            b = b - learning_rate*db
            if (i%100==0):
                costs.append(cost)
                print("cost after iteration %i == %f" %(i,cost))
        param = {"w" : w ,"b" : b}
        grads = {"dw" : dw, "db": db}
        return param,grads,costs

    def _predict(self,w,b,X):

        m = X.shape[1]
        A = self._sigmoid(np.dot(w,X)+b)
        Y_pred = np.zeros((1,m))
        for i in range(A.shape[1]):
            Y_pred[0,i] = 1 if A[0,i] > 0.5 else 0
        assert (Y_pred.shape==(1,m))
        return Y_pred


    def _model(self,train_x, train_y, test_x, test_y, num_iterations, learning_rate):

        # initialize parameters with zeros (≈ 1 line of code)
        w, b = self._initialize_with_zero(train_x.shape[0])
        # Gradient descent (≈ 1 line of code)
        parameters, grads, costs = self._optimize(w, b, train_x, train_y, num_iterations, learning_rate)
        # Retrieve parameters w and b from dictionary "parameters"
        w = parameters["w"]
        b = parameters["b"]
        # Predict test/train set examples (≈ 2 lines of code)
        Y_prediction_test = self._predict(w, b, test_x)
        Y_prediction_train = self._predict(w, b, train_x)
        ### END CODE HERE ###
        # Print train/test Errors
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_y)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_y)) * 100))
        d = {"costs": costs,
             "Y_prediction_test": Y_prediction_test,
             "Y_prediction_train": Y_prediction_train,
             "w": w,
             "b": b,
             "learning_rate": learning_rate,
             "num_iterations": num_iterations}
        return d