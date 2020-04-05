from nnlib import NNModel


if __name__ == '__main__':

    nnModel = NNModel()
    train_data_path = 'neuralnet_dataset/train_catvnoncat.h5'
    test_data_path = 'neuralnet_dataset/test_catvnoncat.h5'
    train_x_orig, train_y_orig, test_x_orig, test_y_orig = nnModel._load_dataset(train_data_path,test_data_path)
    train_x, train_y, test_x, test_y = nnModel._reshape(train_x_orig, train_y_orig, test_x_orig, test_y_orig)
    d = nnModel._model(train_x, train_y, test_x, test_y, 2000, 0.05)