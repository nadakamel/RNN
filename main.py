import os, warnings, shutil, glob, random, csv, time, keras
import pandas as pd
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras import optimizers
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras.models import Sequential

def get_session(gpu_fraction=0.3):
    #Assume that you have 12GB of GPU memory and want to allocate ~4GB
    
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    
    if num_threads:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())

# Hide messy TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#Hide messy Numpy warnings
warnings.filterwarnings("ignore")

FILE_PATH = './sp500-10Years.csv'
BATCH_SIZE = 512
EPOCHS = 50
RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
OPTIMIZER = RMSprop

SEQUENCE_LENGTH = 25
LAYERS = [1, 50, 128, 1]
NORMALIZE_WINDOW = True

MODEL_RESULT = None

# Convert a tuple or struct_time representing a time as returned by gmtime() or localtime() in a date and time format
CURRENT_TIME = time.strftime("%c")

# Generating graphs from the TensorFlow processing at different paths for each run (based on time of run)
tensorboard_callback = TensorBoard(log_dir='./logs/' + CURRENT_TIME, histogram_freq = 0, write_graph = True, write_images = False)

# saving the best weights found at current run of the network model, saved to binary file assigned to filepath
checkpointer = ModelCheckpoint(filepath = "./weights.hdf5", verbose = 1, save_best_only = True, monitor = 'val_loss')    # Reference: https://keras.io/callbacks/#modelcheckpoint

# decaying learning rate when a plateau is reached in the validation loss by a factor of 0.2; reference: https://keras.io/callbacks/#reducelronplateau
reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.2, patience = 5, min_lr = 1e-6)

# stop training when a monitored quantity in the validation loss has stopped improving.
early_stopping = EarlyStopping(monitor='loss', min_delta=0.0001, patience=5, verbose=1, mode='min')

CALLBACKS = [tensorboard_callback,checkpointer,reduce_lr,early_stopping]

folder_path = './' + str(CURRENT_TIME) + '/'
DIRECTORY = os.path.dirname(folder_path)

def check_folder_existence():
    global DIRECTORY
    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)

def load_data(filename, sequence_length, normalise_window):
    data = pd.read_csv(filename, usecols=[1], engine='python', skipfooter=3, header=1)
    data = data.values
    data = data.astype('float32')

    sequence_length = sequence_length + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    if normalise_window:
        result = normalise_windows(result)
    
    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    random.shuffle(train)

    '''
    x_train: training sequence
    y_train: training output
    x_test: validation sequence
    y_test: validation output
    '''
    x_train = train[:, :-1]     # x_train now contains the sequence of seq_len only. Without the last element
    y_train = train[:, -1]      # y_train now contains the OUTPUT only, which is the last element

    x_test = result[int(row):, :-1] # x_test now contains the sequence of seq_len for testing
    y_test = result[int(row):, -1]  # y_test now contains the OUTPUT only, which is the last element for testing

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return [x_train, y_train, x_test, y_test]

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation('linear'))

    start = time.time()
    model.compile(loss='mse', optimizer=OPTIMIZER)
    print 'Compilation Time (in secs): ' + str(time.time() - start)
    
    return model

def compile_model(model, x_train, y_train, x_test, y_test):
    result = model.fit(
	    x_train,
	    y_train,
	    batch_size=BATCH_SIZE,
	    nb_epoch=EPOCHS,
	    validation_split=0.05,
        validation_data=(x_test, y_test),
        callbacks=CALLBACKS)
    
    return result

def evaluate_model(model, x_test, y_test):
    # evaluate the result
    test_mse = model.evaluate(x_test, y_test, verbose=1)
    print ('\nThe mean squared error (MSE) on the test data set is %.3f over %d test samples.') %(test_mse, len(y_test))

def plot_loss_vs_epochs():
    global MODEL_RESULT
    plt.plot(MODEL_RESULT.history['loss'])
    plt.title('Loss vs. Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train'], loc='upper right')
    check_folder_existence()
    plt.savefig('./'+DIRECTORY+'/loss_vs_epochs.jpg')

def plot_test_actual_vs_predicted(predictions, test_out):
    fig = plt.figure()
    plt.plot(test_out)
    plt.plot(predictions)
    plt.title('SP500 | Last '+str(len(predictions))+' days in test')
    plt.xlabel('Days')
    plt.ylabel('Stock Closing Prices')
    plt.legend(['actual', 'prediction'], loc='upper right')
    check_folder_existence()
    plt.savefig('./'+DIRECTORY+'/output_prediction.jpg', bbox_inches='tight')

def predict_next_sequence(model, last_sequence, prediction_length):
    prediction_seqs = []
    for i in range(prediction_length):
        normalised_data = [((float(p) / float(last_sequence[0])) - 1) for p in last_sequence[-SEQUENCE_LENGTH:]]
        normalised_data = np.array(normalised_data)
        normalised_data = np.reshape(normalised_data, (normalised_data.shape[0], 1))
        normalised_data = np.array([normalised_data])
        pred_value= (model.predict(normalised_data)[0][0] + 1) * last_sequence[0]
        last_sequence.append(pred_value)
        prediction_seqs.append(pred_value)

    return prediction_seqs

def main():
    print('> Loading data... ')
    train_seq, train_out, test_seq, test_out = load_data(FILE_PATH, SEQUENCE_LENGTH, NORMALIZE_WINDOW)
    print('> Data Loaded. Compiling...')
    
    global_start_time = time.time()
    model = build_model(LAYERS)
    
    global MODEL_RESULT
    MODEL_RESULT = compile_model(model, train_seq, train_out, test_seq, test_out)

    print 'Training Duration (in secs): ' + str(time.time() - global_start_time)
    
    # save model
    model.save('./RNN.h5')

    # plot training loss vs epochs
    plot_loss_vs_epochs()

    # evaluate model
    evaluate_model(model, test_seq, test_out)

    # actual values in test vs predicted values
    predictions = model.predict(test_seq)
    num_test_samples = len(predictions)
    predictions = np.reshape(predictions, (num_test_samples,1))

    # plot results
    plot_test_actual_vs_predicted(predictions, test_out)
    
    # predicte coming days
    data = pd.read_csv(FILE_PATH, usecols=[1], engine='python', skipfooter=3, header=1)
    data = data.values
    data = data.astype('float32')
    num_of_last_days = 15
    data = data[-num_of_last_days:].tolist()
    for i in range(len(data)):
        data[i] = data[i][0]
    lastDays = data
    prediction_length = 5
    predicted_values = predict_next_sequence(model, lastDays, prediction_length)

    # print predictions
    print 'Predicted values for next '+str(prediction_length)+' days..'
    for i in range(len(predicted_values)):
        print(i+1, predicted_values[i])


if __name__ == "__main__":
    main()
