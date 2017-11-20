import os, warnings, shutil, glob, random, csv, time, keras
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

FILE_PATH = './sp500-10yrs.csv'
BATCH_SIZE = 512
EPOCHS = 10
RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
OPTIMIZER = RMSprop

MODEL_RESULT = None
SEQUENCE_LENGTH = 50
LAYERS = [1, 50, 128, 1]
NORMALIZE_WINDOW = True

# Hide messy TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#Hide messy Numpy warnings
warnings.filterwarnings("ignore")

# Load data
def load_data(filename, seq_len, normalise_window):
    f = open(filename, 'rb').read()
    data = f.split('\n')
    data = [float(num) for num in data]

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)
    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    #random.shuffle(train)

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
        input_shape=(layers[1], layers[0]),
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
    model.compile(loss="mse", optimizer=OPTIMIZER, metrics=['accuracy'])
    print 'Compilation Time (in secs): ' + str(time.time() - start)
    return model

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.title('Stock Market Multiple Sequential Predictions')
    plt.xlabel('Days')
    plt.ylabel('Prices')
    #Pad the list of predictions to shift it in the graph to it's correct start
    dates = ['27th of Nov', '28th of Nov', '29th of Nov', '30th of Nov', '1st of Dec']
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label=dates[i]+' Prediction')
        plt.legend(loc='bottom left')
    plt.show()

def plot_loss_vs_epochs():
    global MODEL_RESULT
    plt.plot(MODEL_RESULT.history['loss'])
    plt.title('Loss vs. Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train'], loc='upper right')
    plt.show()

def main():
    print('> Loading data... ')
    train_seq, train_out, test_seq, test_out = load_data(FILE_PATH, SEQUENCE_LENGTH, NORMALIZE_WINDOW)

    print('> Data Loaded. Compiling...')
    global_start_time = time.time()
    model = build_model(LAYERS)
    
    global MODEL_RESULT
    MODEL_RESULT = model.fit(
	    train_seq,
	    train_out,
	    batch_size=BATCH_SIZE,
	    nb_epoch=EPOCHS,
	    validation_split=0.05)

    predictions = predict_sequences_multiple(model, test_seq, SEQUENCE_LENGTH, 50)
	#predicted = predict_sequence_full(model, test_seq, seq_len)
	#predicted = predict_point_by_point(model, test_seq)

    print predictions

    print 'Training Duration (in secs): ' + str(time.time() - global_start_time)
    plot_results_multiple(predictions, test_out, 50)

    plot_loss_vs_epochs()

if __name__ == "__main__":
    main()
