
from datetime import datetime
from typing import List, Union
from keras.optimizers import Adam #Оптимизатор
from keras.models import Sequential,load_model, Model #Два варианты моделей
from keras.layers import concatenate, Input, Dense, Dropout, BatchNormalization, Flatten, Conv1D, LSTM #Стандартные слои
from keras.preprocessing.sequence import TimeseriesGenerator # для генерации выборки временных рядов
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from numpy import newaxis

from src.neural_network.schemas import DenseClass, DropoutClass, Loss, LstmClass, Optimizer, Settings
class Data():
	def __init__(self,Date:list[datetime]=[],Open:list[float]=[],Close:list[float]=[],High:list[float]=[],Low:list[float]=[],Volume:list[int]=[]):
		self.Date   = Date
		self.Open   = Open
		self.Close  = Close
		self.High   = High
		self.Low    = Low
		self.Volume = Volume
	   
		
class Timer():

	def __init__(self):
		self.start_dt = None

	def start(self):
		self.start_dt = datetime.now()

	def stop(self):
		end_dt = datetime.now()
		print('Time taken: %s' % (end_dt - self.start_dt))

class NeuralNetworkModel():
	"""A class for an building and inferencing an lstm model"""

	def __init__(self):
		self.model = Sequential()
		self.log_dir = "tf_logs"
		self.writer = None

	def load_model(self, filepath):
		print('[Model] Loading model from file %s' % filepath)
		self.model = load_model(filepath)

	def build_model(self, loss:Loss,optimizer:Optimizer, layers:List[Union[LstmClass, DropoutClass, DenseClass]]):
		timer = Timer()
		timer.start()

		for layer in layers:
			if isinstance(layer, DenseClass):#layer['type'] == 'dense':
				self.model.add(Dense(layer.neurons, activation=layer.activation))
			elif isinstance(layer, LstmClass):#layer['type'] == 'lstm':
				self.model.add(LSTM(layer.neurons, input_shape=(layer.input_timesteps, layer.input_dim), return_sequences=layer.return_seq))
			elif isinstance(layer, DropoutClass):#layer['type'] == 'dropout':
				self.model.add(Dropout(layer.rate))

		#self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])
		self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

		self.model.summary()

		print('[Model] Model Compiled')
		timer.stop()

	def train(self, x, y, epochs, batch_size,  timeframe):
		
		timer = Timer()
		timer.start()
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
		# self.writer = tf.summary.create_file_writer(self.log_dir)
		save_fname = os.path.join('saved_models', '%s-e%s_%s.h5' % (datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs), timeframe))
		callbacks = [
			# tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1),
			EarlyStopping(monitor='val_loss', patience=2),
			ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
		]
		self.model.fit(
			x,
			y,
			epochs=epochs,
			batch_size=batch_size,
			callbacks=callbacks
		)
		self.model.save(save_fname)
		# self.writer.close()

		print('[Model] Training Completed. Model saved as %s' % save_fname)
		timer.stop()

	def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir, timeframe):
		timer = Timer()
		timer.start()
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))

		self.writer = tf.summary.create_file_writer(self.log_dir)
		save_fname = os.path.join(save_dir, '%s-e%s_%s.h5' % (datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs), timeframe))
		callbacks = [
			tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1),
			ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
		]
		self.model.fit_generator(
			data_gen,
			steps_per_epoch=steps_per_epoch,
			epochs=epochs,
			callbacks=callbacks,
			workers=1
		)
		self.writer.close()
		
		print('[Model] Training Completed. Model saved as %s' % save_fname)
		timer.stop()

	def eval_test(self, x_test,  y_test, verbose):
		return self.model.evaluate(x_test, y_test, verbose=verbose)

	def eval_test2(self, x_test,  y_test, verbose):
		score = self.model.evaluate(x_test, y_test, verbose=verbose)
		return score[0], score[1]

	def predict_point_by_point(self, data):
		#Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
		print('[Model] Predicting Point-by-Point...')
		predicted = self.model.predict(data)
		print("predicted.size =", predicted.size)
		predicted = np.reshape(predicted, (predicted.size,))
		return predicted

	def predict_sequences_multiple(self, data, window_size, prediction_len):
		#Predict sequence of 50 steps before shifting prediction run forward by 50 steps
		print('[Model] Predicting Sequences Multiple...')
		prediction_seqs = []
		for i in range(int(len(data)/prediction_len)):
			curr_frame = data[i*prediction_len]
			predicted = []
			for j in range(prediction_len):
				predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
				curr_frame = curr_frame[1:]
				curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
			prediction_seqs.append(predicted)
		return prediction_seqs

	def predict_sequence_full(self, data, window_size):
		#Shift the window by 1 new prediction each time, re-run predictions on new window
		print('[Model] Predicting Sequences Full...')
		curr_frame = data[0]
		predicted = []
		for i in range(len(data)):
			predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
			curr_frame = curr_frame[1:]
			curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
		return predicted

class DataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self):
    
        self.conn = None
        self.cursor = None
        self.connection_to_db = False
        self.data_train :list[list[float]] = [] #closs volume
        self.data_test :list[list[float]] = [] #closs volume
        self.len_train :int = None #closs volume
        self.len_test  :int = None #closs volume
        self.len_train_windows = None

    

    def de_normalise_predicted(self, price_1st, _data):
        return (_data + 1) * price_1st

    def get_last_data(self, seq_len, normalise):
        last_data = self.data_test[seq_len:]
        data_windows = np.array(last_data).astype(float)
        #data_windows = np.array([data_windows])
        #data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows
        data_windows = self.normalise_windows(data_windows, single_window=True) if normalise else data_windows
        return data_windows

    def get_test_data(self, seq_len, normalise):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_windows = []
        for i in range(self.len_test - seq_len + 1):
            data_windows.append(self.data_test[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        return x,y

 

    def get_train_data(self, seq_len, normalise):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len + 1):
            x, y = self._next_window(i, seq_len, normalise)
            # print('x', x,'y', y)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_train - seq_len + 1):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len + 1):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        '''Generates the next data window from the given index location i'''
        window = self.data_train[i:i+seq_len]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-1]
        y = window[-1, [0]]
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        '''Normalise window with a base value of zero'''
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)