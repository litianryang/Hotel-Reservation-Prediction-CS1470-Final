import numpy as np
import tensorflow as tf
import sys
from DataProcessor import get_data

class Model(tf.keras.Model):
	def __init__(self):
		
		super(Model, self).__init__()
		#parameters
		self.batch_size = 15
		self.epoch = 300 #evaluate model after 300 examples
		self.dropout_rate = 0.3
		
		
		self.input_size = 7
		#final out put size would be 30 to represent the bookings for the next 30 days made on a given day
		self.output_size = 30 
		#other parameters
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=.1)

		#layers 
		lstm_bk, lstm_tm, dense_bk, dense_tm = 128, 128, 30, 30
		self.lstm_bk = tf.keras.layers.LSTM(lstm_bk, return_state=True, return_sequences=True, dropout=self.dropout_rate)
		self.lstm_tm = tf.keras.layers.LSTM(lstm_tm, return_state=True, return_sequences=True, dropout=self.dropout_rate)
		#double check these other variables 
		self.dense_bk = tf.keras.layers.Dense(dense_bk, activation="relu")
		self.dense_tm = tf.keras.layers.Dense(dense_tm, activation="relu") 
		self.flatten = tf.keras.layers.Flatten()
		#final dense layer with simple linear activation that produces an array of 30
		self.dense_final = tf.keras.layers.Dense(self.output_size)
        
        
	@tf.function
	def call(self, month_input, dow_input, hist_input, tm_state, bk_state):
		"""
		:param month_input: batch_size by H by T, represents X1 input, predicted month bookings for H historical days
		:param dow_input: batch_size by H by T, represents X2 input, predicted day bookings for H historical days
		:param hist_input: batch_size by H by T * S, represents Y, predicted number of bookings for H historical days
		:return predicted bookings, final time lstm state, final booking lstm state
		booking is represented as arrival values (month, day, and number) for the next T days after that booking days
		H: historical days to look at in the past, 
		T: next T days we're trying to book
		"""
		#time input passing its layers
		time_input = tf.concat([month_input, dow_input],0)
		tm_output = self.lstm_tm(time_input, initial_state=tm_state)
		tm_dense_output = self.dense_tm(tm_output[0])

		#historical booking input passing its layers
		bk_output = self.lstm_bk(hist_input, initial_state=bk_state)
		bk_dense_output = self.dense_bk(bk_output[0])

		#concat the two ouputs and pass the final layer to produce output desired
		tmbk_input = tf.concat([tm_dense_output, bk_dense_output], 0)
		tmbk_input = self.flatten(tmbk_input)
		final_output = self.dense_final(tmbk_input)

		return final_output, (tm_output[1], tm_output[2]), (bk_output[1], bk_output[2])

	def loss(self, prbs, labels):
		"""
		:param probs: predictions made by our model
		:param labels: true bookings
		:return mean mean_squared_error(MSE) loss
		"""
		return tf.reduce_mean(tf.keras.losses.mean_squared_error(prbs, labels))

def train(model, train_month, train_dow, train_hist):
		""" 
		"""
		tm_state = None
		bk_state = None
		start_day = 7
		start_train_month = train_month[start_day:]
		start_train_dow = train_dow[start_day:]
		start_labels = train_hist[start_day:]
		for i in range((int)(len(train_month)/model.batch_size)):
			#preparing inputs to be fed into model
			month_inputs = start_train_month[i*model.batch_size:(i+1)*model.batch_size , 1:]
			dow_inputs = start_train_dow[i*model.batch_size :(i+1)*model.batch_size, 1:]
			hist_inputs = train_hist[i*model.batch_size : (i+1)*model.batch_size+start_day, 1:]
			final_hist_inputs=[]
			for j in range(model.batch_size):
				entry = hist_inputs[j]
				for x in range(start_day-1):
					entry = tf.concat([entry, hist_inputs[j+x+1]], 1)
				final_hist_inputs.append(entry)
			final_hist_inputs = tf.convert_to_tensor(final_hist_inputs)
			#labels preparation
			labels = start_labels[i*model.batch_size:(i+1)*model.batch_size, :1]
			with tf.GradientTape() as tape:
				probs, tm_state, bk_state = model.call(month_inputs, dow_inputs, final_hist_inputs, tm_state, bk_state)
				loss = model.loss(probs, labels)
				print(probs)
				#print(loss)
			gradients = tape.gradient(loss, model.trainable_variables)
			model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_month, test_dow, test_hist):
		"""
		"""

def main ():
	#specify user input to include a hotel type: if CITY then run model on H1 data; else run model on H2 data
	if len(sys.argv) != 2 or sys.argv[1] not in {"CITY", "RESORT"}:
		print("USAGE: python Model.py <Hotel Type>")
		print("<Hotel Type>: [CITY/RESORT]")
		exit()

	#get train/test data
	if sys.argv[1]=="CITY":
		train_x1, train_x2, train_y = get_data('H1Train.csv', 14)
		test_x1, test_x2, test_y = get_data('H1Test.csv', 14)
	elif sys.argv[1]=="RESORT":
		train_x1, train_x2, train_y = get_data('H2Train.csv', 14)
		test_x1, test_x2, test_y = get_data('H2Test.csv', 14)
	

	#initialize model
	model = Model()
	for i in range(model.epoch):
		train(model, train_x1, train_x2, train_y)

if __name__ == '__main__':
   main()


	

		

