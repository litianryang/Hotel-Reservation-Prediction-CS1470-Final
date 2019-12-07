import numpy as np
import tensorflow as tf
from DataProcessor import get_data

class Model(tf.keras.Model):
	def __init__(self):
		
		super(Model, self).__init__()
		#parameters
		self.batch_size = 15
		self.epoch = 300 #evaluate model after 300 examples
		self.dropout_rate = 0.3
		
		#what do these mean?
		self.input_size = 7
		self.output_size = 30 
		#other parameters
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=.1)

		#layers 
		lstm_bk, lstm_tm, dense_bk, dense_tm, dense_final = 128, 128, 30, 30, 60
		self.lstm_bk = tf.keras.layers.LSTM(lstm_bk, return_state=True, return_sequences=True)
		self.lstm_tm = tf.keras.layers.LSTM(lstm_tm, return_state=True, return_sequences=True)
		#double check these other variables 
		self.dense_bk = tf.keras.layers.Dense(dense_bk, "relu")
		self.dense_tm = tf.keras.layers.Dense(dense_tm, "relu") 
		self.dense_final = tf.keras.layers.Dense(dense_final, "relu")
        
        
	@tf.function
	def call(self, month_input, dow_input, hist_input):
		"""
		:param month_input: batch_size by H by T, represents X1 input, predicted month bookings for H historical days
		:param dow_input: batch_size by H by T, represents X2 input, predicted day bookings for H historical days
		:param hist_input: batch_size by H by T * S, represents Y, predicted number of bookings for H historical days
		:return
		booking is represented as arrival values (month, day, and number) for the next T days after that booking day
		H: historical days to look at in the past, 
		T: next T days we're trying to book

		"""
		pass

	def loss(self, prbs, labels, mask):
		"""
		"""

		pass

	def train(self):
		""" 
		"""

		pass

	def test(self):
		"""
		"""
def main ():
	#train, test model
	x1, x2, y = get_data('H2Formatted.csv', 14)

if __name__ == '__main__':
   main()


	

		

