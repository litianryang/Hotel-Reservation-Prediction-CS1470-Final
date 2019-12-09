import numpy as np
import tensorflow as tf
import sys
from Preprocess import get_data

class Model(tf.keras.Model):
    def __init__(self):
        
        super(Model, self).__init__()
        #parameters
        self.batch_size = 15
        self.epoch = 300 #evaluate model after 300 examples
        self.dropout_rate = 0.6
        
        #Calculations for output sizes for each layer
        
        # Month+DOW = 60 out | 30 + 30 = 60
        # LSTM Time = 86880 | 120 out
        # Dense Time = 14520 | 120 out | (14520 - 120 out) / 120 out = 120 LSTM in
        
        # Past Days Bookings = 210 out | 7 * 30 = 210
        # LSTM Book = 353640 | 210 out
        # Dense Book = 44310 | 210 out | (44310 - 210 out) / 210 out = 210 LSTM in
        
        # Time-Booking Concatenated = 330 out | 120 + 210 = 330
        # Dense Output = 9930 | 30 out | (9930 - 30 out) / 30 out = 330 in
        # Final Output = 30
        
        # Possible Book + Time values
        # 10 = YES, 320 = NO
        # 20 = YES, 310 = NO
        # 30 = YES, 300 = NO
        # 40 = YES, 290 = NO
        # 50 = NO
        # 60 = YES, 270 = NO
        # 70 = NO
        # 80 = NO
        # 90 = NO
        # 100 = NO
        # 110 = YES, 220 = NO
        # 120 = YES, 210 = YES!!!! Square roots...
        
        # Output sizes for the time layers
        time_lstm_units = 120
        time_dense_units = 120
        
        # Output sizes for the bookings layers
        book_lstm_units = 210
        book_dense_units = 210
        
        # Final output size would be 30 to represent the bookings for the next 30 days made on a given day
        self.output_size = 30
        
        #other parameters
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

        # Time Layers
        self.lstm_time = tf.keras.layers.LSTM(units=time_lstm_units, input_shape=(30,2), return_state=True, return_sequences=False, dropout=self.dropout_rate, trainable=True)
        self.dense_time = tf.keras.layers.Dense(units=time_dense_units, activation="relu", trainable=True)
        
        # Booking Layers
        self.lstm_book = tf.keras.layers.LSTM(units=book_lstm_units, input_shape=(30,7), return_state=True,  return_sequences=False, dropout=self.dropout_rate, trainable=True)
        self.dense_book = tf.keras.layers.Dense(units=book_dense_units, activation="relu", trainable=True)
        
        #Final dense layer with simple linear activation that produces an array of 30
        self.dense_final = tf.keras.layers.Dense(units=self.output_size, trainable=True)
        
        
    @tf.function
    def call(self, time_input, book_input, time_state, book_state):
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
        time_output, time_state_h, time_state_c = self.lstm_time(inputs=time_input)#, initial_state=time_state)
        print("TIME LSTM: " + str(time_output.shape))
        time_output = self.dense_time(time_output)
        print("TIME Dense: " + str(time_output.shape))

        #historical booking input passing its layers
        book_output, book_state_h, book_state_c = self.lstm_book(inputs=book_input)#, initial_state=book_state)
        print("BOOK LSTM: " + str(book_output.shape))
        book_output = self.dense_book(book_output)
        print("BOOK Dense: " + str(book_output.shape))

        #concat the two ouputs and pass the final layer to produce output desired
        tmbk_input = tf.concat([time_output, book_output], axis=1)
        print("Concat" + str(tmbk_input.shape))
        final_output = self.dense_final(tmbk_input)
        print("Final Dense: " + str(final_output.shape))

        return final_output, [time_state_h, time_state_c], [book_state_h, book_state_c]

    def loss(self, probs, labels):
        """
        :param probs: predictions made by our model
        :param labels: true bookings
        :return mean mean_squared_error(MSE) loss
        """
        return tf.reduce_mean(tf.keras.losses.mean_squared_error(probs, labels))

def train(model, data):
    """ 
    """
    todayBookings, futureDates, pastBookings = data
    time_state = None
    book_state = None
    
    batch_len = model.batch_size
    #i=0
    for i in range(len(todayBookings) // batch_len):
        batch_labels = tf.convert_to_tensor(todayBookings[i * batch_len : (i+1) * batch_len], dtype=tf.float32)
        batch_dates = tf.convert_to_tensor(futureDates[i * batch_len : (i+1) * batch_len], dtype=tf.float32)
        batch_bookings = tf.convert_to_tensor(pastBookings[i * batch_len : (i+1) * batch_len], dtype=tf.float32)
        with tf.GradientTape() as tape:
            probs, time_state, book_state = model(batch_dates, batch_bookings, time_state, book_state)
            loss = model.loss(probs, batch_labels)
            print("Loss round " + str(i) + ": " + str(loss))
        
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    print(probs)
    print(batch_labels)

def test(model, data):
    todayBookings, futureDates, pastBookings = data

    loss = 0    
    batch_len = model.batch_size
    num_of_batchs = len(todayBookings) // batch_len
    print("NUM OF BATCHES: " + str(num_of_batchs))
    for i in range(num_of_batchs):
        batch_labels = tf.convert_to_tensor(todayBookings[i * batch_len : (i+1) * batch_len], dtype=tf.float32)
        batch_dates = tf.convert_to_tensor(futureDates[i * batch_len : (i+1) * batch_len], dtype=tf.float32)
        batch_bookings = tf.convert_to_tensor(pastBookings[i * batch_len : (i+1) * batch_len], dtype=tf.float32)
        with tf.GradientTape() as tape:
            probs, time_state, book_state = model(batch_dates, batch_bookings, None, None)
            loss += model.loss(probs, batch_labels)
    
    return np.exp(loss / num_of_batchs)

def main ():
    #specify user input to include a hotel type: if CITY then run model on H1 data; else run model on H2 data
    if len(sys.argv) != 2 or sys.argv[1] not in {"CITY", "RESORT"}:
        print("USAGE: python Model.py <Hotel Type>")
        print("<Hotel Type>: [CITY/RESORT]")
        exit()

    #get train/test data
    if sys.argv[1]=="CITY":
        train_data, test_data = get_data('H1Train.csv', 'H1Test.csv')
    elif sys.argv[1]=="RESORT":
        train_data, test_data = get_data('H2Train.csv', 'H2Test.csv')
    
    #initialize model
    model = Model()
    #for i in range(10):
    train(model, train_data)
    
    print("Perplexity: " + str(test(model, test_data)))

if __name__ == '__main__':
   main()


    

        

