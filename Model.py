import numpy as np
import tensorflow as tf
import sys
from Preprocess import get_data
import matplotlib.pyplot as plt

class Model(tf.keras.Model):
    def __init__(self):
        
        super(Model, self).__init__()
        #parameters
        self.batch_size = 15
        self.epoch = 300 
        self.dropout_rate = 0.6
        
        
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
        time_output = self.dense_time(time_output)

        #historical booking input passing its layers
        book_output, book_state_h, book_state_c = self.lstm_book(inputs=book_input)#, initial_state=book_state)
        book_output = self.dense_book(book_output)

        #concat the two ouputs and pass the final layer to produce output desired
        tmbk_input = tf.concat([time_output, book_output], axis=1)
        final_output = self.dense_final(tmbk_input)

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
    for i in range(len(todayBookings) // batch_len):
        batch_labels = tf.convert_to_tensor(todayBookings[i * batch_len : (i+1) * batch_len], dtype=tf.float32)
        batch_dates = tf.convert_to_tensor(futureDates[i * batch_len : (i+1) * batch_len], dtype=tf.float32)
        batch_bookings = tf.convert_to_tensor(pastBookings[i * batch_len : (i+1) * batch_len], dtype=tf.float32)
        with tf.GradientTape() as tape:
            probs, time_state, book_state = model(batch_dates, batch_bookings, time_state, book_state)
            loss = model.loss(probs, batch_labels)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

def test(model, data):
    todayBookings, futureDates, pastBookings = data

    loss = 0    
    batch_len = model.batch_size
    num_of_batchs = len(todayBookings) // batch_len

    preds, labels = [],[]
    for i in range(num_of_batchs):
        batch_labels = tf.convert_to_tensor(todayBookings[i * batch_len : (i+1) * batch_len], dtype=tf.float32)
        batch_dates = tf.convert_to_tensor(futureDates[i * batch_len : (i+1) * batch_len], dtype=tf.float32)
        batch_bookings = tf.convert_to_tensor(pastBookings[i * batch_len : (i+1) * batch_len], dtype=tf.float32)
        with tf.GradientTape() as tape:
            probs, time_state, book_state = model(batch_dates, batch_bookings, None, None)
            for i in range (batch_len):
                preds.append(probs[i])
                labels.append(batch_labels[i])
            loss += model.loss(probs, batch_labels)
    return np.exp(loss / num_of_batchs), tf.convert_to_tensor(labels), tf.convert_to_tensor(preds)

def compute_losses (labels, predictions):
    """ uses labels and predictions to calculate all the errors used as measures of accuracy
    """
    mse = tf.losses.mean_squared_error(predictions, labels)
    rmse = tf.sqrt(mse)
    mae = tf.losses.mae(predictions, labels)
    return mae, rmse, mse
def visualize(labels, predictions):
    """ labels: label for the nex thing
        prediction: predicted bookings"""

    labels, predictions = tf.transpose(labels), tf.transpose(predictions)

    labels, predictions = tf.reduce_mean(labels, axis=1), tf.reduce_mean(predictions, axis=1)
    labels, predictions, days = np.array(labels), np.array(predictions), np.arange(0, 30)
    mae, rmse, mse = compute_losses(labels, predictions)
    print('Losses: '"MAE:,", mae, '\n',  "RMSE:,", rmse, '\n', " MSE: ", mse)
    
    #figures
    plt.plot(days, predictions, label='Predictions', color='blue', linewidth=1)
    plt.plot(days, labels, label='Observed', color='red', linewidth=1)
    plt.ylabel("Number of predictions")
    plt.xlabel("Day of the month")
    plt.title("Observed predicted bookings vs observed bookings", color="black")
    plt.legend()
    plt.show()

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
    for i in range(15):
        train(model, train_data)
    perp, labels, preds = test(model, test_data)
    visualize(labels, preds)
    

if __name__ == '__main__':
   main()


    

        

