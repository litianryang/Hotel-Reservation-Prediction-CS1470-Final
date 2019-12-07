import csv
import datetime
import numpy as np

def read_data_file(file_name):
    """
    :param file_name: Path to a data file.
    
    :return: Dictionary containing tuples for each date.
    """
    
    data = {}
    with open(file_name, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
            else:
                bookDate = datetime.date(int(row['BookDateYear']), int(row['BookDateMonth']), int(row['BookDateDayOfMonth']))
                bookings = np.reshape(list(map(int, row['Bookings'][1:-1].split(","))), (-1,1))
                months = np.reshape(list(map(int, row['Month'][1:-1].split(","))), (-1,1))
                DOWs = np.reshape(list(map(int, row['DOW'][1:-1].split(","))), (-1,1))
                data[bookDate] = (bookings, months, DOWs)
            line_count += 1
    print(f'Processed {line_count} lines.')
    
    dailyData = {}
    date_count = 0
    pastBookings = np.zeros((30,7))
    for item in data.items():
        if date_count > 6:
            bookDate = item[0]
            futureDates = np.concatenate((item[1][1], item[1][2]), axis=1)
            dailyData[item[0]] = (futureDates, pastBookings)
        pastBookings = np.concatenate((item[1][0], pastBookings), axis=1)
        pastBookings = np.delete(pastBookings, 7, axis=1)
        date_count += 1
    return dailyData

def get_data(h1_training_file, h1_test_file, h2_training_file, h2_test_file):
    """
    :param h1_training_file: Path to the h1 training file.
    :param h1_test_file: Path to the h1 test file.
    :param h2_training_file: Path to the h2 training file.
    :param h2_test_file: Path to the h2 test file.
    
    :return: A tuple containing four dicts containing the booking data in the same order as inputs.
    """
    
    train_h1 = read_data_file(h1_training_file)    
    test_h1 = read_data_file(h1_test_file)
    train_h2 = read_data_file(h2_training_file)
    test_h2 = read_data_file(h2_test_file)

    return train_h1, test_h1, train_h2, test_h2
    
tr1, ts1, tr2, ts2 = get_data("./Output/H1Train.csv", "./Output/H1Test.csv", "./Output/H2Train.csv", "./Output/H2Test.csv")

print(tr1[datetime.date(2015,6,15)])