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
    return data

def get_data_dict(data):
    dataDict = {}
    date_count = 0
    pastBookings = np.zeros((30,7))
    for item in data.items():
        if date_count > 6:
            futureDates = np.concatenate((item[1][1], item[1][2]), axis=1)
            dataDict[item[0]] = (item[1][0], futureDates, pastBookings)
        pastBookings = np.concatenate((item[1][0], pastBookings), axis=1)
        pastBookings = np.delete(pastBookings, 7, axis=1)
        date_count += 1
    return dataDict
    
def get_data_array(data):
    todayBookings = []
    futureDates = []
    pastBookings = []
    date_count = 0
    pastWeekBookings = np.zeros((30,7))
    for item in data.items():
        if date_count == 7:
            todayBookings = np.reshape(item[1][0], (1, 30))
            futureDates = np.reshape(np.concatenate((item[1][1], item[1][2]), axis=1), (1, 30, 2))
            pastBookings = np.reshape(pastWeekBookings, (1, 30, 7))
        elif date_count > 7:
            todayBookings = np.concatenate([todayBookings, np.reshape(item[1][0], (1, 30))], axis=0)
            futureDates = np.concatenate([futureDates, np.reshape(np.concatenate((item[1][1], item[1][2]), axis=1), (1, 30, 2))], axis=0)
            pastBookings = np.concatenate([pastBookings, np.reshape(pastWeekBookings, (1, 30, 7))], axis=0)
        pastWeekBookings = np.concatenate((item[1][0], pastWeekBookings), axis=1)
        pastWeekBookings = np.delete(pastWeekBookings, 7, axis=1)
        date_count += 1
    return todayBookings, futureDates, pastBookings

def get_data(training_file, test_file, isArray=True):
    """
    :param training_file: Path to the hotel training file.
    :param test_file: Path to the hotel test file.
    :param array: Are we returning the data in an array format (True) or in a dictionary format (False)
    
    :return: If Array (True): A tuple containing three arrays containing today's bookings for the next 30 days, the next 30 days date info, and the past week's worth of bookings for the next 30 days respectively for every day. [individual dates, 30 days, varies depending on array]
    
    If Dictionary (False): A dictionary containing key values pairs where the key is the current date and the value is a tuple containing the same three arrays as Array returns minus the date axis. [30 days, varies depending on array].
    """
    
    if isArray:
        today, future, past = get_data_array(read_data_file(training_file))
        train = (today, future, past)
        today, future, past = get_data_array(read_data_file(test_file))
        test = (today, future, past)
    else:
        train = get_data_dict(read_data_file(training_file))
        test = get_data_dict(read_data_file(test_file))

    return train, test

# isArray = True
# train, test = get_data("./Output/H1Train.csv", "./Output/H1Test.csv", isArray)

# if isArray:
    # print(train[0][2])
    # print(train[1][2])
    # print(train[2][2])
# else:
    # print(train[datetime.date(2015,6,15)])