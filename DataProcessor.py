import csv;
import numpy as np;
"Class that takes in clean formatted data and processes to make input"

def helper (arr, indx, h):
    """arg arr: the array containing data for that var on the last h days
    index: the indx to start from corresponding to a particular booking day
    h: h days to look back at
    ret: an array containing the last h days' data for that booking day
    """
    ret_arr = []
    # up to h+1 so that day's included
    for i in reversed (range(indx - h, indx + 1)):
        ret_arr.append(arr[i])
    return ret_arr


def get_bookings (string_arr, x):
    ret_arr = []
    string = string_arr.replace('[', '').replace(']', '').replace(',', '').split()
    for item in string:
        try:
            ret_arr.append(float(item))
        except ValueError:
            continue
    return ret_arr

def get_data (filename, h):
    """
    arg: filename: the file to get data from
    h: historical days to look back at
    input/bookday: 
    x1: past h bookings' dow 
    x2: past h bookings' month
    y: past h bookings' month
    h: concat past h bookings
    """
    with open(filename) as hotel_file:
        input_x1 = []
        input_x2 = []
        input_y = []
        reader = csv.DictReader(hotel_file, delimiter=',')
        for row in reader:
           input_x1.append(get_bookings(row['DOW'], 'x1'))
           input_x2.append(get_bookings(row['Month'], 'x2'))
           input_y.append(get_bookings(row['Bookings'], 'y'))
   
    print('lenghts 1,2,y: ', np.array(input_x1).shape, np.array(input_x2).shape, np.array(input_y).shape )
    
    procsd_x1, procsd_x2, procsd_y = [], [], []
    for j in range (h, len(input_x1)):
        x = helper(input_x1, j, h)
        procsd_x1.append(x)
        procsd_x2.append(helper(input_x2, j, h))
        y  = helper(input_y, j, h)
        procsd_y.append(helper(input_y, j, h))
    procsd_x1, procsd_x2, procsd_y = np.array(procsd_x1), np.array(procsd_x2), np.array(procsd_y)
    return procsd_x1, procsd_x2, procsd_y

