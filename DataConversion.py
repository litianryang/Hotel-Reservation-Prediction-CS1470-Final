from collections import Counter
import csv
import datetime

months = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12
}

tau = 30
fileName = 'H2'

booking_tuples = []
with open('./CSV/' + fileName + '.csv') as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=',')
    first_line = True
    
    for row in csv_reader:
        if first_line:
            first_line = False
        else:
            daysPastInt = int(row['LeadTime'])
            if(daysPastInt < tau):
                daysPast = datetime.timedelta(daysPastInt)
                bookingDate = datetime.date(int(row['ArrivalDateYear']), int(months[row['ArrivalDateMonth']]), int(row['ArrivalDateDayOfMonth']))
                bookDate = bookingDate - daysPast
                booking_tuples.append((bookDate, daysPastInt))
                
booking_tuples = sorted(booking_tuples)
booking_counted = Counter(booking_tuples)
booking_by_day_raw = {}
for key in booking_counted.keys():
    if key[0] not in booking_by_day_raw:
        booking_by_day_raw[key[0]] = [0,0,0,0,0,0,0,0,0,0,
                                  0,0,0,0,0,0,0,0,0,0,
                                  0,0,0,0,0,0,0,0,0,0] #tau days of 0s
    booking_by_day_raw[key[0]][key[1]] = booking_counted[key]

oneDay = datetime.timedelta(1)
firstDate = True
currentDate = None
booking_by_day = {}
for key in booking_by_day_raw:
    if firstDate:
        currentDate = key
        months = []
        dow = []
        bookingDate = key
        for i in range(tau):
            months.append(bookingDate.month)
            dow.append(bookingDate.weekday())
            bookingDate = bookingDate + oneDay
        booking_by_day[key] = (booking_by_day_raw[key], months, dow)
        firstDate = False
    else:
        while key != currentDate:
            months = []
            dow = []
            bookingDate = currentDate
            for i in range(tau):
                months.append(bookingDate.month)
                dow.append(bookingDate.weekday())
                bookingDate = bookingDate + oneDay
            booking_by_day[currentDate] = ([0,0,0,0,0,0,0,0,0,0,
                                            0,0,0,0,0,0,0,0,0,0,
                                            0,0,0,0,0,0,0,0,0,0],
                                            months, dow)
            currentDate = currentDate + oneDay
        months = []
        dow = []
        bookingDate = key
        for i in range(tau):
            months.append(bookingDate.month)
            dow.append(bookingDate.weekday())
            bookingDate = bookingDate + oneDay
        booking_by_day[key] = (booking_by_day_raw[key], months, dow)
    currentDate = currentDate + oneDay
    
with open('./Output/' + fileName + 'Train.csv', mode='w') as train_file:
    with open('./Output/' + fileName + 'Test.csv', mode='w') as test_file:
        fieldnames = ['BookDateYear', 'BookDateMonth', 'BookDateDayOfMonth', 'Bookings', 'Month', 'DOW']
        trainWriter = csv.DictWriter(train_file, fieldnames=fieldnames)
        trainWriter.writeheader()
        testWriter = csv.DictWriter(test_file, fieldnames=fieldnames)
        testWriter.writeheader()
        
        entryCount = 0
        trainCount = int(len(booking_by_day.keys()) * 0.7)
        for key in booking_by_day.keys():
            if entryCount < trainCount:
                trainWriter.writerow({'BookDateYear': key.year, 'BookDateMonth': key.month, 'BookDateDayOfMonth': key.day, 'Bookings': booking_by_day[key][0], 'Month': booking_by_day[key][1], 'DOW': booking_by_day[key][2]})
            else:
                testWriter.writerow({'BookDateYear': key.year, 'BookDateMonth': key.month, 'BookDateDayOfMonth': key.day, 'Bookings': booking_by_day[key][0], 'Month': booking_by_day[key][1], 'DOW': booking_by_day[key][2]})
            entryCount += 1
print("Done")