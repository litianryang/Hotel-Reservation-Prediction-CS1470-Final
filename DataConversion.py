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


booking_tuples = []
with open('./CSV/H2.csv') as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=',')
    first_line = True
    
    for row in csv_reader:
        if first_line:
            first_line = False
        else:
            daysPastInt = int(row['LeadTime'])
            if(daysPastInt < tau):
                daysPast = datetime.timedelta(daysPastInt)
               arrivalDate = datetime.date(int(row['ArrivalDateYear']), int(months[row['ArrivalDateMonth']]), int(row['ArrivalDateDayOfMonth']))
                bookDate = arrivalDate - daysPast
                booking_tuples.append((bookDate, daysPastInt)) #might need arrivalDate = bookDate + daysPast
                
booking_tuples = sorted(booking_tuples) 
booking_counted = Counter(booking_tuples)
booking_by_day = {}
for key in booking_counted.keys():
    if key[0] not in booking_by_day:
        booking_by_day[key[0]] = [0,0,0,0,0,0,0,0,0,0,
                                  0,0,0,0,0,0,0,0,0,0,
                                  0,0,0,0,0,0,0,0,0,0] #tau days of 0s
    booking_by_day[key[0]][key[1]] = booking_counted[key]
    print(booking_by_day[key[0]key[1]])

#value looks like it's just its position in the array
#booking_by_day ={[bookDate, daysPast/LeadTime] : }
with open('./Output/H2Formatted.csv', mode='w') as output_file:
    fieldnames = ['BookDateYear', 'BookDateMonth', 'BookDateDayOfMonth', 'Bookings', 'Month', 'DOW']
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    writer.writeheader()
    
    oneDay = datetime.timedelta(1)
    for key in booking_by_day.keys():
        months = []
        dow = []
        #should not be adding a day at this point
        #replace with going a day ahead/per booking
        bookingDate = key + oneDay
        for i in range(tau):
            months.append(bookingDate.month)
            dow.append(bookingDate.weekday())
            bookingDate = bookingDate + oneDay
        writer.writerow({'BookDateYear': key.year, 'BookDateMonth': key.month, 'BookDateDayOfMonth': key.day, 'Bookings': booking_by_day[key], 'Month': months, 'DOW': dow})
print("Done")