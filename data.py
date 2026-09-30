import csv
import time

with open("data/dataset1.csv", "r") as f:
    data = csv.reader(f)
    for row in data:
        print(row)
        time.sleep(.01) # slows down the read so it doesn't crash the console
        
    