import sqlite3
from datetime import datetime, timedelta, date
import random


def create_table():
    conn = sqlite3.connect('intelligence_hub.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS imagery (
            id INTEGER PRIMARY KEY,
            airport TEXT NOT NULL,
            aircrafts INTEGER NOT NULL,
            over_threshold TEXT NOT NULL, 
            year INTEGER NOT NULL,
            month INTEGER NOT NULL,
            day INTEGER NOT NULL,
            hour INTEGER NOT NULL,
            minute INTEGER NOT NULL,
            second INTEGER NOT NULL
        )
    ''')

    conn.commit()
    conn.close()


def add_imagery_data(airport, aircrafts, over_threshold, date):  # adds imagery to db
    conn = sqlite3.connect('intelligence_hub.db')
    cursor = conn.cursor()
    year, month, day, hour, minute, second = date.year, date.month, date.day, date.hour, date.minute, date.second
    # Insert data into the 'imagery' table
    cursor.execute('''
            INSERT INTO imagery (airport, aircrafts, over_threshold, year, month, day, hour, minute, second)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (airport, aircrafts, over_threshold, year, month, day, hour, minute, second))
    conn.commit()
    conn.close()


def get_imagery_entry_count():  # returns the total amount of entries
    # Connect to the database
    conn = sqlite3.connect('intelligence_hub.db')
    cursor = conn.cursor()

    # Execute a query to get the count of entries in the 'imagery' table
    cursor.execute('SELECT COUNT(*) FROM imagery')

    # Fetch the result
    entry_count = cursor.fetchone()[0]

    # Close the connection
    conn.close()

    return entry_count


def total_images_over_airport(airport):  # returns how many images of the airport and how many added in last month
    # Connect to the database
    conn = sqlite3.connect('intelligence_hub.db')
    cursor = conn.cursor()

    # Execute a query to get the count of entries in the 'imagery' table
    # cursor.execute('''SELECT COUNT(*) FROM imagery WHERE airport = ? ''', (airport))
    cursor.execute('''SELECT COUNT(*) FROM imagery WHERE airport = ?''', (airport,))

    # Fetch the result
    total_images = cursor.fetchone()[0]

    now_date = datetime.now()
    date_30_days_ago = now_date - timedelta(days=30)

    cursor.execute('''SELECT COUNT(*) FROM imagery WHERE airport = ? AND strftime('%Y-%m-%d', 'now') >= ?;
    ''', (airport, date_30_days_ago.strftime("%Y-%m-%d"),))

    added_this_month = cursor.fetchone()[0]

    conn.close()

    return total_images, added_this_month


def last_image_over_airport(airport):
    # returns date and aircraft count of last image over AOI and how many aircrafts in last month
    conn = sqlite3.connect('intelligence_hub.db')
    cursor = conn.cursor()
    cursor.execute('''SELECT * FROM imagery WHERE airport = ? ORDER BY year DESC, month DESC, day DESC,
    hour DESC, minute DESC, second DESC LIMIT 1;''', (airport,))
    last_entry = cursor.fetchone()
    if last_entry:
        latest_date = datetime(last_entry[4], last_entry[5], last_entry[6], last_entry[7], last_entry[8], last_entry[9])
        latest_aircraft_count = last_entry[2]
    else:
        latest_date = None
        latest_aircraft_count = None
    conn.close()

    return latest_date, latest_aircraft_count


def averages_over_airport(airport, start_date, end_date):  # returns images/month average and aircrafts/image average over airport
    conn = sqlite3.connect('intelligence_hub.db')
    cursor = conn.cursor()

    cursor.execute('''SELECT * FROM imagery WHERE airport = ?''', (airport,))
    rows = cursor.fetchall()

    results = []
    aircraft_count = 0
    for row in rows:  # checking all results of airport
        row_formatted = datetime(row[4], row[5], row[6], row[7], row[8], row[9])
        if end_date <= row_formatted <= start_date:  # if image is in between dates
            results.append(row)
            aircraft_count += row[2]  # sum all aircrafts
    image_count = len(results)
    number_of_months = (start_date.year - end_date.year) * 12 + start_date.month - end_date.month + 1
    if number_of_months == 0:
        number_of_months = 1
    if image_count == 0:
        image_count = 1

    average_aircrafts = aircraft_count // image_count
    average_images = image_count // number_of_months

    conn.close()
    return average_aircrafts, average_images


def add_information():
    airports = [
        "Amsterdam - Amsterdam Airport Schiphol (AMS)",
        "Atlanta - Hartsfield-Jackson International Airport (ATL)",
        "Bangkok - Suvarnabhumi Airport (BKK)",
        "Beijing - Capital International Airport (PEK)",
        "Chicago - O'Hare International Airport (ORD)",
        "Dallas/Fort Worth - Dallas/Fort Worth International Airport (DFW)",
        "Denver - Denver International Airport (DEN)",
        "Dubai - Dubai International Airport (DXB)",
        "Frankfurt - Frankfurt Airport (FRA)",
        "Guangzhou - Baiyun International Airport (CAN)",
        "Hong Kong - Hong Kong International Airport (HKG)",
        "Istanbul - Istanbul Airport (IST)",
        "London - Heathrow Airport (LHR)",
        "Los Angeles - Los Angeles International Airport (LAX)",
        "New York City - John F. Kennedy International Airport (JFK)",
        "Paris - Charles de Gaulle Airport (CDG)",
        "Seoul - Incheon International Airport (ICN)",
        "Shanghai - Pudong International Airport (PVG)",
        "Singapore - Changi Airport (SIN)",
        "Tokyo - Haneda Airport (HND)"
        ]
    for airport in airports:
        cleaned_airport = str(airport[0:-1].replace(' (', '_'))
        for i in range(500):
            aircrafts = random.randint(10,80)
            # year = random.randint(2020,2023)
            year = 2024
            month = random.randint(1,12)
            if month in (1, 3, 5, 7, 8, 10, 12):
                day = random.randint(1, 31)
            elif month in (4, 6, 9, 11):
                day = random.randint(1,30)
            else:
                day = 29 if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) else 28
            hour = random.randint(7,16)
            minute = random.randint(0,59)
            second = random.randint(0, 59)
            if aircrafts > 40:
                threshold = "Over"
            else:
                threshold = "Not Over"
            date = datetime(year, month, day, hour, minute, second)
            add_imagery_data(cleaned_airport, aircrafts, threshold, date)


def get_info_entry(id):  # returns information about specific id entry
    conn = sqlite3.connect('intelligence_hub.db')
    cursor = conn.cursor()
    cursor.execute('''SELECT * FROM imagery WHERE id = ?''',(id,))
    info = cursor.fetchone()
    conn.close()
    return info


def get_entries_month(airport, year, month):  # returns list of all aircrafts count for specific month
    aircrafts_month = []
    conn = sqlite3.connect('intelligence_hub.db')
    cursor = conn.cursor()
    cursor.execute('''SELECT * FROM imagery WHERE airport = ? AND year = ? AND month = ?''',
                   (airport, year, month))
    rows = cursor.fetchall()
    for row in rows:
        aircrafts_month.append(row[2])
    conn.close()
    return aircrafts_month


def scatter_data(airport, start_date, end_date):  # returns array of all entries of aircrafts for between months
    histogram_data = []
    current_date = start_date

    while current_date <= end_date:  # loop that runs for all in between months
        year = current_date.year
        month = current_date.month
        current_date += timedelta(days=32 - current_date.day)
        histogram_month = get_entries_month(airport, year, month)
        histogram_data.append(histogram_month)

    return histogram_data


def get_airport_rank(target_airport):  # returns rank of airport based on image count
    airports = [
        "Amsterdam - Amsterdam Airport Schiphol (AMS)",
        "Atlanta - Hartsfield-Jackson International Airport (ATL)",
        "Bangkok - Suvarnabhumi Airport (BKK)",
        "Beijing - Capital International Airport (PEK)",
        "Chicago - O'Hare International Airport (ORD)",
        "Dallas/Fort Worth - Dallas/Fort Worth International Airport (DFW)",
        "Denver - Denver International Airport (DEN)",
        "Dubai - Dubai International Airport (DXB)",
        "Frankfurt - Frankfurt Airport (FRA)",
        "Guangzhou - Baiyun International Airport (CAN)",
        "Hong Kong - Hong Kong International Airport (HKG)",
        "Istanbul - Istanbul Airport (IST)",
        "London - Heathrow Airport (LHR)",
        "Los Angeles - Los Angeles International Airport (LAX)",
        "New York City - John F. Kennedy International Airport (JFK)",
        "Paris - Charles de Gaulle Airport (CDG)",
        "Seoul - Incheon International Airport (ICN)",
        "Shanghai - Pudong International Airport (PVG)",
        "Singapore - Changi Airport (SIN)",
        "Tokyo - Haneda Airport (HND)"
        ]
    airport_list = []
    for airport in airports:  # get all number of images and airports
        cleaned_airport = str(airport[0:-1].replace(' (', '_'))
        img_num = total_images_over_airport(cleaned_airport)[0]
        airport_list.append((img_num, cleaned_airport))
    sorted_airport_list = sorted(airport_list, key=lambda x: x[0])  # sort airport list by number of images

    for index, (img_count, airport) in enumerate(sorted_airport_list):
        if airport == target_airport:
            rank = len(airports) - index  # calculates rank of airport
            return sorted_airport_list, rank
    return None


def get_daily_distribution(airport):  # gets daily and hourly image distribution
    week_distribution = [0, 0, 0, 0, 0, 0, 0]
    hour_distribution = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    
    conn = sqlite3.connect('intelligence_hub.db')
    cursor = conn.cursor()
    cursor.execute('''SELECT * FROM imagery WHERE airport = ?''', (airport,))
    rows = cursor.fetchall()
    
    for row in rows:
        day_of_week = datetime(row[4], row[5], row[6]).weekday()  # check which day it is
        week_distribution[day_of_week] += 1  # add to day distribution
        
        which_hour = row[7] - 7  # adjust to start at 07
        if 0 <= which_hour < len(hour_distribution[0]):  # Ensure which_hour is within the range
            hour_distribution[day_of_week][which_hour] += 1  # add to hour distribution
        else:
            print(f"⚠️ Skipping invalid hour: {row[7]} → index {which_hour}")
    
    conn.close()
    return week_distribution, hour_distribution



def get_first_last_img(airport):  # returns first and last image over airport
    conn = sqlite3.connect('intelligence_hub.db')
    cursor = conn.cursor()
    cursor.execute('''SELECT * FROM imagery WHERE airport = ? ORDER BY year DESC, month DESC, day DESC,
        hour DESC, minute DESC, second DESC LIMIT 1;''', (airport,))
    last_entry = cursor.fetchone()
    if last_entry:
        latest_date = date(last_entry[4], last_entry[5], last_entry[6])
    cursor.execute('''SELECT * FROM imagery WHERE airport = ? ORDER BY year ASC, month ASC, day ASC,
            hour ASC, minute ASC, second ASC LIMIT 1;''', (airport,))
    first_entry = cursor.fetchone()
    conn.close()
    if first_entry:
        first_date = date(first_entry[4], first_entry[5], first_entry[6])
    if latest_date and first_date:
        return first_date, latest_date
    return None, None


def get_over_threshold(airport):  # returns list of how many over threshold images on airport
    start_date, end_date = get_first_last_img(airport)
    current_date = start_date
    over_threshold = []
    threshold = 40
    while current_date <= end_date:  # loop that runs for all in between months
        year = current_date.year
        month = current_date.month
        current_date += timedelta(days=32 - current_date.day)
        month_aircrafts = get_entries_month(airport, year, month)
        month_count = 0
        for aircraft in month_aircrafts:  # checks all images if they are above threshold
            if aircraft >= threshold:
                month_count += 1
        over_threshold.append(month_count)
    return over_threshold




def get_all_results(airport):  # returns all data from airport
    conn = sqlite3.connect('intelligence_hub.db')
    cursor = conn.cursor()
    cursor.execute('''SELECT * FROM imagery WHERE airport = ?''', (airport,))
    rows = cursor.fetchall()
    conn.close()
    return rows


def delete_all_enteries():  # deletes all enteries in db to clean db if needed
    # Connect to the database
    conn = sqlite3.connect('intelligence_hub.db')
    cursor = conn.cursor()

    # Execute the SQL query to delete all entries from the 'imagery' table
    cursor.execute('DELETE FROM imagery')

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

add_information()