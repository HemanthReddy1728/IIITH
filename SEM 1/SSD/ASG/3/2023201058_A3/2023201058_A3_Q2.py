import csv
from prettytable import PrettyTable

cricketers_directory = []

def load_cricketers_from_csv(filename):
    global cricketers_directory
    cricketers_directory = []
    try:
        # print(cricketers_directory)
        with open(filename, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            # cricketers_directory.extend(reader)
            for row in reader:

                #Calculating missing values
                if 'Batting Strike Rate' not in row or 'Bowling Strike Rate' not in row or 'Actual Strike Rate' not in row:
                    row['Batting Strike Rate'] = int(row['Runs']) / int(row['Balls']) if int(row['Balls']) > 0 else 0
                    row['Bowling Strike Rate'] = int(row['Wickets']) / int(row['Balls']) if int(row['Balls']) > 0 else 0
                    row['Actual Strike Rate'] = calculate_actual_strike_rate(row['Role'], row['Batting Strike Rate'], row['Bowling Strike Rate'])
                cricketers_directory.append(row)
                
        print("Data loaded from", filename)
    except FileNotFoundError:
        print("File not found. No data loaded.")
    except Exception as e:
        print("An error occurred while loading data:", e)
    # print(cricketers_directory)


def save_cricketers_to_csv(filename):
    try:
        with open(filename, mode='w', newline='') as file:
            fieldnames = cricketers_directory[0].keys() if cricketers_directory else []
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(cricketers_directory)
        print("Data saved to", filename)
    except Exception as e:
        print("An error occurred while saving data:", e)


def display_directory():
    if not cricketers_directory:
        print("Cricketer's Directory is empty.")
    else:
        table = PrettyTable()
        table.field_names = cricketers_directory[0].keys()
        for cricketer in cricketers_directory:
            table.add_row(cricketer.values())
        print(table)


def add_cricketer():
    first_name = input("Enter First Name: ")
    last_name = input("Enter Last Name: ")

    count_cric = 0
    for cricketer1 in cricketers_directory:
        if cricketer1["First Name"] == first_name and cricketer1["Last Name"] == last_name:
            count_cric += 1
    if count_cric != 0:
        print("A cricketer with the same name already exists in the directory. Adding Impossible.")
        return
    

    age = input("Enter Age: ")
    while not age.isdigit():
        print("Age must be a positive integer.")
        age = input("Enter Age: ")
    nationality = input("Enter Nationality: ")
    role = input("Enter Role (Batsmen, Bowler, All-rounder, Wk-Batsmen): ")
    while role not in ["Batsmen", "Bowler", "All-rounder", "Wk-Batsmen"]:
        print("Invalid role. Choose from Batsmen, Bowler, All-rounder, Wk-Batsmen.")
        role = input("Enter Role: ")
    runs = input("Enter Runs: ")
    while not runs.isdigit():
        print("Runs must be a non-negative integer.")
        runs = input("Enter Runs: ")
    balls = input("Enter Balls: ")
    while not balls.isdigit():
        print("Balls must be a non-negative integer.")
        balls = input("Enter Balls: ")
    wickets = input("Enter Wickets: ")
    while not wickets.isdigit():
        print("Wickets must be a non-negative integer.")
        wickets = input("Enter Wickets: ")
    
    batting_strike_rate = int(runs) / int(balls) if int(balls) > 0 else 0
    bowling_strike_rate = int(wickets) / int(balls) if int(balls) > 0 else 0
    actual_strike_rate = calculate_actual_strike_rate(role, batting_strike_rate, bowling_strike_rate)

    cricketer = {
        "First Name": first_name,
        "Last Name": last_name,
        "Age": int(age),
        "Nationality": nationality,
        "Role": role,
        "Runs": int(runs),
        "Balls": int(balls),
        "Wickets": int(wickets),
        "Batting Strike Rate": batting_strike_rate,
        "Bowling Strike Rate": bowling_strike_rate,
        "Actual Strike Rate": actual_strike_rate
    }

    cricketers_directory.append(cricketer)
    print("New entry added to the directory.")

# Function to remove an entry
def remove_cricketer():
    display_directory()
    if not cricketers_directory:
        return
    first_name, last_name = input("Enter the First Name & Last Name of the cricketer you want to remove: ").split()
    #  = input("Enter the Last Name of the cricketer you want to remove: ")
    removed = False

    for cricketer in cricketers_directory:
        if cricketer["First Name"] == first_name and cricketer["Last Name"] == last_name:
            cricketers_directory.remove(cricketer)
            removed = True
            print(f"{cricketer['First Name']} {cricketer['Last Name']} removed from the directory.")
            break

    if not removed:
        print(f"No cricketer with Name: {first_name} {last_name} found in the directory.")


def update_cricketer():
    display_directory()
    if not cricketers_directory:
        return

    first_name, last_name = input("Enter the First Name & Last Name of the cricketer you want to update: ").split()
    updated = False
    # print(first_name, last_name)
    for cricketer in cricketers_directory:
        if cricketer["First Name"] == first_name and cricketer["Last Name"] == last_name:
            print("Leave the field blank if you don't want to update it.")

            new_first_name = input(f"Enter First Name ({cricketer['First Name']}): ") or cricketer["First Name"]
            new_last_name = input(f"Enter Last Name ({cricketer['Last Name']}): ") or cricketer["Last Name"]
            # print(new_first_name, new_last_name)
            count_cric = 0
            for cricketer1 in cricketers_directory:
                if cricketer1["First Name"] == new_first_name and cricketer1["Last Name"] == new_last_name:
                    count_cric += 1
            if count_cric > 1:
                print("A cricketer with the same name already exists in the directory. Update Impossible.")
                return
            # print(count_cric, cricketer["First Name"], cricketer["Last Name"])
            cricketer["First Name"] = new_first_name
            cricketer["Last Name"] = new_last_name
            
            age = input(f"Enter Age ({cricketer['Age']}): ") or cricketer["Age"]
            while not age.isdigit():
                print("Age must be a positive integer.")
                age = input("Enter Age ({cricketer['Age']}): ")
            cricketer["Age"] = int(age)

            cricketer["Nationality"] = input(f"Enter Nationality ({cricketer['Nationality']}): ") or cricketer["Nationality"]

            role = input(f"Enter Role ({cricketer['Role']}): ") or cricketer["Role"]
            while role not in ["Batsmen", "Bowler", "All-rounder", "Wk-Batsmen"]:
                print("Invalid role. Choose from Batsmen, Bowler, All-rounder, Wk-Batsmen.")
                role = input(f"Enter Role ({cricketer['Role']}): ")
            cricketer["Role"] = role
            
            runs = input(f"Enter Runs ({cricketer['Runs']}): ") or cricketer["Runs"]
            while not runs.isdigit():
                print("Runs must be a non-negative integer.")
                runs = input("Enter Runs ({cricketer['Runs']}): ")
            cricketer["Runs"] = int(runs)

            balls = input(f"Enter Balls ({cricketer['Balls']}): ") or cricketer["Balls"]
            while not balls.isdigit():
                print("Balls must be a non-negative integer.")
                balls = input("Enter Balls ({cricketer['Balls']}): ")
            cricketer["Balls"] = int(balls)

            wickets = input(f"Enter Wickets ({cricketer['Wickets']}): ") or cricketer["Wickets"]
            while not wickets.isdigit():
                print("Wickets must be a non-negative integer.")
                wickets = input("Enter Wickets ({cricketer['Wickets']}): ")
            cricketer["Wickets"] = int(wickets)


            batting_strike_rate = cricketer["Runs"] / cricketer["Balls"] if cricketer["Balls"] > 0 else 0
            bowling_strike_rate = cricketer["Wickets"] / cricketer["Balls"] if cricketer["Balls"] > 0 else 0
            cricketer["Batting Strike Rate"] = batting_strike_rate
            cricketer["Bowling Strike Rate"] = bowling_strike_rate
            cricketer["Actual Strike Rate"] = calculate_actual_strike_rate(cricketer["Role"], batting_strike_rate, bowling_strike_rate)
            updated = True
            print(f"{cricketer['First Name']} {cricketer['Last Name']} updated in the directory.")
            break

    if not updated:
        print(f"No cricketer with Name: {first_name} {last_name} found in the directory.")

# Function to search for entries based on attributes
def search_cricketer():
    attribute = input("Enter the attribute to search (First Name, Last Name, Age, Nationality, Role, Runs, Balls, Wickets): ").strip()
    # not done : , Batting Strike Rate, Bowling Strike Rate, Actual Strike Rate
    value = input("Enter the value to search for: ").strip()

    found = False

    table = PrettyTable()
    table.field_names = cricketers_directory[0].keys()

    for cricketer in cricketers_directory:
        if attribute in cricketer and cricketer[attribute] == value:
            found = True
            # display_directory()
            table.add_row(cricketer.values())
            # break

    if not found:
        print(f"No cricketer(s) with {attribute} equal to {value} found in the directory.")
    else:
        print("Search Result:")
        print(table)



def calculate_actual_strike_rate(role, batting_strike_rate, bowling_strike_rate):
    if role == "Batsmen" or role == "Wk-Batsmen":
        return batting_strike_rate
    elif role == "Bowler":
        return bowling_strike_rate
    elif role == "All-rounder":
        return max(batting_strike_rate, bowling_strike_rate)
    else:

        return -1  # Return -1 for unknown roles


def main_menu():
    while True:
        print("\nCricketer's Directory Menu:")
        print("1. Load from CSV")
        print("2. Display Directory")
        print("3. Add Cricketer")
        print("4. Remove Cricketer")
        print("5. Update Cricketer")
        print("6. Search Cricketer")
        print("7. Save to CSV and Quit")

        choice = input("Enter your choice: ")
        
        if choice == '1':
            load_cricketers_from_csv("cricketers.csv")
        elif choice == '2':
            display_directory()
        elif choice == '3':
            add_cricketer()
        elif choice == '4':
            remove_cricketer()
        elif choice == '5':
            update_cricketer()
        elif choice == '6':
            search_cricketer()
        elif choice == '7':
            save_cricketers_to_csv("cricketers.csv")
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a valid option.")

if __name__ == "__main__":
    main_menu()
