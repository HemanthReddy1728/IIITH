import csv
import matplotlib.pyplot as plt

owes_data = {}
gets_back_data = {}
owes_total_amount = 0
gets_back_total_amount = 0


with open("expenses.csv", mode="r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        name = row["Name"]
        amount_owes = float(row["Amount Owes"])
        amount_gets_back = float(row["Amount Gets Back"])

        if amount_owes > 0:
            owes_data[name] = amount_owes
            owes_total_amount += amount_owes

        if amount_gets_back > 0:
            gets_back_data[name] = amount_gets_back
            gets_back_total_amount += amount_gets_back
    print("Data loaded from", file.name)

# Pie Chart 1 - "Owes"
if owes_data:
    plt.figure(figsize=(6, 6))
    labels = list(owes_data.keys())  
    amounts = owes_data.values()
    explode = [0.1 if amount == max(amounts) else 0 for amount in amounts]
    plt.pie(amounts, labels=labels, explode=explode, autopct='%1.1f%%', startangle=140)
    # plt.title("Owes")
    plt.legend(title="Amount = {}\nOwes".format(owes_total_amount), loc="lower right")
    plt.axis('equal')
    plt.show()

# Pie Chart 2 - "Gets Back"
if gets_back_data:
    plt.figure(figsize=(6, 6))
    labels = list(gets_back_data.keys()) 
    amounts = gets_back_data.values()
    explode = [0.1 if amount == max(amounts) else 0 for amount in amounts]
    plt.pie(amounts, labels=labels, explode=explode, autopct='%1.1f%%', startangle=140)
    # plt.title("Gets Back")
    plt.legend(title="Amount = {}\nGets Back".format(gets_back_total_amount), loc="lower right")
    plt.axis('equal')
    plt.show()



cricketers_directory = []


def load_cricketers_from_csv(filename):
    global cricketers_directory
    cricketers_directory = []
    try:
        with open(filename, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Calculating missing values
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


def calculate_actual_strike_rate(role, batting_strike_rate, bowling_strike_rate):
    if role == "Batsmen" or role == "Wk-Batsmen":
        return batting_strike_rate
    elif role == "Bowler":
        return bowling_strike_rate
    elif role == "All-rounder":
        return max(batting_strike_rate, bowling_strike_rate)
    else:
        return -1  # Return -1 for unknown roles


def create_strike_rate_histograms():
    if not cricketers_directory:
        print("Cricketer's Directory is empty.")
        return

    # Lists to store strike rates for each type
    batting_strike_rates = []
    bowling_strike_rates = []
    # actual_strike_rates = []
    player_names = []

    for cricketer in cricketers_directory:
        player_names.append(f"{cricketer['First Name']}\n{cricketer['Last Name']}")
        batting_strike_rates.append(float(cricketer['Batting Strike Rate']))
        bowling_strike_rates.append(float(cricketer['Bowling Strike Rate']))
        # actual_strike_rates.append(float(cricketer['Actual Strike Rate']))

    num_players = len(player_names)
    x = range(num_players)

    fig, ax = plt.subplots()
    bar_width = 0.2
    opacity = 0.9

    plt.bar(x, batting_strike_rates, bar_width, alpha=opacity, label='Batting Strike Rate')
    plt.bar([i + bar_width for i in x], bowling_strike_rates, bar_width, alpha=opacity, label='Bowling Strike Rate')
    # plt.bar([i + 2 * bar_width for i in x], actual_strike_rates, bar_width, alpha=opacity, label='Actual Strike Rate')

    plt.xlabel('Players')
    plt.ylabel('Strike Rates')
    plt.title('Player Comparison of Strike Rates')
    plt.xticks([i + bar_width for i in x], player_names)
    plt.legend()

    plt.show()

load_cricketers_from_csv("cricketers.csv")
create_strike_rate_histograms()