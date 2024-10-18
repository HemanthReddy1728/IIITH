import random
import csv
from prettytable import PrettyTable
import pandas as pd

df = pd.read_csv('stock_data.csv')
df = df.iloc[:, :-6]
df.to_csv('Q1_1.txt', index=False, sep=',') 

# ------------------------------------------------------------

with open("Q1_1.txt", "r", newline='') as input_csv, open("Q1_2.txt", "w", newline='') as output_csv:
    reader = csv.reader(input_csv)
    writer = csv.writer(output_csv)
    header = next(reader)
    writer.writerow(header)
    filtered_rows = filter(lambda row: float(row[-1]) >= -3, reader)
    writer.writerows(filtered_rows)

# ------------------------------------------------------------

remove_commas = lambda x: float(x.replace(',', ''))

with open('Q1_2.txt', 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)

open_values = list(map(lambda x: remove_commas(x['Open']), data))
high_values = list(map(lambda x: remove_commas(x['High']), data))
low_values = list(map(lambda x: remove_commas(x['Low']), data))

open_avg = sum(open_values) / len(open_values)
high_avg = sum(high_values) / len(high_values)
low_avg = sum(low_values) / len(low_values)

with open('Q1_3.txt', 'w') as file:
    file.write(f"{open_avg}\n{high_avg}\n{low_avg}")

# ------------------------------------------------------------

def str_to_float(s):
    return float(s.replace(',', ''))


def search_stocks_by_symbol(input_symbol, data):
    matching_stocks = []
    input_symbol = input_symbol.upper()
    
    for row in data:
        symbol = row['Symbol']
        if symbol.startswith(input_symbol):
            ltp = str_to_float(row['LTP'])
            matching_stocks.append((symbol, ltp))
    
    return matching_stocks

def search_table_by_symbol(input_symbol, data):
    matching_table = PrettyTable(["Symbol", "LTP"])
    input_symbol = input_symbol.upper()
    
    for row in data:
        symbol = row['Symbol']
        if symbol.startswith(input_symbol):
            ltp = str_to_float(row['LTP'])
            matching_table.add_row([symbol, ltp])
    
    return matching_table

data = []
with open('Q1_2.txt', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        data.append(row)

input_symbol = input("Enter the starting symbol (A-Z, a-z): ")
if not (ord('a') <= ord(input_symbol) <= ord('z') or ord('A') <= ord(input_symbol) <= ord('Z')):
    print("Invalid Symbol. Empty Table")
else:

    result_stocks = search_stocks_by_symbol(input_symbol, data)
    result_table = search_table_by_symbol(input_symbol, data)

    # if result_stocks:
    #     print("Symbol, LTP")
    #     for symbol, ltp in result_stocks:
    #         print(f"{symbol}, {ltp}")
    # else:
    #     print("No matching stocks found for the input symbol. Empty Table\n")
    #     print("Symbol, LTP")

    result_str = result_table.get_string()
    if len(result_str) > len("Symbol  |  LTP"):
        print(result_str)
        # print(len(result_str))
    else:
        print("No matching stocks found for the input symbol.")

    with open('Q1_4.txt', 'w') as outfile:
        if result_stocks:
            outfile.write("Symbol,LTP\n")
            for symbol, ltp in result_stocks:
                outfile.write(f"{symbol},{ltp}\n")
        else:
            outfile.write("Symbol,LTP\n")
            # outfile.write("Invalid Symbol OR No matching stocks found for the input symbol.")


    # # Write the result table to Q1_4.txt
    # with open('Q1_4.txt', 'w') as outfile:
    #     outfile.write(result_str)

# ----------------------------------------------------------------------------

salary_range = (10000.00, 50000.0)
age_range = (21, 55)
class_list = ['A', 'B', 'C', 'D', 'E', 'F']

data = []
for i in range(10):
    salary = round(random.uniform(*salary_range), 2)
    age = random.randint(*age_range)
    class_ = random.choice(class_list)
    status = random.choice([True, False])
    data.append((salary, age, class_, status))

with open('Q1_5.txt', 'w') as f:
    f.write('Salary,Age,Class,Status\n')
    for row in data:
        f.write(f'{row[0]},{row[1]},{row[2]},{row[3]}\n')
