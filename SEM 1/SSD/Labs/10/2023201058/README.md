# Lab Activity 10

## Question 1

### Code Overview

The code in the script performs the following tasks:

1. Imports the required libraries: random, csv, prettytable, and pandas.
2. Reads a CSV file named 'stock_data.csv' using pandas and assigns it to a dataframe variable 'df'.
3. Drops the last 6 columns from the dataframe.
4. Writes the resulting dataset to a text file named 'Q1_1.txt' using the `to_csv` function from pandas.
5. Reads the input CSV file 'Q1_1.txt', filters rows based on a condition, and writes the filtered rows to an output CSV file 'Q1_2.txt'.
6. Defines a function `remove_commas` to remove commas and convert a string to a float.
7. Reads the CSV file 'Q1_2.txt' and extracts the 'Open', 'High', and 'Low' values using csv.DictReader.
8. Transforms the data using map and lambda functions and calculates the average values for 'Open', 'High', and 'Low'.
9. Writes the average values to the "Q1_3.txt" file.
10. Defines a function `str_to_float` to remove commas and convert a string to a float.
11. Defines a function `search_stocks_by_symbol` to search for matching stocks based on a given symbol.
12. Defines a function `search_table_by_symbol` to create a table of matching stocks based on a given symbol using the prettytable library.
13. Reads data from the CSV file 'Q1_2.txt' and stores it in a list variable 'data'.
14. Prompts the user to enter a starting symbol (A-Z, a-z) and validates the input.
15. Calls the `search_stocks_by_symbol` and `search_table_by_symbol` functions to search for matching stocks and create a table.
16. Displays the result in the terminal in pretty table format and writes it to the "Q1_4.txt" file in CSV format.
17. Generates 10 rows of random data for salary, age, class, and status.
18. Writes the generated data to a file named "Q1_5.txt".

### Execution

- Open parent folder of this folder (2023201058) in vs code. Open 2023201058.py in vs code. Make sure stock_data.csv in in PARENT folder of 2023201058. Run the code with Ctrl+F5. The output files lie outside this folder (2023201058) in the parent folder.

### Assumption

- None
