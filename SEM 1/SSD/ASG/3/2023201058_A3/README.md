# Assignment 3
## Question 1
### Overview
Expense Tracker

### Features

- Add participants to the expense tracker.
- Record expenses specifying the payer, amount paid, and participants involved.
- View a summary of each participant's outstanding balances.

### Usage

1. Run the program in your Python environment.
2. Follow the menu options to perform the desired actions:
   - 1. Add participant(s): Enter the names of participants, separated by commas.
   - 2. Add expense: Record expenses by specifying the payer, amount paid, and participants involved.
   - 3. Show all participants: Display a list of all participants.
   - 4. Show expenses: View the balances (amount owes and amount gets back) for each participant.
   - 5. Exit/Export: Save the data to a CSV file named "expenses.csv" and exit the program.

### Example

1. Adding participants:

   Enter participant names (comma-separated): Alice, Bob, Carol

2. Adding expenses:

   Paid by (participant's name): Alice
   Amount paid: 100
   Distributed amongst (comma-separated): Bob, Carol

3. Showing all participants:

   Participants:
   Alice
   Bob
   Carol

4. Showing expenses:

   Expenses:
   Name                 Amount Owes     Amount Gets Back
   Alice                $0.00           $50.00
   Bob                  $50.00          $0.00
   Carol                $50.00          $0.00

5. Exiting and exporting data to "expenses.csv."


### Execution
- By executing following commands in the terminal (LINUX) you can run the program.

```python
python3 2023201058_A3_Q1.py
```

### Assumption
- Don't give negative values as input
- 



## Question 2
### Overview
Cricketer's Directory

### Functionality

The program provides the following functionality:

- Load from CSV: Load cricketer data from a CSV file (cricketers.csv).

- Display Directory: Display the current directory of cricketers in a tabular format.

- Add Cricketer: Add a new cricketer to the directory by providing their details.

- Remove Cricketer: Remove a cricketer from the directory by specifying their first name and last name.

- Update Cricketer: Update the details of a cricketer in the directory.

- Search Cricketer: Search for cricketers based on attributes (e.g., First Name, Last Name, Age, Nationality, Role, Runs, Balls, Wickets).

- Save to CSV and Quit: Save the current directory to a CSV file (cricketers.csv) and exit the program.


### Execution
- By executing following commands in the terminal (LINUX) you can run the program.

```python
pip install prettytable
python3 2023201058_A3_Q2.py
```

### Assumption
- None of the input should be negative or zero or decimals. Only positive intgers a nd strings are allowed.
- 


## Question 3
### Overview
Cricketers Statistics and Expenses Analysis

## Cricket Player Statistics Analysis

### Overview

The `load_cricketers_from_csv` function reads and analyzes data from a CSV file named "cricketers.csv." This data contains information about cricket players, including their batting strike rates, bowling strike rates, roles, and other attributes.



### Usage

To use the cricket player statistics analysis:

1. Ensure you have a CSV file named "cricketers.csv" with the necessary columns: "First Name," "Last Name," "Role," "Runs," "Balls," "Wickets," "Batting Strike Rate," "Bowling Strike Rate," and "Actual Strike Rate."

2. Call the `load_cricketers_from_csv("cricketers.csv")` function to load and analyze the data.

3. Once the data is loaded, you can call the `create_strike_rate_histograms()` function to create histograms for player comparison. These histograms display batting and bowling strike rates for each player.

## Expenses Data Analysis

### Overview

The code also includes functionality to analyze expenses data from a CSV file named "expenses.csv." The expenses data is used to calculate and visualize who owes money and who is due to receive money.



### Usage

To use the expenses data analysis:

1. Ensure you have a CSV file named "expenses.csv" with the necessary columns: "Name," "Amount Owes," and "Amount Gets Back."

2. Call the code to load and analyze the data. This code will generate and display two pie charts:
   - The first pie chart shows the distribution of amounts owed by individuals.
   - The second pie chart shows the distribution of amounts due to individuals.


### Execution
- By executing following commands in the terminal (LINUX) you can run the program.


```python
pip install matplotlib
python3 2023201058_A3_Q3.py
```

### Assumption
- Based on Q1 and Q2, Data is generated. Please follow their assumptions.


