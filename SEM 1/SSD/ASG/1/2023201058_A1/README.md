# Assignment 1 
## Question 1
### Overview
- Distance and direction calculations b/w police and thief
- Point 2

### Execution
- By executing following commands in the terminal (LINUX) you can run the program.

```shell
chmod +x 2023201058_A1_q1.sh
./2023201058_A1_q1.sh
```

### Assumption
- Give each input value one by one instead of all values at a time



## Question 2
### Overview
- Profit Maximisation 
- Point 2

### Execution
- By executing following commands in the terminal (LINUX) you can run the program.

```shell
chmod +x 2023201058_A1_q2.sh
./2023201058_A1_q2.sh
```

### Assumption
- If array input is [5,7,1,4,6], give it as 5 7 1 4 6 
- Second Assumption


## Question 3
### Overview
- Reversing words in a sentence but not the special symbols
- Point 2

### Execution
- By executing following commands in the terminal (LINUX) you can run the program.

```shell
chmod +x 2023201058_A1_q3.sh
./2023201058_A1_q3.sh </input_text_file_path/>
```

### Assumption
- None


## Question 4
### Overview
- Creating a stored procedure that utilizes cursors to create a new table with given requirements
- Point 2

### Execution
- Rename folder containing csv files into 'Question4Artifacts' and copy it into /var/lib/mysql-files folder using sudo cp -ir path/to/Question4Artifacts /var/lib/mysql-files/ in terminal.
- Open workbench app. Create a new database schema by right clicking in the sidebar of workbench app (View -> panels -> Show sidebar) and select 'Create Schema' option. Make that new schema as default schema by right clicking on it and selecting 'Set as default schema' option.

- By executing given script in the MySQL Workbench (Shift+Ctrl+O) you can run the program. (Ctrl+A and then click lightning icon)


### Assumptions 
- If atleast one of:
CourseCompletionsCount >= 2 
CertificateCompletionsCount >= 2 
is false then skill level is made NULL and increment=0
- Date format changed using STR_TO_DATE() function


## Question 5
### Overview
- Writing SQL queries and creating a table “report” with columns {UserID, UserName, Result (“Yes” or “No”)} with respective reasonable data types.
- Point 2

### Execution
- Rename folder containing csv files into 'Question5Artifacts' and copy it into /var/lib/mysql-files folder using sudo cp -ir path/to/Question5Artifacts /var/lib/mysql-files/ in terminal.
- Open workbench app. Create a new database schema by right clicking in the sidebar of workbench app (View -> panels -> Show sidebar) and select 'Create Schema' option. Make that new schema as default schema by right clicking on it and selecting 'Set as default schema' option.

- By executing given script in the MySQL Workbench (Shift+Ctrl+O) you can run the program. (Ctrl+A and then click lightning icon)


### Assumption
- First Assumption - First, calculate/filter out users who are able to sell more than two items 
- Second Assumption- Next, out of above sellers, Give result='Yes' only to those sellers who have sold their favorite item on 2nd date from top (using limit and offset keywords, also dates sorted in increasing order). For all others, result='No'.
- Third assumption - Used OrderID INT PRIMARY KEY, OrderDate DATE, BrandID INT, BuyerID INT, SellerID INT instead of space seperated column names (`Order ID`, `Order Date`, etc.) for simplicity.