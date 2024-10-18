import csv

expense_tracker = {}

def add_participants():
    participants = input("Enter participant names (comma-separated): ").split(',')
    for participant in participants:
        expense_tracker[participant.strip()] = {'Amount Owes': 0, 'Amount Gets Back': 0}


def add_expense():
    paid_by = input("Paid by (participant's name): ")
    if paid_by not in expense_tracker:
        print("Participant not found.")
        return

    try:
        amount = float(input("Amount paid: "))
        participants_involved = input("Distributed amongst (comma-separated): ").split(',')

        # Validating participants involved
        for participant in participants_involved:
            participant = participant.strip()
            if participant not in expense_tracker:
                print(f"Participant '{participant}' not found. Expense not added.")
                return

        # Calculate the split amount for each participant
        split_amount = amount / len(participants_involved)

        # Update expenses for all participants involved
        for participant in participants_involved:
            if participant == paid_by and paid_by in participants_involved:
                    expense_tracker[participant]['Amount Gets Back'] += split_amount * len(participants_involved) - split_amount
            
            else:
                expense_tracker[participant]['Amount Owes'] += split_amount

        if paid_by not in participants_involved:
            expense_tracker[paid_by]['Amount Gets Back'] += split_amount * len(participants_involved)
            
    except ValueError:
        print("Invalid input. Please enter a valid numeric amount.")


def show_participants():
    print("Participants:")
    for participant in expense_tracker:
        print(participant)


def show_expenses():
    if not expense_tracker:
        print("No expenses recorded yet.")
        return

    print("\nExpenses:")
    print("{:<20} {:<15} {:<15}".format("Name", "Amount Owes", "Amount Gets Back"))
    for participant, expenses in expense_tracker.items():
        print("{:<20} ${:<15.2f} ${:<15.2f}".format(participant, expenses['Amount Owes'], expenses['Amount Gets Back']))


while True:
    print("\nMenu:")
    print("1. Add participant(s)")
    print("2. Add expense")
    print("3. Show all participants")
    print("4. Show expenses")
    print("5. Exit/Export")

    choice = input("Enter your choice: ")

    if choice == '1':
        add_participants()
    elif choice == '2':
        add_expense()
    elif choice == '3':
        show_participants()
    elif choice == '4':
        show_expenses()
    elif choice == '5':
        
        with open("expenses.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Amount Owes", "Amount Gets Back"])
            for participant, expenses in expense_tracker.items():
                writer.writerow([participant, expenses['Amount Owes'], expenses['Amount Gets Back']])
        break
    else:
        print("Invalid choice. Please try again.")
