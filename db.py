# db.py
# This script manages the support ticket database (support_tickets.csv).

import csv
import os
from datetime import datetime

# Define the name of our database file
CSV_FILE = 'support_tickets.csv'
HEADERS = ['TicketID', 'CustomerName', 'CustomerEmail', 'Product', 'ProblemDescription', 'Status', 'DateCreated']

def get_next_ticket_id():
    """
    Calculates the next ticket ID by reading the number of existing tickets.
    Starts from NEX-12345.
    """
    try:
        # os.path.getsize > 0 checks if the file is not empty
        if os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0:
            with open(CSV_FILE, 'r') as file:
                # Count lines, subtract 1 for the header
                num_tickets = len(file.readlines()) - 1
                return f"NEX-{12345 + num_tickets}"
        else:
            return "NEX-12345" # This is the first ticket
    except Exception:
        return "NEX-12345" # Fallback for the first ticket

def create_support_ticket(customer_name, customer_email, product, problem_description):
    """
    Creates a new support ticket and adds it to the CSV file.

    Args:
        customer_name (str): The customer's full name.
        customer_email (str): The customer's email address.
        product (str): The Nexora product in question.
        problem_description (str): A brief description of the issue.

    Returns:
        str: The TicketID of the newly created ticket.
    """
    # Check if the CSV file exists. If not, create it with the headers.
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(HEADERS)

    # Generate the new ticket details
    ticket_id = get_next_ticket_id()
    status = "Open"
    # Get current time in a clean format (e.g., 2025-09-30 20:30:00)
    date_created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Prepare the new row to be added
    new_ticket = [ticket_id, customer_name, customer_email, product, problem_description, status, date_created]

    # Open the file in 'append' mode to add the new ticket at the end
    with open(CSV_FILE, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(new_ticket)

    # Return the new ticket ID so the bot can show it to the user
    return ticket_id

# This block is for testing the script directly.
# It will only run when you execute `python db.py` in your terminal.
if __name__ == "__main__":
    print("--- Testing the db.py script ---")
    
    # Example 1: Creating a ticket for a known issue
    print("Creating a ticket for Arun Kumar...")
    ticket1_id = create_support_ticket(
        customer_name="Arun Kumar",
        customer_email="arun.k@example.com",
        product="SecureView 5000",
        problem_description="The camera will not power on."
    )
    print(f"Success! Ticket '{ticket1_id}' was created.")
    
    # Example 2: Creating another ticket
    print("\nCreating a ticket for Priya Sharma...")
    ticket2_id = create_support_ticket(
        customer_name="Priya Sharma",
        customer_email="priya.s@example.com",
        product="Thermosmart Pro",
        problem_description="The device is not connecting to the Wi-Fi."
    )
    print(f"Success! Ticket '{ticket2_id}' was created.")
    
    print(f"\nCheck the '{CSV_FILE}' file to see the results.")

# Add this new function to your db.py file

def find_ticket_by_id(ticket_id):
    """
    Searches the CSV file for a ticket with a specific ID.

    Args:
        ticket_id (str): The ID of the ticket to find (e.g., "NEX-12345").

    Returns:
        dict: A dictionary containing the ticket details if found, otherwise None.
    """
    if not os.path.exists(CSV_FILE):
        return None # Return nothing if the database file doesn't even exist

    with open(CSV_FILE, 'r', newline='') as file:
        reader = csv.DictReader(file) # Use DictReader to easily access columns by name
        for row in reader:
            if row['TicketID'] == ticket_id:
                return row # Return the entire row of data as a dictionary
    
    return None # Return nothing if no match was found after checking all rows