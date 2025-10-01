# agent.py
# This script demonstrates how an "agent" can decide whether to
# answer a question or perform an action using a tool.

# We import the "tool" from our db.py file
from db import create_support_ticket

def run_agent(user_query: str):
    """
    Simulates an agent that decides which tool to use based on the user's query.
    """
    print(f"👤 User: {user_query}")
    
    # --- This is the Agent's "brain" or "decision-making logic" ---
    # It checks for keywords to decide if an action is needed.
    action_keywords = ["not working", "broken", "won't turn on", "support", "help"]
    
    # Check if any keyword is in the user's query (case-insensitive)
    if any(keyword in user_query.lower() for keyword in action_keywords):
        
        # --- Decision: Use the Ticket Creator Tool ---
        print("🤖 Agent Decision: The user needs help. I will use the 'create_support_ticket' tool.")
        
        # In a real app, the agent would now ask the user for these details.
        # Here, we'll just simulate it.
        print("🤖 Agent: To help you, I'll create a support ticket. I'll need some details...")
        name = "Saanvi Reddy" # Fetched from user profile or conversation
        email = "saanvi.r@example.com" # Fetched from user profile or conversation
        product = "SecureView 5000" # Inferred from the conversation
        
        # The agent calls the tool (our function from db.py)
        ticket_id = create_support_ticket(
            customer_name=name,
            customer_email=email,
            product=product,
            problem_description=user_query
        )
        
        print(f"🤖 Agent: All done! I've created ticket {ticket_id} for you. A support specialist will be in touch.")
        
    else:
        # --- Decision: Use the RAG (Knowledge) Tool ---
        print("🤖 Agent Decision: This is an informational question. I will use my knowledge base to answer.")
        
        # In a real app, this is where you would call the RAG chain from your app.py
        print("🤖 Agent: [Pretending to call the RAG chain to find an answer...]")
        print("🤖 Agent: The standard warranty period for all Nexora smart devices is 2 years.")

# --- Let's test our new agent ---
if __name__ == "__main__":
    print("--- Testing the Agent's Decision Making ---\n")
    
    # Test Case 1: An informational question
    run_agent("What is the standard warranty period for Nexora devices?")
    
    print("\n--------------------------------------------\n")
    
    # Test Case 2: A problem that requires action
    run_agent("My SecureView 5000 is broken and not working!")