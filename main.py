import time
from config import GROQ_API_KEY
from data_loader import load_candidates
from llm_interface import initialize_groq_llm
from interview_manager import ConversationalHRAssistant

def main():
    if not GROQ_API_KEY:
        print("GROQ_API_KEY environment variable not set.")
        return

    candidate_db = load_candidates()
    groq_llm = initialize_groq_llm(GROQ_API_KEY)
    hr_assistant = ConversationalHRAssistant(groq_llm=groq_llm)

    phone_number = input("Enter phone number of the candidate: ").strip()
    candidate_info = hr_assistant.identify_candidate(phone_number, candidate_db)
    if not candidate_info:
        print("Candidate not found.")
        return

    hr_assistant.candidate_info = candidate_info
    print("\n--- INTERVIEW STARTING ---\n")
    
    greeting = hr_assistant.initial_greeting(candidate_info["name"])
    hr_assistant.add_to_history("HR", greeting)
    print(f"HR: {greeting}")

    while not hr_assistant.interview_ended:
        candidate_response = input("Candidate: ").strip()
        if candidate_response.lower() in ["exit", "quit", "end"]:
            print("Interview simulation ended by user.")
            break
        hr_response = hr_assistant.process_response(candidate_response)
        print(f"HR: {hr_response}")
        time.sleep(0.5)

    print("\n--- INTERVIEW COMPLETED ---\n")
    if input("See full interview transcript? (y/n): ").strip().lower() == 'y':
        print("\n=== INTERVIEW TRANSCRIPT ===\n")
        for entry in hr_assistant.interview_history:
            print(f"{entry['speaker']}: {entry['text']}\n")

if __name__ == "__main__":
    main()