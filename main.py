import os
import json
import time
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from ingest import ingest_candidates, JSON_PATH
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load candidate database
def load_candidates(filepath="dummy-resume.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        candidate_list = json.load(f)

    # Convert list to dict: { phone_number: candidate_data }
    phone_index = {}
    for candidate in candidate_list:
        phone = candidate.get("phone")
        if phone:
            phone_index[phone] = candidate

    return phone_index

# Lookup candidate by phone
def get_candidate_by_phone(phone_number, db):
    return db.get(phone_number)

# Extract resume text for RAG
def extract_resume_text(candidate_data):
    parts = [
        f"Name: {candidate_data.get('name', '')}",
        f"Experience: {candidate_data.get('experience_years', 0)} years",
        f"Skills: {', '.join(candidate_data.get('skills', []))}",
        f"Current Role: {candidate_data.get('current_role', '')} at {candidate_data.get('company', '')}",
        f"Education: {candidate_data.get('education', '')}",
    ]
    for project in candidate_data.get("projects", []):
        parts.append(f"Project: {project['title']} - {project['description']}")
    return "\n".join(parts)

# --- ConversationalHRAssistant Class ---
class ConversationalHRAssistant:
    def __init__(self, groq_llm=None, use_model="groq"):
        self.groq_llm = groq_llm
        self.use_model = use_model
        self.interview_history = []
        self.current_stage = "greeting"  # Track interview stage
        self.candidate_info = None
        self.interview_ended = False

    def _get_llm(self):
        if self.use_model == "groq" and self.groq_llm:
            return self.groq_llm
        else:
            raise ValueError(f"LLM for '{self.use_model}' not initialized or invalid 'use_model' specified.")

    def identify_candidate(self, phone_number, candidate_db):
        return get_candidate_by_phone(phone_number, candidate_db)

    def get_resume_context(self, candidate_info):
        if candidate_info:
            return extract_resume_text(candidate_info)
        return ""

    def _generate_response(self, prompt_template, context, **kwargs):
        llm = self._get_llm()
        prompt = PromptTemplate(template=prompt_template, input_variables=list(kwargs.keys()) + ["context"])
        formatted_prompt = prompt.format(context=context, **kwargs)
        
        try:
            # Use invoke instead of predict (which is deprecated)
            response = llm.invoke(formatted_prompt).content.strip()
        except AttributeError:
            # Fallback to predict if invoke doesn't work
            response = llm.predict(formatted_prompt).strip()
            
        return response

    def initial_greeting(self, candidate_name):
        """Generate the initial greeting when the call starts"""
        return f"Hello, am I speaking with {candidate_name}?"

    def check_availability(self):
        """Ask if this is a good time to talk"""
        return "Is this a good time to talk about your application with us?"

    def end_call_politely(self):
        """End the call if candidate is not available"""
        return "I understand. Thank you for letting me know. I'll call back at a more convenient time. Have a great day!"

    def generate_verification_questions(self, candidate_name, resume_context):
        """Generate verification questions based on resume"""
        prompt_template = """You are an HR representative conducting a phone interview. Based on the following resume of {candidate_name}:

        {context}

        Generate exactly 2 natural-sounding verification questions to confirm the candidate's basic details from their resume.
        Format your response as a numbered list with ONLY the questions (no explanations or preamble):
        1. First question about education
        2. Second question about current role
        """
        questions_raw = self._generate_response(prompt_template, resume_context, candidate_name=candidate_name)
        
        # Parse the response to extract clean questions
        questions = []
        for line in questions_raw.split("\n"):
            line = line.strip()
            if line.startswith("1.") or line.startswith("2."):
                # Remove the number prefix
                question = line[2:].strip()
                questions.append(question)
                
        # Ensure we have at least two questions
        if len(questions) < 2:
            # Add default questions if we don't get properly formatted ones
            if not questions:
                questions = [
                    f"Can you confirm your education background as shown on your resume?",
                    f"Could you tell me about your current role at {self.candidate_info.get('company', 'your company')}?"
                ]
            elif len(questions) == 1:
                questions.append(f"Could you tell me about your current role at {self.candidate_info.get('company', 'your company')}?")
                
        return questions

    def generate_skill_questions(self, candidate_name, resume_context, skills):
        """Generate technical questions about skills listed in resume"""
        prompt_template = """You are an HR representative conducting a phone interview. Based on the following resume of {candidate_name}:

        {context}

        The candidate has these skills: {skills}
        
        Generate exactly 2 conversational questions to verify their knowledge/experience with these skills.
        Format your response as a numbered list with ONLY the questions (no explanations or preamble):
        1. First question about skill 1
        2. Second question about skill 2
        """
        questions_raw = self._generate_response(prompt_template, resume_context, candidate_name=candidate_name, skills=skills)
        
        # Parse the response to extract clean questions
        questions = []
        for line in questions_raw.split("\n"):
            line = line.strip()
            if line.startswith("1.") or line.startswith("2."):
                # Remove the number prefix
                question = line[2:].strip()
                questions.append(question)
                
        # Ensure we have at least two questions
        if len(questions) < 2:
            # Add default questions if we don't get properly formatted ones
            skills_list = skills.split(",")
            if not questions:
                questions = [
                    f"I see {skills_list[0].strip()} is one of your skills. Could you tell me how you've used it in your previous roles?",
                ]
                if len(skills_list) > 1:
                    questions.append(f"Could you also tell me about your experience with {skills_list[1].strip()}?")
                else:
                    questions.append(f"What aspects of {skills_list[0].strip()} do you find most interesting in your work?")
            elif len(questions) == 1:
                if len(skills_list) > 1:
                    questions.append(f"Could you also tell me about your experience with {skills_list[1].strip()}?")
                else:
                    questions.append(f"What aspects of {skills_list[0].strip()} do you find most interesting in your work?")
                
        return questions

    def generate_follow_up(self, candidate_name, resume_context, last_response, current_topic):
        """Generate a follow-up based on candidate's last response"""
        prompt_template = """You are an HR representative conducting a phone interview. Based on the following resume of {candidate_name}:

        {context}

        And the candidate's last response about {current_topic}: "{last_response}"
        
        Generate ONE natural-sounding follow-up question that delves deeper into what they just said. 
        Your response should be ONLY the question with no explanations or preamble.
        """
        follow_up = self._generate_response(
            prompt_template, 
            resume_context, 
            candidate_name=candidate_name, 
            last_response=last_response,
            current_topic=current_topic
        )
        
        # Clean up the response
        lines = follow_up.strip().split("\n")
        for line in lines:
            clean_line = line.strip()
            # Get first non-empty line with a question mark or that seems like a question
            if clean_line and ('?' in clean_line or 
                              any(q in clean_line.lower() for q in ["could you", "can you", "how did", "what was", "tell me"])):
                follow_up = clean_line
                break
        else:
            # Default follow up if we couldn't find a clear question
            follow_up = f"That's interesting. Could you tell me more about your experience with {current_topic}?"
        
        # Remove numbering if present
        if follow_up.startswith("1. "):
            follow_up = follow_up[3:]
            
        return follow_up

    def handle_candidate_questions(self, question):
        """Handle any questions from the candidate"""
        prompt_template = """You are an HR representative finishing up a phone interview. The candidate has asked: "{question}"
        
        Provide a friendly, helpful response as an HR person would. Be brief but informative.
        """
        return self._generate_response(prompt_template, "", question=question)

    def generate_role_exploration_question(self, candidate_info):
        """Generate a question to explore the candidate's current role and experience"""
        current_role = candidate_info.get("current_role", "")
        company = candidate_info.get("company", "")
        experience_years = candidate_info.get("experience_years", 0)
        has_projects = len(candidate_info.get("projects", [])) > 0
        
        # Create different question types based on available information
        if current_role and company:
            return f"Can you tell me more about your current role as {current_role} at {company}? I'd particularly like to hear about your day-to-day responsibilities and any significant achievements."
        elif has_projects and len(candidate_info.get("projects", [])) > 0:
            project = candidate_info.get("projects", [])[0]
            return f"I see you worked on a project called '{project.get('title', '')}'. Could you tell me more about your role in this project and any challenges you faced?"
        elif experience_years > 5:
            return f"With {experience_years} years of experience, you must have handled some challenging situations. Could you share a difficult problem you faced in your career and how you resolved it?"
        else:
            return "Could you tell me about a recent project or task you worked on that you're particularly proud of?"
    
    def generate_role_followup(self, candidate_name, resume_context, last_response):
        """Generate a follow-up question about the candidate's role or project"""
        prompt_template = """You are an HR representative conducting a phone interview. Based on the following resume of {candidate_name}:

        {context}

        The candidate just described their role or project: "{last_response}"
        
        Generate ONE natural follow-up question that explores:
        - If they mentioned a challenge: how they approached solving it
        - If they mentioned teamwork: how they handle conflicts or collaborated
        - If they mentioned a project: their specific contribution or approach
        - If they mentioned achievements: what skills enabled their success
        
        Your response should be ONLY the question with no explanations or preamble.
        """
        
        follow_up = self._generate_response(
            prompt_template, 
            resume_context, 
            candidate_name=candidate_name, 
            last_response=last_response
        )
        
        # Clean up the response
        lines = follow_up.strip().split("\n")
        for line in lines:
            clean_line = line.strip()
            # Get first non-empty line with a question mark or that seems like a question
            if clean_line and ('?' in clean_line or 
                              any(q in clean_line.lower() for q in ["could you", "can you", "how did", "what was", "tell me"])):
                follow_up = clean_line
                break
        else:
            # Default follow up if we couldn't find a clear question
            follow_up = "That's interesting. Could you share a specific challenge you faced in this role and how you overcame it?"
        
        # Remove numbering if present
        if follow_up.startswith("1. "):
            follow_up = follow_up[3:]
            
        return follow_up

    def wrap_up_interview(self):
        """Generate closing remarks for the interview"""
        return "Thank you for your time today and for sharing your experience with me. Do you have any questions about the role or our company that I can answer for you?"

    def add_to_history(self, speaker, text):
        """Add an exchange to the interview history"""
        self.interview_history.append({"speaker": speaker, "text": text})
        
    def process_response(self, candidate_response):
        """Process candidate response based on current interview stage"""
        if self.interview_ended:
            return "The interview has already ended."
            
        if self.current_stage == "greeting":
            if "yes" in candidate_response.lower() or self.candidate_info["name"].lower() in candidate_response.lower():
                self.add_to_history("Candidate", candidate_response)
                self.current_stage = "availability"
                availability_question = self.check_availability()
                self.add_to_history("HR", availability_question)
                return availability_question
            else:
                self.add_to_history("Candidate", candidate_response)
                return "I apologize for the confusion. Have a nice day."
                
        elif self.current_stage == "availability":
            self.add_to_history("Candidate", candidate_response)
            if "no" in candidate_response.lower() or "not" in candidate_response.lower() or "another" in candidate_response.lower():
                end_message = self.end_call_politely()
                self.add_to_history("HR", end_message)
                self.interview_ended = True
                return end_message
            else:
                self.current_stage = "verification"
                resume_context = self.get_resume_context(self.candidate_info)
                verification_questions = self.generate_verification_questions(
                    self.candidate_info["name"], 
                    resume_context
                )
                # Ask first verification question
                first_question = verification_questions[0].strip()
                if first_question.startswith("1. "):
                    first_question = first_question[3:]
                self.add_to_history("HR", first_question)
                self.verification_questions = verification_questions[1:]  # Store remaining questions
                return first_question
                
        elif self.current_stage == "verification":
            self.add_to_history("Candidate", candidate_response)
            
            # If we have more verification questions, ask the next one
            if hasattr(self, 'verification_questions') and self.verification_questions:
                next_question = self.verification_questions[0].strip()
                if next_question.startswith("2. "):
                    next_question = next_question[3:]
                self.add_to_history("HR", next_question)
                self.verification_questions = self.verification_questions[1:]  # Remove the question we just asked
                return next_question
            else:
                # Move to role exploration before skills
                self.current_stage = "role_exploration"
                
                # Generate role exploration questions
                role_question = self.generate_role_exploration_question(self.candidate_info)
                self.add_to_history("HR", role_question)
                return role_question
                
        elif self.current_stage == "role_exploration":
            self.add_to_history("Candidate", candidate_response)
            
            # Check if we should ask a follow-up about their role
            if not hasattr(self, 'role_followup_asked') or not self.role_followup_asked:
                self.role_followup_asked = True
                
                # Generate a follow-up about their role or project
                resume_context = self.get_resume_context(self.candidate_info)
                role_followup = self.generate_role_followup(
                    self.candidate_info["name"],
                    resume_context,
                    candidate_response
                )
                self.add_to_history("HR", role_followup)
                return role_followup
            else:
                # Now move to skills questions
                self.current_stage = "skills"
                resume_context = self.get_resume_context(self.candidate_info)
                skills = ", ".join(self.candidate_info.get("skills", []))
                skill_questions = self.generate_skill_questions(
                    self.candidate_info["name"],
                    resume_context,
                    skills
                )
                
                # Ask first skill question
                first_skill_q = skill_questions[0].strip()
                if first_skill_q.startswith("1. "):
                    first_skill_q = first_skill_q[3:]
                self.add_to_history("HR", first_skill_q)
                self.skill_questions = skill_questions[1:]  # Store remaining questions
                self.current_skill = skills.split(',')[0].strip()  # Track current skill topic
                return first_skill_q
                
        elif self.current_stage == "skills":
            self.add_to_history("Candidate", candidate_response)
            
            # If we have a follow-up, ask it
            if hasattr(self, 'follow_up_mode') and self.follow_up_mode:
                # Switch back to remaining skill questions or wrap up
                self.follow_up_mode = False
                
                if hasattr(self, 'skill_questions') and self.skill_questions:
                    next_question = self.skill_questions[0].strip()
                    if next_question.startswith("2. "):
                        next_question = next_question[3:]
                    self.add_to_history("HR", next_question)
                    self.skill_questions = self.skill_questions[1:]  # Remove the question we just asked
                    
                    # Update current skill topic if we have multiple skills
                    skills = self.candidate_info.get("skills", [])
                    if len(skills) > 1:
                        self.current_skill = skills[1]  # Move to second skill
                    
                    return next_question
                else:
                    # Move to wrap up
                    self.current_stage = "wrap_up"
                    wrap_up = self.wrap_up_interview()
                    self.add_to_history("HR", wrap_up)
                    return wrap_up
            
            # Generate follow-up to their skill answer
            resume_context = self.get_resume_context(self.candidate_info)
            follow_up = self.generate_follow_up(
                self.candidate_info["name"],
                resume_context,
                candidate_response,
                self.current_skill if hasattr(self, 'current_skill') else "skills"
            )
            
            self.add_to_history("HR", follow_up)
            self.follow_up_mode = True  # Mark that next response should continue to skill questions
            return follow_up
            
        elif self.current_stage == "wrap_up":
            self.add_to_history("Candidate", candidate_response)
            
            # Handle candidate questions if any
            if "?" in candidate_response or any(q in candidate_response.lower() for q in ["what", "how", "when", "where", "who", "can you", "could you"]):
                hr_answer = self.handle_candidate_questions(candidate_response)
                self.add_to_history("HR", hr_answer)
                
                # After answering, move to goodbye
                self.current_stage = "goodbye"
                goodbye = self.final_goodbye()
                self.add_to_history("HR", goodbye)
                return f"{hr_answer}\n\n{goodbye}"
            else:
                # Move directly to goodbye
                self.current_stage = "goodbye"
                goodbye = self.final_goodbye()
                self.add_to_history("HR", goodbye)
                self.interview_ended = True
                return goodbye
                
        else:  # Default case or unknown stage
            self.add_to_history("Candidate", candidate_response)
            return "I appreciate your response. Is there anything else you'd like to discuss about the position?"

def main():
    # Load environment variables
    load_dotenv()

    # Load candidate database
    candidate_db = load_candidates()

    # --- Initialize Groq ---
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_llm = None
    if groq_api_key:
        groq_llm = ChatGroq(api_key=groq_api_key, model_name="llama3-8b-8192")
    else:
        print("GROQ_API_KEY environment variable not set. Groq will not be available.")
        return

    # Initialize the HR assistant
    hr_assistant = ConversationalHRAssistant(groq_llm=groq_llm, use_model="groq")

    # Simulate an incoming call
    phone_number = input("Enter phone number of the candidate: ").strip()

    # Identify the candidate
    candidate_info = hr_assistant.identify_candidate(phone_number, candidate_db)
    if not candidate_info:
        print("Candidate not found in system.")
        return

    # Store candidate info
    hr_assistant.candidate_info = candidate_info
    
    # Start the interview
    print("\n--- INTERVIEW STARTING ---\n")
    
    # Initial greeting
    greeting = hr_assistant.initial_greeting(candidate_info["name"])
    hr_assistant.add_to_history("HR", greeting)
    print(f"HR: {greeting}")
    
    # Run interview loop
    while not hr_assistant.interview_ended:
        candidate_response = input("Candidate: ").strip()
        if candidate_response.lower() in ["exit", "quit", "end"]:
            print("Interview simulation ended by user.")
            break
            
        hr_response = hr_assistant.process_response(candidate_response)
        print(f"HR: {hr_response}")
        
        # Add small delay to make it feel more natural
        time.sleep(0.5)
    
    print("\n--- INTERVIEW COMPLETED ---\n")
    
    # Print full interview transcript if desired
    print_transcript = input("Would you like to see the full interview transcript? (y/n): ").strip().lower()
    if print_transcript == 'y':
        print("\n=== INTERVIEW TRANSCRIPT ===\n")
        for entry in hr_assistant.interview_history:
            print(f"{entry['speaker']}: {entry['text']}")
            print()

if __name__ == "__main__":
    # Use the existing ingest functionality from ingest.py - no need to duplicate
    try:
        from ingest import ingest_candidates, JSON_PATH
        print(f"Using candidate data from {JSON_PATH}")
    except Exception as e:
        print(f"Warning: Could not import ingest module: {e}")
        JSON_PATH = "dummy_resumes.json"
    
    main()
    try:
        from ingest import ingest_candidates, JSON_PATH
        model, faiss_index = ingest_candidates(JSON_PATH)
    except Exception as e:
        print(f"Warning: Could not ingest candidates data: {e}")
    
    main()
    