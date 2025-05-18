from prompts import PROMPTS
from data_loader import get_candidate_by_phone
import re
import nltk
from nltk.corpus import words
import random

nltk.download('words', quiet=True)
ENGLISH_WORDS = set(words.words())

class ConversationalHRAssistant:
    def __init__(self, groq_llm=None, use_model="groq"):
        self.groq_llm = groq_llm
        self.use_model = use_model
        self.interview_history = []
        self.current_stage = "greeting"
        self.candidate_info = None
        self.interview_ended = False
        self.question_count = 0
        self.max_questions = 7
        self.awaiting_candidate_questions = False
        self.asked_question_topics = set()
        
        # Define key HR topics that must be covered in logical interview order
        self.hr_topics = [
            "current_role_question",        # Start with current role/background
            "technical_skills_question",    # Technical experience/projects
            "new_opportunity_question",     # What they're looking for
            "career_goals_question",        # Career aspirations
            "location_preferences_question", # Location preferences
            "notice_period_question",       # Notice period
            "salary_expectations_question"  # Salary last, after establishing rapport
        ]
        # Track which topics have been covered
        self.covered_topics = set()

    def _get_llm(self):
        if self.use_model == "groq" and self.groq_llm:
            return self.groq_llm
        else:
            raise ValueError("LLM not initialized or model not supported.")

    def add_to_history(self, speaker, text):
        self.interview_history.append({"speaker": speaker, "text": text})

    def identify_candidate(self, phone_number, candidate_db):
        return get_candidate_by_phone(phone_number, candidate_db)

    def initial_greeting(self, candidate_name):
        greeting = f"Hello, am I speaking with {candidate_name}?"
        self.add_to_history("HR", greeting)
        return greeting

    def check_availability(self):
        availability_msg = "I am Sarah from Jupitos Technologies. Is this a good time to talk about opportunities with us?"
        self.add_to_history("HR", availability_msg)
        return availability_msg

    def end_call_politely(self):
        end_msg = "I understand. Thank you for letting me know. I'll call back at a more convenient time. Have a great day!"
        self.add_to_history("HR", end_msg)
        return end_msg

    def _generate_question(self, prompt_key, **kwargs):
        llm = self._get_llm()
        prompt = PROMPTS.get(prompt_key, "")
        try:
            response = llm.invoke(prompt.format(**kwargs)).content.strip()
            # Filter out any AI disclosures
            if "as an AI" in response.lower() or "language model" in response.lower():
                return "Could you tell me more about that?"
            return response
        except Exception:
            return "Could you elaborate on that?"

    def generate_general_hr_question(self, last_response=None):
        # First, check if we need to cover specific topics still
        uncovered_topics = [topic for topic in self.hr_topics if topic not in self.covered_topics]
        
        if uncovered_topics:
            # Choose the next logical topic in sequence rather than randomly
            for topic in self.hr_topics:
                if topic not in self.covered_topics:
                    chosen_topic = topic
                    break
            
            self.covered_topics.add(chosen_topic)
            
            # Use topic-specific prompt
            question = self._generate_question(chosen_topic)
            
            # Basic validation to ensure we have a single question
            if question.count('?') > 1 or "Here are" in question or "Example:" in question or "*" in question:
                # Try one more time with the same prompt
                question = self._generate_question(chosen_topic)
                # If still problematic, use a simple question form
                if question.count('?') > 1 or "Here are" in question or "Example:" in question or "*" in question:
                    # Extract topic from prompt key
                    topic_name = chosen_topic.replace("_question", "").replace("_", " ")
                    question = f"Could you tell me about your {topic_name}?"
            
        elif last_response and self.question_count < 3:
            # After covering all required topics, use follow-up questions to maintain natural flow
            question = self._generate_question("followup_question", last_response=last_response)
            chosen_topic = "followup"
        else:
            # Fallback to general questions if needed
            question = self._generate_question("general_hr_question")
            chosen_topic = "general"
        
        # Add the topic to the list of asked questions
        self.asked_question_topics.add(chosen_topic)
        
        self.add_to_history("HR", question)
        return question

    def handle_candidate_questions(self, question):
        response = self._generate_question("handle_candidate_question", question=question)
        self.add_to_history("HR", response)
        return response

    def is_gibberish(self, response):
        tokens = re.findall(r'\b\w+\b', response.lower())
        if not tokens:
            return True
        english_word_count = sum(1 for token in tokens if token in ENGLISH_WORDS)
        return english_word_count / len(tokens) < 0.3

    def process_response(self, candidate_response):
        if self.interview_ended:
            return "The interview has already ended."

        self.add_to_history("Candidate", candidate_response)

        # Greeting stage
        if self.current_stage == "greeting":
            if any(word in candidate_response.lower() for word in ["yes", "yeah", "yep", "correct", "speaking"]):
                self.current_stage = "availability"
                return self.check_availability()
            else:
                self.interview_ended = True
                return "I apologize for the confusion. Have a nice day."

        # Availability check
        elif self.current_stage == "availability":
            if any(word in candidate_response.lower() for word in ["no", "not", "another", "busy", "later"]):
                self.interview_ended = True
                return self.end_call_politely()
            else:
                self.current_stage = "interview"
                intro_msg = "Great! I'd like to ask you a few questions about your background and experience to see if there's a good fit with our current openings."
                first_question = self.generate_general_hr_question()
                return f"{intro_msg}\n\n{first_question}"

        # Interview stage
        elif self.current_stage == "interview":
            if self.is_gibberish(candidate_response):
                msg = "I'm sorry, I didn't quite understand that. Could you please clarify?"
                self.add_to_history("HR", msg)
                return msg

            if "?" in candidate_response:
                hr_reply = self.handle_candidate_questions(candidate_response)
                
                # Check if we're ready to wrap up based on covered topics
                if len(self.covered_topics) >= len(self.hr_topics):
                    self.current_stage = "wrap_up"
                    wrap_msg = "Thank you for sharing all this valuable information! Do you have any other questions for me about the position or our company?"
                    self.add_to_history("HR", wrap_msg)
                    return f"{hr_reply}\n\n{wrap_msg}"
                
                follow_up = self.generate_general_hr_question()
                return f"{hr_reply}\n\n{follow_up}"

            self.question_count += 1
            
            # Check if we've covered all topics or reached max questions
            if len(self.covered_topics) >= len(self.hr_topics) or self.question_count >= self.max_questions:
                self.current_stage = "wrap_up"
                wrap_msg = "Thank you for sharing all this valuable information! Do you have any questions for me about the position or our company?"
                self.add_to_history("HR", wrap_msg)
                return wrap_msg
            
            # Add transitional phrases for more natural flow
            transitions = [
                "Thank you for sharing that. ",
                "I appreciate that information. ",
                "That's helpful to know. ",
                "Good to understand your background. ",
                "That's great context. "
            ]
            
            transition = random.choice(transitions) if self.question_count > 1 else ""
            next_q = self.generate_general_hr_question(last_response=candidate_response)
            return f"{transition}{next_q}"

        # Wrap-up stage
        elif self.current_stage == "wrap_up":
            if "?" in candidate_response:
                hr_reply = self.handle_candidate_questions(candidate_response)
                follow_up_msg = "Do you have any other questions for me?"
                self.add_to_history("HR", follow_up_msg)
                return f"{hr_reply}\n\n{follow_up_msg}"

            self.interview_ended = True
            return "Thank you so much for your time today! I've learned a lot about your experience and goals. We'll review your information and be in touch soon about next steps. Have a great day!"

        return "Thank you for your response. Could you tell me more?"