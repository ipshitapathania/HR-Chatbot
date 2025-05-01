from prompts import PROMPTS
from resume_utils import extract_resume_text
from data_loader import get_candidate_by_phone
import re
import nltk
from nltk.corpus import words

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
        self.resume_context = ""
        self.question_count = 0
        self.max_questions = 7
        self.awaiting_candidate_questions = False

    def _get_llm(self):
        if self.use_model == "groq" and self.groq_llm:
            return self.groq_llm
        else:
            raise ValueError("LLM not initialized or model not supported.")

    def identify_candidate(self, phone_number, candidate_db):
        return get_candidate_by_phone(phone_number, candidate_db)

    def initial_greeting(self, candidate_name):
        return f"Hello, am I speaking with {candidate_name}?"

    def check_availability(self):
        return "Is this a good time to talk about your application with us?"

    def end_call_politely(self):
        return "I understand. Thank you for letting me know. I'll call back at a more convenient time. Have a great day!"

    def _generate_question(self, prompt_key, **kwargs):
        llm = self._get_llm()
        prompt = PROMPTS.get(prompt_key, "")
        formatted_prompt = prompt.format(**kwargs)
        try:
            return llm.invoke(formatted_prompt).content.strip()
        except Exception:
            return "Could you elaborate on that?"

    def generate_dynamic_question(self, last_response=None):
        if not self.resume_context:
            # Extract resume text and set question limit based on its length
            self.resume_context = extract_resume_text(self.candidate_info)
            self._set_dynamic_question_limit()

        if not last_response:
            return self._generate_question("initial_question", resume=self.resume_context)
        else:
            return self._generate_question("followup_question", resume=self.resume_context, last_response=last_response)

    def handle_candidate_questions(self, question):
        return self._generate_question("handle_candidate_question", question=question)

    def add_to_history(self, speaker, text):
        self.interview_history.append({"speaker": speaker, "text": text})

    def is_gibberish(self, response):
        # Check if the response contains mostly non-English words
        tokens = re.findall(r'\b\w+\b', response.lower())
        if not tokens:
            return True
        english_word_count = sum(1 for token in tokens if token in ENGLISH_WORDS)
        return english_word_count / len(tokens) < 0.3

    def _set_dynamic_question_limit(self):
        # Adjust the maximum number of questions based on resume word count
        word_count = len(self.resume_context.split())
        if word_count < 150:
            self.max_questions = 5
        elif word_count < 300:
            self.max_questions = 7
        else:
            self.max_questions = 9

    def process_response(self, candidate_response):
        if self.interview_ended:
            return "The interview has already ended."

        self.add_to_history("Candidate", candidate_response)

        # Greeting stage
        if self.current_stage == "greeting":
            if any(word in candidate_response.lower() for word in ["yes", "yeah", "yep", "correct", "speaking"]) or self.candidate_info["name"].lower() in candidate_response.lower():
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
                return self.generate_dynamic_question()

        # Interview stage
        elif self.current_stage == "interview":
            if self.is_gibberish(candidate_response):
                # Handle gibberish responses
                msg = "I’m sorry, I didn’t quite understand that. Could you please clarify or provide more details?"
                self.add_to_history("HR", msg)
                return msg

            if "?" in candidate_response or any(q in candidate_response.lower() for q in ["what", "how", "when", "where", "who", "can you", "could you"]):
                # Handle candidate's questions
                hr_reply = self.handle_candidate_questions(candidate_response)
                follow_up = self.generate_dynamic_question()
                self.add_to_history("HR", hr_reply)
                self.add_to_history("HR", follow_up)
                return f"{hr_reply}\n\n{follow_up}"

            self.question_count += 1
            if self.question_count >= self.max_questions:
                # Transition to wrap-up stage
                self.current_stage = "wrap_up"
                self.awaiting_candidate_questions = True
                wrap_msg = "Thanks for sharing all that! Before we finish, do you have any questions for me about the role or our company?"
                self.add_to_history("HR", wrap_msg)
                return wrap_msg

            next_q = self.generate_dynamic_question(last_response=candidate_response)
            self.add_to_history("HR", next_q)
            return next_q

        # Wrap-up stage
        elif self.current_stage == "wrap_up":
            self.add_to_history("Candidate", candidate_response)

            while "?" in candidate_response or any(q in candidate_response.lower() for q in ["what", "how", "when", "where", "who", "can you", "could you"]):
                # Handle candidate questions in a loop
                hr_reply = self.handle_candidate_questions(candidate_response)
                self.add_to_history("HR", hr_reply)
                follow_up_msg = "Do you have any other questions for me?"
                self.add_to_history("HR", follow_up_msg)
                return f"{hr_reply}\n\n{follow_up_msg}"

            self.interview_ended = True
            return "Thanks again for your time! We'll be in touch soon. Have a great day!"

        return "Thank you for your response. Could you tell me more?"
