from ddgs import DDGS
import json
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("❌ GROQ_API_KEY not found in .env.")

# Global variable to store last generated questions
last_questions_result = None

# Pydantic Schema
class QuestionSet(BaseModel):
    questions: List[str]
    summary: str

# System instruction
system_instruction_text = (
    "You are an experienced technical interviewer. Based on the context provided, generate a list of potential interview questions "
    "that assess key skills, concepts, and problem-solving ability for the given role and company. "
    "Prioritize questions relevant to the candidate's professional background and skills as described in their resume. "
    "The tone should be professional and slightly challenging, but never unsafe or offensive."
)

# Fallback questions if Groq output is invalid
FALLBACK_QUESTIONS = {
    "questions": [
        "What are the key responsibilities of a software engineer?",
        "Explain the difference between an array and a linked list.",
        "How does a hash table work and what are its common use cases?",
        "Explain the time complexity of quicksort in best and worst case.",
        "Write a function to check if a string is a palindrome."
    ],
    "summary": "Interviews for this role typically cover fundamental computer science concepts, data structures, algorithms, and problem-solving skills."
}


def generate_interview_questions(
    job_role: str,
    company_name: str,
    job_description: Optional[str] = None,
    other_details: Optional[str] = None,
    resume_text: Optional[str] = None
) -> dict:
    global last_questions_result

    if not job_role:
        last_questions_result = {"error": "Missing job role."}
        return last_questions_result

    try:
        # Step 1: Search online using DDGS
        query = f"{job_role} interview questions"
        if company_name and company_name != "Not specified":
            query += f" at {company_name}"

        search_context = ""
        if not job_description or len(job_description) < 100 or not resume_text:
            with DDGS() as ddgs:
                search_results_raw = ddgs.text(query, max_results=3)
            search_context = "\n".join([item.get("body", "") for item in search_results_raw if item.get("body")])
            if not search_context.strip():
                search_context = "No significant online information found. Rely on provided details."
        else:
            search_context = "Detailed job description and/or resume provided. Focusing on internal context."

        # Step 2: Build prompt
        prompt = (
            f"Job Role: {job_role}\n"
            f"Company: {company_name}\n"
            f"Job Description: {job_description if job_description != 'No description provided' else ''}\n"
            f"Additional Info (Skills, Experience, Interview Type etc.): {other_details if other_details else ''}\n"
        )

        if resume_text and resume_text.strip():
            truncated_resume_text = resume_text[:min(len(resume_text), 3000)]
            prompt += f"Candidate's Resume Content:\n{truncated_resume_text}\n\n"
            prompt += "Please generate questions that are specifically tailored to the candidate's skills, projects, and experience.\n"

        prompt += (
            f"Background Info from web:\n{search_context}\n\n"
            f"Please generate exactly 5 interview questions of these types:\n"
            f"- 2 easy theory questions (fundamental concepts)\n"
            f"- 1 medium theory question (deeper understanding)\n"
            f"- 1 advanced technical design/algorithm question\n"
            f"- 1 practical coding exercise (moderate difficulty)\n\n"
            f"Also provide a concise 2-3 sentence summary on the typical focus of interviews for this role.\n\n"
            f"Return your answer strictly in this JSON format:\n"
            "```json\n"
            "{\n"
            '  "questions": ["question1", "question2", "question3", "question4", "question5"],\n'
            '  "summary": "summary text"\n'
            "}\n"
            "```"
        )

        # Step 3: Initialize Groq model
        model = ChatGroq(
            temperature=0.7,
            model_name="groq/compound",
            groq_api_key=api_key
        )

        messages = [
            SystemMessage(content=system_instruction_text),
            HumanMessage(content=prompt)
        ]

        # Step 4: Groq call
        response = model.invoke(messages)
        response_text = response.content.strip()

        # Remove code fences if any
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()

        # --- JSON repair helper ---
        def try_parse_json(raw: str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                if "}" in raw:
                    raw = raw[:raw.rfind("}") + 1]  # truncate to last complete brace
                    try:
                        return json.loads(raw)
                    except Exception:
                        return None
                return None

        response_data = try_parse_json(response_text)

        if not response_data:
            last_questions_result = FALLBACK_QUESTIONS
            return last_questions_result

        validated = QuestionSet(**response_data)
        last_questions_result = validated.model_dump()
        return last_questions_result

    except Exception as e:
        last_questions_result = {"error": f"Unexpected error: {str(e)}"}
        return last_questions_result
