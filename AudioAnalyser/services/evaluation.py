import os
import json
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()

api_key = os.getenv('GROQ_API_KEY')
if not api_key:
    raise ValueError("❌ GROQ_API_KEY not found in .env.")

# Define global variable to store last result
last_analysis_result = None  # ✅ Accessible from other files

# Define Pydantic Schemas for Technical Evaluation
class TechnicalEvaluation(BaseModel):
    category: str
    score: float
    feedback: str
    improvement_tip: str

class TechnicalFeedback(BaseModel):
    evaluation: List[TechnicalEvaluation]
    overall_summary: str
    actionable_suggestions: List[str]

# Define system instruction
system_instruction_text = (
    "You are a highly skilled technical interviewer. Your job is to evaluate the following technical answer "
    "in terms of correctness, clarity, depth of explanation, and conciseness. "
    "Provide feedback in a positive, constructive tone along with suggestions for improvement. "
    "Your response must strictly follow the provided JSON schema below:\n\n"
    "Schema:\n"
    "{\n"
    "  'evaluation': [\n"
    "    {'category': str, 'score': float, 'feedback': str, 'improvement_tip': str}\n"
    "  ],\n"
    "  'overall_summary': str,\n"
    "  'actionable_suggestions': [str]\n"
    "}"
)

# --- Main Function ---
def analyze_technical_answer(transcript_text: str) -> dict:
    """
    Evaluate a technical answer using Groq (ChatGroq model)
    Returns structured JSON adhering to TechnicalFeedback schema.
    """
    global last_analysis_result  # ✅ Update global result

    try:
        # Initialize ChatGroq model
        model = ChatGroq(
            temperature=0.4,
            model_name="groq/compound",
            groq_api_key=api_key
        )

        # Compose the message
        human_prompt = (
            f"Please evaluate the following technical answer. Analyze it for correctness, clarity, depth, and conciseness. "
            f"Provide the results in **strict JSON format** as per the schema.\n\n"
            f"Technical Answer:\n{transcript_text}"
        )

        # Send prompt
        messages = [
            SystemMessage(content=system_instruction_text),
            HumanMessage(content=human_prompt)
        ]

        response = model.invoke(messages)
        response_text = response.content.strip()

        # Clean up code fences if LLM includes them
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()

        # Try parsing JSON
        try:
            response_data = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"❌ JSON Parsing Error: {e}")
            last_analysis_result = {"error": "Invalid JSON from Groq."}
            return last_analysis_result

        # Validate with Pydantic schema
        feedback = TechnicalFeedback(**response_data)
        last_analysis_result = feedback.model_dump()  # ✅ Save result globally
        return last_analysis_result

    except Exception as e:
        print(f"❌ General Error: {e}")
        last_analysis_result = {"error": str(e)}
        return last_analysis_result
