import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# Fetch Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("❌ GROQ_API_KEY not found in .env file.")


class QueryGenerator:
    def __init__(self):
        # Initialize ChatGroq model (choose the model you prefer)
        self.model = ChatGroq(
            temperature=0.7,
            model_name="groq/compound",  # or "mixtral-8x7b" if you prefer
            api_key=groq_api_key
        )

    def generate(self, short_prompt: str) -> str:
        """
        Expands a short user prompt into a detailed structured query
        for retrieving interview-related knowledge.

        Args:
            short_prompt (str): A short or vague user input.

        Returns:
            str: Expanded, well-structured query text.
        """
        system_message = SystemMessage(
            content=(
                "You are assisting an AI-powered interview analysis system. "
                "Your goal is to expand short prompts into detailed structured queries "
                "to retrieve diverse and meaningful information about interview preparation, "
                "behavioral expectations, technical pitfalls, communication, and mindset."
            )
        )

        user_prompt = f"""
The user has given this short or vague prompt:
"{short_prompt}"

Expand it into a **detailed, structured query** that retrieves information from an interview knowledge base containing:
- Behavioral and technical interview insights
- Recruiter feedback and common mistakes
- Communication and body language improvement tips
- Preparation and strategy frameworks

Requirements:
- Write 5–6 sentences in total
- Include **keywords** such as "behavioral expectations", "common technical pitfalls", "effective communication", "preparation strategies"
- Use **bullet points (•)** for structure
- Make it natural and information-seeking
- Return only the expanded query (no extra commentary)
"""

        try:
            response = self.model.invoke([system_message, HumanMessage(content=user_prompt)])
            return response.content.strip()
        except Exception as e:
            print(f"❌ Error generating query with ChatGroq: {e}")
            return ""
