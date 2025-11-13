import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from ReportGeneration.Retriever.retriever import ContextRetriever
from ReportGeneration.Query.query_generation import QueryGenerator
from shared_state import stored_job_info, stored_audio_transcripts, stored_video_analysis, questions_generated

# Load environment variables
load_dotenv()

# --- Initialize ChatGroq ---
llm = ChatGroq(
    model="groq/compound",  # You can change to other Groq models if needed
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.4,
    max_tokens=2048
)

# --- Main System Instruction ---
system_instruction_text = """
You are an expert interview analyst AI.
You must create a detailed and professional JSON report analyzing a candidate's mock interview.

You will use:
- Retrieved knowledge base context (technical and behavioral interview insights)
- Job information
- Questions asked
- Audio transcript analysis
- Video emotion analysis

Output Format (strict JSON):
{
  "summary": "...",
  "technical_feedback": "...",
  "behavioral_feedback": "...",
  "communication_feedback": "...",
  "suggestions": ["...", "..."],
  "score" : "overall score according to questions answered is..."
}

Rules:
- Keep tone formal and concise.
- Each section must be complete sentences.
- Return JSON only, no extra text.
"""

# --- Function to Generate Interview Report ---
def generate_interview_report():
    try:
        # Step 1: Generate an expanded query
        query_gen = QueryGenerator()
        expanded_query = query_gen.generate("Generate the best technical and behavioral interview improvement insights")

        # Step 2: Retrieve relevant context
        retriever = ContextRetriever()
        context_chunks = retriever.retrieve(expanded_query)

        formatted_chunks = "\n\n".join(
            [f"Source: {chunk['source']} | Page: {chunk['page']}\n{chunk['text']}" for chunk in context_chunks]
        )

        # Step 3: Construct final prompt
        prompt = f"""
{system_instruction_text}

=== Retrieved Context ===
{formatted_chunks}

=== Job Info ===
{stored_job_info}

=== Questions Asked ===
{questions_generated}

=== Audio Transcript ===
{stored_audio_transcripts}

=== Video Emotion Analysis ===
{stored_video_analysis}

Now generate the full JSON report strictly following the schema above.
"""

        # Step 4: Generate response using ChatGroq
        response = llm.invoke(prompt)
        text_output = response.content.strip()

        # Step 5: Try to parse JSON
        try:
            return json.loads(text_output)
        except json.JSONDecodeError:
            # Handle cases where model adds extra text
            json_str = text_output[text_output.find("{"): text_output.rfind("}") + 1]
            return json.loads(json_str)

    except Exception as e:
        print(f"❌ Error generating interview report: {e}")
        return None
