import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError

from rag.vector_store import load_vector_store
from rag.retriever import retrieve
from rag.reranker import rerank


# --------------------------------------------------
# Gemini Setup
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. Add it to your .env file."
    )

client = genai.Client(api_key=api_key)


# ==================================================
# Resolve Conversational Follow-Up Question
# ==================================================

def resolve_question(query, history, brand, model):

    # --------------------------------------------------
    # First question: no conversation history needed
    # --------------------------------------------------

    if not history:
        return query

    # --------------------------------------------------
    # Build recent conversation context
    # --------------------------------------------------

    history_parts = []

    for message in history[-3:]:

        history_parts.append(
            f"""
User: {message["question"]}
DriveWise: {message["answer"]}
"""
        )

    history_text = "\n".join(history_parts)


    # --------------------------------------------------
    # Question Rewriting Prompt
    # --------------------------------------------------

    prompt = f"""
You are helping an automotive RAG retrieval system understand
a user's follow-up question.

The user is currently asking questions about:

Brand: {brand}
Model: {model}


RECENT CONVERSATION:

{history_text}


CURRENT USER QUESTION:

{query}


TASK:

Rewrite the CURRENT USER QUESTION into one clear,
standalone question that can be understood without seeing
the previous conversation.


RULES:

1. Do NOT answer the question.

2. Return ONLY the rewritten question.

3. Do NOT add automotive facts.

4. Do NOT use outside knowledge.

5. Preserve the user's original intent.

6. Resolve conversational references such as:

   - it
   - this
   - that
   - one
   - ones
   - those
   - they
   - them
   - which one
   - what about it
   - what about the automatic
   - what about the diesel

7. Use the recent conversation only to understand
   what the user is referring to.

8. Include the selected brand and model when useful.

9. Do not change a factual assumption into a new fact.

10. Keep the rewritten question concise.


EXAMPLE:

Recent conversation:

User:
What engine options does Scorpio have?

DriveWise:
The Scorpio offers petrol and diesel engine options.

Current question:

Which one has more torque?

Rewritten question:

Which of the engine options available in the Mahindra Scorpio has more torque?


REWRITTEN QUESTION:
"""


    # --------------------------------------------------
    # Gemini Question Resolution
    # --------------------------------------------------

    try:

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        if response.text:

            rewritten_question = response.text.strip()

            print(
                "Resolved follow-up question:",
                rewritten_question
            )

            return rewritten_question


    except Exception as error:

        # Follow-up rewriting should NOT crash
        # the entire RAG pipeline.

        print(
            "Question resolution error:",
            type(error).__name__,
            error
        )


    # --------------------------------------------------
    # Fallback
    # --------------------------------------------------

    # If Gemini cannot rewrite the question,
    # continue using the original question.

    return query


# ==================================================
# Generate Answer
# ==================================================

def generate_answer(query, ranked_results):

    if not ranked_results:

        return (
            "I could not find this information "
            "in the selected brochure."
        )


    # --------------------------------------------------
    # Build context from retrieved brochure chunks
    # --------------------------------------------------

    context_parts = []

    for result in ranked_results:

        metadata = result["metadata"]

        context_parts.append(
            f"""
Brand: {metadata['brand']}
Model: {metadata['model']}
Page: {metadata['page']}
Section: {metadata['section']}

Content:
{result['text']}
"""
        )

    context = "\n\n".join(context_parts)


    # --------------------------------------------------
    # Grounded RAG Prompt
    # --------------------------------------------------

    prompt = f"""
You are DriveWise, an AI automotive brochure assistant.

Answer the user's question using ONLY the brochure context
provided below.


ANSWER RULES:

1. Do not use outside knowledge.

2. Do not invent vehicle specifications, features,
   variants, prices, availability, or other information.

3. Give the direct answer first.

4. Do NOT include brochure page numbers inside the answer.

5. Do NOT include source citations inside the answer.

6. Do NOT mention internal metadata such as:
   - page numbers
   - section names
   - retrieval results
   - chunks

7. Do NOT say phrases such as:
   - "according to Page 3"
   - "based on Page 4"
   - "according to the provided context"
   - "based on the provided brochure context"

8. For simple factual questions, give a short and
   direct answer.

9. For broad questions such as:
   - "Tell me about this car"
   - "What are its main features?"
   - "What safety features does it have?"

   organize the answer using Markdown headings
   and bullet points when useful.

10. Preserve important conditions from the brochure.

    For example, if a feature is available only on a
    particular variant, transmission, fuel type, or
    drivetrain, clearly mention that condition.

11. Do not claim that a feature is standard across the
    entire model range unless the brochure context
    explicitly says it is standard.

12. If the brochure context does not contain enough
    information to answer the question, respond exactly:

    "I could not find this information in the selected brochure."


USER QUESTION:

{query}


BROCHURE CONTEXT:

{context}


ANSWER:
"""


    # --------------------------------------------------
    # Gemini Call With Retry Handling
    # --------------------------------------------------

    max_retries = 3

    for attempt in range(max_retries):

        try:

            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )

            if not response.text:

                return (
                    "I could not generate an answer "
                    "from the selected brochure."
                )

            return response.text.strip()


        except ServerError as error:

            # Retry if Gemini is temporarily unavailable

            if attempt < max_retries - 1:

                wait_time = 3 * (attempt + 1)

                print(
                    f"Gemini is busy. "
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

            else:

                print(
                    "\nGemini service is currently unavailable."
                )

                print(
                    "Please try again later."
                )

                raise error


# ==================================================
# Test Complete RAG Pipeline
# ==================================================

if __name__ == "__main__":

    # --------------------------------------------------
    # 1. Load FAISS index + stored chunks
    # --------------------------------------------------

    index, chunks = load_vector_store()


    # --------------------------------------------------
    # 2. Selected vehicle
    # --------------------------------------------------

    brand = "Hyundai"
    car_model = "Creta"


    # --------------------------------------------------
    # 3. Test question
    # --------------------------------------------------

    query = "How many airbags does Creta have?"


    # --------------------------------------------------
    # 4. Retrieve relevant brochure chunks
    # --------------------------------------------------

    results = retrieve(
        query=query,
        index=index,
        chunks=chunks,
        brand=brand,
        car_model=car_model,
        top_k=5
    )


    # --------------------------------------------------
    # 5. Re-rank results
    # --------------------------------------------------

    ranked_results = rerank(
        query=query,
        results=results,
        top_k=3
    )


    # --------------------------------------------------
    # 6. Generate answer
    # --------------------------------------------------

    answer = generate_answer(
        query=query,
        ranked_results=ranked_results
    )


    # --------------------------------------------------
    # 7. Display result
    # --------------------------------------------------

    print("\n" + "=" * 60)

    print("Question:")
    print(query)

    print("\nDriveWise:")
    print(answer)

    print("=" * 60)