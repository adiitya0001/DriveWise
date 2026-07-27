import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError

from rag.vector_store import load_vector_store
from rag.retriever import retrieve
from rag.reranker import rerank


# ==================================================
# Gemini Setup
# ==================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. Add it to your .env file."
    )

client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-3.6-flash"


# ==================================================
# Resolve Conversational Follow-Up Question
# ==================================================

def resolve_question(query, history, brand, model):

    # No history = already standalone
    if not history:
        return query

    history_parts = []

    for message in history[-3:]:
        history_parts.append(
            f"""
User: {message["question"]}
DriveWise: {message["answer"]}
"""
        )

    history_text = "\n".join(history_parts)

    prompt = f"""
You rewrite follow-up questions for an automotive brochure
retrieval system.

Selected vehicle:
Brand: {brand}
Model: {model}

RECENT CONVERSATION:
{history_text}

CURRENT QUESTION:
{query}

Rewrite the current question so that it can be understood
without the previous conversation.

Rules:
- Do not answer the question.
- Return only the rewritten question.
- Preserve the user's meaning.
- Resolve words such as it, this, that, one, those and they.
- Use the selected vehicle when useful.
- Do not add facts.
- Keep it concise.

REWRITTEN QUESTION:
"""

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        if response.text:

            rewritten = response.text.strip()

            print(
                "Resolved follow-up question:",
                rewritten
            )

            return rewritten

    except Exception as error:

        # Rewriting failure must not break RAG
        print(
            "Question resolution error:",
            type(error).__name__,
            error
        )

    return query


# ==================================================
# Build Brochure Context
# ==================================================

def build_context(ranked_results):

    context_parts = []

    for number, result in enumerate(
        ranked_results,
        start=1
    ):

        metadata = result.get("metadata", {})

        brand = metadata.get("brand", "")
        model = metadata.get("model", "")
        page = metadata.get("page", "")
        section = metadata.get("section", "")
        text = result.get("text", "")

        context_parts.append(
            f"""
--- BROCHURE EXCERPT {number} ---

Vehicle: {brand} {model}
Section: {section}
Internal page: {page}

{text}
"""
        )

    return "\n\n".join(context_parts)


# ==================================================
# Generate Grounded Answer
# ==================================================

def generate_answer(query, ranked_results):

    if not ranked_results:

        return (
            "I could not find this information "
            "in the selected brochure."
        )

    context = build_context(ranked_results)

    # --------------------------------------------------
    # Debug exactly what is being used
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("GENERATOR INPUT")
    print("=" * 70)

    print("\nQuestion:")
    print(query)

    print("\nContext:")
    print(context)

    print("=" * 70)

    # --------------------------------------------------
    # Grounded RAG Prompt
    # --------------------------------------------------

    prompt = f"""
You are DriveWise, an automotive brochure question-answering
assistant.

Your job is to answer the USER QUESTION using the BROCHURE
EXCERPTS below.

The excerpts are retrieved from the brochure knowledge base.
They may contain unrelated information as well as information
relevant to the question.

IMPORTANT EVIDENCE RULE:

If ANY excerpt contains information that answers all or part
of the user's question, use that information.

Do not reject useful information merely because:
- another excerpt is unrelated,
- the relevant excerpt is not the first excerpt,
- the excerpt contains additional information,
- the information applies only to a specific variant,
  engine, transmission or trim.

For a broad question, combine relevant facts found across
multiple excerpts.


ANSWER RULES:

1. Use ONLY facts explicitly present in the brochure excerpts.

2. Never use outside automotive knowledge.

3. Never invent specifications, features, prices, variants,
   availability or equipment.

4. Give the useful answer first.

5. For broad questions such as:
   "What safety features does it have?"
   "What are its main features?"
   "Tell me about this car"

   summarize all clearly relevant information found in the
   excerpts.

6. Use Markdown bullet points or short headings when they
   make a broad answer easier to read.

7. Preserve availability conditions.

   If an excerpt says a feature belongs to Smart+,
   Accomplished, Accomplished+, a particular transmission,
   engine or other configuration, state that condition.

8. Do not imply that variant-specific equipment is standard
   on every variant.

9. Do not mention:
   - excerpt numbers
   - retrieved chunks
   - retrieval
   - internal page numbers
   - metadata

10. Do not include citations in the answer. The application
    displays sources separately.

11. The fallback sentence should be used ONLY when none of
    the excerpts contain information that answers the
    question.

12. If none of the excerpts contain relevant evidence,
    respond exactly:

I could not find this information in the selected brochure.


USER QUESTION:

{query}


BROCHURE EXCERPTS:

{context}


Now determine whether the excerpts contain evidence relevant
to the question.

If they do, answer using that evidence.

ANSWER:
"""

    # --------------------------------------------------
    # Gemini Call
    # --------------------------------------------------

    max_retries = 3

    for attempt in range(max_retries):

        try:

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )

            if not response.text:

                print(
                    "Gemini returned an empty response."
                )

                return (
                    "I could not generate an answer "
                    "from the selected brochure."
                )

            answer = response.text.strip()

            print("\n" + "=" * 70)
            print("GEMINI ANSWER")
            print("=" * 70)
            print(answer)
            print("=" * 70)

            return answer

        except ServerError as error:

            if attempt < max_retries - 1:

                wait_time = 3 * (attempt + 1)

                print(
                    f"Gemini service error. "
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

            else:

                print(
                    "Gemini service is currently unavailable."
                )

                raise

        except Exception as error:

            print(
                "Gemini generation error:",
                type(error).__name__,
                error
            )

            raise


# ==================================================
# Local RAG Test
# ==================================================

if __name__ == "__main__":

    # Load vector database
    index, chunks = load_vector_store()

    # --------------------------------------------------
    # TEST SIERRA
    # --------------------------------------------------

    brand = "Tata"
    car_model = "Sierra"

    query = "What safety features does the Sierra have?"

    print("\n" + "=" * 70)
    print("DRIVEWISE LOCAL TEST")
    print("=" * 70)

    print("Vehicle:", brand, car_model)
    print("Question:", query)

    # --------------------------------------------------
    # Retrieve more candidates
    # --------------------------------------------------

    results = retrieve(
        query=query,
        index=index,
        chunks=chunks,
        brand=brand,
        car_model=car_model,
        top_k=10,
        search_k=50
    )

    print("\nRetrieved results:", len(results))

    for i, result in enumerate(
        results,
        start=1
    ):

        print("\n" + "-" * 60)
        print("RESULT", i)
        print("Score:", result.get("score"))
        print("Metadata:", result.get("metadata"))
        print(result.get("text", "")[:800])

    # --------------------------------------------------
    # Keep top 5
    # --------------------------------------------------

    ranked_results = rerank(
        query=query,
        results=results,
        top_k=5
    )

    print(
        "\nChunks sent to generator:",
        len(ranked_results)
    )

    # --------------------------------------------------
    # Generate
    # --------------------------------------------------

    answer = generate_answer(
        query=query,
        ranked_results=ranked_results
    )

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(answer)
    print("=" * 70)