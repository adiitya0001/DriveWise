from typing import List
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Gemini errors
from google.genai.errors import ClientError, ServerError

from rag.vector_store import load_vector_store
from rag.retriever import retrieve
from rag.reranker import rerank
from rag.generator import generate_answer, resolve_question


# ==================================================
# FastAPI App
# ==================================================

app = FastAPI(
    title="DriveWise API",
    description="Metadata-aware conversational automotive RAG API",
    version="1.2.0"
)


# ==================================================
# CORS
# ==================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# Load Vector Store
# ==================================================

print("Loading DriveWise vector store...")

index, chunks = load_vector_store()

print("Vector store loaded successfully.")
print(f"FAISS vectors: {index.ntotal}")
print(f"Stored chunks: {len(chunks)}")


# ==================================================
# Request Models
# ==================================================

class ChatMessage(BaseModel):
    question: str
    answer: str


class QuestionRequest(BaseModel):
    brand: str
    model: str
    question: str

    # Safer than using [] directly as default
    history: List[ChatMessage] = Field(default_factory=list)


# ==================================================
# Detect Whether Question Needs Conversation Context
# ==================================================

def needs_question_resolution(question: str, history: list) -> bool:
    """
    Returns True only when the current question appears to depend
    on previous conversation context.

    This avoids unnecessary Gemini calls for standalone questions.
    """

    # No history = nothing to resolve
    if not history:
        return False

    text = question.lower().strip()


    # --------------------------------------------------
    # Common follow-up phrases
    # --------------------------------------------------

    follow_up_phrases = [
        "what about",
        "how about",
        "which one",
        "which ones",
        "and the",
        "and what",
        "what else",
        "anything else",
        "tell me more",
        "more about",
        "compared to that",
        "compared to it",
        "is that",
        "does that",
        "does it",
        "can it",
        "is it",
        "are they",
        "do they",
        "can they",
    ]

    if any(phrase in text for phrase in follow_up_phrases):
        return True


    # --------------------------------------------------
    # Pronouns/references that may depend on history
    # --------------------------------------------------

    reference_words = [
        "it",
        "its",
        "this",
        "that",
        "those",
        "these",
        "they",
        "them",
        "one",
        "ones",
        "former",
        "latter",
    ]

    words = set(
        re.findall(r"\b[a-zA-Z]+\b", text)
    )

    if words.intersection(reference_words):
        return True


    # --------------------------------------------------
    # Very short questions are often follow-ups
    # --------------------------------------------------

    word_count = len(text.split())

    if word_count <= 4:
        return True


    # Otherwise treat it as a standalone question
    return False


# ==================================================
# Home Route
# ==================================================

@app.get("/")
def home():

    return {
        "message": "DriveWise API is running",
        "version": "1.2.0",
        "vectors": index.ntotal,
        "chunks": len(chunks)
    }


# ==================================================
# Available Vehicles
# ==================================================

@app.get("/vehicles")
def get_vehicles():

    vehicles = {}

    for chunk in chunks:

        metadata = chunk.get("metadata", {})

        brand = metadata.get("brand")
        model = metadata.get("model")

        if not brand or not model:
            continue

        if brand not in vehicles:
            vehicles[brand] = set()

        vehicles[brand].add(model)


    formatted_vehicles = {
        brand: sorted(list(models))
        for brand, models in sorted(vehicles.items())
    }


    return {
        "vehicles": formatted_vehicles
    }


# ==================================================
# Ask DriveWise
# ==================================================

@app.post("/ask")
def ask_question(request: QuestionRequest):

    try:

        # --------------------------------------------------
        # Validate Request
        # --------------------------------------------------

        question = request.question.strip()
        brand = request.brand.strip()
        model = request.model.strip()


        if not question:

            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty."
            )


        if not brand:

            raise HTTPException(
                status_code=400,
                detail="Brand is required."
            )


        if not model:

            raise HTTPException(
                status_code=400,
                detail="Model is required."
            )


        # --------------------------------------------------
        # Convert Conversation History
        # --------------------------------------------------

        history = [
            {
                "question": item.question,
                "answer": item.answer
            }
            for item in request.history
        ]


        # Only keep last 3 conversations
        history = history[-3:]


        # --------------------------------------------------
        # Decide Whether Gemini Question Resolution Is Needed
        # --------------------------------------------------

        resolution_needed = needs_question_resolution(
            question=question,
            history=history
        )


        # --------------------------------------------------
        # Resolve Follow-Up ONLY When Necessary
        # --------------------------------------------------

        if resolution_needed:

            retrieval_query = resolve_question(
                query=question,
                history=history,
                brand=brand,
                model=model
            )

            resolution_status = "YES"

        else:

            # Question already makes sense independently.
            # Do NOT waste a Gemini request.

            retrieval_query = question

            resolution_status = "NO"


        # --------------------------------------------------
        # Debug Information
        # --------------------------------------------------

        print("\n" + "=" * 60)
        print("DriveWise Request")
        print("=" * 60)

        print("Brand:", brand)
        print("Model:", model)

        print(
            "Original question:",
            question
        )

        print(
            "Question resolution needed:",
            resolution_status
        )

        print(
            "Retrieval question:",
            retrieval_query
        )

        print(
            "History messages:",
            len(history)
        )


        # --------------------------------------------------
        # 1. Retrieve Brochure Chunks
        # --------------------------------------------------

        results = retrieve(
            query=retrieval_query,
            index=index,
            chunks=chunks,
            brand=brand,
            car_model=model,
            top_k=5
        )


        if not results:

            raise HTTPException(
                status_code=404,
                detail=(
                    f"No relevant brochure information "
                    f"found for {brand} {model}."
                )
            )


        # --------------------------------------------------
        # 2. Re-rank Results
        # --------------------------------------------------

        ranked_results = rerank(
            query=retrieval_query,
            results=results,
            top_k=3
        )


        if not ranked_results:

            raise HTTPException(
                status_code=404,
                detail="No relevant information found."
            )


        # --------------------------------------------------
        # 3. Generate Grounded Answer
        # --------------------------------------------------

        answer = generate_answer(
            query=question,
            ranked_results=ranked_results
        )


        # --------------------------------------------------
        # 4. Build Sources
        # --------------------------------------------------

        sources = []


        for result in ranked_results:

            metadata = result.get(
                "metadata",
                {}
            )


            sources.append({
                "brand": metadata.get("brand"),
                "model": metadata.get("model"),
                "page": metadata.get("page"),
                "section": metadata.get(
                    "section",
                    "general"
                )
            })


        # --------------------------------------------------
        # 5. Return Response
        # --------------------------------------------------

        return {
            "brand": brand,
            "model": model,
            "question": question,
            "answer": answer,
            "sources": sources
        }


    # ==================================================
    # FastAPI Errors
    # ==================================================

    except HTTPException:
        raise


    # ==================================================
    # Gemini Client Errors
    # ==================================================

    except ClientError as error:

        print(
            "Gemini Client Error:",
            error
        )


        status_code = getattr(
            error,
            "status_code",
            None
        )


        # Rate limit / quota
        if status_code == 429:

            raise HTTPException(
                status_code=429,
                detail=(
                    "DriveWise is receiving too many "
                    "requests right now. "
                    "Please wait a moment and try again."
                )
            )


        raise HTTPException(
            status_code=502,
            detail=(
                "DriveWise could not generate "
                "an answer right now. "
                "Please try again."
            )
        )


    # ==================================================
    # Gemini Server Errors
    # ==================================================

    except ServerError as error:

        print(
            "Gemini Server Error:",
            error
        )


        status_code = getattr(
            error,
            "status_code",
            None
        )


        if status_code == 503:

            raise HTTPException(
                status_code=503,
                detail=(
                    "The AI service is temporarily busy. "
                    "Please try again shortly."
                )
            )


        raise HTTPException(
            status_code=502,
            detail=(
                "The AI service is temporarily unavailable. "
                "Please try again."
            )
        )


    # ==================================================
    # Unexpected Errors
    # ==================================================

    except Exception as error:

        print(
            "DriveWise API Error:",
            type(error).__name__,
            error
        )


        raise HTTPException(
            status_code=500,
            detail=(
                "Something went wrong while "
                "generating the answer."
            )
        )