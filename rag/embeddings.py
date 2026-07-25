import os
import time

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError, ServerError


# ==================================================
# Environment
# ==================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing. "
        "Add it to your .env file."
    )


# ==================================================
# Gemini Client
# ==================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)

EMBEDDING_MODEL = "gemini-embedding-001"


# ==================================================
# Normalize Embedding
# ==================================================

def normalize_embedding(values):

    vector = np.asarray(
        values,
        dtype="float32"
    )

    norm = np.linalg.norm(vector)

    if norm > 0:
        vector = vector / norm

    return vector


# ==================================================
# Create ONE Embedding
# Used for live user questions
# ==================================================

def create_embedding(
    text,
    task_type="RETRIEVAL_QUERY"
):

    if not text or not text.strip():
        raise ValueError(
            "Cannot create embedding for empty text."
        )

    while True:

        try:

            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config={
                    "task_type": task_type
                }
            )

            if not response.embeddings:
                raise ValueError(
                    "Gemini returned no embedding."
                )

            return normalize_embedding(
                response.embeddings[0].values
            )

        except ClientError as error:

            # Convert error to text because different
            # google-genai versions expose status differently.
            error_text = str(error)

            if (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
            ):

                print(
                    "\nGemini embedding rate limit reached."
                )

                print(
                    "Waiting 65 seconds..."
                )

                time.sleep(65)

                continue

            raise

        except ServerError:

            print(
                "\nGemini embedding service temporarily busy."
            )

            print(
                "Waiting 15 seconds..."
            )

            time.sleep(15)


# ==================================================
# Create Multiple Embeddings
# Used ONLY when building FAISS
# ==================================================

def create_embeddings(
    texts,
    batch_size=20
):

    if not texts:

        raise ValueError(
            "No texts provided for embedding."
        )

    all_embeddings = []

    total = len(texts)

    print(
        f"Creating embeddings for {total} texts..."
    )

    # Tracks how many individual texts have been
    # submitted during the current quota window.
    texts_this_window = 0

    for start in range(
        0,
        total,
        batch_size
    ):

        end = min(
            start + batch_size,
            total
        )

        batch = texts[start:end]

        # ==================================================
        # Proactive rate-limit protection
        # ==================================================

        if (
            texts_this_window > 0
            and
            texts_this_window + len(batch) > 80
        ):

            print()
            print("=" * 50)
            print(
                "Approaching Gemini free-tier rate limit."
            )
            print(
                "Waiting 65 seconds before continuing..."
            )
            print("=" * 50)
            print()

            time.sleep(65)

            texts_this_window = 0

        # ==================================================
        # Send current batch
        # ==================================================

        while True:

            try:

                print(
                    f"Creating embeddings "
                    f"{start + 1}-{end}/{total}"
                )

                response = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch,
                    config={
                        "task_type":
                            "RETRIEVAL_DOCUMENT"
                    }
                )

                if not response.embeddings:

                    raise ValueError(
                        "Gemini returned no embeddings."
                    )

                if (
                    len(response.embeddings)
                    != len(batch)
                ):

                    raise ValueError(
                        "Gemini embedding count "
                        "does not match batch size."
                    )

                for item in response.embeddings:

                    vector = normalize_embedding(
                        item.values
                    )

                    all_embeddings.append(
                        vector
                    )

                texts_this_window += len(batch)

                break

            except ClientError as error:

                error_text = str(error)

                if (
                    "429" in error_text
                    or
                    "RESOURCE_EXHAUSTED"
                    in error_text
                ):

                    print()
                    print("=" * 50)
                    print(
                        "Gemini rate limit reached."
                    )
                    print(
                        "Waiting 65 seconds "
                        "for quota reset..."
                    )
                    print("=" * 50)
                    print()

                    time.sleep(65)

                    texts_this_window = 0

                    # Retry SAME batch
                    continue

                raise

            except ServerError:

                print()
                print(
                    "Gemini service temporarily busy."
                )
                print(
                    "Waiting 15 seconds and "
                    "retrying same batch..."
                )

                time.sleep(15)

                continue

    # ==================================================
    # Convert to NumPy
    # ==================================================

    embeddings = np.asarray(
        all_embeddings,
        dtype="float32"
    )

    if len(embeddings) != total:

        raise ValueError(
            f"Expected {total} embeddings "
            f"but received {len(embeddings)}."
        )

    print()
    print("=" * 60)

    print(
        f"Successfully created "
        f"{len(embeddings)} embeddings."
    )

    print(
        "Embedding dimension:",
        embeddings.shape[1]
    )

    print("=" * 60)

    return embeddings


# ==================================================
# Local Test
# ==================================================

if __name__ == "__main__":

    test_texts = [
        "Hyundai Creta comes with six airbags.",
        "The vehicle includes advanced safety features.",
        "The engine provides strong performance."
    ]

    embeddings = create_embeddings(
        test_texts
    )

    print(
        "\nEmbedding matrix:",
        embeddings.shape
    )

    query = create_embedding(
        "How many airbags does the Creta have?"
    )

    print(
        "Query embedding:",
        query.shape
    )