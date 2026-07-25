from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def create_embedding(text):
    embedding = model.encode(text)

    return embedding

if __name__ == "__main__":

    sentences = [
        "Hyundai CRETA comes equipped with six airbags as standard.",
        "How many airbags does Creta have?",
        "What is the engine power of Creta?",
        "Does Creta have a Bose sound system?"
    ]

    embeddings = model.encode(sentences)

    similarities = model.similarity(
        embeddings[0],
        embeddings[1:]
    )

    print("Original:")
    print(sentences[0])

    print("\nSimilarity scores:")

    for sentence, score in zip(sentences[1:], similarities[0]):
        print(f"{score.item():.4f} -> {sentence}")