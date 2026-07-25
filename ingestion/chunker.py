from ingestion.pdf_loader import load_pdf

def clean_text(text):
    lines = text.splitlines()

    cleaned_lines = []

    for line in lines:
        line = line.strip()

        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

def detect_section(text):
    text_lower = text.lower()

    section_keywords = {
        "safety": [
            "safety", "airbag", "adas",
            "collision", "blind-spot",
            "lane keeping"
        ],

        "engine_performance": [
            "engine", "petrol", "diesel",
            "turbo", "transmission",
            "torque", "power"
        ],

        "infotainment_connectivity": [
            "infotainment", "bluelink",
            "alexa", "bose", "connectivity",
            "android auto", "apple carplay"
        ],

        "interior_comfort": [
            "interior", "seat",
            "ventilated", "sunroof",
            "climate control"
        ],

        "dimensions": [
            "dimensions", "wheelbase",
            "overall length", "overall width",
            "overall height"
        ]
    }

    scores = {}

    for section, keywords in section_keywords.items():
        scores[section] = sum(
            keyword in text_lower
            for keyword in keywords
        )

    best_section = max(scores, key=scores.get)

    if scores[best_section] == 0:
        return "general"

    return best_section

def split_text(text, chunk_size=800, overlap=100):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start = end - overlap

    return chunks

def create_chunks(pages):

    chunks = []

    for page in pages:

        cleaned_text = clean_text(page["text"])

        if not cleaned_text:
            continue

        text_chunks = split_text(cleaned_text)

        for text_chunk in text_chunks:

            section = detect_section(text_chunk)

            chunk = {
                "text": text_chunk,

                "metadata": {
                    "brand": page["brand"],
                    "model": page["model"],
                    "page": page["page"],
                    "section": section
                }
            }

            chunks.append(chunk)

    return chunks

if __name__ == "__main__":

    pages = load_pdf(
        "data/brochures/creta.pdf",
        brand="Hyundai",
        model="Creta"
    )

    chunks = create_chunks(pages)

    print("Total chunks:", len(chunks))

    for chunk in chunks:

        print("\n" + "=" * 50)

        print("Metadata:")
        print(chunk["metadata"])

        print("\nText:")
        print(chunk["text"][:300])