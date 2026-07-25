from pathlib import Path

from ingestion.pdf_loader import load_pdf
from ingestion.chunker import create_chunks

from rag.vector_store import (
    build_vector_store,
    save_vector_store
)


BROCHURE_DIR = Path("data/brochures")


def main():

    print("Scanning brochures...\n")

    all_chunks = []

    # Find every PDF inside data/brochures and its subfolders
    pdf_files = list(BROCHURE_DIR.rglob("*.pdf"))

    if not pdf_files:
        print("No brochures found.")
        return

    print(f"Found {len(pdf_files)} brochures.\n")

    for pdf_path in pdf_files:

        # Folder name becomes brand
        brand = pdf_path.parent.name

        # PDF filename becomes model
        model = pdf_path.stem

        print(f"Processing: {brand} {model}")

        try:
            # Load PDF
            pages = load_pdf(
                str(pdf_path),
                brand=brand,
                model=model
            )

            # Create chunks
            chunks = create_chunks(pages)

            all_chunks.extend(chunks)

            print(f"  Pages: {len(pages)}")
            print(f"  Chunks: {len(chunks)}")
            print()

        except Exception as error:
            print(f"  ERROR: {error}")
            print()


    if not all_chunks:
        print("No chunks were created.")
        return


    print("=" * 60)
    print("Total chunks:", len(all_chunks))
    print("=" * 60)

    print("\nCreating embeddings and FAISS index...")

    index, chunks = build_vector_store(all_chunks)

    print("Vectors created:", index.ntotal)

    save_vector_store(index, chunks)

    print("\nVector store saved successfully.")
    print("Indexing complete.")


if __name__ == "__main__":
    main()