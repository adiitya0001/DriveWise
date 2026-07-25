import fitz


def load_pdf(pdf_path, brand, model):
    """
    Extract text from a PDF using PyMuPDF.
    """

    print(f"Loading PDF: {pdf_path}")

    try:
        document = fitz.open(pdf_path)
    except Exception as error:
        print(f"Could not open PDF: {error}")
        return []

    pages = []
    successful_pages = 0
    failed_pages = 0

    for page_number, page in enumerate(document):

        try:
            text = page.get_text("text").strip()

            if text:
                successful_pages += 1
            else:
                failed_pages += 1

            pages.append({
                "text": text,
                "page": page_number + 1,
                "brand": brand,
                "model": model
            })

        except Exception as error:
            print(
                f"Failed to extract {brand} {model} "
                f"page {page_number + 1}: {error}"
            )

            failed_pages += 1

            pages.append({
                "text": "",
                "page": page_number + 1,
                "brand": brand,
                "model": model
            })

    document.close()

    print(f"  Extracted pages: {successful_pages}")
    print(f"  Empty/failed pages: {failed_pages}")

    return pages


if __name__ == "__main__":

    pages = load_pdf(
        "data/brochures/Mahindra/Scorpio.pdf",
        brand="Mahindra",
        model="Scorpio"
    )

    print("\nTotal pages:", len(pages))

    for page in pages:
        print("\n" + "=" * 60)
        print(f"PAGE {page['page']}")
        print("=" * 60)

        if page["text"]:
            print(page["text"][:1500])
        else:
            print("[NO TEXT EXTRACTED]")