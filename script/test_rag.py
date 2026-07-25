from rag.vector_store import load_vector_store
from rag.retriever import retrieve
from rag.reranker import rerank
from rag.generator import generate_answer


index, chunks = load_vector_store()


tests = [
    {
        "brand": "Hyundai",
        "model": "Creta",
        "question": "How many airbags does Creta have?"
    },
    {
        "brand": "Mahindra",
        "model": "Scorpio",
        "question": "What engine options does Scorpio have?"
    },
    {
        "brand": "Mahindra",
        "model": "Thar",
        "question": "Does Thar have 4WD?"
    },
    {
        "brand": "Mahindra",
        "model": "XUV_7XO",
        "question": "What ADAS features does XUV 7XO have?"
    },
    {
        "brand": "Tata",
        "model": "Harrier",
        "question": "What safety features does Harrier have?"
    },
    {
        "brand": "Tata",
        "model": "Nexon",
        "question": "What engine options does Nexon have?"
    },
    {
        "brand": "Tata",
        "model": "sierra",
        "question": "What are the main features of Sierra?"
    }
]


for test in tests:

    print("\n" + "=" * 70)

    print(
        f"{test['brand']} {test['model']}"
    )

    print("Question:", test["question"])

    # 1. Retrieve
    results = retrieve(
        query=test["question"],
        index=index,
        chunks=chunks,
        brand=test["brand"],
        car_model=test["model"],
        top_k=5
    )

    if not results:
        print("❌ No results found")
        continue

    # 2. Rerank
    ranked_results = rerank(
        query=test["question"],
        results=results,
        top_k=3
    )

    # 3. Generate answer
    answer = generate_answer(
        query=test["question"],
        ranked_results=ranked_results
    )

    print("\nDriveWise:")
    print(answer)

    print("\nTop source:")

    if ranked_results:
        metadata = ranked_results[0]["metadata"]

        print(
            f"{metadata['brand']} "
            f"{metadata['model']} "
            f"- Page {metadata['page']}"
        )