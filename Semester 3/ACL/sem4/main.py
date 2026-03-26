from sem4.ir_system import SimpleIRSystem

COLLECTION_PATH = "../cacm/cacm.all"

TEST_QUERIES = ["algorithm", "problem", "computer"]

RELEVANCE = {
    "algorithm": [2263, 2146, 2767, 2884, 2916, 2283, 2902],
    "problem": [2196, 3018, 2794, 78, 1435, 1978, 2289],
    "computer": [1621, 670, 2339, 2739, 3142, 675, 3012]
}


def main():
    print("Loading IR system...")
    ir = SimpleIRSystem(COLLECTION_PATH)
    ir.load_documents()
    ir.determine_stopwords()
    print(f"Stopwords (7 most frequent): {ir.stopword_frequencies}")

    build_time = ir.build_index()
    print(f"Index build time: {build_time:.4f} seconds")

    print("\nCorpus statistics:")
    stats = ir.stats()
    for k, v in stats.items():
        print(f"{k}: {v}")

    ir.evaluate(TEST_QUERIES, RELEVANCE)


if __name__ == "__main__":
    main()
