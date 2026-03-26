from ir_system import IRSystem
import json

COLLECTION_PATH = "cacm/cacm.all"
QUERY_PATH = "cacm/query.text"
QRELS_PATH = "cacm/qrels.text"

def main():
    ir = IRSystem(COLLECTION_PATH)
    ir.load_documents()
    print(f"   - Loaded {len(ir.documents)} documents")
    ir.load_queries(QUERY_PATH)
    print(f"   - Loaded {len(ir.queries_data)} queries")
    ir.load_relevance(QRELS_PATH)
    print(f"   - Loaded relevance data for {len(ir.relevance_data)} queries")
    ir.determine_stopwords(n=7)
    print("\nStopword frequencies:")
    for word, freq in sorted(ir.stopword_frequencies.items(), key=lambda x: x[1], reverse=True):
        print(f"  {word}: {freq}")

    build_time = ir.build_index()
    print(f"Index build time: {build_time:.4f} seconds")

    stats = ir.stats()
    print("\nCorpus statistics:")
    print(f"  Tokens before preprocessing: {stats['tokens_before']}")
    print(f"  Tokens after preprocessing:  {stats['tokens_after']}")
    print(f"  Vocabulary size:             {stats['vocab_size']}")
    print(f"  Number of documents:         {stats['num_docs']}")
    print(f"  Index size (postings):       {stats['index_size']}")

    print("\nRunning evaluation on all queries with relevance judgments...")
    results, eval_stats = ir.evaluate_all_queries()
    
    print(f"\nEvaluated {eval_stats['num_queries']} queries")
    print("\nOverall Performance:")
    print(f"\n  P@3:  Min={eval_stats['P@3']['min']:.4f}, Max={eval_stats['P@3']['max']:.4f}, Avg={eval_stats['P@3']['avg']:.4f}")
    print(f"  P@10: Min={eval_stats['P@10']['min']:.4f}, Max={eval_stats['P@10']['max']:.4f}, Avg={eval_stats['P@10']['avg']:.4f}")
    print(f"  R-precision: Min={eval_stats['R-precision']['min']:.4f}, Max={eval_stats['R-precision']['max']:.4f}, Avg={eval_stats['R-precision']['avg']:.4f}")

    sorted_by_p3 = sorted(results, key=lambda x: x['P@3'], reverse=True)

    best_query = sorted_by_p3[0]

    mid_idx = len(sorted_by_p3) // 3
    interesting_query = sorted_by_p3[mid_idx] if mid_idx < len(sorted_by_p3) else sorted_by_p3[-1]

    print("\n" + "-" * 80)
    print("EXAMPLE 1: Best Performing Query")
    print("-" * 80)
    display_query_results(ir, best_query)

    print("\n" + "-" * 80)
    print("EXAMPLE 2: Interesting Query Result")
    print("-" * 80)
    display_query_results(ir, interesting_query)

    save_results(results, eval_stats)


def display_query_results(ir, result):
    query_id = result['query_id']
    query_text = result['query_text']
    
    print(f"\nQuery ID: {query_id}")
    print(f"Query Text: {query_text}")
    print(f"\nQuery Processing:")

    preprocessed = ir.preprocess_query(query_text)
    print(f"  Original terms: {query_text.split()[:10]}...")
    print(f"  After preprocessing: {preprocessed}")
    
    print(f"\nRelevance Information:")
    print(f"  Number of relevant documents: {result['num_relevant']}")
    relevant_docs = ir.relevance_data[query_id]
    print(f"  Relevant doc IDs: {relevant_docs[:10]}{'...' if len(relevant_docs) > 10 else ''}")
    
    print(f"\nRetrieval Results:")
    print(f"  Total matching documents: {result['matches']}")
    
    print(f"\n  Top 3 Retrieved Documents:")
    for i, (doc_id, score) in enumerate(result['top3_scores'], 1):
        is_relevant = "Relevant" if doc_id in relevant_docs else "Not relevant"
        print(f"    {i}. Doc {doc_id:4d} (score: {score:.4f}) {is_relevant}")

        doc_text = ir.documents.get(doc_id, "")[:100]
        print(f"       Snippet: {doc_text}...")
    
    print(f"\n  Top 10 Retrieved Documents:")
    for i, (doc_id, score) in enumerate(result['top10_scores'], 1):
        is_relevant = doc_id in relevant_docs
        print(f"    {i:2d}. Doc {doc_id:4d} (score: {score:.4f}) {is_relevant}")
    
    print(f"\nEvaluation Metrics:")
    print(f"  P@3:  {result['P@3']:.4f} ({int(result['P@3']*3)}/3 relevant in top 3)")
    print(f"  P@10: {result['P@10']:.4f} ({int(result['P@10']*10)}/10 relevant in top 10)")
    print(f"  R-precision: {result['R-precision']:.4f} (precision at {result['num_relevant']} documents)")
    print(f"  Query time: {result['query_time']:.6f} seconds")


def save_results(results, stats):
    with open("evaluation_results.txt", "w", encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RANKED RETRIEVAL SYSTEM - EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        for result in results:
            f.write(f"\nQuery ID: {result['query_id']}\n")
            f.write(f"Query: {result['query_text']}\n")
            f.write(f"Relevant docs: {result['num_relevant']}\n")
            f.write(f"Retrieved docs: {result['matches']}\n")
            f.write(f"Top 10: {result['top10']}\n")
            f.write(f"P@3:  {result['P@3']:.4f}\n")
            f.write(f"P@10: {result['P@10']:.4f}\n")
            f.write(f"R-precision: {result['R-precision']:.4f}\n")
            f.write("-" * 80 + "\n")
    

    with open("evaluation_summary.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()

