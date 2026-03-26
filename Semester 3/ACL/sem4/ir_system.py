import re
import time
from collections import Counter, defaultdict

from nltk.stem import PorterStemmer


class SimpleIRSystem:
    def __init__(self, collection_path):
        self.collection_path = collection_path
        self.documents = {}
        self.stopwords = set()
        self.stopword_frequencies = {}
        self.index = defaultdict(list)
        self.stemmer = PorterStemmer()
        self.tokens_before = 0
        self.tokens_after = 0
        self.vocab = set()

    def load_documents(self):
        current_id = None
        content = []
        record_content = False
        with open(self.collection_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('.I'):
                    if current_id is not None:
                        self.documents[current_id] = ' '.join(content)
                    current_id = int(line.split()[1])
                    content = []
                    record_content = False
                elif line.startswith('.T') or line.startswith('.W'):
                    record_content = True
                elif line.startswith('.'):
                    record_content = False
                elif record_content:
                    content.append(line)
            if current_id is not None:
                self.documents[current_id] = ' '.join(content)

    def determine_stopwords(self, n=7):
        counter = Counter()
        for text in self.documents.values():
            counter.update(self._tokenize(text))

        most_common = counter.most_common(n)
        self.stopwords = set([w for w, _ in most_common])
        self.stopword_frequencies = {w: freq for w, freq in most_common}

    def _tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        return text.split()

    def preprocess(self, text):
        tokens = self._tokenize(text)
        self.tokens_before += len(tokens)
        processed = [self.stemmer.stem(t) for t in tokens if t not in self.stopwords]
        self.tokens_after += len(processed)
        self.vocab.update(processed)
        return processed

    def build_index(self):
        start = time.time()

        for doc_id, text in self.documents.items():
            tokens = self.preprocess(text)
            term_freqs = Counter(tokens)
            for token, freq in term_freqs.items():
                self.index[token].append((doc_id, freq))

        return time.time() - start

    def query(self, word):
        start = time.time()
        w = word.lower()
        if w in self.stopwords:
            return [], 0.0
        postings = self.index.get(w, [])
        ranked_results = sorted(postings, key=lambda x: x[1], reverse=True)

        return [doc_id for doc_id, freq in ranked_results], time.time() - start

    def stats(self):
        return {
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "vocab_size": len(self.vocab),
            "num_docs": len(self.documents),
            "index_size": sum(len(v) for v in self.index.values()),
        }

    def evaluate_query(self, word, relevant_docs):
        posting, query_time = self.query(word)
        posting = list(dict.fromkeys(posting))
        top3 = posting[:3]
        top7 = posting[:7]

        def precision_at(k_list):
            if not k_list:
                return 0.0
            hits = sum(1 for d in k_list if d in relevant_docs)
            return hits / len(k_list)

        with open(f"{word}_docs.txt", "a") as f:
            for doc_id in posting[:25]:
                f.write(f"{doc_id}: {self.documents[doc_id]}\n")

        return {
            "matches": len(posting),
            "query_time": query_time,
            "top3": top3,
            "top7": top7,
            "P@3": precision_at(top3),
            "P@7": precision_at(top7)
        }

    def evaluate(self, queries, relevance):
        for q in queries:
            print(f"Query: '{q}'")
            if q not in relevance:
                print("No relevance judgments provided. Skipping.")
                continue
            results = self.evaluate_query(q, relevance[q])
            print(f"Matching documents: {results['matches']}")
            print(f"Query time: {results['query_time']:.8f} seconds")
            print(f"Top 3 docs: {results['top3']}")
            print(f"Top 7 docs: {results['top7']}")
            print(f"P@3: {results['P@3']:.2f}")
            print(f"P@7: {results['P@7']:.2f}")
