import re
import time
import math
from collections import Counter, defaultdict

from nltk.stem import PorterStemmer


class IRSystem:
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
        self.idf = {}
        self.doc_lengths = {}
        self.queries_data = {}
        self.relevance_data = {}

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

            self.doc_lengths[doc_id] = sum(term_freqs.values())

            for token, freq in term_freqs.items():
                self.index[token].append((doc_id, freq))

        N = len(self.documents)
        for term, postings in self.index.items():
            df = len(postings)
            self.idf[term] = math.log10(N / df)

        return time.time() - start

    def ranked_query(self, query_text):
        start = time.time()

        query_tokens = self.preprocess_query(query_text)

        if not query_tokens:
            return [], time.time() - start

        query_tf = Counter(query_tokens)

        doc_scores = defaultdict(float)

        for term, qtf in query_tf.items():
            if term not in self.index:
                continue

            query_weight = (1 + math.log10(qtf)) * self.idf.get(term, 0)

            for doc_id, dtf in self.index[term]:
                doc_weight = (1 + math.log10(dtf)) * self.idf.get(term, 0)
                doc_scores[doc_id] += query_weight * doc_weight

        for doc_id in doc_scores:
            if self.doc_lengths[doc_id] > 0:
                doc_scores[doc_id] /= math.sqrt(self.doc_lengths[doc_id])

        ranked_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        return ranked_results, time.time() - start

    def preprocess_query(self, query_text):
        tokens = self._tokenize(query_text)
        processed = [self.stemmer.stem(t) for t in tokens if t not in self.stopwords]
        return processed

    def load_queries(self, query_file):
        current_id = None
        content = []
        record_content = False

        with open(query_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('.I'):
                    if current_id is not None:
                        self.queries_data[current_id] = ' '.join(content)
                    current_id = int(line.split()[1])
                    content = []
                    record_content = False
                elif line.startswith('.W'):
                    record_content = True
                elif line.startswith('.'):
                    record_content = False
                elif record_content:
                    content.append(line)
            if current_id is not None:
                self.queries_data[current_id] = ' '.join(content)

    def load_relevance(self, qrels_file):
        """Load relevance judgments from CACM qrels file"""
        with open(qrels_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    query_id = int(parts[0])
                    doc_id = int(parts[1])
                    if query_id not in self.relevance_data:
                        self.relevance_data[query_id] = []
                    self.relevance_data[query_id].append(doc_id)

    def stats(self):
        return {
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "vocab_size": len(self.vocab),
            "num_docs": len(self.documents),
            "index_size": sum(len(v) for v in self.index.values()),
        }

    def evaluate_ranked_query(self, query_text, relevant_docs):
        ranked_results, query_time = self.ranked_query(query_text)

        retrieved_docs = [doc_id for doc_id, score in ranked_results]

        top3 = retrieved_docs[:3]
        top10 = retrieved_docs[:10]

        R = len(relevant_docs)
        topR = retrieved_docs[:R]

        def precision_at(k_list, rel_docs):
            if not k_list:
                return 0.0
            hits = sum(1 for d in k_list if d in rel_docs)
            return hits / len(k_list)

        p_at_3 = precision_at(top3, relevant_docs)
        p_at_10 = precision_at(top10, relevant_docs)
        r_precision = precision_at(topR, relevant_docs)

        return {
            "query_text": query_text,
            "matches": len(retrieved_docs),
            "query_time": query_time,
            "top3": top3,
            "top10": top10,
            "top3_scores": ranked_results[:3],
            "top10_scores": ranked_results[:10],
            "P@3": p_at_3,
            "P@10": p_at_10,
            "R-precision": r_precision,
            "num_relevant": R,
            "retrieved_docs": retrieved_docs
        }

    def evaluate_all_queries(self):
        results = []
        p3_scores = []
        p10_scores = []
        r_prec_scores = []

        for query_id, query_text in self.queries_data.items():
            if query_id not in self.relevance_data:
                continue

            relevant_docs = self.relevance_data[query_id]
            result = self.evaluate_ranked_query(query_text, relevant_docs)
            result['query_id'] = query_id
            results.append(result)

            p3_scores.append(result['P@3'])
            p10_scores.append(result['P@10'])
            r_prec_scores.append(result['R-precision'])

        stats = {
            "num_queries": len(results),
            "P@3": {
                "min": min(p3_scores) if p3_scores else 0,
                "max": max(p3_scores) if p3_scores else 0,
                "avg": sum(p3_scores) / len(p3_scores) if p3_scores else 0
            },
            "P@10": {
                "min": min(p10_scores) if p10_scores else 0,
                "max": max(p10_scores) if p10_scores else 0,
                "avg": sum(p10_scores) / len(p10_scores) if p10_scores else 0
            },
            "R-precision": {
                "min": min(r_prec_scores) if r_prec_scores else 0,
                "max": max(r_prec_scores) if r_prec_scores else 0,
                "avg": sum(r_prec_scores) / len(r_prec_scores) if r_prec_scores else 0
            }
        }

        return results, stats
