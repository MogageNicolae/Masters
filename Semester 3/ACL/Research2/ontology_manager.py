import json
import re
import os

# Basic list of words to ignore in context generation
STOP_WORDS = {
    "the", "a", "an", "of", "in", "to", "for", "with", "on", "at", "by", "from", 
    "is", "are", "and", "or", "as", "be", "that", "which", "used", "refers", 
    "usually", "common", "defined", "see", "also", "known", "include", "group",
    "structure", "process", "related", "causing", "having", "other", "into", "but"
}

class OntologyManager:
    def __init__(self, ontology_path="data/medical_ontology.json", source_path="data/2024_js_elastic_updated_20250722.json"):
        self.ontology_path = ontology_path
        self.source_path = source_path
        
        if not os.path.exists(self.ontology_path):
            print(f"Ontology file not found at {self.ontology_path}. Generating from source...")
            self.generate_ontology()
            
        self.ontology = self.load_ontology(self.ontology_path)

    def extract_keywords(self, definition_text, synonyms):
        """
        Generates a list of relevant keywords from the definition and synonyms.
        """
        unique_keywords = set()
        
        # Add Synonyms as keywords
        if synonyms:
            for syn in synonyms:
                # Clean synonyms (remove punctuation/extra spaces)
                clean_syn = syn.lower().strip()
                unique_keywords.add(clean_syn)

        # Add words from the Definition
        if definition_text:
            # Find all words with 3 or more letters
            tokens = re.findall(r'\b[a-zA-Z]{3,}\b', definition_text.lower())
            for word in tokens:
                if word not in STOP_WORDS:
                    unique_keywords.add(word)
                    
        return list(unique_keywords)

    def generate_ontology(self):
        print("Loading source data...")
        try:
            with open(self.source_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Could not find source file '{self.source_path}'.")
            return

        transformed_data = {}
        
        print(f"Processing {len(raw_data)} entries...")
        
        for entry in raw_data:
            # Filter for main Descriptors only
            if entry.get('db') == 'mesh':
                
                english_term = entry.get("eng")
                french_term = entry.get("trx")
                
                # Skip if we don't have the main terms
                if not english_term or not french_term:
                    continue
                    
                # Use lower case for the main dictionary key
                key = english_term.lower()
                
                # Get definition and synonyms for context
                definition = entry.get("notes", {}).get("scn", "")
                synonyms = entry.get("xtr_en", [])
                
                # Generate the context object
                transformed_data[key] = {
                    "target": french_term,
                    "context_keywords": self.extract_keywords(definition, synonyms)
                }

        print(f"Saving dictionary with {len(transformed_data)} terms to {self.ontology_path}...")
        
        with open(self.ontology_path, 'w', encoding='utf-8') as f:
            json.dump(transformed_data, f, ensure_ascii=False, indent=4)
            
        print("Done! Ontology file ready.")

    def load_ontology(self, path):
        """Loads the ontology from a JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Ontology file not found at {path}")
            return {}

    def identify_concepts(self, text):
        """
        Identifies ontology concepts in the input text.
        Returns a list of dictionaries with 'term', 'target', and 'context_match'.
        Optimized for speed using a lookup index.
        """
        found_concepts = []
        text_lower = text.lower()
        
        # Simple optimization: Check for terms based on their starting word
        # This requires building an index, which we should do in __init__ ideally,
        # but for now we can do a quick check or just iterate if we haven't built it.
        
        # If we haven't built the index yet, do it now (lazy initialization)
        if not hasattr(self, '_term_index'):
            self._term_index = {}
            for term in self.ontology:
                words = term.lower().split()
                if words:
                    first_word = words[0]
                    if first_word not in self._term_index:
                        self._term_index[first_word] = []
                    self._term_index[first_word].append(term)
        
        # Tokenize text roughly
        text_words = text_lower.split()
        
        # Check potential terms
        potential_terms = set()
        for word in text_words:
            # Clean word
            clean_word = re.sub(r'[^\w\s]', '', word)
            if clean_word in self._term_index:
                potential_terms.update(self._term_index[clean_word])
                
        # Now verify potential terms
        for term in potential_terms:
            # Use regex to match whole words
            if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                data = self.ontology[term]
                # Check for context
                context_score = 0
                for keyword in data.get('context_keywords', []):
                    if keyword in text_lower:
                        context_score += 1
                
                found_concepts.append({
                    'term': term,
                    'target': data['target'],
                    'context_score': context_score
                })
        
        # Sort by context score (descending)
        found_concepts.sort(key=lambda x: x['context_score'], reverse=True)
        return found_concepts
