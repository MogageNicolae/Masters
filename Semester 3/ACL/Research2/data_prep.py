import os
import json
import re
import pandas as pd
from ontology_manager import OntologyManager
from tqdm import tqdm
import random

def prepare_data():
    print("Initializing Ontology Manager...")
    ontology_manager = OntologyManager()
    
    csv_path = "data/elrc_en_fr.csv"
    print(f"Loading dataset from {csv_path}...")

    df = pd.read_csv(csv_path)
    
    # Map columns based on inspection
    # Header: id,lang,source_text,target_text
    if 'source_text' in df.columns and 'target_text' in df.columns:
        df = df.rename(columns={'source_text': 'source', 'target_text': 'target'})
    elif 'source' not in df.columns or 'target' not in df.columns:
        if 'en' in df.columns and 'fr' in df.columns:
            df = df.rename(columns={'en': 'source', 'fr': 'target'})
        else:
            print("Warning: Could not identify 'source'/'target' columns. Using first two columns.")
            df.columns = ['source', 'target'] + list(df.columns[2:])
            
    df = df.dropna(subset=['source', 'target'])
    print(f"Original dataset size: {len(df)}")
    
    processed_samples = []
    
    print("Processing dataset and augmenting with constraints...")
    records = df.to_dict('records')
    
    for item in tqdm(records):
        source_text = str(item['source'])
        target_text = str(item['target'])
        
        # 1. Always keep the original sample
        processed_samples.append({
            "source": source_text,
            "target": target_text
        })
        
        # 2. Check for ontology concepts
        concepts = ontology_manager.identify_concepts(source_text)
        
        if concepts:
            # Strict filtering: Only use concepts with context_score > 0
            relevant_concepts = [c for c in concepts if c['context_score'] > 0]
            
            if relevant_concepts:
                # Create augmented sample with Inline Annotations
                # Sort concepts by length of term (descending) to avoid partial replacements of longer terms
                relevant_concepts.sort(key=lambda x: len(x['term']), reverse=True)
                
                augmented_source = source_text
                for c in relevant_concepts:
                    term = c['term']
                    target = c['target']
                    # Regex to match whole word and replace with inline annotation
                    # We use a simple replacement for now, but ideally we should handle case sensitivity better
                    # to preserve original casing of the term if possible, or just use the term as matched.
                    # The identify_concepts method returns the term from the ontology (which might be lowercase).
                    # We should find the actual text in the source.
                    
                    pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
                    
                    def replacement_func(match):
                        original_term = match.group(0)
                        return f"{original_term} <dnt> {target} </dnt>"
                    
                    augmented_source = pattern.sub(replacement_func, augmented_source)
                
                processed_samples.append({
                    "source": augmented_source,
                    "target": target_text
                })
    
    print(f"Processed dataset size: {len(processed_samples)}")
    
    # Shuffle
    random.seed(42)
    random.shuffle(processed_samples)
    
    # Split
    split_idx = int(len(processed_samples) * 0.9)
    train_data = processed_samples[:split_idx]
    val_data = processed_samples[split_idx:]
    
    print(f"Train size: {len(train_data)}")
    print(f"Val size: {len(val_data)}")
    
    # Save as JSON
    with open("data/train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
        
    with open("data/val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
        
    print("Data preparation complete. Saved to data/train.json and data/val.json")

if __name__ == "__main__":
    prepare_data()
