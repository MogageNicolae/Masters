from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch


models=[
    "Helsinki-NLP/opus-mt-en-fr",
    "facebook/nllb-200-distilled-600M"
]

from peft import PeftModel, PeftConfig
import os
import re

class OntologyAwareTranslator:
    def __init__(self, model_name="facebook/nllb-200-distilled-600M", fine_tuned_path="models/fine_tuned_nllb", verbose=False, load_adapters=True):
        self.verbose = verbose
        if self.verbose:
            print(f"Loading model: {model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        if load_adapters and os.path.exists(fine_tuned_path):
            if self.verbose:
                print(f"Found fine-tuned model at {fine_tuned_path}. Loading adapters...")
            
            # 1. Add special tokens to tokenizer
            special_tokens_dict = {'additional_special_tokens': ['<dnt>', '</dnt>']}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            
            # 2. Resize base model embeddings BEFORE loading adapters
            self.model.resize_token_embeddings(len(self.tokenizer))
            
            # 3. Load adapters
            self.model = PeftModel.from_pretrained(self.model, fine_tuned_path)
        else:
            if self.verbose:
                print("Loading base model (no adapters).")

        self.model.to(self.device)
        print(f"Model loaded on {self.device}.")

    def translate(self, text, constraints=None):
        """
        Translates text. If constraints (list of target words) are provided, 
        pre-pends them to the source text to provide context (Context Injection).
        """
        input_text = text
        if constraints:
            # Inline Annotations: Inject <dnt> target </dnt> after the term in source
            # Constraints should be a list of dicts with 'term' and 'target'
            # OR a list of strings "term : target" which we parse.
            # To be safe and consistent with main.py updates, let's assume constraints is a list of dicts 
            # OR we parse the string format if that's what main.py sends.
            # Actually, main.py currently sends strings. Let's update main.py to send dicts, 
            # or parse here. Parsing is safer if we want to keep main.py simple.
            
            # Sort constraints by length of term to avoid partial matches
            # We need to parse "term : target" back to term and target if strings are passed
            parsed_constraints = []
            for c in constraints:
                if isinstance(c, str):
                    if " : " in c:
                        parts = c.split(" : ")
                        parsed_constraints.append({'term': parts[0], 'target': parts[1]})
                elif isinstance(c, dict):
                    parsed_constraints.append(c)
            
            parsed_constraints.sort(key=lambda x: len(x['term']), reverse=True)
            
            for c in parsed_constraints:
                term = c['term']
                target = c['target']
                # Regex replacement
                pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
                def replacement_func(match):
                    original_term = match.group(0)
                    return f"{original_term} <dnt> {target} </dnt>"
                
                input_text = pattern.sub(replacement_func, input_text)
            
            if self.verbose:
                print(f"Translating with inline annotations: '{input_text}'")

        # NLLB requires source language code in tokenizer and target language code in generate
        self.tokenizer.src_lang = "eng_Latn"
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True).to(self.device)
        
        generation_args = {
            "max_length": 512,
            "num_beams": 5,
            "num_return_sequences": 1,
            "early_stopping": True,
            "forced_bos_token_id": self.tokenizer.convert_tokens_to_ids("fra_Latn")
        }

        translated = self.model.generate(**inputs, **generation_args)
        decoded_output = self.tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
        
        # Clean output: Remove <dnt> tags if they appear
        # The model might output "<dnt> target </dnt>" or similar.
        # We want to keep the target but remove the tags.
        # Regex to remove <dnt> and </dnt> tags
        cleaned_output = re.sub(r'<dnt>\s*', '', decoded_output)
        cleaned_output = re.sub(r'\s*</dnt>', '', cleaned_output)
        
        return cleaned_output
