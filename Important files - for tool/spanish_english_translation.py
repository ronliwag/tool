#!/usr/bin/env python3
"""
Spanish to English Translation Component
For complete speech-to-speech translation pipeline
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

class SpanishEnglishTranslator:
    """Spanish to English translation using Helsinki-NLP model"""
    
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "Helsinki-NLP/opus-mt-es-en"
        
        # Load translation model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Spanish-English translator initialized: {self.model_name}")
        print(f"Device: {self.device}")
    
    def translate(self, spanish_text):
        """Translate Spanish text to English"""
        try:
            if not spanish_text.strip():
                return ""
            
            # Tokenize input
            inputs = self.tokenizer(
                spanish_text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate translation
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.get('input_ids', list(inputs.values())[0]),
                    max_length=512,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode translation
            translation = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )[0]
            
            print(f"Translation: {spanish_text} -> {translation}")
            return translation.strip()
            
        except Exception as e:
            print(f"Error in translation: {e}")
            return ""
    
    def translate_batch(self, spanish_texts):
        """Translate multiple Spanish texts to English"""
        try:
            if not spanish_texts:
                return []
            
            translations = []
            for text in spanish_texts:
                translation = self.translate(text)
                translations.append(translation)
            
            return translations
            
        except Exception as e:
            print(f"Error in batch translation: {e}")
            return []
    
    def get_translation_confidence(self, spanish_text, english_text):
        """Get confidence score for translation"""
        try:
            # Tokenize both texts
            src_inputs = self.tokenizer(spanish_text, return_tensors="pt", padding=True, truncation=True)
            tgt_inputs = self.tokenizer(english_text, return_tensors="pt", padding=True, truncation=True)
            
            # Move to device
            src_inputs = {k: v.to(self.device) for k, v in src_inputs.items()}
            tgt_inputs = {k: v.to(self.device) for k, v in tgt_inputs.items()}
            
            # Get model output
            with torch.no_grad():
                outputs = self.model(**src_inputs, labels=tgt_inputs.input_ids)
                loss = outputs.loss
                confidence = torch.exp(-loss).item()
            
            return confidence
            
        except Exception as e:
            print(f"Error calculating translation confidence: {e}")
            return 0.0

def main():
    """Test the Spanish-English translation component"""
    print("Testing Spanish-English Translation Component")
    print("=" * 50)
    
    # Initialize translator
    translator = SpanishEnglishTranslator()
    
    # Test translations
    test_phrases = [
        "Hola, ¿cómo estás?",
        "Buenos días, ¿qué tal?",
        "Me llamo Juan y soy de España.",
        "¿Puedes ayudarme con esto?",
        "Muchas gracias por tu ayuda."
    ]
    
    print("\nTesting Spanish to English translations:")
    print("-" * 40)
    
    for spanish_text in test_phrases:
        english_text = translator.translate(spanish_text)
        confidence = translator.get_translation_confidence(spanish_text, english_text)
        print(f"Spanish: {spanish_text}")
        print(f"English: {english_text}")
        print(f"Confidence: {confidence:.3f}")
        print("-" * 40)
    
    # Interactive mode
    print("\nInteractive translation mode (type 'quit' to exit):")
    while True:
        spanish_input = input("\nEnter Spanish text: ").strip()
        if spanish_input.lower() == 'quit':
            break
        
        if spanish_input:
            english_output = translator.translate(spanish_input)
            if english_output:
                confidence = translator.get_translation_confidence(spanish_input, english_output)
                print(f"English: {english_output}")
                print(f"Confidence: {confidence:.3f}")

if __name__ == "__main__":
    main()
