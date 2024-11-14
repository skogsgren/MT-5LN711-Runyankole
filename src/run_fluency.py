import openai
import subprocess
import sacrebleu
import re
# Set your OpenAI API key
openai.api_key = 'insert api-key here'
# Define paths for eval and NMT model
source_path = './data/original/nyn_test.bpe'
model_path = 'model_nyn_eng_step_4352.pt'
TGT_PATH = './data/original/eng_test.detok'  # Target reference file path
ef translate_with_opennmt(source_path, model_path):
    """
    Translate source text using a pre-trained OpenNMT model.
    Assumes you have a command-line OpenNMT model ready.
    """
    # OpenNMT translation command
    translation_command = [
        "onmt_translate",
        "-model", model_path,
        "-src", source_path,
        "-output", "translation.txt"
    ]
    subprocess.run(translation_command)

 
    with open("translation.txt", "r") as file:
        translations = file.read().strip().splitlines()

    translations = [detokenize_bpe(sentence) for sentence in translations]

    # Overwrite the file with detokenized translations
    with open("translation.txt", "w") as file:
        for sentence in translations:
            file.write(sentence + "\n")
    return translations

def detokenize_bpe(text):
    """
    Detokenize BPE (Byte Pair Encoding) text by removing BPE markers.
    """
    return text.replace('@@ ', '').replace('@@', '')

def enhance_fluency_with_openai(translations, batch_size=5):
    """
    Use OpenAI to enhance the fluency and style of translations in batches.
    """
    enhanced_translations = []
    
    # Process translations in batches
    for i in range(0, len(translations), batch_size):
        # Get the current batch
        batch = translations[i:i + batch_size]
        # Join the batch into a single prompt
        prompt = "Improve the fluency and readability of these translations:\n\n"
        prompt += "\n\n".join(f"{idx+1}. {sentence}" for idx, sentence in enumerate(batch, start=i + 1))
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  
            messages=[{"role": "system", "content": "You are an assistant that improves translation fluency."},
                      {"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7
        )
        
        # Process the response and split by lines if multiple outputs
        enhanced_text = response.choices[0].message['content'].strip().splitlines()
        
        # Remove numbers at the beginning of each line
        cleaned_text = [re.sub(r'^\d+\.\s*', '', line) for line in enhanced_text if line.strip()]
        
        # Append cleaned translations
        enhanced_translations.extend(cleaned_text)

    return enhanced_translations

def calculate_bleu_score(reference_path, hypothesis_text):
    """
    Calculate BLEU score using sacrebleu, comparing reference and hypothesis lists.
    """
    # Load the reference file (target sentences)
    with open(reference_path, "r") as file:
        reference_text = file.read().strip().splitlines()

    # Calculate BLEU score
    bleu_score = sacrebleu.corpus_bleu(hypothesis_text, [reference_text])
    return bleu_score.score

# Translate all sentences in the source file with OpenNMT
translations = translate_with_opennmt(source_path, model_path)
print("Initial Translations (BPE):", translations[:5])  # Show a sample of BPE-tokenized translations

# Calculate and print the BLEU score for the original translations
original_bleu_score = calculate_bleu_score(TGT_PATH, translations)
print("Original Translations BLEU Score:", original_bleu_score)

# Enhance Fluency with OpenAI in batches
enhanced_translations = enhance_fluency_with_openai(translations)
print("Enhanced Translations:", enhanced_translations[:5])  # Show a sample of the enhanced translations

# Save the enhanced translations to a file
with open("enhanced_translations.txt", "w") as file:
    for sentence in enhanced_translations:
        file.write(sentence + "\n")

# Calculate and print the BLEU score for the enhanced translations
enhanced_bleu_score = calculate_bleu_score(TGT_PATH, enhanced_translations)
print("Enhanced Translations BLEU Score:", enhanced_bleu_score)
