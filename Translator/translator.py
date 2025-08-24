from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from langdetect import detect
import torch
import PyPDF2

def load_model(model_name="facebook/mbart-large-50-many-to-many-mmt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)
    return model, tokenizer, device

def translate_text(text, src_lang, model, tokenizer, device):
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt", truncation=True).to(device)

    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],
        max_length=512,
        num_beams=4
    )

    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

def translate_pdf(pdf_path, output_file, model, tokenizer, device):
    translated_text = ""

    # Extract text from PDF
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text:
                continue

            # Detect language
            detected_lang = detect(text[:200])  # detect on first 200 chars
            if detected_lang.startswith("fr"):
                src = "fr_XX"
            elif detected_lang.startswith("zh"):
                src = "zh_CN"
            else:
                continue

            # Translate page
            translated_page = translate_text(text, src, model, tokenizer, device)
            translated_text += f"\n--- Page {page_num+1} ---\n{translated_page}\n"

    # Save output
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(translated_text)

    print(f"✅ Translation complete! Saved to {output_file}")

if __name__ == "__main__":
    model, tokenizer, device = load_model()

    pdf_file = input("Enter PDF file path: ")
    output_file = "translated_output.txt"

    translate_pdf(pdf_file, output_file, model, tokenizer, device)
