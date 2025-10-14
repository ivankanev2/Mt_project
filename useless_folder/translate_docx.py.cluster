from docx import Document
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Load model
model_name = "facebook/m2m100_418M"
tok = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)
tok.src_lang = "en"

def translate_text(text):
    if not text.strip():
        return text
    enc = tok(text, return_tensors="pt", truncation=True)
    gen = model.generate(**enc, forced_bos_token_id=tok.get_lang_id("bg"))
    return tok.batch_decode(gen, skip_special_tokens=True)[0]

# Open your Word doc
doc = Document("Copy of APP-GW-G0R-004_Revision_A_eng.docx")

# Translate each paragraph
for para in doc.paragraphs:
    para.text = translate_text(para.text)

# Save new doc
doc.save("output_bg.docx")
print("✅ Translated document saved as output_bg.docx")
