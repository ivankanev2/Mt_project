from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

model_name = "facebook/m2m100_418M"  # smaller and lighter than 1.2B
tok = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

tok.src_lang = "en"
text = "The reactor will enter a planned outage next week."
enc = tok(text, return_tensors="pt")
gen = model.generate(**enc, forced_bos_token_id=tok.get_lang_id("bg"))
print(tok.batch_decode(gen, skip_special_tokens=True)[0])
