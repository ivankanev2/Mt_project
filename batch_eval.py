import argparse, csv, os, re, sys, json
import re
from pathlib import Path
from huggingface_hub.utils import HfHubHTTPError
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from lxml import etree
import torch
from tqdm import tqdm


# for progress bars
def pbar(it, **kwargs):
    """Show a tqdm bar only if the session looks interactive."""
    return tqdm(it, **kwargs) if sys.stderr.isatty() else it
    
# ---------------- Common helpers ----------------

def save_table_csv_and_xlsx(path_csv, columns, rows):
    """
    Save both a CSV (UTF-8 BOM) and a matching .xlsx file for easy opening in Excel.
    """
    import pandas as pd, os
    df = pd.DataFrame(rows, columns=columns)
    # CSV with BOM so Excel opens Bulgarian characters correctly
    df.to_csv(path_csv, index=False, encoding="utf-8-sig")
    # Excel file
    path_xlsx = os.path.splitext(path_csv)[0] + ".xlsx"
    df.to_excel(path_xlsx, index=False)
    print("💾 Wrote", path_csv, "and", path_xlsx)


def normalize_lines(lines):
    out = []
    for s in lines:
        s = " ".join(s.split()).strip()
        if s:
            out.append(s)
    return out

# really annoying piece of code that breaks things every time
def drop_furniture(lines):
    out = []
    for s in lines:
        t = (s or "").strip()
        if t:
            out.append(t)
    return out

#very important code || tweaked many times for accuracy
def extract_plain_text_docx(path):
    """
    DOCX extractor for alignment:
      - Preserves body order (paragraphs, tables)
      - Recurses into content controls (w:sdt)
      - Emits EACH paragraph inside table cells as its own line
      - Preserves hard line breaks (<w:br/>, <w:cr/>)
      - Extracts text boxes (w:txbxContent) in-place
      - Resolves altChunk -> reads word/altChunk#.xml and extracts visible text
      - Appends footnotes/endnotes text at the end (best-effort)
      - Skips headers/footers
    """
    from zipfile import ZipFile
    from lxml import etree

    W   = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    REL = "http://schemas.openxmlformats.org/package/2006/relationships"
    NS  = {"w": W, "rel": REL,
           "wps": "http://schemas.microsoft.com/office/word/2010/wordprocessingShape",
           "v":   "urn:schemas-microsoft-com:vml"}

    def clean(s: str) -> str:
        return " ".join((s or "").split()).strip()

    def para_to_lines(p_el):
        # collect runs + explicit line breaks
        parts = []
        for node in p_el.iter():
            q = etree.QName(node)
            if q.namespace == W and q.localname == "t":
                parts.append(node.text or "")
            elif q.namespace == W and q.localname in ("br", "cr"):
                parts.append("\n")
        text = "".join(parts)
        lines = [clean(x) for x in text.split("\n")]
        return [x for x in lines if x]

    def textboxes_in(el):
        # DrawingML and legacy VML text boxes
        lines = []
        # DrawingML (Word 2010 shapes use wps:txbx; plain txbxContent also appears)
        for tx in el.xpath(".//wps:txbx//w:txbxContent | .//w:txbxContent", namespaces=NS):
            t = "".join(tn.text or "" for tn in tx.xpath(".//w:t", namespaces=NS) if tn.text)
            t = clean(t)
            if t:
                # split by line breaks inside the text box content
                for ln in t.splitlines():
                    ln = clean(ln)
                    if ln:
                        lines.append(ln)
        # VML pathway (rare)
        for vtx in el.xpath(".//w:pict//v:shape//v:textbox//w:txbxContent", namespaces=NS):
            t = "".join(tn.text or "" for tn in vtx.xpath(".//w:t", namespaces=NS) if tn.text)
            t = clean(t)
            if t:
                for ln in t.splitlines():
                    ln = clean(ln)
                    if ln:
                        lines.append(ln)
        return lines

    def parse_xml_bytes(xml_bytes):
        return etree.fromstring(xml_bytes, etree.XMLParser(recover=True))

    out = []
    with ZipFile(path, "r") as zf:
        # === document.xml ===
        doc_root = parse_xml_bytes(zf.read("word/document.xml"))
        body = doc_root.find(".//w:body", namespaces=NS)

        # rels to resolve altChunk targets
        rels_map = {}
        try:
            rels_root = parse_xml_bytes(zf.read("word/_rels/document.xml.rels"))
            for rel in rels_root.findall(".//rel:Relationship", namespaces=NS):
                rId = rel.get("Id")
                tgt = rel.get("Target")
                if rId and tgt:
                    # resolve relative path
                    rels_map[rId] = "word/" + tgt if not tgt.startswith("word/") else tgt
        except KeyError:
            pass

        # helper to extract altChunk text
        def altchunk_texts(p_el):
            lines = []
            for ac in p_el.findall(".//w:altChunk", namespaces=NS):
                rId = ac.get("{%s}id" % REL)
                if not rId or rId not in rels_map:
                    continue
                target = rels_map[rId]
                try:
                    data = zf.read(target)
                except KeyError:
                    continue
                # altChunk usually HTML or WordprocessingML; try both
                try:
                    node = parse_xml_bytes(data)
                except Exception:
                    continue
                # Extract visible text (HTML-ish or WML)
                # Try generic text pass:
                txt = " ".join(x.strip() for x in node.itertext() if x and x.strip())
                txt = clean(txt)
                if txt:
                    for ln in txt.splitlines():
                        ln = clean(ln)
                        if ln:
                            lines.append(ln)
            return lines

        # in-order traversal of body
        def traverse(el):
            for child in el.iterchildren():
                lname = etree.QName(child).localname
                if lname == "p":
                    # paragraph content
                    out.extend(para_to_lines(child))
                    # any text boxes anchored in this paragraph
                    out.extend(textboxes_in(child))
                    # altChunk inside/after the paragraph
                    out.extend(altchunk_texts(child))
                elif lname == "tbl":
                    # walk rows/cells; emit each cell paragraph as its own line
                    for tr in child.findall(".//w:tr", namespaces=NS):
                        for tc in tr.findall(".//w:tc", namespaces=NS):
                            # recurse into content controls in the cell
                            sdtc = tc.find(".//w:sdtContent", namespaces=NS)
                            if sdtc is not None:
                                # visit sdt content preserving order
                                for p in sdtc.findall(".//w:p", namespaces=NS):
                                    out.extend(para_to_lines(p))
                                    out.extend(textboxes_in(p))
                                    out.extend(altchunk_texts(p))
                            # paragraphs directly in cell
                            for p in tc.findall("./w:p", namespaces=NS):
                                out.extend(para_to_lines(p))
                                out.extend(textboxes_in(p))
                                out.extend(altchunk_texts(p))
                elif lname == "sdt":
                    sdtc = child.find(".//w:sdtContent", namespaces=NS)
                    if sdtc is not None:
                        traverse(sdtc)
                elif lname in ("sectPr",):
                    continue
                else:
                    continue

        traverse(body)

        # === footnotes/endnotes (append after body) ===
        for name in ("word/footnotes.xml", "word/endnotes.xml"):
            try:
                notes_root = parse_xml_bytes(zf.read(name))
            except KeyError:
                continue
            for p in notes_root.findall(".//w:p", namespaces=NS):
                out.extend(para_to_lines(p))

    # final cleanup (keep non-empty)
    return [s for s in (clean(x) for x in out) if s]


#useless code for a pdf file even though there are none in the documents
def extract_plain_text_pdf(path):
    from pypdf import PdfReader
    reader = PdfReader(path)
    lines = []
    for page in reader.pages:
        text = page.extract_text() or ""
        for line in text.split("\n"):
            line = line.strip()
            if line:
                lines.append(line)
    return lines

def extract_text_xml(path, xpath_override=None):
    """
    Returns a list of Bulgarian reference lines.
    - XLIFF (1.x / 2.0): returns all <target> texts in document order
    - TMX: returns BG <seg> for TUs; falls back to EN if BG missing (rare)
    - Generic XML: returns all text nodes, or XPath-selected nodes if xpath_override is set
    """
    from lxml import etree
    from pathlib import Path

    parser = etree.XMLParser(recover=True)
    root = etree.parse(str(Path(path)), parser=parser).getroot()
    ns = {k if k else "ns": v for k, v in (root.nsmap or {}).items() if v}
    lname = etree.QName(root).localname.lower()

    # XLIFF
    if lname == "xliff":
        targets = root.xpath(".//*[local-name()='target']", namespaces=ns)
        lines = []
        for t in targets:
            txt = " ".join(t.itertext()).strip()
            if txt:
                lines.append(txt)
        return lines

    # TMX
    if lname == "tmx":
        tus = root.xpath(".//*[local-name()='tu']", namespaces=ns)
        lines = []
        for tu in tus:
            # Bulgarian seg
            bg = tu.xpath(".//*[local-name()='tuv' and (translate(@lang,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')='bg' or translate(@xml:lang,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')='bg' or contains(translate(@lang,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'bg-') or contains(translate(@xml:lang,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'bg-'))]/*[local-name()='seg']", namespaces=ns)
            if bg:
                txt = " ".join(bg[0].itertext()).strip()
                if txt:
                    lines.append(txt)
                continue
            # fallback: English seg (avoid empty)
            en = tu.xpath(".//*[local-name()='tuv' and (translate(@lang,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')='en' or translate(@xml:lang,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')='en' or contains(translate(@lang,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'en-') or contains(translate(@xml:lang,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'en-'))]/*[local-name()='seg']", namespaces=ns)
            if en:
                txt = " ".join(en[0].itertext()).strip()
                if txt:
                    lines.append(txt)
        return lines

    # Generic XML
    if xpath_override:
        nodes = root.xpath(xpath_override, namespaces=ns)
        texts = []
        for n in nodes:
            if isinstance(n, str):
                t = n.strip()
            else:
                t = " ".join(n.itertext()).strip()
            if t:
                texts.append(t)
        return texts

    texts = []
    for el in root.iter():
        t = (el.text or "").strip()
        if t:
            texts.append(t)
        tail = (el.tail or "").strip()
        if tail:
            texts.append(tail)
    return [s for s in texts if len(s) > 1]

#etracts the text, then checks for format
def extract_text_auto(path, xpath_override=None):
    ext = Path(path).suffix.lower()
    if ext == ".docx":
        return extract_plain_text_docx(path)
    elif ext == ".pdf":
        return extract_plain_text_pdf(path)
    elif ext == ".xml":
        return extract_text_xml(path, xpath_override=xpath_override)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .docx, .pdf, or .xml")


# ------------- Glossary / BG normalization -------------
#glossary was something i was testing in the earlier itterations of my projection, as of now i dont use it,
#its supposed to be a dictionary of terms to replace in the output, but i dont use it as of now
def load_glossary(path):
    pairs = []
    if not Path(path).exists():
        return pairs
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            en = row["en"].strip().lower()
            bg = row["bg"].strip()
            if en and bg:
                pairs.append((en, bg))
    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    return pairs

def enforce_glossary(bg_text, glossary_pairs):
    # Replace English phrases that leaked into BG output with the BG term
    s = bg_text
    for en, bg in glossary_pairs:
        pattern = re.compile(rf"(?i)\b{re.escape(en)}\b")
        s = pattern.sub(bg, s)
    return " ".join(s.split())

BG_NORMALIZE = [
    (re.compile(r"(?i)\bконтейнер(ът|а|и|ите)?\b"), "защитна херметична конструкция (контаймент)"),
    (re.compile(r"(?i)\bядрото на реактора\b"), "активната зона на реактора"),
    (re.compile(r"(?i)\bпремахване на остатъчната топлина\b"), "отвеждане на остатъчната топлина"),
    (re.compile(r"(?i)\bотбрана в дълбочина\b"), "дълбоко ешелонирана защита (ДЕЗ)"),
]
def normalize_bg_variants(text):
    s = text
    for patt, repl in BG_NORMALIZE:
        s = patt.sub(repl, s)
    return " ".join(s.split())

# ------------- Anchors + embedding alignment -------------
REQ_EN = re.compile(r"\bRequirement\s+(\d+)\b", flags=re.IGNORECASE)
REQ_BG = re.compile(r"\bИзискване\s+(\d+)\b", flags=re.IGNORECASE)

def anchor_map(lines, regex):
    m = {}
    for idx, s in enumerate(lines):
        mo = regex.search(s)
        if mo:
            n = int(mo.group(1))
            m.setdefault(n, []).append(idx)
    return m

def windowed_embed_align(src_chunk, ref_chunk, win=6, thr=0.5, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    from sentence_transformers import SentenceTransformer
    import numpy as np
    src = normalize_lines(src_chunk)
    ref = normalize_lines(ref_chunk)
    if not src or not ref:
        return []
    st = SentenceTransformer(model_name)
    e_src = st.encode(src, convert_to_numpy=True, normalize_embeddings=True)
    e_ref = st.encode(ref, convert_to_numpy=True, normalize_embeddings=True)
    aligned = []
    j = 0
    for i, v in enumerate(pbar(e_src, total=len(e_src), desc="DP align", unit="lines")):
        start = j
        end = min(len(e_ref), j + win)
        if start >= end:
            aligned.append((i, None, src[i], ""))
            continue
        sims = e_ref[start:end] @ v
        k = int(sims.argmax())
        best_j = start + k
        sim = float(sims[k])
        if sim >= thr:
            aligned.append((i, best_j, src[i], ref[best_j]))
            j = best_j + 1
        else:
            aligned.append((i, None, src[i], ""))
    return aligned

# ------------- sacreBLEU metrics -------------
def score_bleu_chrf(sys_lines, ref_lines):
    try:
        import sacrebleu
    except ImportError:
        return None, None
    if not sys_lines or not ref_lines:
        return None, None
    m = min(len(sys_lines), len(ref_lines))
    if m < 10:
        return None, None
    sys_small = [s.lower().strip() for s in sys_lines[:m]]
    ref_small = [[r.lower().strip() for r in ref_lines[:m]]]
    bleu = sacrebleu.corpus_bleu(sys_small, ref_small).score
    chrf = sacrebleu.corpus_chrf(sys_small, ref_small).score
    return bleu, chrf

# ------------- Model adapters -------------
def detect_family(model_id):
    mid = model_id.lower()
    if "m2m100" in mid:
        return "m2m"
    if "nllb" in mid:
        return "nllb"
    if "helsinki-nlp/opus-mt-" in mid or "opus-mt-" in mid or "marian" in mid:
        return "marian"
    if any(x in mid for x in ["bggpt", "qwen", "llama", "mistral", "mixtral", "gemma", "phi-3", "xglm"]):
        return "causal_llm"

    return "auto"  # try M2M -> Marian -> NLLB in that order

def translator_for(model_id, device=None, lora_adapter=None):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    fam = detect_family(model_id)
    if fam == "m2m":
        from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration
        tok = M2M100Tokenizer.from_pretrained(model_id)
        model = M2M100ForConditionalGeneration.from_pretrained(model_id)
        return ("m2m", tok, model, device)
    elif fam == "nllb":
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        return ("nllb", tok, model, device)
    elif fam == "causal_llm":
        tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id

        device = None
        model = None
        try:
            # 4-bit if available
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",                 # accelerate handles placement
                quantization_config=bnb,
            )
            # when using device_map="auto", pick a device just for tensors:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        except Exception:
            # Fallback: no 4-bit, place model on a single device
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto").to(device)

        if lora_adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_adapter)
            model = model.to(device)

        return ("causal_llm", tok, model, device)



    else:  # marian default
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        return ("marian", tok, model, device)

def translate_batch(model_id, fam, tok, model, lines, device=None, batch_size=16):
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    out = []

    if fam == "m2m":
        # M2M-100: use forced BOS for bg + declare source language
        tok.src_lang = "en"
        bos = tok.get_lang_id("bg")
        pbar = tqdm(total=len(lines), desc=f"Translating ({model_id})", unit="line", dynamic_ncols=True, leave=False)
        for i in range(0, len(lines), batch_size):
            batch = lines[i:i+batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            gen = model.generate(**enc, forced_bos_token_id=bos)
            dec = tok.batch_decode(gen, skip_special_tokens=True)
            out.extend([d.strip() for d in dec])
            pbar.update(len(batch))
        pbar.close()

    elif fam == "nllb":
        # NLLB-200: FLORES codes eng_Latn -> bul_Cyrl
        # Some tokenizer variants lack `lang_code_to_id`. Use a robust fallback.
        try:
            bos = tok.lang_code_to_id.get("bul_Cyrl")  # may not exist on Fast tokenizers
        except Exception:
            bos = None
        if bos is None:
            bos = tok.convert_tokens_to_ids("bul_Cyrl")  # safe fallback that always works

        # Set the source language (important for NLLB encoders)
        if hasattr(tok, "src_lang"):
            tok.src_lang = "eng_Latn"

        pbar = tqdm(total=len(lines), desc=f"Translating ({model_id})", unit="line", dynamic_ncols=True, leave=False)
        for i in range(0, len(lines), batch_size):
            batch = lines[i:i+batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            gen = model.generate(**enc, forced_bos_token_id=bos)
            dec = tok.batch_decode(gen, skip_special_tokens=True)
            out.extend([d.strip() for d in dec])
            pbar.update(len(batch))
        pbar.close()

    elif fam == "causal_llm":
        # Build a strict prompt so the model outputs ONLY BG text
        def build_prompt(text: str) -> str:
            is_bggpt = "bggpt" in model_id.lower()
            if is_bggpt:
                # Gemma2/BgGPT: no system role; user-only instruction
                if hasattr(tok, "apply_chat_template"):
                    messages = [
                        {"role": "user", "content": f"You are not a chatbot. You are a translation API. Task: translate from English to Bulgarian (bg-BG, Cyrillic). Output only the Bulgarian translation, starting immediately. No explanations, notes, or extra punctuation. Only Bulgarian text, digits, and punctuation from the input are allowed. English: {text} Bulgarian: "},
                    ]
                    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if hasattr(tok, "apply_chat_template") and not is_bggpt:
                messages = [
                    {"role": "system", "content": "You are a professional technical translator. Translate English to Bulgarian. Output ONLY the translation."},
                    {"role": "user", "content": text},
                ]
                return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return (
                "You are a professional technical translator. "
                "Translate the following English text into Bulgarian. "
                "Output ONLY the translation, no explanations, no quotes.\n"
                f"English: {text}\nBulgarian:"
            )

        # Ensure correct padding for decoder-only models
        tok.padding_side = "left"
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id

        out = []
        pbar = tqdm(total=len(lines), desc=f"Translating ({model_id})", unit="line", dynamic_ncols=True, leave=False)
        for i in range(0, len(lines), batch_size):
            prompts = [build_prompt(s) for s in lines[i:i+batch_size]]

            enc = tok(
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024
            ).to(device)

            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=256,          # allow long paragraphs; drop to 128 if RAM is tight
                    temperature=0.0,
                    do_sample=False,
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.pad_token_id,
                    return_dict_in_generate=True  # so we can slice by token length
                )

            seqs = gen.sequences  # [batch, input_len + output_len]
            in_len = enc.input_ids.shape[1]
            for seq in seqs:
                gen_ids = seq[in_len:]  # ← strip the prompt by token length (no prompt bleed)
                text = tok.decode(gen_ids, skip_special_tokens=True).strip()
                # Defensive: remove any leftover role tag text
                text = re.sub(r"^(?:assistant\s*:?\s*)", "", text, flags=re.I).strip('“”"')
                out.append(text)
            pbar.update(len(prompts))

        pbar.close()
        return out



    else:
        # Marian / OPUS-MT (and most other seq2seq models)
        # Many OPUS models expect the target tag >>bul<< at the start.
        needs_tag = "helsinki-nlp/opus-mt-" in model_id.lower()
        pbar = tqdm(total=len(lines), desc=f"Translating ({model_id})", unit="line", dynamic_ncols=True, leave=False)
        for i in range(0, len(lines), batch_size):
            batch = lines[i:i+batch_size]
            if needs_tag:
                batch = [">>bul<< " + s for s in batch]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            gen = model.generate(**enc)
            dec = tok.batch_decode(gen, skip_special_tokens=True)
            out.extend([d.strip() for d in dec])
            pbar.update(len(batch))
        pbar.close()

    return out

def dp_align(sys_lines, ref_lines, sim_thr=0.50, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """
    Needleman–Wunsch style global alignment over sentence embeddings.
    Returns list of pairs (i, j) with i=index in sys_lines, j=index in ref_lines (or None for gap).
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np

    a = [s.strip() for s in sys_lines]
    b = [t.strip() for t in ref_lines]
    if not a or not b:
        return []

    st = SentenceTransformer(model_name)
    Ea = st.encode(a, normalize_embeddings=True)
    Eb = st.encode(b, normalize_embeddings=True)
    S = Ea @ Eb.T  # cosine similarity matrix: shape [len(a), len(b)]

    n, m = S.shape
    gap = -0.40  # gap penalty (tune  -0.30 .. -0.60)

    dp  = np.zeros((n+1, m+1), dtype=np.float32)
    ptr = np.zeros((n+1, m+1), dtype=np.int8)  # 1=diag, 2=up (gap in ref), 3=left (gap in sys)

    for i in range(1, n+1):
        dp[i,0] = dp[i-1,0] + gap
        ptr[i,0] = 2
    for j in range(1, m+1):
        dp[0,j] = dp[0,j-1] + gap
        ptr[0,j] = 3


    for i in tqdm(range(1, n+1), desc="DP fill (rows)", unit="row", dynamic_ncols=True, leave=False):
        Si = S[i-1]
        for j in range(1, m+1):
            match = dp[i-1,j-1] + Si[j-1]
            up    = dp[i-1,j]   + gap
            left  = dp[i,  j-1] + gap
            # pick best
            if match >= up and match >= left:
                dp[i,j] = match; ptr[i,j] = 1
            elif up >= left:
                dp[i,j] = up;    ptr[i,j] = 2
            else:
                dp[i,j] = left;  ptr[i,j] = 3

    # backtrace
    i, j = n, m
    pairs = []
    while i > 0 or j > 0:
        move = ptr[i,j]
        if move == 1:
            pairs.append((i-1, j-1)); i -= 1; j -= 1
        elif move == 2:
            pairs.append((i-1, None)); i -= 1
        else:
            pairs.append((None, j-1)); j -= 1
    pairs.reverse()

    # filter weak matches: if similarity < sim_thr, treat as gap on ref side
    aligned = []
    for pi, pj in pairs:
        if pi is not None and pj is not None:
            if S[pi, pj] >= sim_thr:
                aligned.append((pi, pj))
            else:
                aligned.append((pi, None))
        elif pi is not None:
            aligned.append((pi, None))
        # rows that are only in ref are not needed for EN-driven table
    return aligned

# ------------- Main -------------
#all of the arguments you can pass to the script. use a chatbot for that since it is very annoying writing it normally
def main():
    ap = argparse.ArgumentParser(description="Batch-evaluate multiple MT models with anchored alignment and sacreBLEU/chrF.")
    ap.add_argument("--src", required=True, help="English DOCX/PDF")
    ap.add_argument("--ref", required=True, help="Bulgarian reference DOCX/PDF")
    ap.add_argument("--models", required=True, help="Comma-separated HF model IDs")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--glossary", default="glossary.csv")
    ap.add_argument("--win", type=int, default=8)
    ap.add_argument("--thr", type=float, default=0.55)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--export_disagreements", action="store_true")
    ap.add_argument("--lora_adapter", default="",
                help="Path to a PEFT LoRA adapter to load onto causal LLMs (e.g. adapters/qwen2_5_05b_bg_lora_test)")
    ap.add_argument("--no_glossary", action="store_true",
                help="Do NOT enforce glossary replacements")
    ap.add_argument("--no_bg_normalize", action="store_true",
                help="Do NOT apply Bulgarian normalization rules")
    ap.add_argument("--clean_outdir", action="store_true",
                help="Delete existing files in the output directory before writing new results")
    ap.add_argument("--ref_xpath", default="", help="XPath to select text from XML refs (e.g. //seg or //p)")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.clean_outdir:
        for f in outdir.iterdir():
            if f.is_file():
                f.unlink()
        print(f"🧹 Cleaned old files from {outdir}")

    # Extract + clean
    eng_raw = extract_text_auto(args.src)
    bul_raw = extract_text_auto(args.ref, xpath_override=(args.ref_xpath or None))

    dbg_dir = Path(args.outdir)
    (dbg_dir / "debug").mkdir(parents=True, exist_ok=True)
    # Raw (exactly what the extractor returned, before cleaning)
    dbg_en_raw = dbg_dir / "debug" / "dbg_en_raw.txt"
    dbg_bg_raw = dbg_dir / "debug" / "dbg_bg_raw.txt"
    with open(dbg_en_raw, "w", encoding="utf-8-sig") as f:
        for i, s in enumerate(tqdm(eng_raw, desc="Dump EN (raw)", unit="line", dynamic_ncols=True, leave=False)):
            f.write(f"{i:05d} | {s}\n")
    with open(dbg_bg_raw, "w", encoding="utf-8-sig") as f:
        for i, s in enumerate(tqdm(bul_raw, desc="Dump BG (raw)", unit="line", dynamic_ncols=True, leave=False)):
            f.write(f"{i:05d} | {s}\n")

    eng = normalize_lines(drop_furniture(eng_raw))
    bul = normalize_lines(drop_furniture(bul_raw))

    # We will align the FULL documents end-to-end for now
    src_en = eng      # English lines to translate
    ref_bg = bul      # Bulgarian reference lines

    print(f"DP input sizes: EN={len(src_en)} | BG_REF={len(ref_bg)}")

    # Cleaned (after drop_furniture + normalize_lines)
    dbg_en_clean = dbg_dir / "debug" / "dbg_en_clean.txt"
    dbg_bg_clean = dbg_dir / "debug" / "dbg_bg_clean.txt"
    with open(dbg_en_clean, "w", encoding="utf-8-sig") as f:
        for i, s in enumerate(tqdm(eng, desc="Dump EN (clean)", unit="line", dynamic_ncols=True, leave=False)):
            f.write(f"{i:05d} | {s}\n")
    with open(dbg_bg_clean, "w", encoding="utf-8-sig") as f:
        for i, s in enumerate(tqdm(bul, desc="Dump BG (clean)", unit="line", dynamic_ncols=True, leave=False)):
            f.write(f"{i:05d} | {s}\n")

    print(f"Cleaned lines: EN={len(eng)}  BG_REF={len(bul)}")
    print("✅ Wrote debug dumps:",
      dbg_en_raw, dbg_bg_raw, dbg_en_clean, dbg_bg_clean)
    
    def drop_sections_by_markers(lines, start_markers, stop_markers=None):
        keep = []
        skip = False
        for s in lines:
            if any(m in s for m in start_markers):
                skip = True
            if not skip:
                keep.append(s)
            if stop_markers and any(m in s for m in stop_markers):
                skip = False
        return keep

    # Example markers — adjust to your documents
    toc_markers_en = ["TABLE OF CONTENTS"]
    acro_markers_en = ["LIST OF ACRONYMS", "LIST OF ACRONYMS AND ABBREVIATIONS"]

    toc_markers_bg = ["СЪДЪРЖАНИЕ"]
    acro_markers_bg = ["СПИСЪК НА АКРОНИМИТЕ", "СПИСЪК НА АКРОНИМИТЕ И СЪКРАЩЕНИЯТА"]

    # Create filtered versions for SCORING ONLY
    eng_for_scoring = drop_sections_by_markers(eng, toc_markers_en)
    eng_for_scoring = drop_sections_by_markers(eng_for_scoring, acro_markers_en)

    bul_for_scoring = drop_sections_by_markers(bul, toc_markers_bg)
    bul_for_scoring = drop_sections_by_markers(bul_for_scoring, acro_markers_bg)

    # Anchors in both
    en_anchor = anchor_map(eng_for_scoring, REQ_EN)
    bg_anchor = anchor_map(bul_for_scoring, REQ_BG)
    anchor_numbers = sorted(set(en_anchor.keys()) & set(bg_anchor.keys()))
    if not anchor_numbers:
        print("Warning: no requirement-number anchors found in both docs. Alignment may be noisier.", file=sys.stderr)

    glossary_pairs = load_glossary(args.glossary)

    summary_rows = []
    for model_id in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== Evaluating model: {model_id} ===")
        fam, tok, model, device = translator_for(model_id, device=None, lora_adapter=args.lora_adapter)
        if fam == "skip":
            continue


        # Translate all EN (for scoring only)
        sys_bg = translate_batch(model_id, fam, tok, model, src_en, batch_size=args.batch)
        print(f"Translated {len(sys_bg)} lines.")
        # Apply glossary + BG normalization

        if glossary_pairs and not args.no_glossary:
            sys_bg = [enforce_glossary(t, glossary_pairs) for t in sys_bg]

        if not args.no_bg_normalize:
            sys_bg = [normalize_bg_variants(t) for t in sys_bg]


        # Save system.out
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_id).strip("_").lower()
        sys_out_path = Path(args.outdir) / f"system_{slug}.out"
        with open(sys_out_path, "w", encoding="utf-8-sig") as f:
            for t in sys_bg:
                f.write(t + "\n")
        print(f"  -> wrote {sys_out_path}")

        # ---------- Global DP alignment over the whole doc ----------
        # We align the model output (sys_bg) to the reference BG (ref_bg) using a global
        # dynamic-programming aligner so we get ~one row per EN line with explicit gaps.
        pairs = dp_align(sys_bg, ref_bg, sim_thr=args.thr)

        rows = []
        for i, j in tqdm(pairs, desc="Materialize DP rows", unit="row", dynamic_ncols=True, leave=False):
            en_txt  = src_en[i]                  # NOTE: src_en, not eng
            sys_txt = sys_bg[i]
            ref_txt = ref_bg[j] if j is not None else ""   # gap on BG side if no good match
            rows.append([en_txt, sys_txt, ref_txt, i, ("" if j is None else j), "dp"])

        compare_path = Path(args.outdir) / f"compare_anchored_{slug}.csv"
        cols = ["English", "System BG", "Reference BG", "en_idx", "bg_idx", "segment"]
        save_table_csv_and_xlsx(str(compare_path), cols, rows)
        print(f"  -> DP rows: {len(rows)} (should be close to number of EN lines)")


        print(f"  -> wrote {compare_path} (rows: {len(rows)})")

        # Metrics on aligned subset
        sys_subset, ref_subset = [], []
        for _, sys_txt, ref_txt, _, _, _ in rows:
            if ref_txt:
                sys_subset.append(sys_txt)
                ref_subset.append(ref_txt)
        bleu, chrf = score_bleu_chrf(sys_subset, ref_subset)
        n_pairs = len(sys_subset)
        print(f"  -> aligned subset: n={n_pairs}, BLEU={bleu if bleu is not None else 'NA'}, chrF={chrf if chrf is not None else 'NA'}")

        # Optional: disagreements export
        if args.export_disagreements:
            dis_path = Path(args.outdir) / f"disagreements_{slug}.csv"
            cols = ["English", "System BG", "Reference BG", "en_idx", "bg_idx", "segment"]
            save_table_csv_and_xlsx(str(dis_path), cols, rows)


        summary_rows.append({
            "model": model_id,
            "family": fam,
            "aligned_pairs": n_pairs,
            "BLEU": None if bleu is None else round(bleu, 2),
            "chrF": None if chrf is None else round(chrf, 2),
            "system_out": str(sys_out_path),
            "compare_csv": str(compare_path),
            "disagreements_csv": str(Path(args.outdir) / f"disagreements_{slug}.csv") if args.export_disagreements else ""
        })

    # summary.csv
    # summary.csv / .xlsx
    summary_path = Path(args.outdir) / "summary.csv"
    if summary_rows:
        cols = ["model", "family", "aligned_pairs", "BLEU", "chrF", "system_out", "compare_csv", "disagreements_csv"]
        rows = [
            [r["model"], r["family"], r["aligned_pairs"], r["BLEU"], r["chrF"], r["system_out"], r["compare_csv"], r["disagreements_csv"]]
            for r in summary_rows
        ]
        save_table_csv_and_xlsx(str(summary_path), cols, rows)
        print(f"== Summary written: {summary_path} ==")


if __name__ == "__main__":
    main()
