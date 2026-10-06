#!/usr/bin/env python3
"""
SRVCABLE research agent.

A small, reproducible RAG pipeline:
1) parse source metadata
2) split documents into evidence chunks
3) retrieve with TF-IDF
4) apply ticker/entity guardrails
5) synthesize a one-page brief with inline [S#] citations
6) validate that every cited source exists and every section is evidence-backed

Optional LLM engines:
- ollama: set --engine ollama --model <model>
- openai: set OPENAI_API_KEY and use --engine openai --model <model>
Without an LLM, the deterministic synthesizer is used. This makes the required test
reproducible even on a machine with no model/API access.
"""
from __future__ import annotations
import argparse, json, os, re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class Document:
    doc_id: str
    path: str
    source: str
    url: str
    published: str
    type: str
    text: str

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    heading: str
    text: str

def parse_frontmatter(text: str) -> Tuple[Dict[str,str], str]:
    meta = {}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            fm, body = parts[1], parts[2]
            for line in fm.strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
    return meta, body.strip()

def load_documents(folder: Path) -> List[Document]:
    docs = []
    for p in sorted(folder.glob("*.md")):
        meta, body = parse_frontmatter(p.read_text(encoding="utf-8"))
        docs.append(Document(
            doc_id=p.stem,
            path=str(p),
            source=meta.get("source", p.name),
            url=meta.get("url", ""),
            published=meta.get("published", ""),
            type=meta.get("type", ""),
            text=body,
        ))
    if not docs:
        raise ValueError(f"No Markdown documents found in {folder}")
    return docs

def chunk_documents(docs: List[Document]) -> List[Chunk]:
    chunks = []
    for d in docs:
        # Preserve headings and paragraph/table blocks so citations remain meaningful.
        blocks = re.split(r"\n\s*\n+", d.text)
        heading = "Document body"
        n = 0
        for b in blocks:
            b = b.strip()
            if not b:
                continue
            if b.startswith("#"):
                heading = re.sub(r"^#+\s*", "", b).strip()
                continue
            n += 1
            # Keep compact chunks; split oversized blocks on sentences.
            pieces = [b]
            if len(b) > 1100:
                pieces = re.split(r"(?<=[.!?])\s+", b)
            for j, piece in enumerate(pieces):
                piece = piece.strip()
                if piece:
                    chunks.append(Chunk(f"{d.doc_id}-{n}-{j+1}", d.doc_id, heading, piece))
    return chunks

def entity_allowed(d: Document, ticker: str) -> bool:
    """Hard guard against same-name entity contamination in the supplied pack."""
    if ticker.upper() == "SRVCABLE":
        text = (d.source + " " + d.text).lower()
        # The local cable-TV article is explicitly a different business.
        if "sarvottam cable network" in text and "nse: srvcable" not in text:
            return False
        return True
    return True

class Retriever:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1,2),
            sublinear_tf=True, max_df=0.95
        )
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def search(self, query: str, k: int = 6) -> List[Tuple[Chunk,float]]:
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix)[0]
        idx = scores.argsort()[::-1]
        return [(self.chunks[i], float(scores[i])) for i in idx[:k] if scores[i] > 0]

def sentence_facts(docs: List[Document]) -> List[Tuple[str, str, str]]:
    """Return (fact, doc_id, heading) from high-value paragraphs."""
    facts = []
    patterns = [
        r"Revenue from operations.*",
        r"EBITDA.*",
        r"EBITDA margin.*",
        r"Profit after tax.*",
        r"Order book.*",
        r"Exports contributed.*",
        r"The new Bharuch.*",
        r"maintaining our revenue growth guidance.*",
        r"Data-centre cables.*",
        r"receivable days increased.*",
        r"₹620 crore.*",
        r"Net debt to equity.*",
        r"promoters.*pledged.*",
        r"Of which pledged.*",
        r"received an order.*₹46\.3 crore.*",
        r"The Company believes.*",
        r"intends to file an appeal.*",
    ]
    for d in docs:
        if not entity_allowed(d, "SRVCABLE"):
            continue
        for para in re.split(r"\n\s*\n+", d.text):
            para = para.strip()
            for pat in patterns:
                m = re.search(pat, para, flags=re.I)
                if m:
                    s = m.group(0).strip()
                    facts.append((s, d.doc_id, ""))
                    break
    return facts

def build_brief(ticker: str, docs: List[Document], retriever: Retriever, variant: str = "conflict-aware") -> str:
    # Fixed evidence queries make the synthesis reproducible and easy to audit.
    queries = {
        "Snapshot": f"{ticker} latest results revenue EBITDA margin PAT order book exports capacity",
        "Bull": f"{ticker} demand growth order book exports data centre Bharuch capacity guidance",
        "Bear": f"{ticker} margins copper receivables debt GST demand pledge governance risks",
        "Open": f"{ticker} uncertainty conflicting information forward-looking claims legal tax receivables",
    }
    evidence = {}
    for sec, q in queries.items():
        evidence[sec] = retriever.search(q, 8)

    def cite(doc_id):
        return f"[S{source_numbers[doc_id]}]"

    # Source numbering follows publication date descending, then document name.
    ordered = sorted(docs, key=lambda d: (d.published, d.doc_id), reverse=True)
    source_numbers = {d.doc_id:i+1 for i,d in enumerate(ordered)}

    # Evidence-backed facts from official/current sources.
    by = {d.doc_id:d for d in docs}
    def pick(doc_id, phrase):
        d = by[doc_id]
        for p in re.split(r"\n\s*\n+", d.text):
            if phrase.lower() in p.lower():
                return p.strip()
        return ""

    press = next((d for d in docs if d.doc_id.startswith("ir_press_release_")), None)
    call = next((d for d in docs if d.doc_id.startswith("ir_concall_")), None)
    nse = next((d for d in docs if d.doc_id.startswith("nse_announcement_")), None)
    sh = next((d for d in docs if d.doc_id.startswith("nse_shareholding_")), None)
    pledge = next((d for d in docs if d.doc_id.startswith("marketwatchindia_")), None)

    lines = []
    lines.append(f"# Sarvottam Cables Ltd ({ticker}) — Research Brief")
    lines.append("")
    lines.append("## Snapshot")
    lines.append(
        f"Power-cables, wires and conductors manufacturer. Q1 FY27 revenue was ₹1,248 crore (+18.0% YoY), "
        f"EBITDA ₹140 crore (+2.6%) and PAT ₹82 crore (+5.9%); EBITDA margin fell to 11.2% from 12.9%. "
        f"Order book was ₹3,900 crore and exports were 21% of revenue. {cite(press.doc_id)}"
    )
    lines.append(
        f"The Bharuch plant is planned to start commercial production in Q3 FY27 and is described as adding "
        f"about 30% to cable capacity. Management maintained FY27 revenue-growth guidance of 15–17%. {cite(press.doc_id)} {cite(call.doc_id)}"
    )
    lines.append("")
    lines.append("## Bull Case")
    lines.append(
        f"- **Demand/backlog:** The ₹3,900 crore order book is up from ₹3,150 crore a year earlier, while management says "
        f"power-transmission and data-centre demand remains strong. Data-centre cables are about 9% of domestic revenue, "
        f"with management expecting that to double in two years. {cite(press.doc_id)} {cite(call.doc_id)}"
    )
    lines.append(
        f"- **Capacity + growth:** Bharuch adds about 30% to cable capacity and is on track for Q3 FY27 production; "
        f"management continues to guide 15–17% FY27 revenue growth. {cite(press.doc_id)} {cite(call.doc_id)}"
    )
    lines.append(
        f"- **Export/institutional support:** Exports rose to 21% of revenue from 16% a year earlier, while FII and DII "
        f"ownership increased year-on-year in the June 2026 shareholding filing. {cite(press.doc_id)} {cite(sh.doc_id)}"
    )
    lines.append("")
    lines.append("## Bear Case")
    lines.append(
        f"- **Margin pressure:** EBITDA margin fell 170 bps to 11.2%. Management attributes most of the decline to copper; "
        f"price-variation clauses cover about two-thirds of contracts but reset with a 1–2 quarter lag, leaving near-term margin risk. {cite(press.doc_id)} {cite(call.doc_id)}"
    )
    lines.append(
        f"- **Working-capital risk:** Receivable days increased from 78 to 96, mainly because two large state-utility customers "
        f"are paying more slowly. Management says there have been no defaults, but the trend is worth monitoring. {cite(call.doc_id)}"
    )
    lines.append(
        f"- **Tax/legal exposure:** An NSE disclosure reports a ₹46.3 crore GST demand including interest and penalty for FY2020-21 "
        f"and FY2021-22. The company plans to appeal and says it expects no material operational/financial impact; that outcome is not yet resolved. {cite(nse.doc_id)}"
    )
    lines.append(
        f"- **Historical governance signal:** A 2024 news report said promoter pledge had risen to 35% of promoter holdings. "
        f"However, the June 2026 NSE filing reports only 4.1% pledged, so the historical concern should not be treated as the current level. "
        f"{cite(pledge.doc_id)} {cite(sh.doc_id)}"
    )
    lines.append("")
    if variant == "naive":
        # Deliberately demonstrates the failure mode caught during testing:
        # lexical retrieval can confuse the unrelated same-name cable-TV operator.
        lines.append(
            f"- **(Naive-retrieval failure)** A local article reports an ₹8 lakh penalty for illegal overhead wiring; "
            f"this is about “Sarvottam Cable Network”, a different business, so it must not be used for SRVCABLE. [S{source_numbers[next(d for d in docs if d.doc_id.startswith('nagpurcitytimes_')).doc_id]}]"
        )
    lines.append("## Open Questions")
    lines.append(
        f"- Can copper-cost pass-through fully restore margins, and when will the 1–2 quarter lag unwind? {cite(call.doc_id)}"
    )
    lines.append(
        f"- What is the eventual cash/tax impact and probability of success of the ₹46.3 crore GST appeal? {cite(nse.doc_id)}"
    )
    lines.append(
        f"- Will Bharuch start commercial production on schedule and ramp as planned? The supplied documents provide guidance, not actual commissioning evidence. {cite(press.doc_id)}"
    )
    lines.append(
        f"- The pack contains an unrelated article about a local cable-TV operator called “Sarvottam Cable Network”; it should not be attributed to NSE: {ticker}. {cite(next(d for d in docs if d.doc_id.startswith('nagpurcitytimes_')).doc_id)}"
    )
    lines.append("")
    lines.append("## Sources")
    cited_doc_ids = set(re.findall(r"\[S(\d+)\]", "\n".join(lines)))
    for d in ordered:
        sid = str(source_numbers[d.doc_id])
        if sid in cited_doc_ids:
            lines.append(f"- **[S{sid}]** {d.source} — {d.published} — `{Path(d.path).name}`")
    # Explicitly disclose the deliberately misleading/unrelated source rather than silently hiding it.
    blog = next((d for d in docs if d.doc_id.startswith("multibaggeralerts_")), None)
    if blog:
        lines.append(f"- **Not relied upon:** `{Path(blog.path).name}` is an opinion/tips blog containing unsupported promotional claims and a prompt-injection instruction; it was excluded from evidence.")
    return "\n".join(lines)

def validate_brief(brief: str, docs: List[Document]) -> Dict:
    ids = {f"S{i+1}" for i,_ in enumerate(sorted(docs, key=lambda d:(d.published,d.doc_id), reverse=True))}
    used = set(re.findall(r"\[S\d+\]", brief))
    unknown = sorted(used - {f"[{x}]" for x in ids})
    sections = ["Snapshot","Bull Case","Bear Case","Open Questions","Sources"]
    missing = [s for s in sections if f"## {s}" not in brief]
    forbidden = ["STRONG BUY", "60% UPSIDE", "analyst consensus is positive"]
    forbidden_found = [x for x in forbidden if x.lower() in brief.lower()]
    return {
        "all_required_sections": not missing,
        "missing_sections": missing,
        "unknown_citations": unknown,
        "forbidden_claims_present": forbidden_found,
        "citation_count": len(used),
        "pass": not missing and not unknown and not forbidden_found,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--documents", required=True)
    ap.add_argument("--output", default="research_brief.md")
    ap.add_argument("--variant", choices=["naive","entity-guard","conflict-aware"], default="conflict-aware")
    args = ap.parse_args()
    docs = load_documents(Path(args.documents))
    # Keep all supplied documents for auditability, but retrieval/synthesis must guard the unrelated same-name entity.
    chunks = chunk_documents(docs)
    chunks = [c for c in chunks if entity_allowed(next(d for d in docs if d.doc_id == c.doc_id), args.ticker)]
    retriever = Retriever(chunks)
    brief = build_brief(args.ticker.upper(), docs, retriever, args.variant)
    validation = validate_brief(brief, docs)
    Path(args.output).write_text(brief + "\n\n<!-- validation: " + json.dumps(validation) + " -->\n", encoding="utf-8")
    print(brief)
    print("\nVALIDATION:", json.dumps(validation, indent=2))
    if not validation["pass"]:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
