from __future__ import annotations

import base64
import cgi
import datetime as dt
import hashlib
import html
import json
import mimetypes
import os
import re
import shutil
import sqlite3
import ssl
import sys
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from pipeline_trace import StepTracer, TraceStep, calculate_model_cost_usd

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


ROOT = Path(__file__).resolve().parent
CASE_STUDY_DIR = Path(os.environ.get("CASE_STUDY_DIR", ROOT / "case_study"))
DB_PATH = Path(os.environ.get("NORTHWIND_DB", ROOT / "northwind.sqlite"))
UPLOAD_DIR = Path(os.environ.get("NORTHWIND_UPLOADS", ROOT / "runtime_uploads"))


TIER_1 = {
    "new york", "san francisco", "boston", "washington", "los angeles",
    "seattle", "london", "zurich", "tokyo", "singapore",
}
TIER_2 = {
    "chicago", "denver", "atlanta", "austin", "dallas", "houston",
    "miami", "portland", "san diego", "toronto", "amsterdam", "berlin",
    "sydney",
}
MEAL_CAPS = {"breakfast": 25.0, "lunch": 35.0, "dinner": 75.0}
ALCOHOL_WORDS = {
    "beer", "wine", "hefeweizen", "ale", "cocktail", "vodka", "whiskey",
    "whisky", "tequila", "mezcal", "bourbon", "champagne", "martini",
    "pinot", "merlot", "cabernet", "lager", "stout", "ipa",
}
POLICY_SCOPE_TERMS = {
    "accommodation", "airfare", "airline", "alcohol", "allowance", "approval",
    "approve", "audit", "bag", "baggage", "breakfast", "business", "cap",
    "card", "claim", "client", "conference", "concur", "corporate", "dinner",
    "employee", "expense", "fare", "finance", "flight", "fuel", "gas", "grade",
    "hotel", "incidentals", "lodging", "lunch", "manager", "meal", "mileage",
    "parking", "per-diem", "policy", "receipt", "reimburs", "rental", "report",
    "review", "rideshare", "submit", "taxi", "training", "travel", "trip",
    "uber", "vendor", "wifi", "wi-fi", "lyft", "limit", "override",
}
POLICY_ID_PATTERN = re.compile(r"\b[A-Z]{2,5}-\d{3}\b", re.I)


def now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def json_dumps(data: Any) -> bytes:
    return json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")


def slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return value or "item"


def money(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"${value:,.2f}"


def normalize_text(text: str) -> str:
    text = text.replace("\u2014", "-").replace("\u2013", "-").replace("\u00a0", " ")
    return re.sub(r"[ \t]+", " ", text)


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9'-]{1,}", text.lower())
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "are", "was",
        "were", "into", "must", "may", "not", "all", "any", "per", "its",
        "has", "have", "when", "where", "what", "which", "who", "how",
        "does", "can", "should", "will", "about", "under", "over",
    }
    return [w for w in words if w not in stop]


def scope_token_forms(token: str) -> set[str]:
    forms = {token}
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            forms.add(token[: -len(suffix)])
    return forms


def policy_scope_trace(question: str) -> tuple[bool, list[str], dict[str, Any]]:
    started = time.perf_counter()
    doc_ids = [match.upper() for match in POLICY_ID_PATTERN.findall(question)]
    matches = set(doc_ids)
    for token in tokenize(question):
        forms = scope_token_forms(token)
        for form in forms:
            if form in POLICY_SCOPE_TERMS or any(term in form for term in POLICY_SCOPE_TERMS if len(term) >= 5):
                matches.add(form)
    ordered = sorted(matches)
    in_scope = bool(ordered)
    notes = (
        "in_scope: matched " + ", ".join(f"'{match}'" for match in ordered[:6])
        if in_scope
        else "out_of_scope: no policy-domain terms detected"
    )
    return in_scope, ordered, {
        "step_name": "scope_check",
        "model_used": "n/a",
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "cost_usd": 0.0,
        "status": "ok",
        "notes": notes,
    }


def direct_policy_answer(question: str, policy: "PolicyIndex") -> dict[str, Any] | None:
    tokens = set(tokenize(question))
    if "dinner" in tokens and ("expense" in tokens or "cap" in tokens or "limit" in tokens):
        citation = policy.clause("TEP-002", "2", "Meal expenses are subject to breakfast, lunch, and dinner caps.")
        return {
            "answer": f"{citation.doc_id} §{citation.section}: {citation.quote}",
            "confidence": 0.9,
            "citations": [citation_to_dict(citation)],
            "refused": False,
        }
    if "alcohol" in tokens:
        citation = policy.clause("TEP-003", "2", "Alcoholic beverages are reimbursable only during sanctioned client entertainment.")
        return {
            "answer": f"{citation.doc_id} §{citation.section}: {citation.quote}",
            "confidence": 0.9,
            "citations": [citation_to_dict(citation)],
            "refused": False,
        }
    if tokens & {"uber", "lyft", "rideshare", "taxi"}:
        citation = policy.clause("TEP-006", "2", "Rideshare and taxi rules govern Uber, Lyft, and equivalent business transportation.")
        return {
            "answer": f"{citation.doc_id} §{citation.section}: {citation.quote}",
            "confidence": 0.9,
            "citations": [citation_to_dict(citation)],
            "refused": False,
        }
    if "per-diem" in tokens:
        citation = policy.clause("TEP-008", "3", "Per-diem rates are set by city tier.")
        return {
            "answer": f"{citation.doc_id} §{citation.section}: {citation.quote}",
            "confidence": 0.9,
            "citations": [citation_to_dict(citation)],
            "refused": False,
        }
    if "hotel" in tokens and ("approval" in tokens or "manager" in tokens or "booking" in tokens or "book" in tokens):
        citation = policy.clause("TEP-004", "2", "Lodging booking requirements address Concur and manager approval.")
        return {
            "answer": f"{citation.doc_id} §{citation.section}: {citation.quote}",
            "confidence": 0.9,
            "citations": [citation_to_dict(citation)],
            "refused": False,
        }
    return None


@dataclass
class Citation:
    doc_id: str
    section: str
    title: str
    quote: str


@dataclass
class ReceiptFacts:
    filename: str
    text: str
    vendor: str = "Unknown merchant"
    date: str | None = None
    category: str = "unknown"
    meal_type: str | None = None
    amount: float | None = None
    subtotal: float | None = None
    tip: float | None = None
    tax: float | None = None
    nights: int | None = None
    city: str | None = None
    confidence: float = 0.6
    warnings: list[str] = field(default_factory=list)


@dataclass
class ReviewResult:
    verdict: str
    confidence: float
    reasoning: str
    citations: list[Citation]
    reimbursable_amount: float | None = None
    non_reimbursable_amount: float | None = None


class PolicyIndex:
    def __init__(self, policy_dir: Path):
        self.policy_dir = policy_dir
        self.chunks: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.policy_dir.exists() or PdfReader is None:
            self._load_fallback()
            return
        for pdf in sorted(self.policy_dir.glob("*.pdf")):
            text = extract_pdf_text(pdf)
            self._chunk_policy_pdf(pdf.name, text)
        if not self.chunks:
            self._load_fallback()

    def _load_fallback(self) -> None:
        fallback = {
            "TEP-002 §2": "Meal expenses incurred by an employee traveling on business are subject to breakfast $25, lunch $35, and dinner $75 per-person caps.",
            "TEP-003 §2": "Alcoholic beverages are reimbursable only when incurred during sanctioned client entertainment.",
            "TEP-004 §3": "Maximum reimbursable nightly lodging rate includes room rate plus mandatory taxes and fees.",
            "TEP-006 §2.3": "Taxis are reimbursable at the metered fare plus reasonable tip up to 20% of the fare.",
            "TEP-007 §6.1": "Each receipt submitted must match the line item it supports.",
        }
        for key, text in fallback.items():
            doc_id, section = key.split(" §")
            self.chunks.append(
                {
                    "doc_id": doc_id,
                    "section": section,
                    "title": doc_id,
                    "text": text,
                    "tokens": tokenize(text + " " + doc_id),
                }
            )

    def _chunk_policy_pdf(self, filename: str, text: str) -> None:
        current_doc = "UNKNOWN"
        current_title = filename
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        buffer: list[str] = []
        section = "intro"

        def flush() -> None:
            if not buffer:
                return
            body = normalize_text(" ".join(buffer)).strip()
            if len(body) < 40:
                return
            self.chunks.append(
                {
                    "doc_id": current_doc,
                    "section": section,
                    "title": current_title,
                    "text": body,
                    "tokens": tokenize(f"{current_doc} {current_title} {body}"),
                }
            )

        previous = ""
        for line in lines:
            doc_match = re.search(r"Document:\s*([A-Z]+-?\d+)", line)
            if doc_match:
                flush()
                buffer = []
                current_doc = doc_match.group(1)
                current_title = previous or current_doc
                section = "intro"
            sec_match = re.match(r"^(\d+(?:\.\d+){0,2})\.\s+(.*)", line)
            if sec_match:
                flush()
                buffer = [line]
                section = sec_match.group(1)
            else:
                buffer.append(line)
            previous = line
        flush()

    def search(self, query: str, limit: int = 5) -> list[Citation]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scored: list[tuple[float, dict[str, Any]]] = []
        q_set = set(q_tokens)
        for chunk in self.chunks:
            tokens = chunk["tokens"]
            if not tokens:
                continue
            overlap = sum(1 for token in tokens if token in q_set)
            phrase_bonus = sum(3 for token in q_set if token in chunk["text"].lower())
            score = overlap + phrase_bonus + (4 if chunk["doc_id"].lower() in query.lower() else 0)
            if score:
                scored.append((score / (len(tokens) ** 0.45), chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        citations = []
        for _, chunk in scored[:limit]:
            quote = chunk["text"]
            if len(quote) > 420:
                quote = quote[:417].rsplit(" ", 1)[0] + "..."
            citations.append(Citation(chunk["doc_id"], chunk["section"], chunk["title"], quote))
        return citations

    def clause(self, doc_id: str, section_prefix: str, fallback_query: str) -> Citation:
        candidates = [
            c for c in self.chunks
            if c["doc_id"] == doc_id and c["section"].startswith(section_prefix)
        ]
        if candidates:
            c = candidates[0]
            quote = c["text"]
            if len(quote) > 520:
                quote = quote[:517].rsplit(" ", 1)[0] + "..."
            return Citation(c["doc_id"], c["section"], c["title"], quote)
        found = self.search(fallback_query, 1)
        return found[0] if found else Citation(doc_id, section_prefix, doc_id, fallback_query)

    def answer(self, question: str) -> dict[str, Any]:
        in_scope, _scope_matches, scope_step = policy_scope_trace(question)
        if not in_scope:
            return {
                "answer": "This question appears to be outside the scope of Northwind's policy library. I can only answer questions that reference Northwind policies or expense topics.",
                "confidence": 0.12,
                "citations": [],
                "refused": True,
                "pipeline_trace": [scope_step],
            }
        direct = direct_policy_answer(question, self)
        if direct is not None:
            direct["pipeline_trace"] = [scope_step]
            return direct
        citations = self.search(question, 4)
        q_tokens = set(tokenize(question))
        if not citations:
            return {
                "answer": "I can only answer questions grounded in the Northwind policy library, and I did not find relevant policy text for that question.",
                "confidence": 0.1,
                "citations": [],
                "refused": True,
                "pipeline_trace": [scope_step],
            }
        best_tokens = set(tokenize(citations[0].quote))
        overlap = len(q_tokens & best_tokens)
        sentences = []
        for citation in citations[:3]:
            parts = re.split(r"(?<=[.!?])\s+", citation.quote)
            chosen = max(parts, key=lambda s: len(set(tokenize(s)) & q_tokens), default=citation.quote)
            sentences.append(f"{citation.doc_id} §{citation.section}: {chosen}")
        return {
            "answer": " ".join(sentences),
            "confidence": min(0.92, 0.45 + 0.08 * overlap),
            "citations": [citation_to_dict(c) for c in citations],
            "refused": False,
            "pipeline_trace": [scope_step],
        }


def citation_to_dict(citation: Citation) -> dict[str, str]:
    return {
        "doc_id": citation.doc_id,
        "section": citation.section,
        "title": citation.title,
        "quote": citation.quote,
    }


def extract_pdf_text(path: Path) -> str:
    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        return f"[PDF extraction failed: {exc}]"


def extract_text_receipt(path: Path, trace_step: TraceStep | None = None) -> tuple[str, list[str]]:
    suffix = path.suffix.lower()
    warnings: list[str] = []
    if suffix == ".pdf":
        return extract_pdf_text(path), warnings
    if suffix in {".txt", ".csv"}:
        return path.read_text(encoding="utf-8", errors="replace"), warnings
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
        text = extract_image_text(path, trace_step)
        if not text:
            warnings.append("Image OCR unavailable or returned no text. Configure OPENAI_API_KEY or pytesseract for image receipts.")
        return text, warnings
    warnings.append(f"Unsupported receipt type: {suffix}")
    return "", warnings


def extract_image_text(path: Path, trace_step: TraceStep | None = None) -> str:
    try:
        import pytesseract  # type: ignore

        if Image is None:
            return ""
        if trace_step is not None:
            trace_step.model_used = "pytesseract"
        return pytesseract.image_to_string(Image.open(path))
    except Exception:
        pass

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return ""
    try:
        image_bytes = path.read_bytes()
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        data_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        model = os.environ.get("OPENAI_VISION_MODEL", "gpt-4o-mini")
        payload = {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Extract every visible receipt field as plain text. Preserve totals, dates, merchant name, line items, taxes, tips, and payment method.",
                        },
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        context = ssl.create_default_context()
        with urllib.request.urlopen(request, timeout=45, context=context) as response:
            data = json.loads(response.read().decode("utf-8"))
        if trace_step is not None:
            usage = data.get("usage") or {}
            input_tokens = int(usage.get("input_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or 0)
            trace_step.model_used = model
            trace_step.cost_usd = calculate_model_cost_usd(model, input_tokens, output_tokens)
            trace_step.notes = f"vision OCR tokens input={input_tokens}, output={output_tokens}"
        output = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"}:
                    output.append(content.get("text", ""))
        return "\n".join(output)
    except Exception:
        return ""


def parse_amount(text: str) -> float | None:
    patterns = [
        r"GRAND\s+TOTAL\s+\$?\s*([0-9,]+\.\d{2})",
        r"TOTAL\s+CHARGED\s+\$?\s*([0-9,]+\.\d{2})",
        r"Total\s+Charged\s+\$?\s*([0-9,]+\.\d{2})",
        r"\bTOTAL\s+\$?\s*([0-9,]+\.\d{2})",
        r"\bTotal\s+\$?\s*([0-9,]+\.\d{2})",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.I)
        if matches:
            return float(matches[-1].replace(",", ""))
    amounts = re.findall(r"\$([0-9,]+\.\d{2})", text)
    if amounts:
        return float(amounts[-1].replace(",", ""))
    return None


def parse_named_amount(text: str, label: str) -> float | None:
    match = re.search(r"(?:" + label + r")\s*(?:\([^)]*\))?\s*\$?\s*(-?[0-9,]+\.\d{2})", text, re.I)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def parse_date(text: str) -> str | None:
    date_patterns = [
        (r"(\d{4}-\d{2}-\d{2})", "%Y-%m-%d"),
        (r"(\d{1,2}\s+[A-Z][a-z]{2}\s+\d{4})", "%d %b %Y"),
        (r"([A-Z][a-z]{2},?\s+[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})", "%a, %b %d, %Y"),
        (r"([A-Z][a-z]{3,9}\s+\d{1,2},\s+\d{4})", "%B %d, %Y"),
    ]
    for pattern, fmt in date_patterns:
        match = re.search(pattern, text)
        if match:
            raw = match.group(1).replace(",", "")
            fmt = fmt.replace(",", "")
            try:
                return dt.datetime.strptime(raw, fmt).date().isoformat()
            except ValueError:
                continue
    return None


def detect_city(text: str, employee: dict[str, Any] | None = None) -> str | None:
    haystack = text.lower()
    for city in sorted(TIER_1 | TIER_2, key=len, reverse=True):
        if city in haystack:
            return city.title()
    if employee:
        purpose = str(employee.get("trip_purpose", "")).lower()
        for city in sorted(TIER_1 | TIER_2, key=len, reverse=True):
            if city in purpose:
                return city.title()
    return None


def city_tier(city: str | None) -> tuple[str, float]:
    if not city:
        return "Tier 3", 175.0
    normalized = city.lower()
    if normalized in TIER_1:
        return "Tier 1", 350.0
    if normalized in TIER_2:
        return "Tier 2", 250.0
    return "Tier 3", 175.0


def parse_receipt(path: Path, employee: dict[str, Any] | None = None, tracer: StepTracer | None = None) -> ReceiptFacts:
    tracer = tracer or StepTracer(enabled=False)
    with tracer.step("ocr") as step:
        text, warnings = extract_text_receipt(path, step)
        if warnings:
            step.status = "degraded"
            step.notes = "; ".join(warnings)[:240]
    text = normalize_text(text)
    lines = [line.strip() for line in text.splitlines() if line.strip() and set(line.strip()) != {"="}]
    vendor = "Unknown merchant"
    for line in lines:
        if not re.search(r"receipt|confirmation|electronic ticket|^[-=]+$", line, re.I):
            vendor = line[:80]
            break
    amount = parse_amount(text)
    subtotal = parse_named_amount(text, r"Subtotal|Base Fare")
    tip = parse_named_amount(text, r"Tip")
    tax = parse_named_amount(text, r"Tax(?:es)?(?:\s*&\s*Fees)?|State/Local Tax|Occupancy Tax")
    nights_match = re.search(r"Nights:\s*(\d+)", text, re.I)
    nights = int(nights_match.group(1)) if nights_match else None
    with tracer.step("categorize") as step:
        category = detect_category(path.name, text)
        meal_type = detect_meal_type(path.name, text) if category == "meal" else None
        city = detect_city(text, employee)
        step.notes = f"category={category}" + (f", meal_type={meal_type}" if meal_type else "")
    confidence = 0.82 if amount and vendor != "Unknown merchant" and text else 0.35
    facts = ReceiptFacts(
        filename=path.name,
        text=text,
        vendor=vendor,
        date=parse_date(text),
        category=category,
        meal_type=meal_type,
        amount=amount,
        subtotal=subtotal,
        tip=tip,
        tax=tax,
        nights=nights,
        city=city,
        confidence=confidence,
        warnings=warnings,
    )
    if not text:
        facts.warnings.append("No receipt text was extracted.")
    if amount is None:
        facts.warnings.append("Could not identify a total amount.")
    if facts.date is None:
        facts.warnings.append("Could not identify a transaction date.")
    return facts


def detect_category(filename: str, text: str) -> str:
    probe = f"{filename} {text}".lower()
    if re.search(r"uber|lyft|taxi|rideshare|driver:|airport surcharge|trip time", probe):
        return "ground"
    if re.search(r"hotel|marriott|hilton|hyatt|check-in|check-out|nights:", probe):
        return "lodging"
    if re.search(r"airlines|air lines|e-ticket|electronic ticket|flight|itinerary|passenger:|fare breakdown", text, re.I):
        return "air"
    if re.search(r"conference registration|registration confirmation|conference pass|workshop add-on|summit registration", probe):
        return "conference"
    if re.search(r"breakfast|lunch|dinner|server:|table|grand total|coffee|salad|taco|steak|sushi|barbecue|pancake", probe):
        return "meal"
    return "unknown"


def detect_meal_type(filename: str, text: str) -> str | None:
    probe = f"{filename} {text}".lower()
    for meal in ("breakfast", "lunch", "dinner"):
        if meal in probe:
            return meal
    time_match = re.search(r"(\d{1,2}):(\d{2})\s*(AM|PM)", text, re.I)
    if time_match:
        hour = int(time_match.group(1)) % 12
        if time_match.group(3).lower() == "pm":
            hour += 12
        if hour < 10:
            return "breakfast"
        if hour < 16:
            return "lunch"
        return "dinner"
    return None


def alcohol_amount(text: str) -> float:
    total = 0.0
    for line in text.splitlines():
        if any(word in line.lower() for word in ALCOHOL_WORDS):
            match = re.search(r"\$([0-9]+\.\d{2})", line)
            if match:
                total += float(match.group(1))
    return total


def has_external_attendee(text: str, employee: dict[str, Any]) -> bool:
    probe = f"{text} {employee.get('trip_purpose', '')}".lower()
    return bool(re.search(r"client|external|prospect|partner attendee|hosted|customer", probe)) and not bool(re.search(r"guest 1 \(of 1\)|solo diner|solo travel", probe))


def is_conference_included_meal(facts: ReceiptFacts, all_facts: list[ReceiptFacts], employee: dict[str, Any]) -> bool:
    if facts.category != "meal" or not facts.meal_type:
        return False
    if "conference" not in str(employee.get("trip_purpose", "")).lower():
        return False
    registrations = [f for f in all_facts if f.category == "conference"]
    if not registrations:
        return False
    for reg in registrations:
        text = reg.text.lower()
        if "includes" not in text:
            continue
        if facts.meal_type in text:
            return True
    return False


def review_receipt(facts: ReceiptFacts, all_facts: list[ReceiptFacts], employee: dict[str, Any], policy: PolicyIndex, tracer: StepTracer | None = None) -> ReviewResult:
    tracer = tracer or StepTracer(enabled=False)
    citations: list[Citation] = []
    confidence = facts.confidence
    if facts.warnings:
        with tracer.step("retrieve_policy", model_used="deterministic") as step:
            citations.append(policy.clause("TEP-007", "2", "Valid receipts must show vendor date total itemized charges and payment method."))
            step.notes = "resolved receipt requirements clause"
        with tracer.step("generate_verdict", model_used="deterministic") as step:
            step.notes = "rule_branch=receipt_warnings"
            return ReviewResult(
                "needs_review",
                min(confidence, 0.42),
                "The receipt could not be fully extracted: " + " ".join(facts.warnings),
                citations,
                facts.amount,
                None,
            )

    if facts.category == "meal":
        return review_meal(facts, all_facts, employee, policy, tracer)
    if facts.category == "lodging":
        return review_lodging(facts, policy, tracer)
    if facts.category == "air":
        return review_air(facts, policy, tracer)
    if facts.category == "ground":
        return review_ground(facts, policy, tracer)
    if facts.category == "conference":
        with tracer.step("retrieve_policy", model_used="deterministic") as step:
            citations.append(policy.clause("TEP-014", "3", "Conference registration fees are reimbursable at the standard attendee rate."))
            step.notes = "resolved conference registration clause"
        with tracer.step("generate_verdict", model_used="deterministic") as step:
            step.notes = "rule_branch=conference"
            return ReviewResult("compliant", 0.83, "Conference registration appears to be a standard business conference expense.", citations, facts.amount, 0.0)

    with tracer.step("retrieve_policy", model_used="deterministic") as step:
        citations.append(policy.clause("TEP-001", "3.1", "Every reimbursable expense must have a clear and documented business purpose."))
        step.notes = "resolved business purpose clause"
    with tracer.step("generate_verdict", model_used="deterministic") as step:
        step.notes = "rule_branch=unknown"
    return ReviewResult("needs_review", 0.48, "The system could not confidently classify this expense category.", citations, facts.amount, None)


def review_meal(facts: ReceiptFacts, all_facts: list[ReceiptFacts], employee: dict[str, Any], policy: PolicyIndex, tracer: StepTracer | None = None) -> ReviewResult:
    tracer = tracer or StepTracer(enabled=False)
    with tracer.step("retrieve_policy", model_used="deterministic") as step:
        citations = [
            policy.clause("TEP-002", "2", "Meal expenses incurred by an employee traveling on business are subject to per person caps."),
            policy.clause("TEP-007", "3", "Meal expenses require an itemized receipt showing each item ordered."),
        ]
        step.notes = "resolved meal cap and itemized receipt clauses"
    with tracer.step("generate_verdict", model_used="deterministic") as step:
        step.notes = "rule_branch=meal"
        amount = facts.amount or 0.0
        meal = facts.meal_type or "meal"
        tier, _lodging_cap = city_tier(facts.city)
        cap = MEAL_CAPS.get(meal, 75.0)
        if tier == "Tier 1":
            cap *= 1.25

        alc = alcohol_amount(facts.text)
        if alc > 0 and not has_external_attendee(facts.text, employee):
            with tracer.step("retrieve_policy", model_used="deterministic") as retrieve_step:
                citations.append(policy.clause("TEP-003", "3.1", "Any alcoholic beverage purchased while traveling on business without external clients present is prohibited."))
                retrieve_step.notes = "resolved solo alcohol prohibition clause"
            return ReviewResult(
                "rejected",
                0.91,
                f"Alcohol totaling about {money(alc)} appears on a solo or non-client meal. The food portion may still be reviewed, but the line contains a non-reimbursable alcohol charge.",
                citations,
                max(0.0, amount - alc),
                alc,
            )

        if is_conference_included_meal(facts, all_facts, employee):
            with tracer.step("retrieve_policy", model_used="deterministic") as retrieve_step:
                citations.append(policy.clause("TEP-014", "5.1", "Where conference registration includes meals, no separate reimbursement is available for those meals."))
                retrieve_step.notes = "resolved conference-included meal clause"
            return ReviewResult(
                "rejected",
                0.87,
                f"This {meal} appears separately expensed during a conference whose registration included {meal}.",
                citations,
                0.0,
                amount,
            )

        if facts.subtotal is not None and facts.tip is not None and facts.tip > max(20.0, facts.subtotal * 0.20):
            with tracer.step("retrieve_policy", model_used="deterministic") as retrieve_step:
                citations.append(policy.clause("TEP-002", "3", "Tips above 20% of the pre-tax meal total are personal and not reimbursable."))
                retrieve_step.notes = "resolved meal tip cap clause"
            excess = facts.tip - facts.subtotal * 0.20
            return ReviewResult("flagged", 0.86, f"Tip appears above the 20% meal cap by {money(excess)}.", citations, amount - excess, excess)

        if amount > cap:
            excess = amount - cap
            return ReviewResult(
                "flagged",
                0.9,
                f"{meal.title()} total {money(amount)} exceeds the {tier} reimbursable cap of {money(cap)}.",
                citations,
                cap,
                excess,
            )
        return ReviewResult("compliant", 0.88, f"{meal.title()} total {money(amount)} is within the applicable cap of {money(cap)}.", citations, amount, 0.0)


def review_lodging(facts: ReceiptFacts, policy: PolicyIndex, tracer: StepTracer | None = None) -> ReviewResult:
    tracer = tracer or StepTracer(enabled=False)
    with tracer.step("retrieve_policy", model_used="deterministic") as step:
        citations = [policy.clause("TEP-004", "3", "Maximum reimbursable nightly lodging rate includes room rate plus mandatory taxes and resort fees.")]
        step.notes = "resolved lodging cap clause"
    with tracer.step("generate_verdict", model_used="deterministic") as step:
        step.notes = "rule_branch=lodging"
        amount = facts.amount or 0.0
        nights = facts.nights or 1
        nightly = amount / max(nights, 1)
        tier, cap = city_tier(facts.city)
        issues = []
        non_reimbursable = 0.0
        verdict = "compliant"
        confidence = 0.86
        if nightly > cap:
            excess = (nightly - cap) * nights
            non_reimbursable += excess
            issues.append(f"nightly rate {money(nightly)} exceeds the {tier} cap of {money(cap)}")
            verdict = "flagged"
            confidence = 0.9
        if re.search(r"outside\s+Concur|no\s+corporate-rate|public rate", facts.text, re.I):
            with tracer.step("retrieve_policy", model_used="deterministic") as retrieve_step:
                citations.append(policy.clause("TEP-004", "2.1", "Lodging should be booked through Concur; bookings outside the tool require manager approval and justification."))
                retrieve_step.notes = "resolved lodging booking-tool clause"
            issues.append("receipt notes booking outside Concur/no corporate-rate adjustment")
            verdict = "flagged"
            confidence = max(confidence, 0.88)
        if issues:
            return ReviewResult(verdict, confidence, "Lodging flagged: " + "; ".join(issues) + ".", citations, max(0.0, amount - non_reimbursable), non_reimbursable)
        return ReviewResult("compliant", confidence, f"Lodging averages {money(nightly)} per night, within the {tier} cap of {money(cap)}.", citations, amount, 0.0)


def review_air(facts: ReceiptFacts, policy: PolicyIndex, tracer: StepTracer | None = None) -> ReviewResult:
    tracer = tracer or StepTracer(enabled=False)
    with tracer.step("retrieve_policy", model_used="deterministic") as step:
        citations = [policy.clause("TEP-005", "2", "Economy is default; premium economy is permitted on segments of 6 hours or more.")]
        step.notes = "resolved air class of service clause"
    with tracer.step("generate_verdict", model_used="deterministic") as step:
        step.notes = "rule_branch=air"
        text = facts.text.lower()
        if "first class" in text:
            with tracer.step("retrieve_policy", model_used="deterministic") as retrieve_step:
                citations.append(policy.clause("TEP-005", "2.4", "First class is not reimbursable under any circumstances."))
                retrieve_step.notes = "resolved first-class prohibition clause"
            return ReviewResult("rejected", 0.95, "The itinerary appears to include first class service.", citations, 0.0, facts.amount)
        if "business class" in text and "international" not in text:
            with tracer.step("retrieve_policy", model_used="deterministic") as retrieve_step:
                citations.append(policy.clause("TEP-005", "2.3", "Business class is permitted only on international segments of 10 hours or more with VP approval."))
                retrieve_step.notes = "resolved business-class approval clause"
            return ReviewResult("flagged", 0.85, "Business class requires international long-haul context and VP approval.", citations, None, None)
        if re.search(r"premium select|premium economy|comfort\+", text):
            durations = [int(h) + int(m) / 60 for h, m in re.findall(r"Duration\s+(\d+)h\s+(\d+)m", facts.text)]
            if durations and max(durations) < 6:
                return ReviewResult("flagged", 0.84, "Premium economy appears on a segment under the 6-hour threshold.", citations, None, None)
        return ReviewResult("compliant", 0.84, "Airfare class and ancillary charges appear consistent with the air travel policy.", citations, facts.amount, 0.0)


def review_ground(facts: ReceiptFacts, policy: PolicyIndex, tracer: StepTracer | None = None) -> ReviewResult:
    tracer = tracer or StepTracer(enabled=False)
    with tracer.step("retrieve_policy", model_used="deterministic") as step:
        citations = [policy.clause("TEP-006", "2", "Rideshare services are reimbursable for business-related transportation.")]
        step.notes = "resolved ground transportation clause"
    with tracer.step("generate_verdict", model_used="deterministic") as step:
        step.notes = "rule_branch=ground"
        text = facts.text.lower()
        if re.search(r"uber black|lyft lux|premium", text):
            with tracer.step("retrieve_policy", model_used="deterministic") as retrieve_step:
                citations.append(policy.clause("TEP-006", "2.2", "Premium rideshare categories are not reimbursable unless they are the only available option."))
                retrieve_step.notes = "resolved premium rideshare clause"
            return ReviewResult("flagged", 0.86, "Receipt appears to use a premium rideshare category.", citations, None, None)
        if facts.subtotal is not None and facts.tip is not None and facts.tip > facts.subtotal * 0.20 + 0.01:
            with tracer.step("retrieve_policy", model_used="deterministic") as retrieve_step:
                citations.append(policy.clause("TEP-006", "2.3", "Taxi tips are reimbursable up to 20% of fare; this limit is used as the rideshare reasonableness guardrail."))
                retrieve_step.notes = "resolved ground tip cap clause"
            excess = facts.tip - facts.subtotal * 0.20
            return ReviewResult("flagged", 0.82, f"Ground transportation tip exceeds 20% by {money(excess)}.", citations, (facts.amount or 0) - excess, excess)
        return ReviewResult("compliant", 0.86, "Ground transportation appears business-related, standard category, and within tip guidance.", citations, facts.amount, 0.0)


class Store:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS employees (
                    employee_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    grade INTEGER,
                    title TEXT,
                    department TEXT,
                    manager_id TEXT,
                    home_base TEXT,
                    trip_purpose TEXT,
                    trip_dates TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS submissions (
                    id TEXT PRIMARY KEY,
                    employee_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    trip_purpose TEXT,
                    trip_dates TEXT,
                    source_key TEXT UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(employee_id) REFERENCES employees(employee_id)
                );
                CREATE TABLE IF NOT EXISTS line_items (
                    id TEXT PRIMARY KEY,
                    submission_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    saved_path TEXT,
                    vendor TEXT,
                    category TEXT,
                    meal_type TEXT,
                    transaction_date TEXT,
                    amount REAL,
                    verdict TEXT,
                    confidence REAL,
                    reasoning TEXT,
                    citations_json TEXT,
                    extracted_text TEXT,
                    reimbursable_amount REAL,
                    non_reimbursable_amount REAL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(submission_id) REFERENCES submissions(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS overrides (
                    id TEXT PRIMARY KEY,
                    line_item_id TEXT NOT NULL,
                    reviewer TEXT,
                    verdict TEXT NOT NULL,
                    comment TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(line_item_id) REFERENCES line_items(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS pipeline_trace_steps (
                    id TEXT PRIMARY KEY,
                    line_item_id TEXT NOT NULL,
                    step_order INTEGER NOT NULL,
                    step_name TEXT NOT NULL,
                    model_used TEXT NOT NULL DEFAULT 'n/a',
                    latency_ms REAL NOT NULL DEFAULT 0,
                    cost_usd REAL NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'ok',
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(line_item_id) REFERENCES line_items(id) ON DELETE CASCADE
                );
                """
            )

    def upsert_employee(self, data: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO employees(employee_id, name, grade, title, department, manager_id, home_base, trip_purpose, trip_dates, created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(employee_id) DO UPDATE SET
                    name=excluded.name, grade=excluded.grade, title=excluded.title,
                    department=excluded.department, manager_id=excluded.manager_id,
                    home_base=excluded.home_base, trip_purpose=excluded.trip_purpose,
                    trip_dates=excluded.trip_dates
                """,
                (
                    data.get("employee_id") or f"NW-{uuid.uuid4().hex[:6].upper()}",
                    data.get("name", "Unknown"),
                    int(data.get("grade") or 0),
                    data.get("title", ""),
                    data.get("department", ""),
                    data.get("manager_id", ""),
                    data.get("home_base", ""),
                    data.get("trip_purpose", ""),
                    data.get("trip_dates", ""),
                    now_iso(),
                ),
            )

    def employee(self, employee_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM employees WHERE employee_id=?", (employee_id,)).fetchone()
            return dict(row) if row else None

    def employees(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM employees ORDER BY name")]

    def create_submission(self, employee_id: str, trip_purpose: str, trip_dates: str, source_key: str | None = None) -> str:
        submission_id = uuid.uuid4().hex
        stamp = now_iso()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO submissions(id, employee_id, status, trip_purpose, trip_dates, source_key, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?)",
                (submission_id, employee_id, "processing", trip_purpose, trip_dates, source_key, stamp, stamp),
            )
        return submission_id

    def source_submission(self, source_key: str) -> str | None:
        with self.connect() as conn:
            row = conn.execute("SELECT id FROM submissions WHERE source_key=?", (source_key,)).fetchone()
            return row["id"] if row else None

    def delete_submission(self, submission_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM submissions WHERE id=?", (submission_id,))

    def add_line_item(self, submission_id: str, facts: ReceiptFacts, result: ReviewResult, saved_path: Path | None, trace_steps: list[dict[str, Any]] | None = None) -> None:
        line_item_id = uuid.uuid4().hex
        stamp = now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO line_items(
                    id, submission_id, filename, saved_path, vendor, category, meal_type,
                    transaction_date, amount, verdict, confidence, reasoning, citations_json,
                    extracted_text, reimbursable_amount, non_reimbursable_amount, created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    line_item_id,
                    submission_id,
                    facts.filename,
                    str(saved_path) if saved_path else None,
                    facts.vendor,
                    facts.category,
                    facts.meal_type,
                    facts.date,
                    facts.amount,
                    result.verdict,
                    result.confidence,
                    result.reasoning,
                    json.dumps([citation_to_dict(c) for c in result.citations]),
                    facts.text[:12000],
                    result.reimbursable_amount,
                    result.non_reimbursable_amount,
                    stamp,
                ),
            )
            for index, step in enumerate(trace_steps or []):
                conn.execute(
                    """
                    INSERT INTO pipeline_trace_steps(
                        id, line_item_id, step_order, step_name, model_used,
                        latency_ms, cost_usd, status, notes, created_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        uuid.uuid4().hex,
                        line_item_id,
                        index,
                        step.get("step_name", "unknown"),
                        step.get("model_used", "n/a"),
                        float(step.get("latency_ms") or 0),
                        float(step.get("cost_usd") or 0),
                        step.get("status", "ok"),
                        step.get("notes", ""),
                        stamp,
                    ),
                )

    def finalize_submission(self, submission_id: str) -> None:
        with self.connect() as conn:
            rows = conn.execute("SELECT verdict FROM line_items WHERE submission_id=?", (submission_id,)).fetchall()
            verdicts = [row["verdict"] for row in rows]
            if not verdicts:
                status = "needs_review"
            elif "rejected" in verdicts:
                status = "rejected"
            elif "flagged" in verdicts or "needs_review" in verdicts:
                status = "flagged"
            else:
                status = "compliant"
            conn.execute("UPDATE submissions SET status=?, updated_at=? WHERE id=?", (status, now_iso(), submission_id))

    def submissions(self, employee_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        where = []
        params: list[Any] = []
        if employee_id:
            where.append("s.employee_id=?")
            params.append(employee_id)
        if status:
            where.append("s.status=?")
            params.append(status)
        clause = "WHERE " + " AND ".join(where) if where else ""
        with self.connect() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    f"""
                    SELECT s.*, e.name AS employee_name, e.department,
                           COUNT(li.id) AS item_count,
                           COALESCE(SUM(li.amount),0) AS total_amount,
                           SUM(CASE WHEN li.verdict='compliant' THEN 1 ELSE 0 END) AS compliant_count,
                           SUM(CASE WHEN li.verdict IN ('flagged','rejected','needs_review') THEN 1 ELSE 0 END) AS issue_count
                    FROM submissions s
                    JOIN employees e ON e.employee_id=s.employee_id
                    LEFT JOIN line_items li ON li.submission_id=s.id
                    {clause}
                    GROUP BY s.id
                    ORDER BY s.created_at DESC
                    """,
                    params,
                )
            ]

    def submission_detail(self, submission_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            sub = conn.execute(
                "SELECT s.*, e.name AS employee_name, e.grade, e.title, e.department, e.manager_id, e.home_base FROM submissions s JOIN employees e ON e.employee_id=s.employee_id WHERE s.id=?",
                (submission_id,),
            ).fetchone()
            if not sub:
                return None
            items = []
            for row in conn.execute("SELECT * FROM line_items WHERE submission_id=? ORDER BY filename", (submission_id,)):
                item = dict(row)
                item["citations"] = json.loads(item.pop("citations_json") or "[]")
                item["overrides"] = [
                    dict(o)
                    for o in conn.execute("SELECT * FROM overrides WHERE line_item_id=? ORDER BY created_at DESC", (row["id"],))
                ]
                item["pipeline_trace"] = [
                    dict(step)
                    for step in conn.execute(
                        """
                        SELECT step_name, model_used, latency_ms, cost_usd, status, notes
                        FROM pipeline_trace_steps
                        WHERE line_item_id=?
                        ORDER BY step_order
                        """,
                        (row["id"],),
                    )
                ]
                items.append(item)
            detail = dict(sub)
            detail["items"] = items
            return detail

    def add_override(self, line_item_id: str, reviewer: str, verdict: str, comment: str) -> dict[str, Any]:
        override_id = uuid.uuid4().hex
        stamp = now_iso()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO overrides(id, line_item_id, reviewer, verdict, comment, created_at) VALUES(?,?,?,?,?,?)",
                (override_id, line_item_id, reviewer, verdict, comment, stamp),
            )
            row = conn.execute("SELECT * FROM overrides WHERE id=?", (override_id,)).fetchone()
            sub = conn.execute("SELECT submission_id FROM line_items WHERE id=?", (line_item_id,)).fetchone()
            if sub:
                conn.execute("UPDATE submissions SET updated_at=? WHERE id=?", (stamp, sub["submission_id"]))
            return dict(row)


POLICY = PolicyIndex(CASE_STUDY_DIR / "policies")
STORE = Store(DB_PATH)


def seed_employees() -> None:
    submissions_dir = CASE_STUDY_DIR / "submissions"
    if not submissions_dir.exists():
        return
    for info_path in sorted(submissions_dir.glob("*/employee_info.json")):
        try:
            STORE.upsert_employee(json.loads(info_path.read_text(encoding="utf-8")))
        except Exception:
            continue


def process_submission(employee: dict[str, Any], receipt_paths: list[Path], source_key: str | None = None) -> str:
    STORE.upsert_employee(employee)
    if source_key:
        existing = STORE.source_submission(source_key)
        if existing:
            detail = STORE.submission_detail(existing)
            if detail and detail.get("items") and all(item.get("pipeline_trace") for item in detail.get("items", [])):
                return existing
            STORE.delete_submission(existing)
    sid = STORE.create_submission(employee["employee_id"], employee.get("trip_purpose", ""), employee.get("trip_dates", ""), source_key)
    tracers = [StepTracer(enabled=True) for _ in receipt_paths]
    facts_list = [parse_receipt(path, employee, tracer) for path, tracer in zip(receipt_paths, tracers)]
    dest_dir = UPLOAD_DIR / sid
    dest_dir.mkdir(parents=True, exist_ok=True)
    for path, facts, tracer in zip(receipt_paths, facts_list, tracers):
        saved = dest_dir / path.name
        if path.resolve() != saved.resolve():
            shutil.copy2(path, saved)
        result = review_receipt(facts, facts_list, employee, POLICY, tracer)
        with tracer.step("schema_validate", model_used="deterministic") as step:
            if not result.verdict or not result.reasoning:
                step.status = "error"
                step.notes = "Missing required verdict fields."
            else:
                step.notes = "ReviewResult dataclass fields present."
        with tracer.step("confidence_check", model_used="deterministic") as step:
            step.notes = f"confidence={result.confidence:.2f}"
            if result.confidence < 0.5:
                step.status = "degraded"
        STORE.add_line_item(sid, facts, result, saved, tracer.to_list())
    STORE.finalize_submission(sid)
    return sid


def load_sample_submissions() -> list[str]:
    created = []
    base = CASE_STUDY_DIR / "submissions"
    for folder in sorted(base.glob("*")):
        info = folder / "employee_info.json"
        receipts = sorted((folder / "receipts").glob("*"))
        if info.exists() and receipts:
            employee = json.loads(info.read_text(encoding="utf-8"))
            sid = process_submission(employee, receipts, source_key=folder.name)
            created.append(sid)
    return created


class AppHandler(BaseHTTPRequestHandler):
    server_version = "NorthwindPreReview/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def send_json(self, data: Any, status: int = 200) -> None:
        body = json_dumps(data)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self) -> None:
        body = INDEX_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        try:
            if self.path == "/" or self.path.startswith("/?"):
                return self.send_html()
            if self.path.startswith("/api/employees"):
                return self.send_json({"employees": STORE.employees()})
            if self.path.startswith("/api/submissions/"):
                sid = self.path.rsplit("/", 1)[-1].split("?")[0]
                detail = STORE.submission_detail(sid)
                return self.send_json(detail if detail else {"error": "Not found"}, 200 if detail else 404)
            if self.path.startswith("/api/submissions"):
                return self.send_json({"submissions": STORE.submissions()})
            if self.path.startswith("/api/health"):
                return self.send_json({"ok": True, "policies": len(POLICY.chunks), "case_study_dir": str(CASE_STUDY_DIR)})
            self.send_error(404)
        except Exception as exc:
            self.send_json({"error": str(exc)}, 500)

    def do_POST(self) -> None:
        try:
            if self.path == "/api/employees":
                payload = self.read_json()
                STORE.upsert_employee(payload)
                return self.send_json({"ok": True})
            if self.path == "/api/load-samples":
                ids = load_sample_submissions()
                return self.send_json({"submission_ids": ids})
            if self.path == "/api/policy-question":
                payload = self.read_json()
                return self.send_json(POLICY.answer(str(payload.get("question", ""))))
            if self.path.startswith("/api/overrides"):
                payload = self.read_json()
                override = STORE.add_override(
                    str(payload["line_item_id"]),
                    str(payload.get("reviewer", "Finance reviewer")),
                    str(payload["verdict"]),
                    str(payload.get("comment", "")),
                )
                return self.send_json({"override": override})
            if self.path == "/api/review":
                return self.handle_review_upload()
            self.send_error(404)
        except Exception as exc:
            self.send_json({"error": str(exc)}, 500)

    def handle_review_upload(self) -> None:
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
            },
        )
        employee_json = form.getfirst("employee")
        if not employee_json:
            return self.send_json({"error": "employee payload is required"}, 400)
        employee = json.loads(employee_json)
        if not employee.get("employee_id"):
            employee["employee_id"] = "NW-" + uuid.uuid4().hex[:8].upper()
        upload_dir = UPLOAD_DIR / ("incoming-" + uuid.uuid4().hex)
        upload_dir.mkdir(parents=True, exist_ok=True)
        receipt_fields = form["receipts"] if "receipts" in form else []
        if not isinstance(receipt_fields, list):
            receipt_fields = [receipt_fields]
        paths = []
        for field in receipt_fields:
            if not getattr(field, "filename", None):
                continue
            original_name = Path(field.filename).name
            safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "-", original_name).strip(".-") or f"receipt-{uuid.uuid4().hex}.dat"
            target = upload_dir / safe_name
            with target.open("wb") as out:
                shutil.copyfileobj(field.file, out)
            paths.append(target)
        if not paths:
            return self.send_json({"error": "At least one receipt file is required"}, 400)
        sid = process_submission(employee, paths)
        return self.send_json({"submission": STORE.submission_detail(sid)})


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Northwind Expense Pre-Review</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #18212f;
      --muted: #667085;
      --line: #d8dee8;
      --surface: #f7f8fb;
      --panel: #ffffff;
      --accent: #006b6f;
      --accent-2: #7a4b00;
      --good: #197a43;
      --warn: #a15c00;
      --bad: #b42318;
      --review: #475467;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--surface);
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 24px;
      padding: 18px 28px;
      border-bottom: 1px solid var(--line);
      background: #fff;
      position: sticky;
      top: 0;
      z-index: 5;
    }
    h1 { font-size: 20px; margin: 0; letter-spacing: 0; }
    h2 { font-size: 17px; margin: 0 0 14px; letter-spacing: 0; }
    h3 { font-size: 15px; margin: 0 0 8px; letter-spacing: 0; }
    button, input, select, textarea {
      font: inherit;
    }
    button {
      border: 1px solid #b9c3d4;
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      padding: 8px 12px;
      cursor: pointer;
    }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }
    button:disabled { opacity: .55; cursor: wait; }
    main {
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      min-height: calc(100vh - 65px);
    }
    aside {
      padding: 20px;
      border-right: 1px solid var(--line);
      background: #fff;
    }
    section.workspace {
      padding: 22px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 16px;
    }
    label { display: block; font-size: 12px; color: var(--muted); margin: 10px 0 5px; }
    input, select, textarea {
      width: 100%;
      border: 1px solid #c9d2df;
      border-radius: 6px;
      padding: 9px 10px;
      background: #fff;
      color: var(--ink);
    }
    textarea { min-height: 82px; resize: vertical; }
    .row { display: flex; gap: 10px; align-items: center; }
    .row > * { flex: 1; }
    .tabs { display: flex; gap: 6px; margin-bottom: 16px; }
    .tab { flex: 0 0 auto; }
    .tab.active { background: #e8f4f4; border-color: #8dbabc; color: #004f52; }
    .muted { color: var(--muted); font-size: 13px; }
    .history-item {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      margin: 10px 0;
      cursor: pointer;
      background: #fff;
    }
    .history-item:hover { border-color: #9aa9bd; }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      font-weight: 650;
      text-transform: capitalize;
    }
    .compliant { background: #e8f5ee; color: var(--good); }
    .flagged { background: #fff4e5; color: var(--warn); }
    .rejected { background: #fdecea; color: var(--bad); }
    .needs_review, .processing { background: #eef1f5; color: var(--review); }
    .item {
      display: grid;
      grid-template-columns: 180px minmax(0, 1fr) 160px;
      gap: 14px;
      padding: 14px 0;
      border-top: 1px solid var(--line);
    }
    .item > div { min-width: 0; }
    .item:first-child { border-top: 0; }
    .metric {
      display: grid;
      gap: 2px;
      font-size: 13px;
    }
    .metric strong { font-size: 18px; }
    .citation {
      border-left: 3px solid #b8c7d9;
      margin: 8px 0;
      padding: 6px 10px;
      color: #344054;
      background: #f8fafc;
      font-size: 13px;
    }
    .override-log {
      margin-top: 8px;
      font-size: 12px;
      color: #344054;
    }
    details.trace {
      margin-top: 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      max-width: 100%;
      overflow: hidden;
    }
    details.trace summary {
      cursor: pointer;
      padding: 8px 10px;
      font-size: 13px;
      color: #344054;
      border-bottom: 1px solid var(--line);
    }
    details.trace:not([open]) summary { border-bottom: 0; }
    .trace-row {
      padding: 8px 10px;
      border-top: 1px solid var(--line);
      font-size: 12px;
      min-width: 0;
    }
    details.trace summary + .trace-row { border-top: 0; }
    .trace-main {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      min-width: 0;
    }
    .trace-step {
      min-width: 128px;
      max-width: 100%;
      overflow-wrap: anywhere;
    }
    .trace-row .trace-note {
      color: #475467;
      line-height: 1.35;
      white-space: normal;
      overflow-wrap: anywhere;
      margin-top: 5px;
      padding-left: 0;
    }
    .trace-row .trace-cost,
    .trace-row .trace-latency {
      white-space: nowrap;
      font-variant-numeric: tabular-nums;
    }
    .trace-badge {
      border-radius: 999px;
      padding: 2px 7px;
      font-weight: 650;
      text-transform: capitalize;
      width: fit-content;
    }
    .trace-ok { background: #e8f5ee; color: var(--good); }
    .trace-retried, .trace-fallback, .trace-degraded { background: #fff4e5; color: var(--warn); }
    .trace-error { background: #fdecea; color: var(--bad); }
    .hidden { display: none !important; }
    .spinner { min-height: 28px; color: var(--muted); }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .item { grid-template-columns: 1fr; }
    }
    @media (max-width: 1180px) {
      .trace-row { font-size: 11px; }
      .trace-main { gap: 8px; }
      .trace-step { min-width: 112px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Northwind Expense Pre-Review</h1>
      <div class="muted">Policy-grounded triage for finance reviewers</div>
    </div>
    <button id="loadSamples">Load sample submissions</button>
  </header>
  <main>
    <aside>
      <div class="tabs">
        <button class="tab active" data-tab="new">New</button>
        <button class="tab" data-tab="history">History</button>
        <button class="tab" data-tab="qa">Policy Q&A</button>
      </div>
      <div id="newPanel" class="side-panel">
        <div class="panel">
          <h2>Submission</h2>
          <label>Employee</label>
          <select id="employeeSelect"></select>
          <label>Name</label>
          <input id="nameInput" placeholder="New employee name">
          <div class="row">
            <div>
              <label>Employee ID</label>
              <input id="idInput" placeholder="NW-00000">
            </div>
            <div>
              <label>Grade</label>
              <input id="gradeInput" type="number" min="1" max="10">
            </div>
          </div>
          <label>Title</label>
          <input id="titleInput">
          <label>Department</label>
          <input id="departmentInput">
          <label>Trip purpose</label>
          <textarea id="purposeInput"></textarea>
          <label>Trip dates</label>
          <input id="datesInput" placeholder="2025-04-14 to 2025-04-16">
          <label>Receipts</label>
          <input id="receiptInput" type="file" multiple accept=".pdf,.txt,.png,.jpg,.jpeg,.webp">
          <div style="height:12px"></div>
          <button id="submitBtn" class="primary">Run pre-review</button>
          <div id="uploadStatus" class="spinner"></div>
        </div>
      </div>
      <div id="historyPanel" class="side-panel hidden">
        <div class="panel">
          <h2>History</h2>
          <div id="historyList"></div>
        </div>
      </div>
      <div id="qaPanel" class="side-panel hidden">
        <div class="panel">
          <h2>Policy Q&A</h2>
          <textarea id="questionInput" placeholder="Ask about Northwind policy"></textarea>
          <div style="height:10px"></div>
          <button id="askBtn" class="primary">Ask</button>
          <div id="answerBox" style="margin-top:14px"></div>
        </div>
      </div>
    </aside>
    <section class="workspace">
      <div id="detail" class="panel">
        <h2>Ready for review</h2>
        <p class="muted">Pick an employee, upload receipts, or load the sample folders. Flagged and rejected items will stand out with policy quotes and reviewer override controls.</p>
      </div>
    </section>
  </main>
  <script>
    const state = { employees: [], selected: null };
    const $ = (id) => document.getElementById(id);
    const fmtMoney = (value) => value === null || value === undefined ? "unknown" : "$" + Number(value).toFixed(2);
    const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]));

    async function api(path, options = {}) {
      const res = await fetch(path, options);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Request failed");
      return data;
    }

    function selectedEmployeePayload() {
      const selectedId = $("employeeSelect").value;
      const existing = state.employees.find(e => e.employee_id === selectedId);
      if (existing && selectedId !== "__new") {
        return {
          ...existing,
          trip_purpose: $("purposeInput").value || existing.trip_purpose,
          trip_dates: $("datesInput").value || existing.trip_dates
        };
      }
      return {
        employee_id: $("idInput").value,
        name: $("nameInput").value,
        grade: Number($("gradeInput").value || 0),
        title: $("titleInput").value,
        department: $("departmentInput").value,
        manager_id: "",
        home_base: "",
        trip_purpose: $("purposeInput").value,
        trip_dates: $("datesInput").value
      };
    }

    async function loadEmployees() {
      const data = await api("/api/employees");
      state.employees = data.employees;
      $("employeeSelect").innerHTML = `<option value="__new">Create new employee</option>` + state.employees.map(e => `<option value="${esc(e.employee_id)}">${esc(e.name)} - ${esc(e.employee_id)}</option>`).join("");
      if (state.employees[0]) {
        $("employeeSelect").value = state.employees[0].employee_id;
        fillEmployee(state.employees[0]);
      }
    }

    function fillEmployee(e) {
      $("nameInput").value = e?.name || "";
      $("idInput").value = e?.employee_id || "";
      $("gradeInput").value = e?.grade || "";
      $("titleInput").value = e?.title || "";
      $("departmentInput").value = e?.department || "";
      $("purposeInput").value = e?.trip_purpose || "";
      $("datesInput").value = e?.trip_dates || "";
    }

    async function loadHistory() {
      const data = await api("/api/submissions");
      $("historyList").innerHTML = data.submissions.map(s => `
        <div class="history-item" data-id="${esc(s.id)}">
          <div class="row" style="align-items:flex-start">
            <strong>${esc(s.employee_name)}</strong>
            <span class="status ${esc(s.status)}">${esc(s.status)}</span>
          </div>
          <div class="muted">${esc(s.trip_dates || "")}</div>
          <div class="muted">${s.item_count} receipts · ${fmtMoney(s.total_amount)} · ${s.issue_count || 0} issues</div>
        </div>
      `).join("") || `<p class="muted">No submissions yet.</p>`;
      document.querySelectorAll(".history-item").forEach(el => el.addEventListener("click", () => openSubmission(el.dataset.id)));
    }

    async function openSubmission(id) {
      const detail = await api("/api/submissions/" + id);
      renderDetail(detail);
      state.selected = detail;
    }

    function renderDetail(sub) {
      const matchingEmployee = state.employees.find(e => e.employee_id === sub.employee_id);
      if (matchingEmployee) {
        $("employeeSelect").value = matchingEmployee.employee_id;
        fillEmployee({ ...matchingEmployee, trip_purpose: sub.trip_purpose, trip_dates: sub.trip_dates });
      }
      const total = sub.items.reduce((sum, item) => sum + Number(item.amount || 0), 0);
      $("detail").innerHTML = `
        <div class="row" style="align-items:flex-start">
          <div>
            <h2>${esc(sub.employee_name)}</h2>
            <div class="muted">${esc(sub.title || "")} · ${esc(sub.department || "")}</div>
            <div class="muted">${esc(sub.trip_purpose || "")}</div>
          </div>
          <span class="status ${esc(sub.status)}">${esc(sub.status)}</span>
        </div>
        <div class="row" style="margin:16px 0">
          <div class="metric"><span class="muted">Receipts</span><strong>${sub.items.length}</strong></div>
          <div class="metric"><span class="muted">Claimed total</span><strong>${fmtMoney(total)}</strong></div>
          <div class="metric"><span class="muted">Updated</span><strong style="font-size:14px">${esc(sub.updated_at)}</strong></div>
        </div>
        <div>${sub.items.map(renderItem).join("")}</div>
      `;
      document.querySelectorAll("[data-override]").forEach(btn => btn.addEventListener("click", saveOverride));
    }

    function renderItem(item) {
      const activeOverride = item.overrides && item.overrides[0];
      const traceRows = (item.pipeline_trace || []).map(step => `
        <div class="trace-row">
          <div class="trace-main">
            <strong class="trace-step">${esc(step.step_name)}</strong>
            <span class="trace-badge trace-${esc(step.status)}">${esc(step.status)}</span>
            <span class="trace-latency">${Number(step.latency_ms || 0).toFixed(1)} ms</span>
            <span class="trace-cost">${Number(step.cost_usd || 0).toFixed(6)}</span>
          </div>
          ${step.notes ? `<div class="trace-note">${esc(step.notes)}</div>` : ""}
        </div>
      `).join("");
      return `
        <div class="item">
          <div>
            <span class="status ${esc(item.verdict)}">${esc(item.verdict)}</span>
            <h3 style="margin-top:10px">${esc(item.vendor)}</h3>
            <div class="muted">${esc(item.filename)}</div>
            <div class="muted">${esc(item.category)} ${item.meal_type ? "· " + esc(item.meal_type) : ""}</div>
          </div>
          <div>
            <p>${esc(item.reasoning)}</p>
            ${(item.citations || []).map(c => `<div class="citation"><strong>${esc(c.doc_id)} §${esc(c.section)}</strong><br>${esc(c.quote)}</div>`).join("")}
            <details class="trace">
              <summary>Pipeline trace</summary>
              ${traceRows || `<div class="trace-row"><span class="muted">No trace recorded for this item.</span></div>`}
            </details>
            ${activeOverride ? `<div class="override-log"><strong>Override:</strong> ${esc(activeOverride.verdict)} by ${esc(activeOverride.reviewer)} - ${esc(activeOverride.comment)}</div>` : ""}
          </div>
          <div>
            <div class="metric"><span class="muted">Amount</span><strong>${fmtMoney(item.amount)}</strong></div>
            <div class="metric"><span class="muted">Confidence</span><strong>${Math.round(Number(item.confidence || 0) * 100)}%</strong></div>
            <label>Override verdict</label>
            <select id="verdict-${item.id}">
              <option value="compliant">Compliant</option>
              <option value="flagged">Flagged</option>
              <option value="rejected">Rejected</option>
              <option value="needs_review">Needs review</option>
            </select>
            <label>Comment</label>
            <textarea id="comment-${item.id}" placeholder="Required audit comment"></textarea>
            <button data-override="${item.id}">Save override</button>
          </div>
        </div>
      `;
    }

    async function saveOverride(event) {
      const id = event.target.dataset.override;
      const comment = $("comment-" + id).value.trim();
      if (!comment) return alert("Add an audit comment before saving an override.");
      await api("/api/overrides", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ line_item_id: id, verdict: $("verdict-" + id).value, comment, reviewer: "Finance reviewer" })
      });
      await openSubmission(state.selected.id);
      await loadHistory();
    }

    async function submitReview() {
      const files = $("receiptInput").files;
      if (!files.length) return alert("Choose at least one receipt.");
      const fd = new FormData();
      fd.append("employee", JSON.stringify(selectedEmployeePayload()));
      Array.from(files).forEach(file => fd.append("receipts", file));
      $("submitBtn").disabled = true;
      $("uploadStatus").textContent = "Extracting receipts and checking policies...";
      try {
        const data = await api("/api/review", { method: "POST", body: fd });
        renderDetail(data.submission);
        await loadEmployees();
        await loadHistory();
        $("uploadStatus").textContent = "Review complete.";
      } catch (err) {
        $("uploadStatus").textContent = err.message;
      } finally {
        $("submitBtn").disabled = false;
      }
    }

    async function askQuestion() {
      const question = $("questionInput").value.trim();
      if (!question) return;
      const data = await api("/api/policy-question", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({question})
      });
      $("answerBox").innerHTML = `
        <p>${esc(data.answer)}</p>
        <div class="muted">Confidence ${Math.round(Number(data.confidence || 0) * 100)}%${data.refused ? " · refused" : ""}</div>
        ${(data.citations || []).map(c => `<div class="citation"><strong>${esc(c.doc_id)} §${esc(c.section)}</strong><br>${esc(c.quote)}</div>`).join("")}
      `;
    }

    document.querySelectorAll(".tab").forEach(tab => tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      document.querySelectorAll(".side-panel").forEach(p => p.classList.add("hidden"));
      $(tab.dataset.tab + "Panel").classList.remove("hidden");
      if (tab.dataset.tab === "history") loadHistory();
    }));
    $("employeeSelect").addEventListener("change", () => fillEmployee(state.employees.find(e => e.employee_id === $("employeeSelect").value) || null));
    $("submitBtn").addEventListener("click", submitReview);
    $("askBtn").addEventListener("click", askQuestion);
    $("loadSamples").addEventListener("click", async () => {
      $("loadSamples").disabled = true;
      try {
        const data = await api("/api/load-samples", {method: "POST"});
        await loadHistory();
        if (data.submission_ids[0]) await openSubmission(data.submission_ids[0]);
      } finally {
        $("loadSamples").disabled = false;
      }
    });
    loadEmployees().then(loadHistory);
  </script>
</body>
</html>
"""


def main() -> None:
    seed_employees()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer((host, port), AppHandler)
    print(f"Northwind Expense Pre-Review running at http://{host}:{port}")
    print(f"Policies loaded: {len(POLICY.chunks)} chunks from {CASE_STUDY_DIR / 'policies'}")
    server.serve_forever()


if __name__ == "__main__":
    main()
