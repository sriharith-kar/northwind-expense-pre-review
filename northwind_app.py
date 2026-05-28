from __future__ import annotations

import base64
import datetime as dt
import hashlib
import html
import json
import math
import mimetypes
import os
import shutil
import sqlite3
import ssl
import struct
import sys
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from email import policy as email_policy
from email.parser import BytesParser
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


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file(ROOT / ".env")

CASE_STUDY_DIR = Path(os.environ.get("CASE_STUDY_DIR", ROOT / "case_study"))
DB_PATH = Path(os.environ.get("NORTHWIND_DB", ROOT / "northwind.sqlite"))
UPLOAD_DIR = Path(os.environ.get("NORTHWIND_UPLOADS", ROOT / "runtime_uploads"))
GEMINI_API_BASE = os.environ.get("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_VISION_MODEL = os.environ.get("GEMINI_VISION_MODEL", GEMINI_MODEL)
GEMINI_EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
RETRIEVAL_TOP_K = int(os.environ.get("NORTHWIND_RETRIEVAL_TOP_K", "6"))


def now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def json_dumps(data: Any) -> bytes:
    return json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")


def money(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"${value:,.2f}"


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return " ".join(text.replace("\u2014", "-").replace("\u2013", "-").replace("\u00a0", " ").split())


def model_resource(model: str) -> str:
    return model if model.startswith("models/") else f"models/{model}"


def clean_json_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def clamp(value: Any, low: float, high: float, default: float) -> float:
    number = coerce_float(value)
    if number is None:
        return default
    return max(low, min(high, number))


def pack_vector(values: list[float]) -> bytes:
    return struct.pack(f"{len(values)}f", *values)


def unpack_vector(blob: bytes) -> list[float]:
    if not blob:
        return []
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


def normalize_vector(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if not norm:
        return values
    return [value / norm for value in values]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class GeminiClient:
    def __init__(self) -> None:
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        self.generate_model = GEMINI_MODEL
        self.vision_model = GEMINI_VISION_MODEL
        self.embedding_model = GEMINI_EMBEDDING_MODEL

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _request(self, endpoint: str, payload: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
        if not self.configured:
            raise RuntimeError("GEMINI_API_KEY is not configured.")
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            method="POST",
        )
        context = ssl.create_default_context()
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            return json.loads(response.read().decode("utf-8"))

    def _set_usage(self, step: TraceStep | None, model: str, data: dict[str, Any]) -> None:
        if step is None:
            return
        usage = data.get("usageMetadata") or {}
        input_tokens = int(usage.get("promptTokenCount") or 0)
        output_tokens = int(usage.get("candidatesTokenCount") or 0)
        step.model_used = model
        step.cost_usd = calculate_model_cost_usd(model, input_tokens, output_tokens)
        step.notes = f"tokens input={input_tokens}, output={output_tokens}"

    def _candidate_text(self, data: dict[str, Any]) -> str:
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = ((candidates[0].get("content") or {}).get("parts") or [])
        output = []
        for part in parts:
            if "text" in part:
                output.append(str(part.get("text") or ""))
        return "\n".join(output).strip()

    def generate_json(self, prompt: str, schema: dict[str, Any], trace_step: TraceStep | None = None, model: str | None = None, temperature: float = 0.1) -> dict[str, Any]:
        model_name = model or self.generate_model
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json",
                "responseJsonSchema": schema,
            },
        }
        data = self._request(f"{GEMINI_API_BASE}/{model_resource(model_name)}:generateContent", payload)
        self._set_usage(trace_step, model_name, data)
        text = clean_json_text(self._candidate_text(data))
        if not text:
            raise RuntimeError("Gemini returned an empty JSON response.")
        return json.loads(text)

    def generate_text_with_image(self, path: Path, prompt: str, trace_step: TraceStep | None = None) -> str:
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {"inlineData": {"mimeType": mime, "data": base64.b64encode(path.read_bytes()).decode("ascii")}},
                    ],
                }
            ],
            "generationConfig": {"temperature": 0.0},
        }
        data = self._request(f"{GEMINI_API_BASE}/{model_resource(self.vision_model)}:generateContent", payload, timeout=90)
        self._set_usage(trace_step, self.vision_model, data)
        return self._candidate_text(data)

    def embed_text(self, text: str, task_type: str, title: str = "", trace_step: TraceStep | None = None) -> list[float]:
        payload: dict[str, Any] = {"content": {"parts": [{"text": text}]}, "taskType": task_type}
        if title and task_type == "RETRIEVAL_DOCUMENT":
            payload["title"] = title
        data = self._request(f"{GEMINI_API_BASE}/{model_resource(self.embedding_model)}:embedContent", payload)
        self._set_usage(trace_step, self.embedding_model, data)
        values = data.get("embedding", {}).get("values") or []
        return normalize_vector([float(value) for value in values])


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


def nullable(kind: str) -> dict[str, Any]:
    return {"type": [kind, "null"]}


RECEIPT_FACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "vendor": {"type": "string"},
        "date": nullable("string"),
        "category": {"type": "string", "enum": ["air", "lodging", "ground", "meal", "conference", "unknown"]},
        "meal_type": {"type": ["string", "null"], "enum": ["breakfast", "lunch", "dinner", "snack", "unknown", None]},
        "amount": nullable("number"),
        "subtotal": nullable("number"),
        "tip": nullable("number"),
        "tax": nullable("number"),
        "nights": nullable("integer"),
        "city": nullable("string"),
        "confidence": {"type": "number"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["vendor", "date", "category", "meal_type", "amount", "subtotal", "tip", "tax", "nights", "city", "confidence", "warnings"],
    "additionalProperties": False,
}


REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["compliant", "flagged", "rejected", "needs_review"]},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
        "citation_chunk_ids": {"type": "array", "items": {"type": "string"}},
        "reimbursable_amount": nullable("number"),
        "non_reimbursable_amount": nullable("number"),
    },
    "required": ["verdict", "confidence", "reasoning", "citation_chunk_ids", "reimbursable_amount", "non_reimbursable_amount"],
    "additionalProperties": False,
}


QA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
        "citation_chunk_ids": {"type": "array", "items": {"type": "string"}},
        "refused": {"type": "boolean"},
    },
    "required": ["answer", "confidence", "citation_chunk_ids", "refused"],
    "additionalProperties": False,
}


class PolicyIndex:
    def __init__(self, policy_dir: Path, client: GeminiClient, db_path: Path):
        self.policy_dir = policy_dir
        self.client = client
        self.db_path = db_path
        self.chunks: list[dict[str, Any]] = []
        self.index_error = ""
        self._load()
        self._init_vector_store()
        if self.client.configured:
            try:
                self._ensure_indexed()
            except Exception as exc:
                self.index_error = str(exc)[:240]

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_vector_store(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS policy_documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    indexed_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS policy_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    section TEXT NOT NULL,
                    title TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES policy_documents(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS policy_embeddings (
                    chunk_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    embedding_blob BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(chunk_id, model),
                    FOREIGN KEY(chunk_id) REFERENCES policy_chunks(id) ON DELETE CASCADE
                );
                """
            )

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
        fallback = [
            ("TEP-002", "fallback-1", "Meal expense caps", "Meal expenses incurred by an employee traveling on business are subject to breakfast $25, lunch $35, and dinner $75 per-person caps."),
            ("TEP-003", "fallback-2", "Alcohol", "Alcoholic beverages are reimbursable only when incurred during sanctioned client entertainment."),
            ("TEP-004", "fallback-3", "Lodging", "Maximum reimbursable nightly lodging rate includes room rate plus mandatory taxes and fees."),
            ("TEP-006", "fallback-4", "Ground transportation", "Taxis are reimbursable at the metered fare plus reasonable tip up to 20% of the fare."),
            ("TEP-007", "fallback-5", "Receipts", "Each receipt submitted must match the line item it supports."),
        ]
        for index, (doc_id, section, title, text) in enumerate(fallback):
            self._append_chunk("fallback", doc_id, section, title, index, text)

    def _chunk_policy_pdf(self, filename: str, text: str) -> None:
        raw_lines = [line.strip() for line in text.splitlines()]
        doc_id = Path(filename).stem.upper()
        title = Path(filename).stem
        paragraphs: list[str] = []
        buffer: list[str] = []
        for line in raw_lines:
            if line.lower().startswith("document:"):
                doc_id = (line.split(":", 1)[1].strip().split() or [doc_id])[0].upper()
            if not line:
                if buffer:
                    paragraphs.append(normalize_text(" ".join(buffer)))
                    buffer = []
                continue
            buffer.append(line)
        if buffer:
            paragraphs.append(normalize_text(" ".join(buffer)))

        pending: list[str] = []
        pending_words = 0
        chunk_index = 0
        for paragraph in paragraphs:
            words = paragraph.split()
            if len(words) > 650:
                if pending:
                    chunk_index = self._flush_policy_chunk(filename, doc_id, title, chunk_index, pending)
                    pending = []
                    pending_words = 0
                for offset in range(0, len(words), 560):
                    piece = " ".join(words[offset:offset + 650])
                    chunk_index = self._flush_policy_chunk(filename, doc_id, title, chunk_index, [piece])
                continue
            if pending_words + len(words) > 650 and pending:
                chunk_index = self._flush_policy_chunk(filename, doc_id, title, chunk_index, pending)
                pending = []
                pending_words = 0
            pending.append(paragraph)
            pending_words += len(words)
        if pending:
            self._flush_policy_chunk(filename, doc_id, title, chunk_index, pending)

    def _flush_policy_chunk(self, filename: str, doc_id: str, title: str, chunk_index: int, paragraphs: list[str]) -> int:
        text = normalize_text("\n".join(paragraphs))
        if len(text) >= 40:
            self._append_chunk(filename, doc_id, f"chunk-{chunk_index + 1}", title, chunk_index, text)
            chunk_index += 1
        return chunk_index

    def _append_chunk(self, filename: str, doc_id: str, section: str, title: str, chunk_index: int, text: str) -> None:
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        chunk_id = hashlib.sha256(f"{filename}:{chunk_index}:{content_hash}".encode("utf-8")).hexdigest()[:24]
        self.chunks.append(
            {
                "id": chunk_id,
                "document_id": hashlib.sha256(filename.encode("utf-8")).hexdigest()[:24],
                "filename": filename,
                "doc_id": doc_id,
                "section": section,
                "title": title,
                "chunk_index": chunk_index,
                "text": text,
                "content_hash": content_hash,
            }
        )

    def _ensure_indexed(self) -> None:
        by_document: dict[str, dict[str, Any]] = {}
        for chunk in self.chunks:
            by_document[chunk["document_id"]] = chunk
        with self.connect() as conn:
            active_ids = [chunk["id"] for chunk in self.chunks]
            if active_ids:
                placeholders = ",".join("?" for _ in active_ids)
                conn.execute(f"DELETE FROM policy_chunks WHERE id NOT IN ({placeholders})", active_ids)
            for document_id, chunk in by_document.items():
                conn.execute(
                    """
                    INSERT INTO policy_documents(id, filename, content_hash, indexed_at)
                    VALUES(?,?,?,?)
                    ON CONFLICT(id) DO UPDATE SET filename=excluded.filename, content_hash=excluded.content_hash
                    """,
                    (document_id, chunk["filename"], chunk["content_hash"], now_iso()),
                )
            for chunk in self.chunks:
                conn.execute(
                    """
                    INSERT INTO policy_chunks(id, document_id, doc_id, section, title, chunk_index, text, content_hash)
                    VALUES(?,?,?,?,?,?,?,?)
                    ON CONFLICT(id) DO UPDATE SET text=excluded.text, content_hash=excluded.content_hash
                    """,
                    (chunk["id"], chunk["document_id"], chunk["doc_id"], chunk["section"], chunk["title"], chunk["chunk_index"], chunk["text"], chunk["content_hash"]),
                )
        for chunk in self.chunks:
            self._ensure_embedding(chunk)

    def _ensure_embedding(self, chunk: dict[str, Any]) -> None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT chunk_id FROM policy_embeddings WHERE chunk_id=? AND model=?",
                (chunk["id"], self.client.embedding_model),
            ).fetchone()
        if row:
            return
        vector = self.client.embed_text(chunk["text"], "RETRIEVAL_DOCUMENT", chunk.get("title", ""))
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO policy_embeddings(chunk_id, model, dimension, embedding_blob, created_at) VALUES(?,?,?,?,?)",
                (chunk["id"], self.client.embedding_model, len(vector), pack_vector(vector), now_iso()),
            )

    def retrieve_chunks(self, query: str, limit: int = RETRIEVAL_TOP_K, tracer: StepTracer | None = None) -> list[dict[str, Any]]:
        tracer = tracer or StepTracer(enabled=False)
        if not self.client.configured:
            raise RuntimeError("GEMINI_API_KEY is required for policy retrieval.")
        if not query.strip():
            return []
        if not self._has_embeddings():
            self._ensure_indexed()
        with tracer.step("embed_query", model_used=self.client.embedding_model) as step:
            query_vector = self.client.embed_text(query, "RETRIEVAL_QUERY", trace_step=step)
        with tracer.step("retrieve_policy", model_used=self.client.embedding_model) as step:
            rows = []
            with self.connect() as conn:
                rows = [
                    dict(row)
                    for row in conn.execute(
                        """
                        SELECT pc.*, pe.embedding_blob
                        FROM policy_chunks pc
                        JOIN policy_embeddings pe ON pe.chunk_id=pc.id
                        WHERE pe.model=?
                        """,
                        (self.client.embedding_model,),
                    )
                ]
            scored = []
            for row in rows:
                score = cosine_similarity(query_vector, unpack_vector(row.pop("embedding_blob")))
                row["score"] = score
                scored.append(row)
            scored.sort(key=lambda item: item["score"], reverse=True)
            selected = scored[:limit]
            step.notes = "retrieved=" + ", ".join(f"{item['doc_id']}:{item['section']}:{item['score']:.3f}" for item in selected)
            return selected

    def _has_embeddings(self) -> bool:
        with self.connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM policy_embeddings WHERE model=?", (self.client.embedding_model,)).fetchone()
            return bool(row and row["count"])

    def citations_from_chunks(self, chunks: list[dict[str, Any]], selected_ids: list[str] | None = None) -> list[Citation]:
        selected = set(selected_ids or [])
        output = []
        for chunk in chunks:
            if selected and chunk["id"] not in selected:
                continue
            quote = chunk["text"]
            if len(quote) > 520:
                quote = quote[:517].rsplit(" ", 1)[0] + "..."
            output.append(Citation(chunk["doc_id"], chunk["section"], chunk["title"], quote))
        return output

    def answer(self, question: str) -> dict[str, Any]:
        tracer = StepTracer(enabled=True)
        try:
            chunks = self.retrieve_chunks(question, RETRIEVAL_TOP_K, tracer)
            if not chunks:
                return {
                    "answer": "I can only answer questions grounded in the Northwind policy library, and I did not find relevant policy text for that question.",
                    "confidence": 0.1,
                    "citations": [],
                    "refused": True,
                    "pipeline_trace": tracer.to_list(),
                }
            prompt = policy_qa_prompt(question, chunks)
            with tracer.step("generate_answer", model_used=self.client.generate_model) as step:
                data = self.client.generate_json(prompt, QA_SCHEMA, step)
            selected_ids = [str(item) for item in data.get("citation_chunk_ids") or []]
            citations = self.citations_from_chunks(chunks, selected_ids)
            refused = bool(data.get("refused")) or not citations
            return {
                "answer": str(data.get("answer") or "I did not find enough policy support to answer that question."),
                "confidence": clamp(data.get("confidence"), 0.0, 1.0, 0.2),
                "citations": [] if refused else [citation_to_dict(citation) for citation in citations],
                "refused": refused,
                "pipeline_trace": tracer.to_list(),
            }
        except Exception as exc:
            with tracer.step("generate_answer", model_used=self.client.generate_model) as step:
                step.status = "error"
                step.notes = str(exc)[:240]
            return {
                "answer": "Policy Q&A requires a configured Gemini API key and reachable Gemini service.",
                "confidence": 0.0,
                "citations": [],
                "refused": True,
                "pipeline_trace": tracer.to_list(),
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
            warnings.append("Image OCR unavailable or returned no text. Configure GEMINI_API_KEY or pytesseract for image receipts.")
        return text or "", warnings
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

    try:
        return GEMINI.generate_text_with_image(
            path,
            "Extract every visible receipt field as plain text. Preserve totals, dates, merchant name, line items, taxes, tips, payment method, attendee notes, and any mismatch warnings.",
            trace_step,
        )
    except Exception:
        return ""


def policy_chunks_payload(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": chunk["id"],
            "doc_id": chunk["doc_id"],
            "section": chunk["section"],
            "title": chunk["title"],
            "text": chunk["text"],
        }
        for chunk in chunks
    ]


def policy_qa_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "You answer Northwind policy questions using only the supplied retrieved policy chunks.",
            "If the question is not about the supplied Northwind policy text, or the chunks do not support an answer, set refused=true and explain that you can only answer from the policy library.",
            "Cite only chunk_id values that directly support the answer.",
            "Return JSON matching the schema.",
            "",
            "Question:",
            question,
            "",
            "Retrieved policy chunks:",
            json.dumps(policy_chunks_payload(chunks), ensure_ascii=False, indent=2),
        ]
    )


def receipt_extraction_prompt(filename: str, text: str, employee: dict[str, Any] | None) -> str:
    return "\n".join(
        [
            "Extract normalized expense receipt facts from the supplied OCR/PDF text.",
            "Do not use keyword rules. Read the receipt and infer fields from the evidence. Use null for missing fields.",
            "Choose category from air, lodging, ground, meal, conference, or unknown.",
            "For confidence, use a 0 to 1 score based on extraction completeness and ambiguity.",
            "Add warnings for missing amount, missing date, unclear merchant, line-item mismatch, unsupported receipt type, or ambiguous category.",
            "Return JSON matching the schema.",
            "",
            "Filename:",
            filename,
            "",
            "Employee/trip context:",
            json.dumps(employee or {}, ensure_ascii=False, indent=2),
            "",
            "Receipt text:",
            text[:14000],
        ]
    )


def review_retrieval_query(facts: ReceiptFacts, all_facts: list[ReceiptFacts], employee: dict[str, Any]) -> str:
    return json.dumps(
        {
            "task": "Find Northwind travel and expense policy clauses needed to review this receipt.",
            "employee": employee,
            "receipt": facts_to_dict(facts, include_text=True),
            "submission_receipts": [facts_summary(item) for item in all_facts],
        },
        ensure_ascii=False,
    )


def review_prompt(facts: ReceiptFacts, all_facts: list[ReceiptFacts], employee: dict[str, Any], chunks: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "You are Northwind's finance expense pre-review engine.",
            "Use only the employee/trip context, receipt facts, related submission receipts, and retrieved policy chunks supplied below.",
            "Return a verdict of compliant, flagged, rejected, or needs_review. Flag policy exceptions, reject clearly non-reimbursable expenses, and use needs_review when evidence is insufficient.",
            "Amounts must be based on receipt facts and policy text. Use null when the reimbursable split cannot be calculated from the provided evidence.",
            "Cite only chunk_id values from the retrieved policy chunks that directly support the verdict.",
            "Return JSON matching the schema.",
            "",
            "Employee/trip context:",
            json.dumps(employee, ensure_ascii=False, indent=2),
            "",
            "Receipt under review:",
            json.dumps(facts_to_dict(facts, include_text=True), ensure_ascii=False, indent=2),
            "",
            "Other receipts in the same submission:",
            json.dumps([facts_summary(item) for item in all_facts if item.filename != facts.filename], ensure_ascii=False, indent=2),
            "",
            "Retrieved policy chunks:",
            json.dumps(policy_chunks_payload(chunks), ensure_ascii=False, indent=2),
        ]
    )


def facts_summary(facts: ReceiptFacts) -> dict[str, Any]:
    return facts_to_dict(facts, include_text=False)


def facts_to_dict(facts: ReceiptFacts, include_text: bool = False) -> dict[str, Any]:
    data = {
        "filename": facts.filename,
        "vendor": facts.vendor,
        "date": facts.date,
        "category": facts.category,
        "meal_type": facts.meal_type,
        "amount": facts.amount,
        "subtotal": facts.subtotal,
        "tip": facts.tip,
        "tax": facts.tax,
        "nights": facts.nights,
        "city": facts.city,
        "confidence": facts.confidence,
        "warnings": facts.warnings,
    }
    if include_text:
        data["text"] = facts.text[:12000]
    return data


def facts_from_model(filename: str, text: str, data: dict[str, Any], warnings: list[str]) -> ReceiptFacts:
    model_warnings = [str(item) for item in data.get("warnings") or []]
    vendor = str(data.get("vendor") or "Unknown merchant")[:120]
    category = str(data.get("category") or "unknown")
    if category not in {"air", "lodging", "ground", "meal", "conference", "unknown"}:
        category = "unknown"
    meal_type = data.get("meal_type")
    if meal_type not in {"breakfast", "lunch", "dinner", "snack", "unknown", None}:
        meal_type = "unknown"
    return ReceiptFacts(
        filename=filename,
        text=text,
        vendor=vendor or "Unknown merchant",
        date=str(data.get("date")) if data.get("date") else None,
        category=category,
        meal_type=str(meal_type) if meal_type else None,
        amount=coerce_float(data.get("amount")),
        subtotal=coerce_float(data.get("subtotal")),
        tip=coerce_float(data.get("tip")),
        tax=coerce_float(data.get("tax")),
        nights=coerce_int(data.get("nights")),
        city=str(data.get("city")) if data.get("city") else None,
        confidence=clamp(data.get("confidence"), 0.0, 1.0, 0.45),
        warnings=warnings + model_warnings,
    )


def fallback_facts(filename: str, text: str, warnings: list[str]) -> ReceiptFacts:
    combined = list(warnings)
    if not text:
        combined.append("No receipt text was extracted.")
    combined.append("Gemini structured extraction was unavailable, so this receipt requires human review.")
    return ReceiptFacts(filename=filename, text=text, confidence=0.15, warnings=combined)


def parse_receipt(path: Path, employee: dict[str, Any] | None = None, tracer: StepTracer | None = None) -> ReceiptFacts:
    tracer = tracer or StepTracer(enabled=False)
    with tracer.step("ocr") as step:
        text, warnings = extract_text_receipt(path, step)
        if warnings:
            step.status = "degraded"
            step.notes = "; ".join(warnings)[:240]
    text = normalize_text(text)
    with tracer.step("extract_receipt_facts", model_used=GEMINI.generate_model) as step:
        try:
            data = GEMINI.generate_json(receipt_extraction_prompt(path.name, text, employee), RECEIPT_FACT_SCHEMA, step)
            facts = facts_from_model(path.name, text, data, warnings)
            if facts.amount is None:
                facts.warnings.append("Could not identify a total amount.")
            if facts.date is None:
                facts.warnings.append("Could not identify a transaction date.")
            step.notes = (step.notes + f"; category={facts.category}, confidence={facts.confidence:.2f}").strip("; ")
            return facts
        except Exception as exc:
            step.status = "error"
            step.notes = str(exc)[:240]
            return fallback_facts(path.name, text, warnings)


def review_receipt(facts: ReceiptFacts, all_facts: list[ReceiptFacts], employee: dict[str, Any], policy: PolicyIndex, tracer: StepTracer | None = None) -> ReviewResult:
    tracer = tracer or StepTracer(enabled=False)
    if facts.warnings and facts.confidence < 0.3:
        return ReviewResult(
            "needs_review",
            facts.confidence,
            "The receipt could not be reliably extracted: " + " ".join(facts.warnings),
            [],
            facts.amount,
            None,
        )
    try:
        chunks = policy.retrieve_chunks(review_retrieval_query(facts, all_facts, employee), RETRIEVAL_TOP_K, tracer)
    except Exception as exc:
        with tracer.step("retrieve_policy", model_used=policy.client.embedding_model) as step:
            step.status = "error"
            step.notes = str(exc)[:240]
        return ReviewResult(
            "needs_review",
            min(facts.confidence, 0.35),
            "Policy retrieval requires a configured Gemini API key and reachable Gemini embedding service.",
            [],
            facts.amount,
            None,
        )
    with tracer.step("generate_verdict", model_used=policy.client.generate_model) as step:
        try:
            data = policy.client.generate_json(review_prompt(facts, all_facts, employee, chunks), REVIEW_SCHEMA, step)
        except Exception as exc:
            step.status = "error"
            step.notes = str(exc)[:240]
            return ReviewResult(
                "needs_review",
                min(facts.confidence, 0.35),
                "Gemini verdict generation failed, so this receipt needs human review.",
                policy.citations_from_chunks(chunks[:1]),
                facts.amount,
                None,
            )
    verdict = str(data.get("verdict") or "needs_review")
    if verdict not in {"compliant", "flagged", "rejected", "needs_review"}:
        verdict = "needs_review"
    selected_ids = [str(item) for item in data.get("citation_chunk_ids") or []]
    citations = policy.citations_from_chunks(chunks, selected_ids)
    if not citations and chunks:
        verdict = "needs_review"
        citations = policy.citations_from_chunks(chunks[:1])
    return ReviewResult(
        verdict,
        clamp(data.get("confidence"), 0.0, 1.0, min(facts.confidence, 0.6)),
        str(data.get("reasoning") or "Gemini did not provide a review rationale."),
        citations,
        coerce_float(data.get("reimbursable_amount")),
        coerce_float(data.get("non_reimbursable_amount")),
    )


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


GEMINI = GeminiClient()
POLICY = PolicyIndex(CASE_STUDY_DIR / "policies", GEMINI, DB_PATH)
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

    def read_multipart_form(self) -> tuple[dict[str, list[str]], list[tuple[str, str, bytes]]]:
        """Parse browser multipart uploads without deprecated cgi.FieldStorage."""
        length = int(self.headers.get("Content-Length", "0"))
        content_type = self.headers.get("Content-Type", "")
        body = self.rfile.read(length)
        message = BytesParser(policy=email_policy.default).parsebytes(
            b"Content-Type: " + content_type.encode("utf-8") + b"\r\n"
            b"MIME-Version: 1.0\r\n\r\n" + body
        )
        fields: dict[str, list[str]] = {}
        files: list[tuple[str, str, bytes]] = []
        for part in message.iter_parts():
            disposition = part.get("Content-Disposition", "")
            if not disposition:
                continue
            name = part.get_param("name", header="content-disposition")
            if not name:
                continue
            payload = part.get_payload(decode=True) or b""
            filename = part.get_filename()
            if filename:
                files.append((name, filename, payload))
            else:
                fields.setdefault(name, []).append(payload.decode(part.get_content_charset() or "utf-8"))
        return fields, files

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
                return self.send_json({
                    "ok": True,
                    "policies": len(POLICY.chunks),
                    "case_study_dir": str(CASE_STUDY_DIR),
                    "gemini_configured": GEMINI.configured,
                    "embedding_model": GEMINI.embedding_model,
                    "index_error": POLICY.index_error,
                })
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
        fields, files = self.read_multipart_form()
        employee_json = fields.get("employee", [None])[0]
        if not employee_json:
            return self.send_json({"error": "employee payload is required"}, 400)
        employee = json.loads(employee_json)
        if not employee.get("employee_id"):
            employee["employee_id"] = "NW-" + uuid.uuid4().hex[:8].upper()
        upload_dir = UPLOAD_DIR / ("incoming-" + uuid.uuid4().hex)
        upload_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for field_name, filename, payload in files:
            if field_name != "receipts" or not filename:
                continue
            original_name = Path(filename).name
            safe_name = "".join(ch if ch.isalnum() or ch in "_.-" else "-" for ch in original_name).strip(".-") or f"receipt-{uuid.uuid4().hex}.dat"
            target = upload_dir / safe_name
            target.write_bytes(payload)
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
    .loading-row {
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--muted);
      font-size: 13px;
    }
    .loading-dot {
      width: 18px;
      height: 18px;
      border: 2px solid #cbd5e1;
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin .8s linear infinite;
      flex: 0 0 auto;
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
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
          <div class="muted">${s.item_count} receipts Â· ${fmtMoney(s.total_amount)} Â· ${s.issue_count || 0} issues</div>
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
            <div class="muted">${esc(sub.title || "")} Â· ${esc(sub.department || "")}</div>
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
            <div class="muted">${esc(item.category)} ${item.meal_type ? "Â· " + esc(item.meal_type) : ""}</div>
          </div>
          <div>
            <p>${esc(item.reasoning)}</p>
            ${(item.citations || []).map(c => `<div class="citation"><strong>${esc(c.doc_id)} Â§${esc(c.section)}</strong><br>${esc(c.quote)}</div>`).join("")}
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
      $("askBtn").disabled = true;
      $("answerBox").innerHTML = `
        <div class="loading-row" role="status" aria-live="polite">
          <span class="loading-dot" aria-hidden="true"></span>
          <span>Generating answer...</span>
        </div>
      `;
      try {
        const data = await api("/api/policy-question", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({question})
        });
        $("answerBox").innerHTML = `
          <p>${esc(data.answer)}</p>
          <div class="muted">Confidence ${Math.round(Number(data.confidence || 0) * 100)}%${data.refused ? " Â· refused" : ""}</div>
          ${(data.citations || []).map(c => `<div class="citation"><strong>${esc(c.doc_id)} Â§${esc(c.section)}</strong><br>${esc(c.quote)}</div>`).join("")}
        `;
      } catch (err) {
        $("answerBox").innerHTML = `<p class="muted">${esc(err.message)}</p>`;
      } finally {
        $("askBtn").disabled = false;
      }
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
