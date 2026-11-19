from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import re
import textwrap
from typing import Any

import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

AICTE_CATALOGUE_URL = "https://internship.aicte-india.org/api/v1/catalogue/internships"
SNAPSHOT_PATH = Path(__file__).parent / "src" / "data" / "aicteSnapshot.json"

st.set_page_config(
    page_title="nextstepz — AICTE Internship Recommender",
    layout="wide",
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+#\s]", " ", text.lower())).strip()


def skill_matches_corpus(skill: str, norm_corpus: str) -> bool:
    norm_skill = normalize_text(skill)
    if not norm_skill:
        return False
    pattern = rf"(?:^|\s){re.escape(norm_skill)}(?:$|\s)"
    return re.search(pattern, norm_corpus) is not None


def sanitize_pdf_text(text: str) -> str:
    cleaned = (
        text.replace("₹", "Rs. ")
        .replace("—", "-")
        .replace("–", "-")
        .replace("•", "-")
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
    )
    cleaned = re.sub(r"[^\x20-\x7E\n]", " ", cleaned)
    return re.sub(r"[ \t]+", " ", cleaned).strip()


def normalize_item(raw: dict[str, Any]) -> dict[str, Any]:
    item_id = str(raw.get("id") or raw.get("uid") or "")
    stipend = int(raw.get("stipend") or 0)
    stipend_display = str(raw.get("stipendDisplay") or "").strip()
    if not stipend_display:
        stipend_display = f"₹{stipend:,}" if stipend > 0 else "Unpaid"

    return {
        "id": item_id,
        "title": str(raw.get("title") or "AICTE Internship"),
        "organization": str(raw.get("organization") or "AICTE Partner Organization"),
        "category": str(raw.get("category") or "Corporate"),
        "location": str(raw.get("location") or "Pan India"),
        "mode": str(raw.get("mode") or "Remote"),
        "duration": str(raw.get("duration") or "2 Months"),
        "stipendDisplay": stipend_display,
        "daysLeft": int(raw.get("daysLeft") or 7),
        "skills": [str(s) for s in (raw.get("skills") or [])],
        "description": str(raw.get("description") or ""),
        "eligibility": [str(e) for e in (raw.get("eligibility") or [])],
        "verified": bool(raw.get("verified", True)),
        "portalUrl": f"https://internship.aicte-india.org/internships/{item_id}" if item_id else "",
    }


@st.cache_data(ttl=600, show_spinner=False)
def fetch_aicte_internships() -> tuple[list[dict[str, Any]], int, str]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    total_on_portal = 0

    try:
        for page in (1, 2, 3):
            res = requests.get(
                AICTE_CATALOGUE_URL,
                params={"sort": "newest", "page": page, "pageSize": 24},
                headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
                timeout=8,
            )
            res.raise_for_status()
            payload = res.json()
            if page == 1:
                total_on_portal = int(payload.get("total") or 0)
            for raw in payload.get("items") or []:
                norm = normalize_item(raw)
                if norm["id"] and norm["id"] not in seen:
                    seen.add(norm["id"])
                    items.append(norm)
        if items:
            return items, total_on_portal or len(items), "live"
    except Exception:
        pass

    if SNAPSHOT_PATH.exists():
        snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
        fallback_items = [normalize_item(raw) for raw in snapshot.get("items") or []]
        return (
            fallback_items,
            int(snapshot.get("totalOnPortal") or len(fallback_items)),
            "cached_snapshot",
        )

    return [], 0, "empty"


def build_internship_doc(item: dict[str, Any]) -> str:
    return (
        f"{item['title']} | {item['organization']} ({item['category']}) | "
        f"Skills: {', '.join(item['skills'][:8])} | {item['mode']} | {item['description'][:220]}"
    )


def compute_gemini_similarities(
    resume_text: str, internships: list[dict[str, Any]]
) -> list[float] | None:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        docs = [f"Candidate Resume: {resume_text[:1800]}"] + [
            build_internship_doc(item) for item in internships
        ]
        vectors: list[list[float]] = []
        batch_size = 50
        for start in range(0, len(docs), batch_size):
            chunk = docs[start : start + batch_size]
            res = client.models.embed_content(
                model="gemini-embedding-2",
                contents=[{"parts": [{"text": text}]} for text in chunk],
            )
            for emb in res.embeddings or []:
                vectors.append(list(emb.values or []))

        if len(vectors) != len(docs):
            return None

        matrix = np.array(vectors, dtype=np.float32)
        resume_vec = matrix[0:1]
        item_vecs = matrix[1:]
        raw_sims = cosine_similarity(resume_vec, item_vecs)[0]
        return [float(max(0.0, min(0.98, (s - 0.56) / 0.24))) for s in raw_sims]
    except Exception:
        return None


def compute_tfidf_similarities(
    resume_text: str, internships: list[dict[str, Any]]
) -> list[float]:
    docs = [resume_text] + [build_internship_doc(item) for item in internships]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(docs)
    raw_sims = cosine_similarity(matrix[0:1], matrix[1:])[0]
    return [float(max(0.0, min(0.94, np.sqrt(max(0.0, s)) * 1.18))) for s in raw_sims]


def recommend_internships(
    resume_text: str, internships: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], str]:
    cleaned = resume_text.strip()
    if not cleaned or not internships:
        return [], "none"

    gemini_sims = compute_gemini_similarities(cleaned, internships)
    if gemini_sims is not None:
        sims = gemini_sims
        method = "Gemini Embeddings"
    else:
        sims = compute_tfidf_similarities(cleaned, internships)
        method = "Local TF-IDF"

    norm_resume = normalize_text(cleaned)
    ranked: list[dict[str, Any]] = []

    for idx, item in enumerate(internships):
        semantic = sims[idx]
        matched: list[str] = []
        missing: list[str] = []

        if item["skills"]:
            for skill in item["skills"]:
                if skill_matches_corpus(skill, norm_resume):
                    matched.append(skill)
                else:
                    missing.append(skill)
            skill_overlap = min(
                1.0,
                0.65 * (len(matched) / max(3, len(item["skills"])))
                + 0.35 * min(1.0, len(matched) / 4.0),
            )
        else:
            title_tokens = [
                tok for tok in normalize_text(item["title"]).split() if len(tok) >= 4
            ]
            hits = [tok for tok in title_tokens if skill_matches_corpus(tok, norm_resume)]
            matched = hits[:4]
            skill_overlap = min(0.7, len(hits) / max(3, len(title_tokens) or 4))

        blended = 0.54 * semantic + 0.36 * skill_overlap + 0.10
        match_score = max(15, min(98, round(blended * 100)))

        ranked.append(
            {
                "internship": item,
                "matchScore": match_score,
                "matchedSkills": matched,
                "missingSkills": missing,
            }
        )

    ranked.sort(key=lambda r: r["matchScore"], reverse=True)
    return ranked[:12], method


def build_resume_readiness_report(
    resume_text: str, recommendations: list[dict[str, Any]]
) -> dict[str, Any]:
    cleaned = resume_text.strip()
    top8 = recommendations[:8]

    matched_counter: Counter[str] = Counter()
    missing_counter: Counter[str] = Counter()
    for rec in top8:
        for skill in rec["matchedSkills"]:
            matched_counter[skill] += 1
        for gap in rec["missingSkills"]:
            missing_counter[gap] += 1

    top_matched_skills = [skill for skill, _ in matched_counter.most_common(6)]
    top_missing_skills = missing_counter.most_common(4)

    has_education = bool(
        re.search(
            r"\b(b\.?tech|b\.?e\.?|b\.?sc|mca|bca|m\.?tech|diploma|cgpa|gpa|university|college|semester)\b",
            cleaned,
            flags=re.IGNORECASE,
        )
    )
    has_projects = bool(
        re.search(
            r"\b(project|projects|built|developed|deployed|designed|implemented|created|internship)\b",
            cleaned,
            flags=re.IGNORECASE,
        )
    )
    has_quantified_impact = bool(
        re.search(
            r"\b\d+(\.\d+)?\s*(%|x|ms|s|users|records|queries|hours|days|months|lakhs|cgpa)\b",
            cleaned,
            flags=re.IGNORECASE,
        )
    )
    has_links = bool(
        re.search(r"(github\.com|linkedin\.com|https?://|portfolio)", cleaned, flags=re.IGNORECASE)
    )
    word_count = len(cleaned.split()) if cleaned else 0

    avg_top_match = (
        sum(r["matchScore"] for r in top8) / len(top8) if top8 else 35.0
    )

    structure_points = 0
    if word_count >= 60:
        structure_points += 12
    elif word_count >= 25:
        structure_points += 6
    if has_education:
        structure_points += 8
    if has_projects:
        structure_points += 10
    if has_quantified_impact:
        structure_points += 6
    if has_links:
        structure_points += 4

    readiness_score = max(25, min(96, round(avg_top_match * 0.68 + structure_points)))
    if readiness_score >= 72:
        readiness_tier = "Ready to Apply"
    elif readiness_score >= 52:
        readiness_tier = "Competitive"
    else:
        readiness_tier = "Needs Strengthening"

    what_not_to_change: list[str] = []
    if top_matched_skills:
        what_not_to_change.append(
            f"Keep your core matched skills prominently listed: {', '.join(top_matched_skills)}. "
            "They directly align with your top AICTE matches."
        )
    if has_projects:
        what_not_to_change.append(
            "Keep your hands-on project and implementation descriptions—employers under AICTE "
            "and PM Internship Scheme prioritize practical build evidence."
        )
    if has_education:
        what_not_to_change.append(
            "Keep your academic degree, stream, and CGPA/standing details clearly visible for "
            "scheme eligibility verification."
        )
    if has_quantified_impact:
        what_not_to_change.append(
            "Keep the concrete metrics and measurable outcomes in your bullet points, as they "
            "demonstrate execution depth."
        )
    if has_links:
        what_not_to_change.append(
            "Keep your GitHub, portfolio, or LinkedIn links intact so reviewers can inspect "
            "live code and deliverables."
        )
    if len(what_not_to_change) < 2:
        top_cat = top8[0]["internship"]["category"] if top8 else "domain"
        top_title = top8[0]["internship"]["title"] if top8 else "AICTE Technical Internships"
        what_not_to_change.append(
            f'Keep your {top_cat} focus clear—your resume aligns best with roles like "{top_title}".'
        )

    what_to_change: list[str] = []
    if top_missing_skills:
        high_gap_list = ", ".join(
            f"{skill} ({count} top roles)" for skill, count in top_missing_skills[:3]
        )
        what_to_change.append(
            f"Add or build demonstrable proof for frequently requested gap skills across your top matches: {high_gap_list}."
        )
    if not has_projects:
        what_to_change.append(
            "Add a dedicated Projects section with 2–3 concrete builds, naming the tools/frameworks "
            "used and what problem each solved."
        )
    if not has_quantified_impact:
        what_to_change.append(
            "Quantify results in your experience bullets (e.g., dataset size, accuracy, response "
            "time, users served, or delivery timeline)."
        )
    if not has_education:
        what_to_change.append(
            "Include your degree program, branch, graduation year, and CGPA explicitly so AICTE "
            "eligibility checks pass smoothly."
        )
    if not has_links:
        what_to_change.append(
            "Add a GitHub or portfolio URL at the top of your resume to back up your technical claims."
        )
    if word_count < 60:
        what_to_change.append(
            "Expand your resume with fuller coursework, toolchains, and project responsibilities "
            "(aim for 120–350 words for stronger semantic matching)."
        )
    if not what_to_change:
        what_to_change.append(
            "Tailor your top summary line to mirror the exact role title and domain terminology "
            "of your #1 target AICTE listing before applying."
        )

    lead_title = top8[0]["internship"]["title"] if top8 else "AICTE internships"
    lead_score = top8[0]["matchScore"] if top8 else 0
    gap_focus = (
        " & ".join(skill for skill, _ in top_missing_skills[:2])
        if top_missing_skills
        else "role-specific"
    )
    summary = (
        f'Your resume scores {readiness_score}/100 ({readiness_tier}) against live AICTE listings, '
        f'led by "{lead_title}" ({lead_score}% match). Focus on closing {gap_focus} gaps while '
        "preserving your verified technical strengths."
    )

    return {
        "generatedAt": datetime.now().strftime("%d %b %Y"),
        "readinessScore": readiness_score,
        "readinessTier": readiness_tier,
        "summary": summary,
        "whatNotToChange": what_not_to_change[:5],
        "whatToChange": what_to_change[:5],
        "topTargetRoles": [
            {
                "title": r["internship"]["title"],
                "organization": r["internship"]["organization"],
                "matchScore": r["matchScore"],
                "mode": r["internship"]["mode"],
                "stipendDisplay": r["internship"]["stipendDisplay"],
            }
            for r in recommendations[:5]
        ],
    }


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_pure_python_pdf(report: dict[str, Any]) -> bytes:
    """Zero-dependency fallback PDF 1.4 generator if fpdf2 is not installed."""
    page_w, page_h = 595, 842  # A4 in points
    margin_x = 48
    top_y = page_h - 54
    bottom_y = 54

    # Build a stream of styled lines: (font, size, r, g, b, text, extra_gap_before)
    entries: list[tuple[str, int, float, float, float, str, int]] = []
    entries.append(("F2", 18, 0.06, 0.09, 0.16, "nextstepz - Resume Readiness Report", 0))
    meta_line = (
        f"Generated {report['generatedAt']}  |  "
        f"Readiness Score: {report['readinessScore']}/100 ({report['readinessTier']})"
    )
    entries.append(("F1", 10, 0.39, 0.45, 0.55, sanitize_pdf_text(meta_line), 4))
    entries.append(("RULE", 0, 0.89, 0.91, 0.94, "", 6))

    for line in textwrap.wrap(sanitize_pdf_text(report["summary"]), width=86):
        entries.append(("F1", 11, 0.12, 0.16, 0.23, line, 0))

    entries.append(("F2", 13, 0.08, 0.50, 0.24, "What Not to Change (Keep These Strengths)", 14))
    for item in report["whatNotToChange"]:
        wrapped = textwrap.wrap(f"- {sanitize_pdf_text(item)}", width=86)
        for i, line in enumerate(wrapped):
            entries.append(("F1", 10, 0.12, 0.16, 0.23, line if i == 0 else f"  {line}", 3 if i == 0 else 0))

    entries.append(("F2", 13, 0.71, 0.33, 0.04, "What to Change (Recommended Improvements)", 14))
    for item in report["whatToChange"]:
        wrapped = textwrap.wrap(f"- {sanitize_pdf_text(item)}", width=86)
        for i, line in enumerate(wrapped):
            entries.append(("F1", 10, 0.12, 0.16, 0.23, line if i == 0 else f"  {line}", 3 if i == 0 else 0))

    if report["topTargetRoles"]:
        entries.append(("F2", 13, 0.06, 0.09, 0.16, "Top Recommended AICTE Internships", 14))
        for idx, role in enumerate(report["topTargetRoles"], start=1):
            role_line = (
                f"{idx}. {sanitize_pdf_text(role['title'])} "
                f"({sanitize_pdf_text(role['organization'])}) - "
                f"{role['matchScore']}% Match | {sanitize_pdf_text(role['mode'])} | "
                f"{sanitize_pdf_text(role['stipendDisplay'])}"
            )
            wrapped = textwrap.wrap(role_line, width=88)
            for i, line in enumerate(wrapped):
                entries.append(("F1", 10, 0.12, 0.16, 0.23, line if i == 0 else f"   {line}", 3 if i == 0 else 0))

    # Paginate into PDF content streams
    pages_cmds: list[list[str]] = [[]]
    y = top_y

    for font, size, r, g, b, text, gap_before in entries:
        line_h = 10 if font == "RULE" else int(size * 1.4)
        needed = gap_before + line_h
        if y - needed < bottom_y:
            pages_cmds.append([])
            y = top_y
        y -= gap_before
        if font == "RULE":
            pages_cmds[-1].append(
                f"{r:.3f} {g:.3f} {b:.3f} RG 1 w {margin_x} {y} m {page_w - margin_x} {y} l S"
            )
            y -= 14
        else:
            escaped = _pdf_escape(text)
            pages_cmds[-1].append(
                f"BT /{font} {size} Tf {r:.3f} {g:.3f} {b:.3f} rg {margin_x} {y} Td ({escaped}) Tj ET"
            )
            y -= line_h

    # Assemble PDF objects
    objects: list[bytes] = []
    # 1: Catalog, 2: Pages, 3: Helvetica, 4: Helvetica-Bold
    num_pages = len(pages_cmds)
    page_obj_ids = [5 + i * 2 for i in range(num_pages)]
    kids_ref = " ".join(f"{pid} 0 R" for pid in page_obj_ids)

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids_ref}] /Count {num_pages} >>".encode("latin-1"))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")

    for i, cmds in enumerate(pages_cmds):
        page_id = page_obj_ids[i]
        content_id = page_id + 1
        stream_bytes = "\n".join(cmds).encode("latin-1", errors="replace")
        page_dict = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_w} {page_h}] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode("latin-1")
        content_obj = (
            f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("latin-1")
            + stream_bytes
            + b"\nendstream"
        )
        objects.append(page_dict)
        objects.append(content_obj)

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = [0]
    for obj_num, obj_body in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{obj_num} 0 obj\n".encode("latin-1"))
        out.extend(obj_body)
        out.extend(b"\nendobj\n")

    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("latin-1"))

    out.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode("latin-1")
    )
    return bytes(out)


def generate_readiness_pdf(report: dict[str, Any]) -> bytes:
    """Generate a downloadable PDF readiness report using fpdf2 (with pure-Python fallback)."""
    try:
        from fpdf import FPDF

        pdf = FPDF(unit="pt", format="A4")
        pdf.set_auto_page_break(auto=True, margin=48)
        pdf.set_margins(48, 48, 48)
        pdf.add_page()

        # Header
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 22, "nextstepz - Resume Readiness Report", ln=1)

        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(100, 116, 139)
        meta_line = (
            f"Generated {report['generatedAt']}  |  "
            f"Readiness Score: {report['readinessScore']}/100 ({report['readinessTier']})"
        )
        pdf.cell(0, 16, sanitize_pdf_text(meta_line), ln=1)

        pdf.set_draw_color(226, 232, 240)
        y_line = pdf.get_y() + 4
        pdf.line(48, y_line, pdf.w - 48, y_line)
        pdf.ln(14)

        # Summary
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(30, 41, 59)
        pdf.multi_cell(0, 16, sanitize_pdf_text(report["summary"]))
        pdf.ln(12)

        # What Not to Change
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(21, 128, 61)
        pdf.cell(0, 20, "What Not to Change (Keep These Strengths)", ln=1)
        pdf.set_font("Helvetica", "", 10.5)
        pdf.set_text_color(30, 41, 59)
        for item in report["whatNotToChange"]:
            pdf.multi_cell(0, 15, f"- {sanitize_pdf_text(item)}")
            pdf.ln(3)
        pdf.ln(8)

        # What to Change
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(180, 83, 9)
        pdf.cell(0, 20, "What to Change (Recommended Improvements)", ln=1)
        pdf.set_font("Helvetica", "", 10.5)
        pdf.set_text_color(30, 41, 59)
        for item in report["whatToChange"]:
            pdf.multi_cell(0, 15, f"- {sanitize_pdf_text(item)}")
            pdf.ln(3)
        pdf.ln(8)

        # Top Recommended AICTE Internships
        if report["topTargetRoles"]:
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(15, 23, 42)
            pdf.cell(0, 20, "Top Recommended AICTE Internships", ln=1)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(30, 41, 59)
            for idx, role in enumerate(report["topTargetRoles"], start=1):
                role_text = (
                    f"{idx}. {sanitize_pdf_text(role['title'])} "
                    f"({sanitize_pdf_text(role['organization'])}) - "
                    f"{role['matchScore']}% Match | {sanitize_pdf_text(role['mode'])} | "
                    f"{sanitize_pdf_text(role['stipendDisplay'])}"
                )
                pdf.multi_cell(0, 15, role_text)
                pdf.ln(2)

        raw_out = pdf.output()
        return bytes(raw_out) if isinstance(raw_out, (bytes, bytearray)) else raw_out.encode("latin-1")
    except Exception:
        return _build_pure_python_pdf(report)


def read_uploaded_resume(uploaded_file: Any) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    return uploaded_file.getvalue().decode("utf-8", errors="ignore")


def render_readiness_report(report: dict[str, Any]) -> None:
    with st.container(border=True):
        top_left, top_right = st.columns([3, 1], vertical_alignment="center")
        with top_left:
            st.markdown(
                f"### Resume Readiness Report &nbsp; `{report['readinessScore']}/100` &nbsp; "
                f"**{report['readinessTier']}**"
            )
            st.write(report["summary"])
        with top_right:
            pdf_bytes = generate_readiness_pdf(report)
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name="nextstepz-resume-readiness-report.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )

        st.divider()
        keep_col, change_col = st.columns(2, gap="large")
        with keep_col:
            st.markdown("#### What Not to Change")
            for item in report["whatNotToChange"]:
                st.markdown(f"- {item}")
        with change_col:
            st.markdown("#### What to Change")
            for item in report["whatToChange"]:
                st.markdown(f"- {item}")


def render_about_section() -> None:
    st.divider()
    col_intro, col_matching, col_readiness = st.columns([1, 1, 1], gap="large")
    with col_intro:
        st.markdown("#### About this website")
        st.caption(
            "Resume-driven internship discovery for AICTE and the PM Internship Scheme."
        )
    with col_matching:
        st.markdown("**Live AICTE & Semantic Matching**")
        st.caption(
            "nextstepz collects active internship openings directly from the AICTE National "
            "Internship Portal and compares them against your resume using neural sentence "
            "embeddings and skill-overlap scoring to surface your strongest matches."
        )
    with col_readiness:
        st.markdown("**Actionable Resume Readiness**")
        st.caption(
            "Alongside your ranked internship matches, nextstepz evaluates your resume against "
            "live role requirements—highlighting what strengths to keep, which missing skills or "
            "sections to update, and letting you download the full report as a PDF."
        )


def main() -> None:
    internships, total_on_portal, _ = fetch_aicte_internships()

    header_left, header_right = st.columns([3, 2])
    with header_left:
        st.markdown("## nextstepz")
    with header_right:
        st.markdown(
            f"<p style='text-align:right;color:#64748b;padding-top:14px;'>"
            f"{len(internships)} of {total_on_portal:,} live AICTE internships</p>",
            unsafe_allow_html=True,
        )
    st.divider()

    if "resume_text" not in st.session_state:
        st.session_state.resume_text = ""
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = None
    if "ranking_method" not in st.session_state:
        st.session_state.ranking_method = ""
    if "readiness_report" not in st.session_state:
        st.session_state.readiness_report = None
    if "last_uploaded_key" not in st.session_state:
        st.session_state.last_uploaded_key = ""

    left_col, right_col = st.columns([5, 7], gap="large")

    with left_col:
        st.subheader("Add Your Resume")
        st.caption(
            "Paste your resume or upload a PDF/TXT file, then click Generate Internships."
        )

        uploaded = st.file_uploader(
            "Upload PDF or TXT resume",
            type=["pdf", "txt", "md"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            upload_key = f"{uploaded.name}:{uploaded.size}"
            if upload_key != st.session_state.last_uploaded_key:
                extracted = read_uploaded_resume(uploaded)
                st.session_state.last_uploaded_key = upload_key
                if extracted:
                    st.session_state.resume_text = extracted
                    st.session_state.recommendations = None
                    st.session_state.readiness_report = None

        resume_input = st.text_area(
            "Resume text",
            value=st.session_state.resume_text,
            height=340,
            placeholder="Paste your resume here (Education, Technical Skills, Projects, Experience)...",
            label_visibility="collapsed",
        )
        st.session_state.resume_text = resume_input

        btn_col, clear_col = st.columns([3, 1])
        with btn_col:
            submitted = st.button(
                "Generate Internships",
                type="primary",
                use_container_width=True,
            )
        with clear_col:
            if st.button("Clear", use_container_width=True):
                st.session_state.resume_text = ""
                st.session_state.recommendations = None
                st.session_state.ranking_method = ""
                st.session_state.readiness_report = None
                st.session_state.last_uploaded_key = ""
                st.rerun()

        if submitted:
            if not resume_input.strip():
                st.error("Please paste your resume or upload a PDF/TXT file first.")
                st.session_state.recommendations = None
                st.session_state.readiness_report = None
            else:
                with st.spinner("Generating internship recommendations..."):
                    recs, method = recommend_internships(resume_input, internships)
                    st.session_state.recommendations = recs
                    st.session_state.ranking_method = method
                    st.session_state.readiness_report = (
                        build_resume_readiness_report(resume_input, recs) if recs else None
                    )

    with right_col:
        st.subheader("Recommended Internships")
        recs = st.session_state.recommendations
        if recs is None:
            st.info(
                "Your recommended internships will appear here. Add your resume on the left and click Generate Internships."
            )
        elif not recs:
            st.warning(
                "No matching internships found. Try adding more skills or project details to your resume."
            )
        else:
            st.caption(
                f"Top {len(recs)} of {len(internships)} AICTE listings · {st.session_state.ranking_method}"
            )
            for idx, rec in enumerate(recs, start=1):
                item = rec["internship"]
                with st.container(border=True):
                    top_a, top_b = st.columns([4, 1])
                    with top_a:
                        verified_label = (
                            "AICTE Verified" if item["verified"] else "Partner Listing"
                        )
                        st.caption(
                            f"#{idx:02d} · {item['organization']} · {item['category']} · {verified_label}"
                        )
                        st.markdown(f"**{item['title']}**")
                        st.caption(
                            f"{item['mode']} · {item['location']} · {item['duration']} · "
                            f"{item['stipendDisplay']} · {item['daysLeft']}d left"
                        )
                    with top_b:
                        st.markdown(f"**{rec['matchScore']}% Match**")
                        if item["portalUrl"]:
                            st.link_button("Apply", item["portalUrl"], use_container_width=True)

                    if rec["matchedSkills"]:
                        st.markdown(
                            f"**Matched:** {', '.join(rec['matchedSkills'][:5])}"
                        )
                    if rec["missingSkills"]:
                        st.caption(f"Gap: {', '.join(rec['missingSkills'][:3])}")

                    if item["description"] or item["skills"]:
                        with st.expander("View details"):
                            if item["description"]:
                                st.write(item["description"])
                            if item["skills"]:
                                st.write(f"**Required Skills:** {', '.join(item['skills'])}")
                            if item["eligibility"]:
                                st.write(
                                    f"**Eligibility:** {' · '.join(item['eligibility'])}"
                                )

            if st.session_state.readiness_report is not None:
                st.markdown("")
                render_readiness_report(st.session_state.readiness_report)

    render_about_section()


if __name__ == "__main__":
    main()
