import json
import os
import re
from pathlib import Path
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
                if normalize_text(skill) in norm_resume:
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
            hits = [tok for tok in title_tokens if tok in norm_resume]
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


def read_uploaded_resume(uploaded_file: Any) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    return uploaded_file.getvalue().decode("utf-8", errors="ignore")


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

    left_col, right_col = st.columns([5, 7], gap="large")

    with left_col:
        st.subheader("Add Your Resume")
        st.caption(
            "Paste your resume or upload a PDF/TXT file to get matching AICTE internships."
        )

        uploaded = st.file_uploader(
            "Upload PDF or TXT resume",
            type=["pdf", "txt", "md"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            extracted = read_uploaded_resume(uploaded)
            if extracted and extracted != st.session_state.resume_text:
                st.session_state.resume_text = extracted

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
                "Get Recommended Internships",
                type="primary",
                use_container_width=True,
            )
        with clear_col:
            if st.button("Clear", use_container_width=True):
                st.session_state.resume_text = ""
                st.session_state.recommendations = None
                st.session_state.ranking_method = ""
                st.rerun()

        if submitted:
            if not resume_input.strip():
                st.error("Please paste your resume or upload a PDF/TXT file first.")
                st.session_state.recommendations = None
            else:
                with st.spinner("Finding matching internships..."):
                    recs, method = recommend_internships(resume_input, internships)
                    st.session_state.recommendations = recs
                    st.session_state.ranking_method = method

    with right_col:
        st.subheader("Recommended Internships")
        recs = st.session_state.recommendations
        if recs is None:
            st.info(
                "Your recommended internships will appear here. Add your resume on the left and click Get Recommended Internships."
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


if __name__ == "__main__":
    main()
