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
    gap_focus = (
        " & ".join(skill for skill, _ in top_missing_skills[:2])
        if top_missing_skills
        else "role-specific"
    )
    summary = (
        f'Your resume scores {readiness_score}/100 ({readiness_tier}) against live AICTE listings, '
        f'led by "{lead_title}". Focus on closing {gap_focus} gaps while '
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


EXPERTISE_BANK: list[dict[str, Any]] = [
    {
        "area": "Cloud & DevOps",
        "keywords": [
            "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "devops",
            "ci/cd", "jenkins", "linux", "ec2", "s3", "cloud", "ansible", "nginx",
        ],
        "questions": [
            {
                "id": "cloud-1",
                "topic": "Containers & Docker",
                "prompt": "When writing a production Dockerfile for a Python or Node service, what is the primary benefit of using a multi-stage build?",
                "options": [
                    "It lets a single container run two separate kernels simultaneously.",
                    "It leaves build compilers and dev dependencies out of the final runtime image, shrinking size and attack surface.",
                    "It automatically deploys the image across multiple AWS availability zones.",
                    "It disables layer caching so every rebuild installs packages from scratch.",
                ],
                "correctIndex": 1,
                "explanation": "Multi-stage builds copy only compiled artifacts or runtime packages into a slim final image, reducing image size and vulnerabilities.",
            },
            {
                "id": "cloud-2",
                "topic": "Kubernetes Architecture",
                "prompt": "In Kubernetes, what resource ensures a stable DNS name and load-balanced endpoint even as individual Pods restart with new IP addresses?",
                "options": ["A ConfigMap", "A Service", "A PersistentVolumeClaim", "A DaemonSet"],
                "correctIndex": 1,
                "explanation": "A Kubernetes Service maps a stable virtual IP and DNS name to dynamic Pods matched by label selectors.",
            },
            {
                "id": "cloud-3",
                "topic": "AWS Security & IAM",
                "prompt": "An application running on an AWS EC2 instance needs read-only access to an S3 bucket. What is the recommended security practice?",
                "options": [
                    "Store root account access keys inside a .env file baked into the AMI.",
                    "Attach an IAM Role with a least-privilege S3 read policy directly to the EC2 instance.",
                    "Make the S3 bucket public and restrict access by checking the User-Agent header.",
                    "Embed temporary STS tokens inside the frontend JavaScript bundle.",
                ],
                "correctIndex": 1,
                "explanation": "IAM instance roles provide short-lived, automatically rotated credentials without storing static keys on disk.",
            },
            {
                "id": "cloud-4",
                "topic": "CI/CD Pipelines",
                "prompt": "Why should automated linting, unit tests, and vulnerability scans run before the artifact build/push step in a CI/CD pipeline?",
                "options": [
                    "To fail fast on broken code before spending compute and registry storage on invalid builds.",
                    "Because Docker cannot build images unless a unit-test coverage XML file is inside /etc.",
                    "To bypass branch protection rules on pull requests.",
                    "Because production load balancers execute unit tests on incoming HTTP traffic.",
                ],
                "correctIndex": 0,
                "explanation": "Running fast verification stages first prevents broken commits from producing deployable artifacts.",
            },
            {
                "id": "cloud-5",
                "topic": "Linux & Observability",
                "prompt": "A containerized service starts returning HTTP 502 behind Nginx after high traffic. Which check best identifies whether the container was OOM-killed?",
                "options": [
                    "Checking `ping -c 1 localhost` inside the host shell.",
                    "Inspecting container exit status / `dmesg` or `kubectl describe pod` for Exit Code 137 (OOMKilled).",
                    "Converting the Nginx config from tabs to spaces.",
                    "Clearing the browser DNS cache on a client laptop.",
                ],
                "correctIndex": 1,
                "explanation": "Exit Code 137 (SIGKILL) and kernel/pod OOMKilled events directly confirm memory limit exhaustion.",
            },
        ],
    },
    {
        "area": "Data Science & AI/ML",
        "keywords": [
            "machine learning", "deep learning", "data science", "data analysis", "nlp",
            "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "matplotlib",
            "power bi", "tableau", "llm", "rag", "ai", "ml", "computer vision",
        ],
        "questions": [
            {
                "id": "ds-1",
                "topic": "Preventing Data Leakage",
                "prompt": "When building a tabular ML pipeline with missing-value imputation and StandardScaler, when should `.fit()` be called?",
                "options": [
                    "On the combined train + test dataset before splitting so both share identical statistics.",
                    "Only on the training split inside cross-validation, then `.transform()` on validation/test splits.",
                    "Only on the test set so accuracy on unseen samples is maximized.",
                    "After exporting the trained model to ONNX format.",
                ],
                "correctIndex": 1,
                "explanation": "Fitting scalers or imputers on test data leaks distribution information and produces overly optimistic validation scores.",
            },
            {
                "id": "ds-2",
                "topic": "Imbalanced Classification Metrics",
                "prompt": "You are predicting fraudulent transactions where only 1% of rows are positive. Why is raw Accuracy misleading here?",
                "options": [
                    "A trivial classifier predicting 'not fraud' for every row still achieves 99% accuracy while missing all fraud.",
                    "Accuracy cannot be computed for binary classification targets.",
                    "Accuracy penalizes false positives twice as heavily as false negatives.",
                    "Gradient descent diverges whenever accuracy exceeds 90%.",
                ],
                "correctIndex": 0,
                "explanation": "On skewed classes, Precision-Recall AUC, F1-score, and Recall on the minority class reflect actual detection quality.",
            },
            {
                "id": "ds-3",
                "topic": "Embeddings & Cosine Similarity",
                "prompt": "Why is cosine similarity widely preferred over raw Euclidean distance when comparing high-dimensional text embeddings?",
                "options": [
                    "Cosine similarity measures the angle (directional alignment) between vectors rather than sensitivity to vector magnitude.",
                    "Cosine similarity guarantees zero false positives without requiring vector normalization.",
                    "Euclidean distance only works for 2D coordinates and throws errors above 3 dimensions.",
                    "Cosine similarity runs in O(1) time regardless of vocabulary size.",
                ],
                "correctIndex": 0,
                "explanation": "Directional similarity captures semantic overlap even when document lengths or embedding norms differ.",
            },
            {
                "id": "ds-4",
                "topic": "Regularization & Overfitting",
                "prompt": "A model reaches 99% training accuracy but drops to 64% validation accuracy. Which adjustment directly addresses this?",
                "options": [
                    "Increase model depth and train for twice as many epochs without early stopping.",
                    "Apply stronger regularization (L2/dropout), simplify features, or gather more diverse training data.",
                    "Remove the validation split and evaluate strictly on the training set.",
                    "Replace ReLU activations with identity linear functions across every layer.",
                ],
                "correctIndex": 1,
                "explanation": "A wide train-validation gap signals overfitting; regularization, pruning capacity, and data augmentation improve generalization.",
            },
            {
                "id": "ds-5",
                "topic": "Pandas & Data Cleaning",
                "prompt": "In pandas, what is the cleanest way to prevent silent `SettingWithCopyWarning` bugs when filtering a DataFrame and mutating a column?",
                "options": [
                    "Use chained indexing like `df[df['x'] > 0]['y'] = 1`.",
                    "Use explicit `.loc[mask, 'y'] = 1` or call `.copy()` before modifying the filtered slice.",
                    "Convert the entire DataFrame to Python lists on every row update.",
                    "Suppress all Python warnings globally.",
                ],
                "correctIndex": 1,
                "explanation": "`.loc` or an explicit `.copy()` guarantees mutations apply deterministically rather than to an ambiguous view.",
            },
        ],
    },
    {
        "area": "Full-Stack Web Engineering",
        "keywords": [
            "react", "next.js", "nextjs", "javascript", "typescript", "node.js", "nodejs",
            "express", "html", "css", "tailwind", "frontend", "backend", "full stack",
            "full-stack", "rest api", "graphql", "mongodb", "redux",
        ],
        "questions": [
            {
                "id": "web-1",
                "topic": "React Rendering & Keys",
                "prompt": "Why should you avoid using array indices as the `key` prop in a React list where items can be reordered, inserted, or deleted?",
                "options": [
                    "Array indices cause syntax errors in strict TypeScript.",
                    "React reuses component state and DOM nodes by key, causing stale inputs or visual glitches when positions shift.",
                    "Using unique IDs disables virtual DOM diffing.",
                    "Browser CSS grid layouts ignore elements with string keys.",
                ],
                "correctIndex": 1,
                "explanation": "Stable unique IDs let React track each item identity across insertions, deletions, and sorting.",
            },
            {
                "id": "web-2",
                "topic": "REST API Semantics",
                "prompt": "Which HTTP status code should an Express/Node endpoint return when a client submits malformed JSON or missing required fields?",
                "options": ["200 OK", "302 Found", "400 Bad Request", "503 Service Unavailable"],
                "correctIndex": 2,
                "explanation": "400 Bad Request signals a client input validation error rather than a server outage.",
            },
            {
                "id": "web-3",
                "topic": "Async Race Conditions",
                "prompt": "A search box fires an API fetch on every keystroke. How do you ensure a slower older response never overwrites a newer query result?",
                "options": [
                    "Store responses inside `localStorage` first.",
                    "Pass an `AbortController.signal` to `fetch` and abort the previous in-flight request in effect cleanup.",
                    "Use `setTimeout(..., 0)` inside the JSON parser.",
                    "Replace HTTPS with HTTP/1.0.",
                ],
                "correctIndex": 1,
                "explanation": "Aborting stale requests eliminates out-of-order state overwrites.",
            },
            {
                "id": "web-4",
                "topic": "Web Security (XSS)",
                "prompt": "Why is passing untrusted user input directly into `dangerouslySetInnerHTML` in React dangerous?",
                "options": [
                    "It triggers Cross-Site Scripting (XSS), allowing injected scripts or event handlers to execute in the browser.",
                    "It strips all CSS classes from sibling components.",
                    "It converts UTF-8 characters into binary.",
                    "It blocks CORS preflight requests.",
                ],
                "correctIndex": 0,
                "explanation": "Standard JSX interpolation escapes text safely, whereas raw HTML injection allows arbitrary script execution unless sanitized.",
            },
            {
                "id": "web-5",
                "topic": "Authentication & Secrets",
                "prompt": "Where should third-party API keys and database connection secrets be stored in a full-stack Vite + Express application?",
                "options": [
                    "Prefixed with `VITE_` in client source files so React can read them directly.",
                    "In server-side environment variables (`process.env`) accessed exclusively by backend `/api/*` routes.",
                    "Inside a public `secrets.json` file in `/public`.",
                    "Hardcoded in `App.tsx` inside a `useMemo` hook.",
                ],
                "correctIndex": 1,
                "explanation": "Frontend environment variables are bundled into browser code; private secrets must remain server-side.",
            },
        ],
    },
    {
        "area": "Cybersecurity & Networking",
        "keywords": [
            "cybersecurity", "cyber security", "siem", "wazuh", "nmap", "wireshark",
            "burp suite", "owasp", "penetration testing", "soc", "firewall", "networking",
            "tcp/ip", "vulnerability", "cryptography", "infosec", "splunk",
        ],
        "questions": [
            {
                "id": "sec-1",
                "topic": "Injection Defense",
                "prompt": "What is the most effective primary defense against SQL Injection in a backend web service?",
                "options": [
                    "Hiding database table names on the frontend.",
                    "Using parameterized queries / prepared statements that separate SQL structure from user data.",
                    "Base64-encoding user inputs before concatenating them into the SQL string.",
                    "Running the web server on a non-standard port.",
                ],
                "correctIndex": 1,
                "explanation": "Parameterized queries treat user inputs strictly as data values, never as executable SQL syntax.",
            },
            {
                "id": "sec-2",
                "topic": "SIEM & Log Triage",
                "prompt": "A SIEM alert shows 400 failed SSH logins from a single external IP followed immediately by one successful login. What should a SOC analyst do first?",
                "options": [
                    "Ignore the alert because only one login succeeded.",
                    "Verify whether the login is legitimate, isolate/block the source if unauthorized, revoke credentials, and inspect post-login process/audit logs.",
                    "Delete `/var/log/auth.log` to stop further alerts.",
                    "Restart DNS resolution on the client browser.",
                ],
                "correctIndex": 1,
                "explanation": "A brute-force pattern ending in success indicates likely compromise requiring containment, credential rotation, and host investigation.",
            },
            {
                "id": "sec-3",
                "topic": "TLS & Transport Security",
                "prompt": "During an HTTPS TLS handshake, what does verifying the server certificate chain primarily protect the client against?",
                "options": [
                    "Hard drive sector corruption on the client device.",
                    "Man-in-the-middle impersonation by confirming the server public key belongs to the trusted domain.",
                    "Excessive RAM usage inside the client browser.",
                    "Cross-site request forgery in local cookies.",
                ],
                "correctIndex": 1,
                "explanation": "Certificate validation binds the domain identity to a cryptographic key signed by a trusted Certificate Authority.",
            },
            {
                "id": "sec-4",
                "topic": "Access Control (OWASP)",
                "prompt": "An API endpoint `/api/invoices/1042` returns another customer's invoice simply by changing the numeric ID in the URL. Which OWASP Top 10 flaw is this?",
                "options": [
                    "Broken Access Control (IDOR)",
                    "Insufficient Logging & Monitoring",
                    "Weak PRNG Seeding",
                    "DNS Cache Poisoning",
                ],
                "correctIndex": 0,
                "explanation": "Insecure Direct Object Reference (IDOR) occurs when the server fails to verify ownership for the requested object ID.",
            },
            {
                "id": "sec-5",
                "topic": "Network Scanning & Ports",
                "prompt": "In Nmap, why might a host respond differently to a TCP SYN scan (`-sS`) versus a full TCP connect scan (`-sT`)?",
                "options": [
                    "`-sS` sends only SYN and tears down with RST before completing the 3-way handshake, making it faster and stealthier.",
                    "`-sS` scans UDP ports exclusively while `-sT` scans ICMP.",
                    "`-sT` decrypts TLS traffic automatically without keys.",
                    "Both flags perform identical ICMP echo pings only.",
                ],
                "correctIndex": 0,
                "explanation": "A half-open SYN scan avoids completing the full TCP connection unless elevated raw-socket privileges are unavailable.",
            },
        ],
    },
    {
        "area": "Embedded Systems & IoT",
        "keywords": [
            "embedded", "iot", "arduino", "esp32", "raspberry pi", "stm32", "microcontroller",
            "rtos", "mqtt", "uart", "spi", "i2c", "verilog", "vhdl", "vlsi", "pcb",
        ],
        "questions": [
            {
                "id": "iot-1",
                "topic": "Serial Buses (I2C vs SPI)",
                "prompt": "When connecting multiple sensors to a microcontroller, what is a key electrical difference between I2C and SPI?",
                "options": [
                    "I2C uses 2 shared open-drain lines (SDA/SCL) with device addresses, whereas SPI uses separate MOSI/MISO/SCLK lines plus a Chip Select per slave.",
                    "SPI requires external pull-up resistors to enforce 100 kbps speed.",
                    "I2C is always full-duplex while SPI is half-duplex only.",
                    "SPI does not use a clock line.",
                ],
                "correctIndex": 0,
                "explanation": "I2C saves pins via addressing on 2 wires; SPI achieves higher throughput using dedicated data lines and chip-select pins.",
            },
            {
                "id": "iot-2",
                "topic": "Interrupt Service Routines",
                "prompt": "Why should you avoid calling blocking delays (`delay()`) or heavy network calls inside a microcontroller Interrupt Service Routine (ISR)?",
                "options": [
                    "ISRs block lower-priority interrupts and main loop execution, causing missed events or watchdog resets.",
                    "ISRs automatically wipe EEPROM flash memory if they run longer than 1 microsecond.",
                    "Compilers prohibit `if` statements inside ISRs.",
                    "PWM timers reverse polarity when `delay()` is compiled.",
                ],
                "correctIndex": 0,
                "explanation": "Best practice is to set a `volatile` flag or push to an RTOS queue inside the ISR and handle heavy work in the main task.",
            },
            {
                "id": "iot-3",
                "topic": "IoT Telemetry (MQTT)",
                "prompt": "In MQTT, what does QoS Level 1 ('At least once') guarantee when publishing sensor readings over an unstable cellular/Wi-Fi link?",
                "options": [
                    "Messages are sent once and dropped silently if offline.",
                    "The broker/receiver acknowledges delivery via PUBACK, retrying until received (possible duplicates if ACK is lost).",
                    "End-to-end AES-256 encryption without TLS.",
                    "Deterministic sub-millisecond hardware interrupts.",
                ],
                "correctIndex": 1,
                "explanation": "QoS 1 guarantees delivery at least once using PUBACK acknowledgments.",
            },
            {
                "id": "iot-4",
                "topic": "Firmware Memory Safety",
                "prompt": "Why must a global flag shared between an ISR and the main `while(1)` loop in C be declared with the `volatile` keyword?",
                "options": [
                    "To allocate the variable inside external SD card storage.",
                    "To stop the compiler from caching its value in a CPU register and skipping real memory reads.",
                    "To automatically enable mutex locking across multiple CPU cores.",
                    "To double the integer bit width from 16 to 32 bits.",
                ],
                "correctIndex": 1,
                "explanation": "`volatile` tells the compiler the value can change asynchronously outside normal control flow, forcing fresh memory loads.",
            },
            {
                "id": "iot-5",
                "topic": "ADC Resolution & Sampling",
                "prompt": "An ESP32 or STM32 reads an analog temperature sensor with a 12-bit ADC referenced at 3.3V. Approximately how fine is 1 LSB step?",
                "options": [
                    "~0.8 mV (3.3V / 4096)",
                    "~12.9 mV (3.3V / 256)",
                    "~330 mV (3.3V / 10)",
                    "~3.3 V (3.3V / 1)",
                ],
                "correctIndex": 0,
                "explanation": "A 12-bit ADC divides the reference range into 4096 levels, giving ~0.805 mV per count at 3.3V.",
            },
        ],
    },
    {
        "area": "Core Software & Databases",
        "keywords": [
            "python", "java", "c++", "c#", "sql", "postgresql", "mysql", "sqlite",
            "data structures", "algorithms", "oop", "git", "backend", "api", "software",
        ],
        "questions": [
            {
                "id": "core-1",
                "topic": "SQL Indexing & Performance",
                "prompt": "A table with 5 million rows executes `SELECT * FROM applications WHERE student_email = ?` very slowly. What is the first database optimization to inspect?",
                "options": [
                    "Adding an index on `student_email` and checking the query plan with `EXPLAIN`.",
                    "Converting every column in the table to `TEXT`.",
                    "Disabling foreign key constraints across the schema.",
                    "Running `SELECT *` twice before rendering.",
                ],
                "correctIndex": 0,
                "explanation": "Without an index, the database performs a full sequential scan; a B-tree index drops lookup cost from O(N) to O(log N).",
            },
            {
                "id": "core-2",
                "topic": "Algorithms & Complexity",
                "prompt": "You need to check whether any pair of numbers in an unsorted list of 100,000 integers sums to a target `K`. Which approach gives O(N) average time?",
                "options": [
                    "Nested `for` loops comparing every pair `(i, j)`.",
                    "A single pass storing seen values in a hash set and checking if `(K - x)` already exists.",
                    "Bubble sort followed by linear scan from index 0.",
                    "Recursion branching 3 ways per element without memoization.",
                ],
                "correctIndex": 1,
                "explanation": "Hash-set complement lookup solves Two-Sum in O(N) average time and O(N) space instead of O(N^2).",
            },
            {
                "id": "core-3",
                "topic": "SQL Joins",
                "prompt": "You want to list all students—including those who have 0 internship applications—along with their application count. Which join should you use?",
                "options": [
                    "INNER JOIN between `students` and `applications`",
                    "LEFT JOIN from `students` to `applications` with `GROUP BY student_id`",
                    "CROSS JOIN without any `ON` condition",
                    "SELF JOIN on `applications` alone",
                ],
                "correctIndex": 1,
                "explanation": "LEFT JOIN preserves every row from `students` even when no matching rows exist in `applications`.",
            },
            {
                "id": "core-4",
                "topic": "Git Workflows",
                "prompt": "What is the key difference between `git fetch` and `git pull`?",
                "options": [
                    "`git fetch` downloads remote commits/refs safely without altering your local working branch, whereas `git pull` fetches and immediately merges/rebases.",
                    "`git pull` deletes local untracked files automatically.",
                    "`git fetch` pushes local branches to origin.",
                    "They are identical aliases.",
                ],
                "correctIndex": 0,
                "explanation": "`git fetch` lets you inspect upstream changes before integrating them into your current branch.",
            },
            {
                "id": "core-5",
                "topic": "API Error Handling & Retries",
                "prompt": "An external service occasionally returns HTTP 503 (Service Unavailable). Why should client retry logic add exponential backoff rather than looping immediately?",
                "options": [
                    "Immediate tight-loop retries amplify traffic spikes (thundering herd) and keep the overloaded service down.",
                    "HTTP 503 permanently deletes the endpoint if retried under 10 milliseconds.",
                    "Backoff automatically converts 503 into 200 OK on the client without hitting the server.",
                    "TCP sockets cannot open more than once per minute.",
                ],
                "correctIndex": 0,
                "explanation": "Exponential backoff spreads retries out, giving upstream services room to recover.",
            },
        ],
    },
]


def detect_expertise_area(
    resume_text: str, matched_skills: list[str] | None = None
) -> tuple[str, list[str]]:
    skills_list = matched_skills or []
    combined_corpus = normalize_text(f"{resume_text} {' '.join(skills_list)}")

    best_area = "Core Software & Databases"
    best_hits: list[str] = []
    best_score = 0.0

    for domain in EXPERTISE_BANK:
        hits = [kw for kw in domain["keywords"] if skill_matches_corpus(kw, combined_corpus)]
        specificity_bonus = 0.0 if domain["area"] == "Core Software & Databases" else 0.25
        score = len(hits) + specificity_bonus if hits else 0.0
        if score > best_score:
            best_score = score
            best_area = str(domain["area"])
            best_hits = hits

    seen: set[str] = set()
    detected_skills: list[str] = []
    for raw in [*skills_list, *[h.upper() for h in best_hits]]:
        val = raw.strip()
        if val and val.lower() not in seen:
            seen.add(val.lower())
            detected_skills.append(val)

    return best_area, detected_skills[:6]


def generate_expertise_quiz(
    resume_text: str, matched_skills: list[str] | None = None
) -> dict[str, Any]:
    area, detected_skills = detect_expertise_area(resume_text, matched_skills)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if api_key:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            prompt = (
                f'Create a practical 5-question multiple-choice technical test tailored strictly '
                f'to this candidate\'s resume and detected expertise area: "{area}".\n'
                f"Detected skills: {', '.join(detected_skills) or 'General technical skills'}\n"
                f"Resume:\n{resume_text.strip()[:2200]}\n"
                "Return JSON with keys `expertiseArea` (string) and `questions` (list of 5 objects "
                "with `topic`, `prompt`, `options` (exactly 4 strings), `correctIndex` (0-3), `explanation`)."
            )
            res = client.models.generate_content(
                model="gemini-4-hydrogen",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            parsed = json.loads((res.text or "{}").strip())
            raw_qs = parsed.get("questions") or []
            valid_qs: list[dict[str, Any]] = []
            for idx, q in enumerate(raw_qs[:5], start=1):
                opts = [str(o).strip() for o in (q.get("options") or [])]
                c_idx = q.get("correctIndex")
                if (
                    str(q.get("prompt") or "").strip()
                    and str(q.get("topic") or "").strip()
                    and str(q.get("explanation") or "").strip()
                    and len(opts) == 4
                    and isinstance(c_idx, int)
                    and 0 <= c_idx <= 3
                ):
                    valid_qs.append(
                        {
                            "id": f"ai-exp-{idx}",
                            "topic": str(q["topic"]).strip(),
                            "prompt": str(q["prompt"]).strip(),
                            "options": opts,
                            "correctIndex": c_idx,
                            "explanation": str(q["explanation"]).strip(),
                        }
                    )
            if len(valid_qs) == 5:
                return {
                    "expertiseArea": str(parsed.get("expertiseArea") or area).strip(),
                    "detectedSkills": detected_skills,
                    "source": "gemini",
                    "questions": valid_qs,
                }
        except Exception:
            pass

    domain_cfg = next(
        (d for d in EXPERTISE_BANK if d["area"] == area), EXPERTISE_BANK[-1]
    )
    return {
        "expertiseArea": area,
        "detectedSkills": detected_skills,
        "source": "curated",
        "questions": domain_cfg["questions"],
    }


def grade_expertise_quiz(
    quiz: dict[str, Any], selected_answers: dict[str, int]
) -> dict[str, Any]:
    outcomes: list[dict[str, Any]] = []
    for q in quiz["questions"]:
        sel_idx = selected_answers.get(q["id"], -1)
        is_correct = sel_idx == q["correctIndex"]
        options = q["options"]
        outcomes.append(
            {
                "id": q["id"],
                "topic": q["topic"],
                "prompt": q["prompt"],
                "selectedIndex": sel_idx,
                "correctIndex": q["correctIndex"],
                "isCorrect": is_correct,
                "selectedOption": options[sel_idx] if 0 <= sel_idx < len(options) else "Not answered",
                "correctOption": options[q["correctIndex"]],
                "explanation": q["explanation"],
            }
        )

    correct_count = sum(1 for o in outcomes if o["isCorrect"])
    total_count = len(quiz["questions"])
    score_percent = round((correct_count / total_count) * 100) if total_count > 0 else 0

    if score_percent >= 80:
        band = "Strong Mastery"
    elif score_percent >= 60:
        band = "Competent Foundation"
    else:
        band = "Needs Focused Review"

    strong_topics = list(dict.fromkeys(o["topic"] for o in outcomes if o["isCorrect"]))
    weak_topics = list(dict.fromkeys(o["topic"] for o in outcomes if not o["isCorrect"]))

    return {
        "completedAt": datetime.now().strftime("%d %b %Y"),
        "expertiseArea": quiz["expertiseArea"],
        "correctCount": correct_count,
        "totalCount": total_count,
        "scorePercent": score_percent,
        "proficiencyBand": band,
        "strongTopics": strong_topics,
        "weakTopics": weak_topics,
        "outcomes": outcomes,
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
                f"{sanitize_pdf_text(role['mode'])} | "
                f"{sanitize_pdf_text(role['stipendDisplay'])}"
            )
            wrapped = textwrap.wrap(role_line, width=88)
            for i, line in enumerate(wrapped):
                entries.append(("F1", 10, 0.12, 0.16, 0.23, line if i == 0 else f"   {line}", 3 if i == 0 else 0))

    exp = report.get("expertiseResult")
    if exp:
        entries.append(("F2", 13, 0.12, 0.25, 0.69, "Expertise Evaluation Test Summary", 14))
        exp_summary = (
            f"Area Tested: {sanitize_pdf_text(exp['expertiseArea'])}  |  "
            f"Score: {exp['correctCount']}/{exp['totalCount']} "
            f"({exp['scorePercent']}% - {sanitize_pdf_text(exp['proficiencyBand'])})"
        )
        for line in textwrap.wrap(exp_summary, width=88):
            entries.append(("F1", 10, 0.12, 0.16, 0.23, line, 2))
        if exp.get("strongTopics"):
            for line in textwrap.wrap(
                f"Verified Strong Topics: {sanitize_pdf_text(', '.join(exp['strongTopics']))}",
                width=88,
            ):
                entries.append(("F1", 10, 0.12, 0.16, 0.23, line, 2))
        if exp.get("weakTopics"):
            for line in textwrap.wrap(
                f"Topics to Review: {sanitize_pdf_text(', '.join(exp['weakTopics']))}",
                width=88,
            ):
                entries.append(("F1", 10, 0.12, 0.16, 0.23, line, 2))

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
                    f"{sanitize_pdf_text(role['mode'])} | "
                    f"{sanitize_pdf_text(role['stipendDisplay'])}"
                )
                pdf.multi_cell(0, 15, role_text)
                pdf.ln(2)

        exp = report.get("expertiseResult")
        if exp:
            pdf.ln(8)
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(30, 64, 175)
            pdf.cell(0, 20, "Expertise Evaluation Test Summary", ln=1)
            pdf.set_font("Helvetica", "", 10.5)
            pdf.set_text_color(30, 41, 59)
            pdf.multi_cell(
                0,
                15,
                sanitize_pdf_text(
                    f"Area Tested: {exp['expertiseArea']}  |  Score: {exp['correctCount']}/{exp['totalCount']} "
                    f"({exp['scorePercent']}% - {exp['proficiencyBand']})"
                ),
            )
            if exp.get("strongTopics"):
                pdf.multi_cell(
                    0,
                    15,
                    sanitize_pdf_text(f"Verified Strong Topics: {', '.join(exp['strongTopics'])}"),
                )
            if exp.get("weakTopics"):
                pdf.multi_cell(
                    0,
                    15,
                    sanitize_pdf_text(f"Topics to Review: {', '.join(exp['weakTopics'])}"),
                )

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

        exp = report.get("expertiseResult")
        if exp:
            st.divider()
            st.caption(
                f"Expertise Test ({exp['expertiseArea']}): "
                f"{exp['correctCount']}/{exp['totalCount']} ({exp['scorePercent']}%) — "
                f"{exp['proficiencyBand']} · Included in PDF export"
            )


def render_expertise_test_section(
    resume_text: str, matched_skills: list[str]
) -> None:
    area, detected_skills = detect_expertise_area(resume_text, matched_skills)

    with st.container(border=True):
        head_left, head_right = st.columns([3, 1], vertical_alignment="center")
        with head_left:
            st.markdown(f"### Expertise Evaluation Test &nbsp; **{area}**")
            st.caption(
                f"Verify practical proficiency in your detected resume domain ({area}) "
                "with a 5-question technical assessment."
            )
            if detected_skills:
                st.caption(f"Focus skills: {', '.join(detected_skills)}")
        with head_right:
            quiz = st.session_state.expertise_quiz
            result = st.session_state.expertise_result
            if quiz is None and result is None:
                if st.button("Start Expertise Test", type="primary", use_container_width=True):
                    with st.spinner(f"Preparing {area} evaluation test..."):
                        st.session_state.expertise_quiz = generate_expertise_quiz(
                            resume_text, matched_skills
                        )
                        st.session_state.expertise_result = None
                        st.rerun()
            elif result is not None:
                if st.button("Retake Test", use_container_width=True):
                    st.session_state.expertise_result = None
                    st.rerun()

        quiz = st.session_state.expertise_quiz
        result = st.session_state.expertise_result

        if quiz is not None and result is None:
            with st.form("expertise_quiz_form"):
                selected_answers: dict[str, int] = {}
                for idx, q in enumerate(quiz["questions"], start=1):
                    st.markdown(f"**Q{idx} · {q['topic']}** — {q['prompt']}")
                    choice = st.radio(
                        label=f"Question {idx}",
                        options=list(range(len(q["options"]))),
                        format_func=lambda opt_idx, opts=q["options"]: opts[opt_idx],
                        index=None,
                        key=f"exp_ans_{q['id']}",
                        label_visibility="collapsed",
                    )
                    if choice is not None:
                        selected_answers[q["id"]] = int(choice)

                submitted_quiz = st.form_submit_button(
                    "Submit Assessment", type="primary"
                )
                if submitted_quiz:
                    if len(selected_answers) < len(quiz["questions"]):
                        remaining = len(quiz["questions"]) - len(selected_answers)
                        st.error(
                            f"Please answer all {len(quiz['questions'])} questions before submitting ({remaining} remaining)."
                        )
                    else:
                        st.session_state.expertise_result = grade_expertise_quiz(
                            quiz, selected_answers
                        )
                        st.rerun()

        if result is not None:
            st.success(
                f"Score: {result['correctCount']}/{result['totalCount']} "
                f"({result['scorePercent']}%) — {result['proficiencyBand']}"
            )
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Verified Strengths**")
                st.caption(
                    " · ".join(result["strongTopics"])
                    if result["strongTopics"]
                    else "Review the explanations below to build your core baseline."
                )
            with c2:
                st.markdown("**Topics to Brush Up**")
                st.caption(
                    " · ".join(result["weakTopics"])
                    if result["weakTopics"]
                    else "None — 100% accuracy across all tested topics."
                )
            for idx, outcome in enumerate(result["outcomes"], start=1):
                status = "Correct" if outcome["isCorrect"] else "Incorrect"
                with st.expander(f"Q{idx}. {outcome['topic']} ({status})"):
                    st.write(outcome["prompt"])
                    if not outcome["isCorrect"]:
                        st.error(f"Your answer: {outcome['selectedOption']}")
                    st.success(f"Correct answer: {outcome['correctOption']}")
                    st.caption(outcome["explanation"])


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
    if "expertise_quiz" not in st.session_state:
        st.session_state.expertise_quiz = None
    if "expertise_result" not in st.session_state:
        st.session_state.expertise_result = None
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
                    st.session_state.expertise_quiz = None
                    st.session_state.expertise_result = None

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
                st.session_state.expertise_quiz = None
                st.session_state.expertise_result = None
                st.session_state.last_uploaded_key = ""
                st.rerun()

        if submitted:
            if not resume_input.strip():
                st.error("Please paste your resume or upload a PDF/TXT file first.")
                st.session_state.recommendations = None
                st.session_state.readiness_report = None
                st.session_state.expertise_quiz = None
                st.session_state.expertise_result = None
            else:
                with st.spinner("Generating internship recommendations..."):
                    recs, method = recommend_internships(resume_input, internships)
                    st.session_state.recommendations = recs
                    st.session_state.ranking_method = method
                    st.session_state.expertise_quiz = None
                    st.session_state.expertise_result = None
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
                matched_pool: list[str] = []
                for rec in recs[:8]:
                    for s in rec["matchedSkills"]:
                        if s not in matched_pool:
                            matched_pool.append(s)
                st.markdown("")
                render_expertise_test_section(st.session_state.resume_text, matched_pool[:8])
                st.markdown("")
                report_with_test = dict(st.session_state.readiness_report)
                if st.session_state.expertise_result is not None:
                    report_with_test["expertiseResult"] = st.session_state.expertise_result
                render_readiness_report(report_with_test)

    render_about_section()


if __name__ == "__main__":
    main()
