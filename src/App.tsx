import React from 'react';
import { FileText, Loader2, Search, Upload } from 'lucide-react';
import snapshotData from './data/aicteSnapshot.json';
import type { Internship, ScoredRecommendation, ScrapeMeta } from './types/internship';
import { buildResumeReadinessReport, recommendFromResume } from './utils/recommender';
import { RecommendationCard } from './components/RecommendationCard';
import { ResumeReadinessSection } from './components/ResumeReadinessSection';

const RESUME_STORAGE_KEY = 'nextstepz_resume_text_v1';

export default function App() {
  const [resumeText, setResumeText] = React.useState<string>(() => {
    try {
      return localStorage.getItem(RESUME_STORAGE_KEY) ?? '';
    } catch {
      return '';
    }
  });
  const [resumeFileName, setResumeFileName] = React.useState<string>('');
  const [internships, setInternships] = React.useState<Internship[]>(
    () => (snapshotData.items as Internship[]) ?? [],
  );
  const [scrapeMeta, setScrapeMeta] = React.useState<ScrapeMeta>(() => ({
    scrapedAt: snapshotData.scrapedAt,
    sourceUrl: snapshotData.sourceUrl,
    sourceMode: 'cached_snapshot',
    totalOnPortal: snapshotData.totalOnPortal,
    scrapedCount: snapshotData.items.length,
    portalTitle: 'Browse Internships | National Internship Portal',
    flagshipMissions: [],
  }));

  // Recommendations are ONLY shown after clicking "Generate Internships"
  const [recommendations, setRecommendations] = React.useState<ScoredRecommendation[] | null>(null);
  const [generatedResumeSnapshot, setGeneratedResumeSnapshot] = React.useState<string>('');

  const [isRecommending, setIsRecommending] = React.useState(false);
  const [isParsingPdf, setIsParsingPdf] = React.useState(false);
  const [formError, setFormError] = React.useState<string | null>(null);
  const [embeddingNotice, setEmbeddingNotice] = React.useState<string | null>(null);

  const fileInputRef = React.useRef<HTMLInputElement | null>(null);
  const recommendAbortRef = React.useRef<AbortController | null>(null);

  // Sync live AICTE catalogue in background on mount
  React.useEffect(() => {
    let active = true;
    fetch('/api/internships')
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (!active || !data) return;
        if (Array.isArray(data.items) && data.items.length > 0) {
          setInternships(data.items);
        }
        if (data.meta) {
          setScrapeMeta(data.meta);
        }
      })
      .catch(() => {
        // Retain snapshot fallback silently
      });
    return () => {
      active = false;
      recommendAbortRef.current?.abort();
    };
  }, []);

  const runRecommendations = React.useCallback(
    async (textToMatch: string) => {
      const cleaned = textToMatch.trim();
      if (!cleaned) {
        setFormError('Please paste your resume or upload a PDF/TXT file first.');
        setRecommendations(null);
        return;
      }

      setFormError(null);
      setEmbeddingNotice(null);
      setGeneratedResumeSnapshot(cleaned);
      try {
        localStorage.setItem(RESUME_STORAGE_KEY, cleaned);
      } catch {
        // ignore quota errors
      }

      // Show immediate deterministic TF-IDF ranking while neural embeddings load
      const localRanked = recommendFromResume({
        resumeText: cleaned,
        internships,
      });
      setRecommendations(localRanked);

      recommendAbortRef.current?.abort();
      const controller = new AbortController();
      recommendAbortRef.current = controller;

      setIsRecommending(true);
      try {
        const res = await fetch('/api/recommend-resume', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          signal: controller.signal,
          body: JSON.stringify({ resumeText: cleaned }),
        });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || 'Unable to compute neural resume embeddings.');
        }
        if (controller.signal.aborted) return;
        if (Array.isArray(data.recommendations)) {
          setRecommendations(data.recommendations);
        }
        if (data.scrapeMeta) {
          setScrapeMeta(data.scrapeMeta);
        }
      } catch (err) {
        if ((err as { name?: string })?.name === 'AbortError') return;
        const msg = err instanceof Error ? err.message : 'Embedding request failed';
        setEmbeddingNotice(`${msg} Showing local TF-IDF resume matches.`);
      } finally {
        if (!controller.signal.aborted) {
          setIsRecommending(false);
        }
      }
    },
    [internships],
  );

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;

    setFormError(null);
    setEmbeddingNotice(null);

    if (file.size > 4 * 1024 * 1024) {
      setFormError('Resume file must be under 4 MB.');
      return;
    }

    const lower = file.name.toLowerCase();
    if (lower.endsWith('.txt') || lower.endsWith('.md') || file.type === 'text/plain') {
      const content = await file.text();
      setResumeText(content);
      setResumeFileName(file.name);
      return;
    }

    if (lower.endsWith('.pdf') || file.type === 'application/pdf') {
      setIsParsingPdf(true);
      setResumeFileName(file.name);
      const reader = new FileReader();
      reader.onload = async () => {
        try {
          const result = typeof reader.result === 'string' ? reader.result : '';
          const base64 = result.includes(',') ? result.split(',')[1] : result;
          if (!base64) {
            throw new Error('Could not read PDF file bytes.');
          }
          const res = await fetch('/api/parse-resume', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ fileBase64: base64, mimeType: 'application/pdf' }),
          });
          const data = await res.json();
          if (!res.ok) {
            throw new Error(data.error || 'Could not extract text from PDF.');
          }
          const extracted = [
            data.fullName,
            data.degree && data.branch ? `${data.degree} in ${data.branch}` : data.degree || data.branch,
            Array.isArray(data.skills) && data.skills.length > 0
              ? `Skills: ${data.skills.join(', ')}`
              : '',
            data.experienceSummary,
            data.extractedText,
          ]
            .filter(Boolean)
            .join('\n\n');

          setResumeText(extracted);
        } catch (err) {
          setFormError(err instanceof Error ? err.message : 'Unable to parse uploaded PDF.');
        } finally {
          setIsParsingPdf(false);
        }
      };
      reader.onerror = () => {
        setIsParsingPdf(false);
        setFormError('Failed to read selected PDF file.');
      };
      reader.readAsDataURL(file);
      return;
    }

    setFormError('Please upload a .pdf, .txt, or .md file, or paste your resume text.');
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    void runRecommendations(resumeText);
  };

  const handleClear = () => {
    recommendAbortRef.current?.abort();
    setResumeText('');
    setResumeFileName('');
    setRecommendations(null);
    setGeneratedResumeSnapshot('');
    setFormError(null);
    setEmbeddingNotice(null);
    setIsRecommending(false);
    try {
      localStorage.removeItem(RESUME_STORAGE_KEY);
    } catch {
      // ignore
    }
  };

  const topRecommendations = recommendations ? recommendations.slice(0, 12) : [];
  const activeMethodLabel =
    topRecommendations[0]?.modelMethod === 'gemini-embedding-2'
      ? 'Gemini Embeddings'
      : 'Local TF-IDF';

  const readinessReport = React.useMemo(() => {
    if (!recommendations || recommendations.length === 0 || !generatedResumeSnapshot) return null;
    return buildResumeReadinessReport({
      resumeText: generatedResumeSnapshot,
      recommendations,
    });
  }, [recommendations, generatedResumeSnapshot]);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col">
      <header className="bg-white border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-8 h-16 flex items-center justify-between gap-8">
          <span className="font-display text-lg font-bold tracking-tight text-slate-900 whitespace-nowrap shrink-0">
            nextstepz
          </span>
          <span className="text-xs text-slate-500 font-mono tabular-nums whitespace-nowrap shrink-0">
            {scrapeMeta.scrapedCount} of {scrapeMeta.totalOnPortal.toLocaleString('en-IN')} live AICTE internships
          </span>
        </div>
      </header>

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-8 py-8 space-y-12">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Resume Input Column */}
          <section className="lg:col-span-5 lg:sticky lg:top-8" aria-label="Resume Input">
            <form
              onSubmit={handleSubmit}
              className="bg-white border border-slate-200 rounded-lg p-6 space-y-4"
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h1 className="font-display text-xl font-bold text-slate-900 text-balance">
                    Add Your Resume
                  </h1>
                  <p className="text-xs text-slate-600 mt-1 leading-relaxed">
                    Paste your resume or upload a PDF/TXT file, then click Generate Internships.
                  </p>
                </div>

                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,.txt,.md,application/pdf,text/plain"
                  onChange={handleFileChange}
                  aria-label="Upload resume file"
                  className="hidden"
                />
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isParsingPdf}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-200 rounded-md hover:bg-slate-50 disabled:opacity-60 whitespace-nowrap shrink-0"
                >
                  {isParsingPdf ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <Upload className="h-3.5 w-3.5" />
                  )}
                  <span>{isParsingPdf ? 'Reading PDF…' : 'Upload PDF / TXT'}</span>
                </button>
              </div>

              <div>
                <label htmlFor="resume-input" className="sr-only">
                  Resume text
                </label>
                <textarea
                  id="resume-input"
                  rows={13}
                  value={resumeText}
                  onChange={(e) => {
                    setFormError(null);
                    setResumeText(e.target.value);
                  }}
                  placeholder="Paste your resume here (Education, Technical Skills, Projects, Experience)..."
                  className="w-full px-3.5 py-2.5 text-sm border border-slate-200 rounded-md focus:border-blue-800 focus:outline-none leading-relaxed resize-y"
                />
              </div>

              <div className="flex items-center justify-between gap-3 text-xs text-slate-500">
                <div className="flex items-center gap-2.5 min-w-0">
                  {resumeFileName && (
                    <span className="inline-flex items-center gap-1 text-slate-700 font-medium truncate">
                      <FileText className="h-3.5 w-3.5 shrink-0" />
                      <span className="truncate">{resumeFileName}</span>
                    </span>
                  )}
                  <span className="font-mono tabular-nums">{resumeText.trim().length} chars</span>
                </div>

                {(resumeText.trim() || resumeFileName) && (
                  <button
                    type="button"
                    onClick={handleClear}
                    className="text-xs text-slate-500 hover:text-red-700 whitespace-nowrap shrink-0"
                  >
                    Clear
                  </button>
                )}
              </div>

              {formError && (
                <div
                  role="alert"
                  className="bg-red-50 border border-red-200 rounded-md px-3.5 py-2.5 text-xs text-red-900 break-words"
                >
                  {formError}
                </div>
              )}

              <button
                type="submit"
                disabled={isParsingPdf || isRecommending}
                className="w-full inline-flex items-center justify-center gap-2 px-4 py-2.5 text-sm font-medium text-white bg-blue-800 rounded-md hover:bg-blue-900 disabled:opacity-70 transition-colors whitespace-nowrap"
              >
                {isRecommending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Search className="h-4 w-4" />
                )}
                <span>{isRecommending ? 'Generating Internships…' : 'Generate Internships'}</span>
              </button>
            </form>
          </section>

          {/* Recommended Internships & Readiness Output Column */}
          <section className="lg:col-span-7 space-y-4" aria-label="Recommended Internships">
            <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-2">
              <h2 className="font-display text-xl font-bold text-slate-900 text-balance">
                Recommended Internships
              </h2>
              {topRecommendations.length > 0 && (
                <p className="text-xs text-slate-500 font-mono tabular-nums">
                  Top {topRecommendations.length} of {internships.length} AICTE listings ·{' '}
                  {isRecommending ? 'Ranking with Gemini…' : activeMethodLabel}
                </p>
              )}
            </div>

            {embeddingNotice && (
              <div
                role="status"
                className="bg-amber-50 border border-amber-200 rounded-md px-3.5 py-2.5 text-xs text-amber-900 break-words"
              >
                {embeddingNotice}
              </div>
            )}

            {recommendations === null ? (
              isRecommending ? (
                <div className="space-y-3" aria-busy="true">
                  {[1, 2, 3].map((n) => (
                    <div
                      key={n}
                      className="h-32 bg-white border border-slate-200 rounded-lg animate-pulse"
                    />
                  ))}
                </div>
              ) : (
                <div className="bg-white border border-slate-200 rounded-lg p-10 text-center space-y-2">
                  <p className="text-sm font-semibold text-slate-900">
                    Your recommended internships will appear here
                  </p>
                  <p className="text-xs text-slate-500 max-w-md mx-auto leading-relaxed">
                    Add your resume on the left and click Generate Internships to view matching
                    AICTE portal openings and your downloadable Resume Readiness Report.
                  </p>
                </div>
              )
            ) : topRecommendations.length === 0 ? (
              <div className="bg-white border border-slate-200 rounded-lg p-8 text-center">
                <p className="text-sm text-slate-600">
                  No matching internships found. Try adding more skills or project details to your resume.
                </p>
              </div>
            ) : (
              <>
                <div className="space-y-3" data-testid="recommendations-list">
                  {topRecommendations.map((rec, idx) => (
                    <RecommendationCard
                      key={rec.internship.id}
                      recommendation={rec}
                      rank={idx + 1}
                    />
                  ))}
                </div>

                {readinessReport && <ResumeReadinessSection report={readinessReport} />}
              </>
            )}
          </section>
        </div>

        {/* About this website section */}
        <section
          aria-label="About this website"
          className="pt-8 border-t border-slate-200 grid grid-cols-1 md:grid-cols-3 gap-6 text-sm"
        >
          <div className="md:col-span-1">
            <h2 className="font-display text-base font-bold text-slate-900">About this website</h2>
            <p className="text-xs text-slate-500 mt-1">
              Resume-driven internship discovery for AICTE and the PM Internship Scheme.
            </p>
          </div>
          <div className="md:col-span-2 grid grid-cols-1 sm:grid-cols-2 gap-6 text-xs text-slate-600 leading-relaxed">
            <div className="space-y-1.5">
              <h3 className="font-semibold text-slate-900">Live AICTE & Semantic Matching</h3>
              <p>
                nextstepz collects active internship openings directly from the AICTE National
                Internship Portal and compares them against your resume using neural sentence
                embeddings and skill-overlap scoring to surface your strongest matches.
              </p>
            </div>
            <div className="space-y-1.5">
              <h3 className="font-semibold text-slate-900">Actionable Resume Readiness</h3>
              <p>
                Alongside your ranked internship matches, nextstepz evaluates your resume against
                live role requirements—highlighting what strengths to keep, which missing skills or
                sections to update, and letting you download the full report as a PDF.
              </p>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
