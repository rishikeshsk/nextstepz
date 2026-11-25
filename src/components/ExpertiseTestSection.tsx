import React from 'react';
import { Award, CheckCircle2, ClipboardCheck, Loader2, RotateCcw, XCircle } from 'lucide-react';
import type { ExpertiseQuiz, ExpertiseTestResult } from '../types/internship';
import {
  buildDeterministicExpertiseQuiz,
  detectExpertiseArea,
  gradeExpertiseQuiz,
} from '../utils/expertiseTest';

interface ExpertiseTestSectionProps {
  resumeText: string;
  matchedSkills: string[];
  result: ExpertiseTestResult | null;
  onComplete: (result: ExpertiseTestResult | null) => void;
}

export function ExpertiseTestSection({
  resumeText,
  matchedSkills,
  result,
  onComplete,
}: ExpertiseTestSectionProps) {
  const detected = React.useMemo(
    () => detectExpertiseArea(resumeText, matchedSkills),
    [resumeText, matchedSkills],
  );

  const [quiz, setQuiz] = React.useState<ExpertiseQuiz | null>(null);
  const [answers, setAnswers] = React.useState<Record<string, number>>({});
  const [isLoading, setIsLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [notice, setNotice] = React.useState<string | null>(null);

  // Reset local quiz state when resume snapshot changes
  React.useEffect(() => {
    setQuiz(null);
    setAnswers({});
    setError(null);
    setNotice(null);
  }, [resumeText]);

  const handleStartTest = async () => {
    setIsLoading(true);
    setError(null);
    setNotice(null);
    setAnswers({});
    onComplete(null);

    try {
      const res = await fetch('/api/expertise-test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ resumeText, matchedSkills }),
      });
      const data = await res.json();
      if (!res.ok || !Array.isArray(data.questions) || data.questions.length === 0) {
        throw new Error(data.error || 'Unable to generate custom AI test.');
      }
      setQuiz(data as ExpertiseQuiz);
    } catch {
      const fallbackQuiz = buildDeterministicExpertiseQuiz({
        resumeText,
        matchedSkills,
      });
      setQuiz(fallbackQuiz);
      setNotice(`Loaded curated ${fallbackQuiz.expertiseArea} assessment.`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!quiz) return;
    const unanswered = quiz.questions.filter((q) => typeof answers[q.id] !== 'number');
    if (unanswered.length > 0) {
      setError(`Please answer all ${quiz.questions.length} questions before submitting (${unanswered.length} remaining).`);
      return;
    }
    setError(null);
    const graded = gradeExpertiseQuiz(quiz, answers);
    onComplete(graded);
  };

  const handleRetake = () => {
    setAnswers({});
    setError(null);
    onComplete(null);
    if (!quiz) {
      void handleStartTest();
    }
  };

  return (
    <section
      aria-label="Expertise Evaluation Test"
      data-testid="expertise-test-section"
      className="bg-white border border-slate-200 rounded-lg p-6 space-y-5 mt-6"
    >
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-200">
        <div>
          <div className="flex flex-wrap items-baseline gap-2.5">
            <h3 className="font-display text-lg font-bold text-slate-900">
              Expertise Evaluation Test
            </h3>
            <span className="text-xs font-semibold text-blue-800">{detected.area}</span>
          </div>
          <p className="text-xs text-slate-600 mt-1 leading-relaxed">
            Verify practical proficiency in your detected resume domain ({detected.area}) with a
            5-question technical assessment.
          </p>
          {detected.detectedSkills.length > 0 && (
            <p className="text-xs text-slate-500 mt-1">
              Focus skills: {detected.detectedSkills.join(', ')}
            </p>
          )}
        </div>

        {!quiz && !result && (
          <button
            type="button"
            onClick={handleStartTest}
            disabled={isLoading}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium text-white bg-blue-800 rounded-md hover:bg-blue-900 disabled:opacity-60 transition-colors whitespace-nowrap shrink-0 self-start sm:self-auto"
          >
            {isLoading ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <ClipboardCheck className="h-3.5 w-3.5" />
            )}
            <span>{isLoading ? 'Preparing Test…' : 'Start Expertise Test'}</span>
          </button>
        )}

        {result && (
          <button
            type="button"
            onClick={handleRetake}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-200 rounded-md hover:bg-slate-50 transition-colors whitespace-nowrap shrink-0 self-start sm:self-auto"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            <span>Retake Test</span>
          </button>
        )}
      </div>

      {notice && <p className="text-xs text-slate-500">{notice}</p>}

      {quiz && !result && (
        <form onSubmit={handleSubmit} className="space-y-5">
          {quiz.questions.map((q, qIdx) => (
            <fieldset key={q.id} className="space-y-2.5 pb-4 border-b border-slate-100 last:border-b-0">
              <legend className="text-sm font-semibold text-slate-900 leading-snug">
                <span className="font-mono text-xs text-slate-500 mr-2">
                  Q{qIdx + 1} · {q.topic}
                </span>
                <span className="block mt-1">{q.prompt}</span>
              </legend>
              <div className="space-y-2 pt-1">
                {q.options.map((opt, optIdx) => {
                  const inputId = `${q.id}-opt-${optIdx}`;
                  const checked = answers[q.id] === optIdx;
                  return (
                    <label
                      key={inputId}
                      htmlFor={inputId}
                      className={`flex items-start gap-2.5 p-2.5 rounded-md border text-xs cursor-pointer transition-colors ${
                        checked
                          ? 'border-blue-800 bg-blue-50/50 text-slate-900 font-medium'
                          : 'border-slate-200 hover:bg-slate-50 text-slate-700'
                      }`}
                    >
                      <input
                        id={inputId}
                        type="radio"
                        name={q.id}
                        value={optIdx}
                        checked={checked}
                        onChange={() => {
                          setError(null);
                          setAnswers((prev) => ({ ...prev, [q.id]: optIdx }));
                        }}
                        className="mt-0.5 accent-blue-800"
                      />
                      <span>{opt}</span>
                    </label>
                  );
                })}
              </div>
            </fieldset>
          ))}

          {error && (
            <p role="alert" className="text-xs text-rose-700 font-medium">
              {error}
            </p>
          )}

          <div className="flex items-center justify-between gap-4 pt-1">
            <span className="text-xs text-slate-500 font-mono tabular-nums">
              {Object.keys(answers).length}/{quiz.questions.length} answered
            </span>
            <button
              type="submit"
              className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-medium text-white bg-blue-800 rounded-md hover:bg-blue-900 transition-colors whitespace-nowrap"
            >
              <Award className="h-3.5 w-3.5" />
              <span>Submit Assessment</span>
            </button>
          </div>
        </form>
      )}

      {result && (
        <div className="space-y-4" data-testid="expertise-test-results">
          <div className="flex flex-wrap items-baseline justify-between gap-3 bg-slate-50 border border-slate-200 rounded-md p-4">
            <div>
              <span className="text-xs uppercase tracking-wide text-slate-500 font-semibold">
                Assessment Score · {result.expertiseArea}
              </span>
              <div className="flex items-baseline gap-2.5 mt-0.5">
                <span className="font-mono text-2xl font-bold text-slate-900 tabular-nums">
                  {result.correctCount}/{result.totalCount} ({result.scorePercent}%)
                </span>
                <span
                  className={`text-xs font-semibold ${
                    result.scorePercent >= 80
                      ? 'text-emerald-700'
                      : result.scorePercent >= 60
                        ? 'text-blue-800'
                        : 'text-amber-700'
                  }`}
                >
                  {result.proficiencyBand}
                </span>
              </div>
            </div>
            <p className="text-xs text-slate-600">
              Included in your Resume Readiness PDF report.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
            <div className="border border-slate-200 rounded-md p-3.5 space-y-1">
              <div className="font-semibold text-emerald-800">Verified Strengths</div>
              <p className="text-slate-600">
                {result.strongTopics.length > 0
                  ? result.strongTopics.join(' · ')
                  : 'Review the explanations below to build your core baseline.'}
              </p>
            </div>
            <div className="border border-slate-200 rounded-md p-3.5 space-y-1">
              <div className="font-semibold text-amber-800">Topics to Brush Up</div>
              <p className="text-slate-600">
                {result.weakTopics.length > 0
                  ? result.weakTopics.join(' · ')
                  : 'None — 100% accuracy across all tested topics.'}
              </p>
            </div>
          </div>

          <div className="space-y-2.5">
            {result.outcomes.map((item, idx) => (
              <div
                key={item.id}
                className="border border-slate-200 rounded-md p-3.5 text-xs space-y-1.5"
              >
                <div className="flex items-start justify-between gap-2">
                  <span className="font-semibold text-slate-900">
                    Q{idx + 1}. {item.topic}: {item.prompt}
                  </span>
                  {item.isCorrect ? (
                    <CheckCircle2 className="h-4 w-4 text-emerald-700 shrink-0" />
                  ) : (
                    <XCircle className="h-4 w-4 text-rose-700 shrink-0" />
                  )}
                </div>
                {!item.isCorrect && (
                  <p className="text-rose-700">Your answer: {item.selectedOption}</p>
                )}
                <p className="text-emerald-800 font-medium">
                  Correct answer: {item.correctOption}
                </p>
                <p className="text-slate-600 leading-relaxed">{item.explanation}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
