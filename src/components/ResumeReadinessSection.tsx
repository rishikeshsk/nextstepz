import { CheckCircle2, Download, Wrench } from 'lucide-react';
import type { ResumeReadinessReport } from '../types/internship';
import { downloadReadinessPdf } from '../utils/pdfReport';

interface ResumeReadinessSectionProps {
  report: ResumeReadinessReport;
}

export function ResumeReadinessSection({ report }: ResumeReadinessSectionProps) {
  const tierTone =
    report.readinessScore >= 72
      ? 'text-emerald-700'
      : report.readinessScore >= 52
        ? 'text-blue-800'
        : 'text-amber-700';

  return (
    <section
      aria-label="Resume Readiness Report"
      className="bg-white border border-slate-200 rounded-lg p-6 space-y-5 mt-6"
    >
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-200">
        <div>
          <div className="flex flex-wrap items-baseline gap-2.5">
            <h3 className="font-display text-lg font-bold text-slate-900">
              Resume Readiness Report
            </h3>
            <span className="font-mono text-sm font-bold text-slate-900 tabular-nums">
              {report.readinessScore}/100
            </span>
            <span className={`text-xs font-semibold ${tierTone}`}>{report.readinessTier}</span>
          </div>
          <p className="text-xs text-slate-600 mt-1 leading-relaxed">{report.summary}</p>
        </div>

        <button
          type="button"
          onClick={() => downloadReadinessPdf(report)}
          className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium text-white bg-slate-900 rounded-md hover:bg-slate-800 transition-colors whitespace-nowrap shrink-0 self-start sm:self-auto"
        >
          <Download className="h-3.5 w-3.5" />
          <span>Download PDF Report</span>
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="space-y-2.5">
          <div className="flex items-center gap-1.5 text-sm font-semibold text-emerald-800">
            <CheckCircle2 className="h-4 w-4 shrink-0" />
            <h4>What Not to Change</h4>
          </div>
          <ul className="space-y-2 text-xs text-slate-700 leading-relaxed">
            {report.whatNotToChange.map((item, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <span className="text-emerald-700 font-bold mt-0.5" aria-hidden="true">
                  •
                </span>
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>

        <div className="space-y-2.5">
          <div className="flex items-center gap-1.5 text-sm font-semibold text-amber-800">
            <Wrench className="h-4 w-4 shrink-0" />
            <h4>What to Change</h4>
          </div>
          <ul className="space-y-2 text-xs text-slate-700 leading-relaxed">
            {report.whatToChange.map((item, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <span className="text-amber-700 font-bold mt-0.5" aria-hidden="true">
                  •
                </span>
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {report.expertiseResult && (
        <div className="pt-3 border-t border-slate-200 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-600">
          <span>
            Expertise Test ({report.expertiseResult.expertiseArea}):{' '}
            <strong className="font-mono text-slate-900 tabular-nums">
              {report.expertiseResult.correctCount}/{report.expertiseResult.totalCount} (
              {report.expertiseResult.scorePercent}%)
            </strong>{' '}
            — {report.expertiseResult.proficiencyBand}
          </span>
          <span className="text-slate-500">Included in PDF export</span>
        </div>
      )}
    </section>
  );
}
