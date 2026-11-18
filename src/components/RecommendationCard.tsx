import React from 'react';
import { ChevronDown, ChevronUp, ExternalLink } from 'lucide-react';
import type { ScoredRecommendation } from '../types/internship';

interface RecommendationCardProps {
  recommendation: ScoredRecommendation;
  rank: number;
}

export function RecommendationCard({ recommendation, rank }: RecommendationCardProps) {
  const [expanded, setExpanded] = React.useState(false);
  const { internship, matchScore, matchedSkills, missingSkills, matchReasons } =
    recommendation;

  const scoreTone =
    matchScore >= 75
      ? 'text-emerald-700 bg-emerald-50 border-emerald-200'
      : matchScore >= 55
        ? 'text-blue-800 bg-blue-50 border-blue-200'
        : 'text-amber-700 bg-amber-50 border-amber-200';

  return (
    <article className="bg-white border border-slate-200 rounded-lg p-5 transition-colors hover:border-slate-300">
      <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
        <div className="min-w-0 flex-1 space-y-2">
          <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
            <span className="font-mono font-semibold text-slate-700 tabular-nums">
              #{String(rank).padStart(2, '0')}
            </span>
            <span aria-hidden="true">·</span>
            <span className="font-medium text-slate-800">{internship.organization}</span>
            <span aria-hidden="true">·</span>
            <span>{internship.category}</span>
            <span aria-hidden="true">·</span>
            <span>{internship.verified ? 'AICTE Verified' : 'Partner Listing'}</span>
          </div>

          <h3 className="font-display text-base sm:text-lg font-semibold text-slate-900 leading-snug">
            {internship.title}
          </h3>

          <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 text-xs text-slate-600 font-mono tabular-nums">
            <span>{internship.mode}</span>
            <span aria-hidden="true">·</span>
            <span>{internship.location}</span>
            <span aria-hidden="true">·</span>
            <span>{internship.duration}</span>
            <span aria-hidden="true">·</span>
            <span className="font-semibold text-slate-800">{internship.stipendDisplay}</span>
            <span aria-hidden="true">·</span>
            <span className={internship.daysLeft <= 3 ? 'text-amber-700 font-semibold' : ''}>
              {internship.daysLeft}d left
            </span>
          </div>

          {matchReasons[0] && (
            <p className="text-xs text-slate-600 leading-relaxed pt-0.5">{matchReasons[0]}</p>
          )}

          {(matchedSkills.length > 0 || missingSkills.length > 0) && (
            <div className="pt-0.5 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
              {matchedSkills.length > 0 && (
                <span className="text-slate-700">
                  <strong className="font-medium text-emerald-700">Matched:</strong>{' '}
                  {matchedSkills.slice(0, 5).join(', ')}
                </span>
              )}
              {missingSkills.length > 0 && (
                <span className="text-slate-500">
                  <strong className="font-medium text-amber-700">Gap:</strong>{' '}
                  {missingSkills.slice(0, 3).join(', ')}
                </span>
              )}
            </div>
          )}
        </div>

        <div className="flex sm:flex-col items-center sm:items-end justify-between gap-2.5 shrink-0 pt-2 sm:pt-0 border-t sm:border-t-0 border-slate-100">
          <div className="text-left sm:text-right">
            <div
              className={`inline-flex items-baseline gap-1.5 px-2.5 py-1 rounded-md border font-mono tabular-nums ${scoreTone}`}
            >
              <span className="text-lg font-bold">{matchScore}%</span>
              <span className="text-[11px] font-medium">Match</span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setExpanded((prev) => !prev)}
              aria-expanded={expanded}
              className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-200 rounded-md hover:bg-slate-50 transition-colors whitespace-nowrap shrink-0"
            >
              <span>{expanded ? 'Hide details' : 'View details'}</span>
              {expanded ? (
                <ChevronUp className="h-3.5 w-3.5" />
              ) : (
                <ChevronDown className="h-3.5 w-3.5" />
              )}
            </button>

            {internship.portalUrl && (
              <a
                href={internship.portalUrl}
                target="_blank"
                rel="noopener noreferrer"
                aria-label={`Apply to ${internship.title} on AICTE portal`}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-white bg-slate-900 rounded-md hover:bg-slate-800 transition-colors whitespace-nowrap shrink-0"
              >
                <span>Apply</span>
                <ExternalLink className="h-3.5 w-3.5" />
              </a>
            )}
          </div>
        </div>
      </div>

      {expanded && (
        <div className="mt-4 pt-4 border-t border-slate-100 space-y-2.5 text-xs text-slate-600 leading-relaxed">
          {internship.description && <p>{internship.description}</p>}
          {internship.skills.length > 0 && (
            <p>
              <strong className="font-semibold text-slate-800">Required Skills:</strong>{' '}
              {internship.skills.join(', ')}
            </p>
          )}
          {internship.eligibility.length > 0 && (
            <p>
              <strong className="font-semibold text-slate-800">Eligibility:</strong>{' '}
              {internship.eligibility.join(' · ')}
            </p>
          )}
        </div>
      )}
    </article>
  );
}
