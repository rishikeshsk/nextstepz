import { jsPDF } from 'jspdf';
import type { ResumeReadinessReport } from '../types/internship';

function sanitizePdfText(text: string): string {
  return text
    .replace(/₹/g, 'Rs. ')
    .replace(/—/g, '-')
    .replace(/–/g, '-')
    .replace(/[^\x20-\x7E\n]/g, ' ')
    .trim();
}

export function downloadReadinessPdf(report: ResumeReadinessReport): void {
  const doc = new jsPDF({ unit: 'pt', format: 'a4' });
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  const margin = 48;
  const maxWidth = pageWidth - margin * 2;
  let y = 52;

  const ensureSpace = (needed: number) => {
    if (y + needed > pageHeight - 48) {
      doc.addPage();
      y = 52;
    }
  };

  // Header
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(18);
  doc.setTextColor(15, 23, 42);
  doc.text('nextstepz - Resume Readiness Report', margin, y);
  y += 20;

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(10);
  doc.setTextColor(100, 116, 139);
  const generatedLabel = new Date(report.generatedAt).toLocaleDateString('en-IN', {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
  });
  doc.text(
    `Generated ${generatedLabel}  |  Readiness Score: ${report.readinessScore}/100 (${report.readinessTier})`,
    margin,
    y,
  );
  y += 14;

  doc.setDrawColor(226, 232, 240);
  doc.line(margin, y, pageWidth - margin, y);
  y += 20;

  // Summary
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(11);
  doc.setTextColor(30, 41, 59);
  const summaryLines = doc.splitTextToSize(sanitizePdfText(report.summary), maxWidth);
  doc.text(summaryLines, margin, y);
  y += summaryLines.length * 15 + 18;

  // Section 1: What Not to Change
  ensureSpace(60);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(13);
  doc.setTextColor(21, 128, 61);
  doc.text('What Not to Change (Keep These Strengths)', margin, y);
  y += 18;

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(10.5);
  doc.setTextColor(30, 41, 59);
  for (const item of report.whatNotToChange) {
    const lines = doc.splitTextToSize(`- ${sanitizePdfText(item)}`, maxWidth - 8);
    ensureSpace(lines.length * 14 + 8);
    doc.text(lines, margin + 4, y);
    y += lines.length * 14 + 6;
  }

  y += 12;

  // Section 2: What to Change
  ensureSpace(60);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(13);
  doc.setTextColor(180, 83, 9);
  doc.text('What to Change (Recommended Improvements)', margin, y);
  y += 18;

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(10.5);
  doc.setTextColor(30, 41, 59);
  for (const item of report.whatToChange) {
    const lines = doc.splitTextToSize(`- ${sanitizePdfText(item)}`, maxWidth - 8);
    ensureSpace(lines.length * 14 + 8);
    doc.text(lines, margin + 4, y);
    y += lines.length * 14 + 6;
  }

  y += 12;

  // Section 3: Top Matched AICTE Internships
  if (report.topTargetRoles.length > 0) {
    ensureSpace(60);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(13);
    doc.setTextColor(15, 23, 42);
    doc.text('Top Recommended AICTE Internships', margin, y);
    y += 18;

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(10);
    for (let i = 0; i < report.topTargetRoles.length; i++) {
      const role = report.topTargetRoles[i];
      const line = `${i + 1}. ${sanitizePdfText(role.title)} (${sanitizePdfText(role.organization)}) - ${sanitizePdfText(role.mode)} | ${sanitizePdfText(role.stipendDisplay)}`;
      const wrapped = doc.splitTextToSize(line, maxWidth);
      ensureSpace(wrapped.length * 14 + 6);
      doc.text(wrapped, margin, y);
      y += wrapped.length * 14 + 4;
    }
  }

  // Section 4: Expertise Evaluation Test (when completed)
  if (report.expertiseResult) {
    const exp = report.expertiseResult;
    ensureSpace(80);
    y += 10;
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(13);
    doc.setTextColor(30, 64, 175);
    doc.text('Expertise Evaluation Test Summary', margin, y);
    y += 18;

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(10.5);
    doc.setTextColor(30, 41, 59);
    const expHeader = `Area Tested: ${sanitizePdfText(exp.expertiseArea)}  |  Score: ${exp.correctCount}/${exp.totalCount} (${exp.scorePercent}% - ${sanitizePdfText(exp.proficiencyBand)})`;
    const expHeaderLines = doc.splitTextToSize(expHeader, maxWidth);
    ensureSpace(expHeaderLines.length * 14 + 10);
    doc.text(expHeaderLines, margin, y);
    y += expHeaderLines.length * 14 + 6;

    if (exp.strongTopics.length > 0) {
      const strongLine = doc.splitTextToSize(
        `Verified Strong Topics: ${sanitizePdfText(exp.strongTopics.join(', '))}`,
        maxWidth,
      );
      ensureSpace(strongLine.length * 14 + 6);
      doc.text(strongLine, margin, y);
      y += strongLine.length * 14 + 4;
    }

    if (exp.weakTopics.length > 0) {
      const weakLine = doc.splitTextToSize(
        `Topics to Review: ${sanitizePdfText(exp.weakTopics.join(', '))}`,
        maxWidth,
      );
      ensureSpace(weakLine.length * 14 + 6);
      doc.text(weakLine, margin, y);
      y += weakLine.length * 14 + 4;
    }
  }

  doc.save('nextstepz-resume-readiness-report.pdf');
}
