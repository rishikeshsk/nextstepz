import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';
import type {
  CandidateProfile,
  InteractionHistory,
  Internship,
  ReadinessReport,
  ScoredRecommendation,
} from '../src/types/internship';
import {
  buildProfileDocument,
  buildResumeDocument,
  computeDeterministicReadiness,
} from '../src/utils/recommender';

const EMBEDDING_MODEL = 'gemini-embedding-2';
const REPORT_MODEL = 'gemini-4-hydrogen';

// Cache embeddings per internship ID and profile document so interactions re-score in <10ms
const internshipEmbeddingCache = new Map<string, number[]>();
const profileEmbeddingCache = new Map<string, number[]>();
let inFlightEmbeddingTask: Promise<void> | null = null;

function buildConciseEmbeddingDoc(item: Internship): string {
  return `${item.title} | ${item.organization} (${item.category}) | Skills: ${item.skills.slice(0, 8).join(', ')} | ${item.mode} | ${item.description.slice(0, 140)}`;
}

function getGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY?.trim();
  if (!apiKey) {
    const error = new Error(
      'GEMINI_API_KEY is not configured in environment secrets. Please set GEMINI_API_KEY in Settings > Secrets.',
    ) as Error & { statusCode?: number; code?: string };
    error.statusCode = 400;
    error.code = 'MISSING_API_KEY';
    throw error;
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

async function batchEmbedTexts(ai: GoogleGenAI, texts: string[]): Promise<number[][]> {
  if (texts.length === 0) return [];
  const BATCH_SIZE = 50;
  const allVectors: number[][] = [];

  for (let i = 0; i < texts.length; i += BATCH_SIZE) {
    if (i > 0) {
      await new Promise((r) => setTimeout(r, 150));
    }
    const chunk = texts.slice(i, i + BATCH_SIZE);
    let result;
    const maxAttempts = 3;
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        // gemini-embedding-2 merges plain string[] into a single multimodal Content;
        // wrapping each string as a Content ({ parts: [{ text }] }) yields one embedding per item.
        result = await ai.models.embedContent({
          model: EMBEDDING_MODEL,
          contents: chunk.map((text) => ({ parts: [{ text }] })),
        });
        break;
      } catch (err) {
        const anyErr = err as { status?: number; message?: string };
        const msg = String(anyErr?.message ?? err);
        const isTransient =
          anyErr?.status === 503 ||
          anyErr?.status === 500 ||
          msg.includes('503') ||
          msg.includes('UNAVAILABLE');
        if (isTransient && attempt < maxAttempts) {
          await new Promise((r) => setTimeout(r, attempt * 600));
          continue;
        }
        throw err;
      }
    }

    const embeddings = result?.embeddings ?? [];
    for (const emb of embeddings) {
      allVectors.push(emb.values ?? []);
    }
  }

  return allVectors;
}

async function embedDocumentAndInternships(
  docText: string,
  internships: Internship[],
): Promise<{
  profileEmbedding: number[];
  embeddingMap: Record<string, number[]>;
  embeddingModel: string;
}> {
  if (inFlightEmbeddingTask) {
    await inFlightEmbeddingTask.catch(() => {});
  }

  const cachedProfileVec = profileEmbeddingCache.get(docText);
  const missingInternships = internships.filter((item) => !internshipEmbeddingCache.has(item.id));

  if (!cachedProfileVec || missingInternships.length > 0) {
    const task = (async () => {
      const ai = getGeminiClient();
      const textsToEmbed: string[] = [];
      const needProfile = !profileEmbeddingCache.has(docText);
      const stillMissing = internships.filter((item) => !internshipEmbeddingCache.has(item.id));

      if (needProfile) {
        textsToEmbed.push(docText);
      }
      for (const item of stillMissing) {
        textsToEmbed.push(buildConciseEmbeddingDoc(item));
      }
      if (textsToEmbed.length === 0) return;

      const t0 = Date.now();
      const vectors = await batchEmbedTexts(ai, textsToEmbed);
      console.log(
        `[aiEngine] embedded ${textsToEmbed.length} texts -> ${vectors.length} vectors in ${Date.now() - t0}ms`,
      );

      let offset = 0;
      if (needProfile) {
        const pVec = vectors[0] ?? [];
        if (pVec.length > 0) {
          profileEmbeddingCache.set(docText, pVec);
        }
        offset = 1;
      }

      stillMissing.forEach((item, idx) => {
        const vec = vectors[idx + offset];
        if (vec && vec.length > 0) {
          internshipEmbeddingCache.set(item.id, vec);
        }
      });
    })();

    inFlightEmbeddingTask = task;
    try {
      await task;
    } finally {
      if (inFlightEmbeddingTask === task) {
        inFlightEmbeddingTask = null;
      }
    }
  }

  const profileEmbedding = profileEmbeddingCache.get(docText) ?? [];
  const embeddingMap: Record<string, number[]> = {};
  for (const item of internships) {
    const vec = internshipEmbeddingCache.get(item.id);
    if (vec) {
      embeddingMap[item.id] = vec;
    }
  }

  return {
    profileEmbedding,
    embeddingMap,
    embeddingModel: EMBEDDING_MODEL,
  };
}

export async function embedCandidateAndInternships(
  profile: CandidateProfile,
  internships: Internship[],
) {
  return embedDocumentAndInternships(buildProfileDocument(profile), internships);
}

export async function embedResumeAndInternships(
  resumeText: string,
  internships: Internship[],
) {
  return embedDocumentAndInternships(buildResumeDocument(resumeText), internships);
}

export async function generateAiReadinessReport(params: {
  profile: CandidateProfile;
  interactions: InteractionHistory;
  recommendations: ScoredRecommendation[];
}): Promise<ReadinessReport> {
  const { profile, interactions, recommendations } = params;
  const ai = getGeminiClient();

  const baseline = computeDeterministicReadiness(profile, recommendations);
  const top5 = recommendations.slice(0, 5).map((r) => ({
    title: r.internship.title,
    organization: r.internship.organization,
    category: r.internship.category,
    mode: r.internship.mode,
    stipend: r.internship.stipendDisplay,
    matchScore: r.matchScore,
    semanticSimilarity: r.semanticSimilarity,
    matchedSkills: r.matchedSkills,
    missingSkills: r.missingSkills,
  }));

  const savedTitles = recommendations
    .filter((r) => interactions.savedIds.includes(r.internship.id))
    .map((r) => r.internship.title);
  const viewedTitles = recommendations
    .filter((r) => interactions.viewedIds.includes(r.internship.id))
    .map((r) => r.internship.title);

  const prompt = [
    'You are an expert AI Career & Policy Advisor for the PM Internship Scheme and AICTE National Internship Portal.',
    'Analyze the candidate profile, their interaction signals (saved and viewed internships), and the top cosine-similarity matched AICTE internships.',
    'Produce a structured, realistic Readiness Report grounded strictly in the provided data.',
    '',
    `Candidate: ${profile.fullName}, ${profile.degree} in ${profile.branch} (Class of ${profile.graduationYear}), CGPA ${profile.cgpa}, Age ${profile.age}.`,
    `Skills: ${profile.skills.join(', ')}`,
    `Interests: ${profile.interests.join(', ')}`,
    `Experience Summary: ${profile.experienceSummary}`,
    profile.resumeText?.trim()
      ? `Candidate Resume Excerpt: ${profile.resumeText.trim().slice(0, 1200)}`
      : '',
    `Saved Internships (${savedTitles.length}): ${savedTitles.join('; ') || 'None yet'}`,
    `Viewed Internships (${viewedTitles.length}): ${viewedTitles.join('; ') || 'None yet'}`,
    `Baseline Deterministic Score: ${baseline.overallReadinessScore}/100 (${baseline.readinessTier})`,
    `PM Scheme Eligibility Checks Passed: ${baseline.pmSchemeEligibility.criteriaChecks.filter((c) => c.passed).length}/${baseline.pmSchemeEligibility.criteriaChecks.length}`,
    `Top 5 Matched AICTE Internships: ${JSON.stringify(top5)}`,
    `Aggregated Skill Gaps: ${JSON.stringify(baseline.topSkillGaps.map((g) => g.skill))}`,
  ]
    .filter(Boolean)
    .join('\n');

  const callGenerate = async () => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 25_000);
    const t0 = Date.now();
    try {
      const result = await ai.models.generateContent({
        model: REPORT_MODEL,
        contents: prompt,
        config: {
          abortSignal: controller.signal,
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          systemInstruction:
            'Provide concise, institutional-grade, actionable readiness guidance tailored to Indian technical students applying under AICTE and the PM Internship Scheme. Keep bullet items sharp and grounded in actual skills and listings.',
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              domainFitSummary: {
                type: Type.STRING,
                description: '1-2 sentence synthesis of the candidate domain fit across AICTE categories.',
              },
              narrativeSummary: {
                type: Type.STRING,
                description: '3-4 sentence executive readiness verdict covering strengths, interaction signals, and PM Scheme readiness.',
              },
              skillGapActions: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    skill: { type: Type.STRING },
                    learningAction: { type: Type.STRING },
                  },
                  required: ['skill', 'learningAction'],
                },
              },
              fourWeekActionPlan: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    week: { type: Type.STRING },
                    focus: { type: Type.STRING },
                    deliverable: { type: Type.STRING },
                  },
                  required: ['week', 'focus', 'deliverable'],
                },
              },
            },
            required: ['domainFitSummary', 'narrativeSummary', 'skillGapActions', 'fourWeekActionPlan'],
          },
        },
      });
      console.log(`[aiEngine] ${REPORT_MODEL} responded in ${Date.now() - t0}ms`);
      return result;
    } finally {
      clearTimeout(timer);
    }
  };

  let response;
  const maxAttempts = 3;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      response = await callGenerate();
      break;
    } catch (err) {
      const anyErr = err as { status?: number; name?: string; message?: string };
      const msg = String(anyErr?.message ?? err);
      console.error(`[aiEngine] generateContent attempt ${attempt} error:`, msg);
      const isTransient =
        anyErr?.status === 503 ||
        msg.includes('503') ||
        msg.includes('UNAVAILABLE') ||
        anyErr?.name === 'AbortError';
      if (isTransient && attempt < maxAttempts) {
        await new Promise((r) => setTimeout(r, attempt * 1000));
        continue;
      }
      throw err;
    }
  }
  if (!response) {
    throw new Error('Failed to generate readiness report response.');
  }

  const rawText = (response.text ?? '').trim();
  const parsed = JSON.parse(rawText) as {
    domainFitSummary?: string;
    narrativeSummary?: string;
    skillGapActions?: { skill: string; learningAction: string }[];
    fourWeekActionPlan?: { week: string; focus: string; deliverable: string }[];
  };

  const mergedGaps = baseline.topSkillGaps.map((gap) => {
    const aiMatch = parsed.skillGapActions?.find(
      (a) => a.skill.toLowerCase() === gap.skill.toLowerCase(),
    );
    return aiMatch ? { ...gap, learningAction: aiMatch.learningAction } : gap;
  });

  return {
    ...baseline,
    generatedAt: new Date().toISOString(),
    domainFitSummary: parsed.domainFitSummary || baseline.domainFitSummary,
    narrativeSummary: parsed.narrativeSummary || baseline.narrativeSummary,
    topSkillGaps: mergedGaps,
    fourWeekActionPlan:
      parsed.fourWeekActionPlan && parsed.fourWeekActionPlan.length >= 3
        ? parsed.fourWeekActionPlan.slice(0, 4)
        : baseline.fourWeekActionPlan,
    modelUsed: REPORT_MODEL,
  };
}

export interface ParsedResumeResult {
  fullName?: string;
  degree?: string;
  branch?: string;
  graduationYear?: number;
  cgpa?: number;
  skills: string[];
  interests: string[];
  experienceSummary: string;
  extractedText: string;
}

export async function parseResumeWithGemini(params: {
  resumeText?: string;
  fileBase64?: string;
  mimeType?: string;
}): Promise<ParsedResumeResult> {
  const ai = getGeminiClient();
  const textInput = params.resumeText?.trim() ?? '';
  const hasFile = Boolean(params.fileBase64 && params.mimeType);

  if (!textInput && !hasFile) {
    const err = new Error('Please paste resume text or upload a PDF/TXT resume first.') as Error & {
      statusCode?: number;
      code?: string;
    };
    err.statusCode = 400;
    err.code = 'EMPTY_RESUME';
    throw err;
  }

  const parts: Array<{ text: string } | { inlineData: { data: string; mimeType: string } }> = [
    {
      text: [
        'Extract structured candidate profile details from the provided resume for an Indian internship portal (AICTE & PM Internship Scheme).',
        'Return accurate skills, career interests, degree, branch, graduation year, CGPA (on a 10-point scale if available), a 2-3 sentence experience summary, and clean plain-text highlights of the resume.',
        textInput ? `\nResume Text:\n${textInput.slice(0, 12000)}` : '',
      ]
        .filter(Boolean)
        .join('\n'),
    },
  ];

  if (hasFile && params.fileBase64 && params.mimeType) {
    parts.unshift({
      inlineData: {
        data: params.fileBase64,
        mimeType: params.mimeType,
      },
    });
  }

  const callParse = async () => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 25_000);
    try {
      return await ai.models.generateContent({
        model: REPORT_MODEL,
        contents: { parts },
        config: {
          abortSignal: controller.signal,
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          systemInstruction:
            'Extract strictly what is supported by the resume without inventing qualifications. Normalize technical skills into concise standard names.',
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              fullName: { type: Type.STRING },
              degree: { type: Type.STRING },
              branch: { type: Type.STRING },
              graduationYear: { type: Type.INTEGER },
              cgpa: { type: Type.NUMBER },
              skills: { type: Type.ARRAY, items: { type: Type.STRING } },
              interests: { type: Type.ARRAY, items: { type: Type.STRING } },
              experienceSummary: { type: Type.STRING },
              extractedText: {
                type: Type.STRING,
                description: 'Clean plain-text summary of education, projects, internships, and skills from the resume.',
              },
            },
            required: ['skills', 'interests', 'experienceSummary', 'extractedText'],
          },
        },
      });
    } finally {
      clearTimeout(timer);
    }
  };

  let response;
  const maxAttempts = 3;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      response = await callParse();
      break;
    } catch (err) {
      const anyErr = err as { status?: number; name?: string; message?: string };
      const msg = String(anyErr?.message ?? err);
      const isTransient =
        anyErr?.status === 503 ||
        msg.includes('503') ||
        msg.includes('UNAVAILABLE') ||
        anyErr?.name === 'AbortError';
      if (isTransient && attempt < maxAttempts) {
        await new Promise((r) => setTimeout(r, attempt * 1000));
        continue;
      }
      throw err;
    }
  }

  const parsed = JSON.parse((response?.text ?? '{}').trim()) as Partial<ParsedResumeResult>;
  return {
    fullName: parsed.fullName?.trim() || undefined,
    degree: parsed.degree?.trim() || undefined,
    branch: parsed.branch?.trim() || undefined,
    graduationYear:
      typeof parsed.graduationYear === 'number' && parsed.graduationYear >= 2015
        ? parsed.graduationYear
        : undefined,
    cgpa: typeof parsed.cgpa === 'number' && parsed.cgpa > 0 && parsed.cgpa <= 10 ? parsed.cgpa : undefined,
    skills: Array.isArray(parsed.skills)
      ? parsed.skills.map((s) => String(s).trim()).filter(Boolean).slice(0, 18)
      : [],
    interests: Array.isArray(parsed.interests)
      ? parsed.interests.map((s) => String(s).trim()).filter(Boolean).slice(0, 8)
      : [],
    experienceSummary: parsed.experienceSummary?.trim() || '',
    extractedText: parsed.extractedText?.trim() || textInput,
  };
}
