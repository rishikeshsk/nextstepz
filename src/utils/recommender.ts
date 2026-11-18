import type {
  CandidateProfile,
  EligibilityCheck,
  InteractionHistory,
  Internship,
  ReadinessReport,
  ResumeReadinessReport,
  ScoredRecommendation,
  SkillGapItem,
} from '../types/internship';

const STOPWORDS = new Set([
  'the', 'and', 'for', 'with', 'that', 'this', 'from', 'into', 'your', 'will',
  'are', 'who', 'want', 'gain', 'practical', 'experience', 'internship', 'intern',
  'student', 'students', 'freshers', 'work', 'based', 'using', 'under', 'all',
  'not', 'india', 'month', 'months', 'remote', 'hybrid', 'onsite', 'role',
]);

export function tokenizeText(text: string): string[] {
  const cleaned = text
    .toLowerCase()
    .replace(/[^a-z0-9+#.\s-]/g, ' ')
    .split(/\s+/)
    .map((t) => t.trim())
    .filter((t) => t.length >= 2 && !STOPWORDS.has(t));

  const tokens = [...cleaned];
  for (let i = 0; i < cleaned.length - 1; i++) {
    tokens.push(`${cleaned[i]}_${cleaned[i + 1]}`);
  }
  return tokens;
}

export function buildResumeDocument(resumeText: string): string {
  return `Candidate Resume: ${resumeText.trim().slice(0, 1800)}`;
}

export function buildProfileDocument(profile: CandidateProfile): string {
  const parts: string[] = [];
  if (profile.degree?.trim() || profile.branch?.trim()) {
    parts.push(
      `Candidate pursuing ${profile.degree ?? ''} in ${profile.branch ?? ''} graduating ${profile.graduationYear ?? ''}.`,
    );
  }
  if (profile.skills?.length) {
    parts.push(`Core skills: ${profile.skills.join(', ')}.`);
  }
  if (profile.interests?.length) {
    parts.push(`Career interests: ${profile.interests.join(', ')}.`);
  }
  if (profile.preferredModes?.length) {
    parts.push(`Preferred work modes: ${profile.preferredModes.join(', ')}.`);
  }
  if (profile.preferredStates?.length) {
    parts.push(`Preferred locations: ${profile.preferredStates.join(', ')}.`);
  }
  if (profile.experienceSummary?.trim()) {
    parts.push(`Experience summary: ${profile.experienceSummary.trim()}`);
  }
  if (profile.resumeText?.trim()) {
    parts.push(`Resume highlights: ${profile.resumeText.trim().slice(0, 1400)}`);
  }
  return parts.join(' ');
}

export function buildInternshipDocument(internship: Internship): string {
  return [
    `${internship.title} at ${internship.organization} (${internship.category}).`,
    `Required skills: ${internship.skills.join(', ')}.`,
    `Work mode: ${internship.mode}. Location: ${internship.location}, ${internship.state}.`,
    `Duration: ${internship.duration}.`,
    internship.description.slice(0, 700),
  ].join(' ');
}

export function normalizeVector(vec: number[]): number[] {
  let normSq = 0;
  for (let i = 0; i < vec.length; i++) {
    normSq += vec[i] * vec[i];
  }
  const norm = Math.sqrt(normSq);
  if (norm === 0 || !Number.isFinite(norm)) {
    return vec.map(() => 0);
  }
  return vec.map((v) => v / norm);
}

export function cosineSimilarity(vecA: number[], vecB: number[]): number {
  const len = Math.min(vecA.length, vecB.length);
  if (len === 0) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < len; i++) {
    dot += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  if (denom === 0 || !Number.isFinite(denom)) return 0;
  return Math.max(-1, Math.min(1, dot / denom));
}

function meanVectors(vectors: number[][]): number[] {
  if (vectors.length === 0) return [];
  const dims = vectors[0].length;
  const sum = new Array<number>(dims).fill(0);
  for (const vec of vectors) {
    for (let i = 0; i < dims && i < vec.length; i++) {
      sum[i] += vec[i];
    }
  }
  return sum.map((v) => v / vectors.length);
}

export function computeWeightedProfileVector(
  profileVec: number[],
  savedVecs: number[][],
  viewedVecs: number[][],
): number[] {
  if (profileVec.length === 0) return [];
  const normProfile = normalizeVector(profileVec);
  const validSaved = savedVecs.filter((v) => v.length === profileVec.length).map(normalizeVector);
  const validViewed = viewedVecs.filter((v) => v.length === profileVec.length).map(normalizeVector);

  let wProfile = 1.0;
  let wSaved = 0;
  let wViewed = 0;

  if (validSaved.length > 0 && validViewed.length > 0) {
    wProfile = 0.60;
    wSaved = 0.28;
    wViewed = 0.12;
  } else if (validSaved.length > 0) {
    wProfile = 0.68;
    wSaved = 0.32;
  } else if (validViewed.length > 0) {
    wProfile = 0.82;
    wViewed = 0.18;
  }

  const savedMean = validSaved.length > 0 ? meanVectors(validSaved) : [];
  const viewedMean = validViewed.length > 0 ? meanVectors(validViewed) : [];

  const combined = normProfile.map((val, idx) => {
    const sVal = savedMean.length > 0 ? savedMean[idx] : 0;
    const vVal = viewedMean.length > 0 ? viewedMean[idx] : 0;
    return wProfile * val + wSaved * sVal + wViewed * vVal;
  });

  return normalizeVector(combined);
}

export function buildLocalTfidfVectors(docs: string[]): { vectors: number[][]; vocab: string[] } {
  const tokenized = docs.map(tokenizeText);
  const df = new Map<string, number>();

  for (const tokens of tokenized) {
    const unique = new Set(tokens);
    for (const token of unique) {
      df.set(token, (df.get(token) ?? 0) + 1);
    }
  }

  const vocab = Array.from(df.keys()).sort();
  const vocabIndex = new Map<string, number>();
  vocab.forEach((term, idx) => vocabIndex.set(term, idx));

  const nDocs = docs.length;
  const vectors = tokenized.map((tokens) => {
    const vec = new Array<number>(vocab.length).fill(0);
    if (tokens.length === 0) return vec;
    const tf = new Map<string, number>();
    for (const token of tokens) {
      tf.set(token, (tf.get(token) ?? 0) + 1);
    }
    for (const [term, count] of tf.entries()) {
      const idx = vocabIndex.get(term);
      if (idx === undefined) continue;
      const docFreq = df.get(term) ?? 1;
      const idf = Math.log((nDocs + 1) / (docFreq + 1)) + 1;
      vec[idx] = (count / tokens.length) * idf;
    }
    return normalizeVector(vec);
  });

  return { vectors, vocab };
}

function normalizeSkill(skill: string): string {
  return skill.toLowerCase().replace(/[^a-z0-9+#]/g, ' ').replace(/\s+/g, ' ').trim();
}

function skillsRoughlyMatch(candidateSkill: string, targetSkill: string): boolean {
  const a = normalizeSkill(candidateSkill);
  const b = normalizeSkill(targetSkill);
  if (!a || !b) return false;
  if (a === b) return true;
  if (a.length >= 3 && b.includes(a)) return true;
  if (b.length >= 3 && a.includes(b)) return true;
  // Shared 5+ char root token (e.g. "Network Security" <-> "Networking", "Cyber Defence" <-> "Cybersecurity")
  const tokensA = a.split(' ').filter((t) => t.length >= 5);
  const tokensB = b.split(' ').filter((t) => t.length >= 5);
  for (const ta of tokensA) {
    for (const tb of tokensB) {
      if (ta.slice(0, 5) === tb.slice(0, 5)) return true;
    }
  }
  return false;
}

export function scoreInternships(params: {
  profile: CandidateProfile;
  internships: Internship[];
  interactions: InteractionHistory;
  embeddingMap?: Record<string, number[]>;
  profileEmbedding?: number[];
}): ScoredRecommendation[] {
  const { profile, internships, interactions, embeddingMap, profileEmbedding } = params;
  if (internships.length === 0) return [];

  const dismissedSet = new Set(interactions.dismissedIds);
  const savedSet = new Set(interactions.savedIds);
  const viewedSet = new Set(interactions.viewedIds);

  const activeInternships = internships.filter((item) => !dismissedSet.has(item.id));
  if (activeInternships.length === 0) return [];

  const hasEmbeddings =
    Boolean(profileEmbedding && profileEmbedding.length > 0) &&
    activeInternships.every((item) => Boolean(embeddingMap?.[item.id]?.length));

  let itemVectors: number[][] = [];
  let baseProfileVec: number[] = [];
  let savedEntries: { id: string; vec: number[] }[] = [];
  let viewedEntries: { id: string; vec: number[] }[] = [];

  if (hasEmbeddings && profileEmbedding && embeddingMap) {
    itemVectors = activeInternships.map((item) => embeddingMap[item.id]);
    baseProfileVec = profileEmbedding;
    savedEntries = interactions.savedIds
      .map((id) => ({ id, vec: embeddingMap[id] }))
      .filter((entry): entry is { id: string; vec: number[] } => Boolean(entry.vec?.length));
    viewedEntries = interactions.viewedIds
      .map((id) => ({ id, vec: embeddingMap[id] }))
      .filter((entry): entry is { id: string; vec: number[] } => Boolean(entry.vec?.length));
  } else {
    const docs = [
      buildProfileDocument(profile),
      ...activeInternships.map(buildInternshipDocument),
    ];
    const { vectors } = buildLocalTfidfVectors(docs);
    baseProfileVec = vectors[0];
    itemVectors = vectors.slice(1);

    const idToIdx = new Map<string, number>();
    activeInternships.forEach((item, idx) => idToIdx.set(item.id, idx));

    savedEntries = interactions.savedIds
      .map((id) => {
        const idx = idToIdx.get(id);
        return idx !== undefined ? { id, vec: itemVectors[idx] } : null;
      })
      .filter((entry): entry is { id: string; vec: number[] } => entry !== null);

    viewedEntries = interactions.viewedIds
      .map((id) => {
        const idx = idToIdx.get(id);
        return idx !== undefined ? { id, vec: itemVectors[idx] } : null;
      })
      .filter((entry): entry is { id: string; vec: number[] } => entry !== null);
  }

  // Extract implicit skills/categories from saved & viewed internships
  const internshipById = new Map(internships.map((item) => [item.id, item]));
  const savedSkills = new Set<string>();
  const savedOrgs = new Set<string>();
  for (const id of interactions.savedIds) {
    const item = internshipById.get(id);
    if (item) {
      item.skills.forEach((s) => savedSkills.add(normalizeSkill(s)));
      savedOrgs.add(item.organization.toLowerCase());
    }
  }
  const viewedSkills = new Set<string>();
  for (const id of interactions.viewedIds) {
    const item = internshipById.get(id);
    if (item) {
      item.skills.forEach((s) => viewedSkills.add(normalizeSkill(s)));
    }
  }

  const candidateCorpus = [
    ...(profile.skills ?? []),
    ...(profile.interests ?? []),
    profile.branch ?? '',
    profile.experienceSummary ?? '',
    profile.resumeText ?? '',
  ]
    .join(' ')
    .toLowerCase();

  const scored: ScoredRecommendation[] = activeInternships.map((item, idx) => {
    // Leave-one-out centroid so an item's own vector never artificially inflates its cosine to 1.0
    const looSaved = savedEntries.filter((e) => e.id !== item.id).map((e) => e.vec);
    const looViewed = viewedEntries.filter((e) => e.id !== item.id).map((e) => e.vec);
    const weightedProfileVec = computeWeightedProfileVector(baseProfileVec, looSaved, looViewed);

    const rawCosine = cosineSimilarity(weightedProfileVec, itemVectors[idx]);
    // gemini-embedding-2 clusters unrelated texts around ~0.58 and domain matches at ~0.68-0.78
    const semanticSimilarity = hasEmbeddings
      ? Math.max(0, Math.min(0.98, (rawCosine - 0.56) / 0.24))
      : Math.max(0, Math.min(0.94, Math.sqrt(Math.max(0, rawCosine)) * 1.18));

    const matchedSkills: string[] = [];
    const missingSkills: string[] = [];
    const normalizedCorpus = normalizeSkill(candidateCorpus);

    for (const reqSkill of item.skills) {
      const explicitMatch = (profile.skills ?? []).some((cSkill) => skillsRoughlyMatch(cSkill, reqSkill));
      const normReq = normalizeSkill(reqSkill);
      const escapedReq = normReq.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const textMatch = Boolean(
        normReq && new RegExp(`(?:^|\\s)${escapedReq}(?:$|\\s)`).test(normalizedCorpus),
      );
      if (explicitMatch || textMatch) {
        matchedSkills.push(reqSkill);
      } else {
        missingSkills.push(reqSkill);
      }
    }

    // When a portal listing has explicit skills, compute smoothed overlap;
    // when skills[] is empty, check candidate skills or resume tokens against title + description
    let skillOverlapScore = 0;
    if (item.skills.length > 0) {
      skillOverlapScore = Math.min(
        1,
        0.65 * (matchedSkills.length / Math.max(3, item.skills.length)) +
          0.35 * Math.min(1, matchedSkills.length / 4),
      );
    } else {
      const itemCorpus = normalizeSkill(`${item.title} ${item.description}`);
      if ((profile.skills ?? []).length > 0) {
        const inferredMatches = (profile.skills ?? []).filter((s) => {
          const norm = normalizeSkill(s);
          return norm.length >= 2 && itemCorpus.includes(norm);
        });
        inferredMatches.forEach((s) => matchedSkills.push(s));
        skillOverlapScore = Math.min(
          0.85,
          inferredMatches.length / Math.max(3, Math.min(5, profile.skills.length)),
        );
      } else {
        const titleTokens = tokenizeText(item.title).filter((t) => !t.includes('_') && t.length >= 4);
        const resumeHits = titleTokens.filter((tok) => normalizedCorpus.includes(tok));
        resumeHits.slice(0, 4).forEach((tok) => matchedSkills.push(tok));
        skillOverlapScore = Math.min(0.7, resumeHits.length / Math.max(3, titleTokens.length || 4));
      }
    }

    // Interaction affinity boost from saved & viewed internships
    let savedSkillAffinity = 0;
    let viewedSkillAffinity = 0;
    if (item.skills.length > 0) {
      const savedHit = item.skills.filter((s) => savedSkills.has(normalizeSkill(s))).length;
      const viewedHit = item.skills.filter((s) => viewedSkills.has(normalizeSkill(s))).length;
      savedSkillAffinity = savedHit / Math.max(2, item.skills.length);
      viewedSkillAffinity = viewedHit / Math.max(2, item.skills.length);
    }

    const orgAffinity = savedOrgs.has(item.organization.toLowerCase()) ? 0.08 : 0;
    const directSavedBoost = savedSet.has(item.id) ? 0.06 : 0;
    const directViewedBoost = viewedSet.has(item.id) && !savedSet.has(item.id) ? 0.02 : 0;

    const interactionBoost = Math.min(
      0.20,
      savedSkillAffinity * 0.13 +
        viewedSkillAffinity * 0.06 +
        orgAffinity +
        directSavedBoost +
        directViewedBoost,
    );

    const preferredModes = profile.preferredModes ?? [];
    const modeFit =
      preferredModes.length === 0 ||
      preferredModes.some((m) => m.toLowerCase() === item.mode.toLowerCase())
        ? 1
        : 0.35;

    const stipendFit = item.stipend >= (profile.minStipend ?? 0) ? 1 : 0.45;
    const preferenceScore = 0.65 * modeFit + 0.35 * stipendFit;

    const blended =
      0.48 * semanticSimilarity +
      0.32 * skillOverlapScore +
      0.12 * preferenceScore +
      interactionBoost;

    const matchScore = Math.max(15, Math.min(98, Math.round(blended * 100)));

    const matchReasons: string[] = [];
    if (matchedSkills.length > 0 && item.skills.length > 0) {
      matchReasons.push(
        `Matches ${matchedSkills.length}/${item.skills.length} listed skills (${matchedSkills.slice(0, 3).join(', ')})`,
      );
    } else if (matchedSkills.length > 0) {
      matchReasons.push(
        `Matches resume keywords (${matchedSkills.slice(0, 3).join(', ')})`,
      );
    } else {
      const targetContext = profile.branch?.trim() || 'your resume';
      matchReasons.push(
        `Semantic alignment (${Math.round(semanticSimilarity * 100)}%) with ${targetContext}`,
      );
    }

    if (savedSkillAffinity > 0.2 && !savedSet.has(item.id)) {
      matchReasons.push('Aligned with skills from your saved internships');
    } else if (savedSet.has(item.id)) {
      matchReasons.push('Bookmarked in your saved watchlist');
    } else if (viewedSkillAffinity > 0.2 && !viewedSet.has(item.id)) {
      matchReasons.push('Similar domain to internships you recently inspected');
    }

    if (preferredModes.some((m) => m.toLowerCase() === item.mode.toLowerCase())) {
      matchReasons.push(`${item.mode} mode fits your preference`);
    }

    return {
      internship: item,
      matchScore,
      semanticSimilarity: Number(semanticSimilarity.toFixed(3)),
      skillOverlapScore: Number(skillOverlapScore.toFixed(3)),
      interactionBoost: Number(interactionBoost.toFixed(3)),
      matchedSkills,
      missingSkills,
      matchReasons: matchReasons.slice(0, 3),
      modelMethod: hasEmbeddings ? 'gemini-embedding-2' : 'semantic-tfidf-vector',
    };
  });

  return scored.sort((a, b) => {
    if (b.matchScore !== a.matchScore) return b.matchScore - a.matchScore;
    return b.semanticSimilarity - a.semanticSimilarity;
  });
}

export function recommendFromResume(params: {
  resumeText: string;
  internships: Internship[];
  embeddingMap?: Record<string, number[]>;
  resumeEmbedding?: number[];
}): ScoredRecommendation[] {
  const cleaned = params.resumeText.trim();
  if (!cleaned) return [];

  const resumeOnlyProfile: CandidateProfile = {
    fullName: 'Candidate',
    degree: '',
    branch: '',
    graduationYear: 2026,
    cgpa: 0,
    age: 22,
    familyGovtJob: false,
    annualFamilyIncomeLakhs: 0,
    skills: [],
    interests: [],
    preferredModes: [],
    preferredStates: [],
    minStipend: 0,
    experienceSummary: '',
    resumeText: cleaned,
  };

  return scoreInternships({
    profile: resumeOnlyProfile,
    internships: params.internships,
    interactions: {
      viewedIds: [],
      savedIds: [],
      dismissedIds: [],
      recentEvents: [],
    },
    embeddingMap: params.embeddingMap,
    profileEmbedding: params.resumeEmbedding,
  });
}

export function evaluatePmSchemeEligibility(profile: CandidateProfile): {
  eligible: boolean;
  criteriaChecks: EligibilityCheck[];
} {
  const ageOk = profile.age >= 21 && profile.age <= 24;
  const incomeOk = profile.annualFamilyIncomeLakhs <= 8;
  const noGovtJobOk = !profile.familyGovtJob;
  const degreeOk = profile.degree.trim().length > 0 && profile.branch.trim().length > 0;

  const criteriaChecks: EligibilityCheck[] = [
    {
      label: 'Age Bracket (21–24 Years)',
      passed: ageOk,
      detail: ageOk
        ? `Candidate age ${profile.age} satisfies PM Internship Scheme bracket (21–24).`
        : `Candidate age ${profile.age} falls outside the 21–24 PM Internship Scheme window.`,
    },
    {
      label: 'Household Income Ceiling (≤ ₹8.0 LPA)',
      passed: incomeOk,
      detail: incomeOk
        ? `Declared family income ₹${profile.annualFamilyIncomeLakhs.toFixed(1)} LPA is within the ₹8.0 LPA limit.`
        : `Declared income ₹${profile.annualFamilyIncomeLakhs.toFixed(1)} LPA exceeds the ₹8.0 LPA cap.`,
    },
    {
      label: 'No Permanent Govt. Employee in Immediate Family',
      passed: noGovtJobOk,
      detail: noGovtJobOk
        ? 'Confirmed no parent or spouse holds a permanent government post.'
        : 'Scheme excludes candidates with permanent government employees in immediate family.',
    },
    {
      label: 'Recognized Academic Qualification',
      passed: degreeOk,
      detail: degreeOk
        ? `${profile.degree} (${profile.branch}) aligns with AICTE & PM Scheme technical tracks.`
        : 'Degree and academic stream are required for verification.',
    },
  ];

  return {
    eligible: criteriaChecks.every((c) => c.passed),
    criteriaChecks,
  };
}

export function computeDeterministicReadiness(
  profile: CandidateProfile,
  recommendations: ScoredRecommendation[],
): ReadinessReport {
  const top10 = recommendations.slice(0, 10);
  const avgTopMatch =
    top10.length > 0
      ? Math.round(top10.reduce((sum, r) => sum + r.matchScore, 0) / top10.length)
      : 50;
  const avgSkillOverlap =
    top10.length > 0
      ? Math.round(
          (top10.reduce((sum, r) => sum + r.skillOverlapScore, 0) / top10.length) * 100,
        )
      : 45;
  const avgSemantic =
    top10.length > 0
      ? Math.round(
          (top10.reduce((sum, r) => sum + r.semanticSimilarity, 0) / top10.length) * 100,
        )
      : 55;

  const eligibility = evaluatePmSchemeEligibility(profile);
  const eligibilityScore =
    Math.round(
      (eligibility.criteriaChecks.filter((c) => c.passed).length /
        eligibility.criteriaChecks.length) *
        100,
    );

  const resumeBonus = profile.resumeText && profile.resumeText.trim().length > 100 ? 6 : 0;
  const experienceScore = Math.min(
    96,
    Math.round(
      42 +
        Math.min(30, profile.experienceSummary.trim().length / 6) +
        Math.min(24, (profile.cgpa / 10) * 24) +
        resumeBonus,
    ),
  );

  const skillDepthScore = Math.min(
    95,
    Math.round(avgSkillOverlap * 0.65 + Math.min(35, profile.skills.length * 4.5)),
  );

  const overallReadinessScore = Math.max(
    25,
    Math.min(
      98,
      Math.round(
        0.30 * skillDepthScore +
          0.25 * avgSemantic +
          0.25 * experienceScore +
          0.20 * eligibilityScore,
      ),
    ),
  );

  const readinessTier =
    overallReadinessScore >= 76
      ? 'Deployment Ready'
      : overallReadinessScore >= 58
        ? 'Competitive Candidate'
        : 'Developing Foundation';

  // Aggregate missing skills across top recommendations
  const gapCounter = new Map<string, number>();
  for (const rec of top10) {
    for (const miss of rec.missingSkills) {
      gapCounter.set(miss, (gapCounter.get(miss) ?? 0) + 1);
    }
  }

  const topSkillGaps: SkillGapItem[] = Array.from(gapCounter.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([skill, count], index) => ({
      skill,
      priority: index < 2 ? 'High' : 'Medium',
      impactCount: count,
      learningAction: `Complete a focused applied mini-project demonstrating ${skill} aligned with AICTE deliverables.`,
    }));

  const embeddingModel = recommendations[0]?.modelMethod ?? 'semantic-tfidf-vector';

  return {
    generatedAt: new Date().toISOString(),
    overallReadinessScore,
    readinessTier,
    pmSchemeEligibility: eligibility,
    domainFitSummary: `Strongest alignment with ${top10[0]?.internship.category ?? 'Technology'} & ${top10[0]?.internship.title ?? 'AICTE'} roles (${avgTopMatch}% average top-10 match).`,
    competencyBreakdown: [
      {
        dimension: 'Technical Skill Coverage',
        score: skillDepthScore,
        benchmark: 72,
        note: `${profile.skills.length} verified profile skills mapped against live AICTE postings.`,
      },
      {
        dimension: 'Semantic Domain Fit',
        score: avgSemantic,
        benchmark: 70,
        note: 'Cosine vector proximity between candidate profile + interaction signals and AICTE descriptions.',
      },
      {
        dimension: 'Academic & Project Depth',
        score: experienceScore,
        benchmark: 68,
        note: `CGPA ${profile.cgpa.toFixed(1)} with documented hands-on implementation background.`,
      },
      {
        dimension: 'PM Scheme Eligibility',
        score: eligibilityScore,
        benchmark: 100,
        note: eligibility.eligible
          ? 'All 4 PM Internship Scheme policy checks satisfied.'
          : 'Review flagged PM Internship Scheme eligibility criteria.',
      },
    ],
    topSkillGaps,
    fourWeekActionPlan: [
      {
        week: 'Week 1',
        focus: topSkillGaps[0] ? `Core Foundations in ${topSkillGaps[0].skill}` : 'AICTE Profile & Portfolio Audit',
        deliverable: 'Publish a clean GitHub repository with documented README and reproducible setup.',
      },
      {
        week: 'Week 2',
        focus: topSkillGaps[1] ? `Applied Integration: ${topSkillGaps[1].skill}` : 'Domain Capstone Sprint',
        deliverable: 'Build an end-to-end prototype solving a public sector or enterprise workflow.',
      },
      {
        week: 'Week 3',
        focus: 'Mock Technical Screening & Case Walkthroughs',
        deliverable: 'Prepare 2-minute architecture explanations for your top 2 projects.',
      },
      {
        week: 'Week 4',
        focus: 'Targeted AICTE & PM Scheme Applications',
        deliverable: 'Submit customized applications to your top 5 ranked postings before closing deadlines.',
      },
    ],
    narrativeSummary: `Based on live AICTE portal listings, ${profile.fullName} scores ${overallReadinessScore}/100 (${readinessTier}). ${eligibility.eligible ? 'Your profile satisfies all PM Internship Scheme eligibility criteria.' : 'One or more PM Internship Scheme policy checks require attention.'}`,
    modelUsed: 'deterministic-vector-baseline',
    embeddingModel,
  };
}

export function buildResumeReadinessReport(params: {
  resumeText: string;
  recommendations: ScoredRecommendation[];
}): ResumeReadinessReport {
  const { resumeText, recommendations } = params;
  const cleaned = resumeText.trim();
  const lower = cleaned.toLowerCase();
  const top8 = recommendations.slice(0, 8);

  const matchedCounter = new Map<string, number>();
  const missingCounter = new Map<string, number>();
  for (const rec of top8) {
    for (const skill of rec.matchedSkills) {
      matchedCounter.set(skill, (matchedCounter.get(skill) ?? 0) + 1);
    }
    for (const gap of rec.missingSkills) {
      missingCounter.set(gap, (missingCounter.get(gap) ?? 0) + 1);
    }
  }

  const topMatchedSkills = Array.from(matchedCounter.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
    .map(([skill]) => skill);

  const topMissingSkills = Array.from(missingCounter.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4);

  const hasEducation = /\b(b\.?tech|b\.?e\.?|b\.?sc|mca|bca|m\.?tech|diploma|cgpa|gpa|university|college|semester)\b/i.test(
    cleaned,
  );
  const hasProjects = /\b(project|projects|built|developed|deployed|designed|implemented|created|internship)\b/i.test(
    cleaned,
  );
  const hasQuantifiedImpact = /\b\d+(\.\d+)?\s*(%|x|ms|s|users|records|queries|hours|days|months|lakhs|cgpa)\b/i.test(
    cleaned,
  );
  const hasLinks = /(github\.com|linkedin\.com|https?:\/\/|portfolio)/i.test(cleaned);
  const wordCount = cleaned ? cleaned.split(/\s+/).length : 0;

  const avgTopMatch =
    top8.length > 0
      ? top8.reduce((sum, r) => sum + r.matchScore, 0) / top8.length
      : 35;

  let structurePoints = 0;
  if (wordCount >= 60) structurePoints += 12;
  else if (wordCount >= 25) structurePoints += 6;
  if (hasEducation) structurePoints += 8;
  if (hasProjects) structurePoints += 10;
  if (hasQuantifiedImpact) structurePoints += 6;
  if (hasLinks) structurePoints += 4;

  const readinessScore = Math.max(
    25,
    Math.min(96, Math.round(avgTopMatch * 0.68 + structurePoints)),
  );

  const readinessTier: ResumeReadinessReport['readinessTier'] =
    readinessScore >= 72
      ? 'Ready to Apply'
      : readinessScore >= 52
        ? 'Competitive'
        : 'Needs Strengthening';

  const whatNotToChange: string[] = [];
  if (topMatchedSkills.length > 0) {
    whatNotToChange.push(
      `Keep your core matched skills prominently listed: ${topMatchedSkills.join(', ')}. They directly align with your top AICTE matches.`,
    );
  }
  if (hasProjects) {
    whatNotToChange.push(
      'Keep your hands-on project and implementation descriptions—employers under AICTE and PM Internship Scheme prioritize practical build evidence.',
    );
  }
  if (hasEducation) {
    whatNotToChange.push(
      'Keep your academic degree, stream, and CGPA/standing details clearly visible for scheme eligibility verification.',
    );
  }
  if (hasQuantifiedImpact) {
    whatNotToChange.push(
      'Keep the concrete metrics and measurable outcomes in your bullet points, as they demonstrate execution depth.',
    );
  }
  if (hasLinks) {
    whatNotToChange.push(
      'Keep your GitHub, portfolio, or LinkedIn links intact so reviewers can inspect live code and deliverables.',
    );
  }
  if (whatNotToChange.length < 2) {
    whatNotToChange.push(
      `Keep your ${top8[0]?.internship.category ?? 'domain'} focus clear—your resume aligns best with roles like "${top8[0]?.internship.title ?? 'AICTE Technical Internships'}".`,
    );
  }

  const whatToChange: string[] = [];
  if (topMissingSkills.length > 0) {
    const highGapList = topMissingSkills
      .slice(0, 3)
      .map(([skill, count]) => `${skill} (${count} top roles)`)
      .join(', ');
    whatToChange.push(
      `Add or build demonstrable proof for frequently requested gap skills across your top matches: ${highGapList}.`,
    );
  }
  if (!hasProjects) {
    whatToChange.push(
      'Add a dedicated Projects section with 2–3 concrete builds, naming the tools/frameworks used and what problem each solved.',
    );
  }
  if (!hasQuantifiedImpact) {
    whatToChange.push(
      'Quantify results in your experience bullets (e.g., dataset size, accuracy, response time, users served, or delivery timeline).',
    );
  }
  if (!hasEducation) {
    whatToChange.push(
      'Include your degree program, branch, graduation year, and CGPA explicitly so AICTE eligibility checks pass smoothly.',
    );
  }
  if (!hasLinks) {
    whatToChange.push(
      'Add a GitHub or portfolio URL at the top of your resume to back up your technical claims.',
    );
  }
  if (wordCount < 60) {
    whatToChange.push(
      'Expand your resume with fuller coursework, toolchains, and project responsibilities (aim for 120–350 words for stronger semantic matching).',
    );
  }
  if (whatToChange.length === 0) {
    whatToChange.push(
      'Tailor your top summary line to mirror the exact role title and domain terminology of your #1 target AICTE listing before applying.',
    );
  }

  const topTitle = top8[0]?.internship.title ?? 'AICTE internships';
  const summary = `Your resume scores ${readinessScore}/100 (${readinessTier}) against live AICTE listings, led by "${topTitle}" (${top8[0]?.matchScore ?? 0}% match). Focus on closing ${topMissingSkills.length > 0 ? topMissingSkills.map(([s]) => s).slice(0, 2).join(' & ') : 'role-specific'} gaps while preserving your verified technical strengths.`;

  return {
    generatedAt: new Date().toISOString(),
    readinessScore,
    readinessTier,
    summary,
    whatNotToChange: whatNotToChange.slice(0, 5),
    whatToChange: whatToChange.slice(0, 5),
    topTargetRoles: recommendations.slice(0, 5).map((r) => ({
      title: r.internship.title,
      organization: r.internship.organization,
      matchScore: r.matchScore,
      mode: r.internship.mode,
      stipendDisplay: r.internship.stipendDisplay,
    })),
  };
}

