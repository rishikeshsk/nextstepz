import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildLocalTfidfVectors,
  buildResumeReadinessReport,
  computeDeterministicReadiness,
  computeWeightedProfileVector,
  cosineSimilarity,
  evaluatePmSchemeEligibility,
  recommendFromResume,
  scoreInternships,
} from './recommender';
import { DEFAULT_CANDIDATE_PROFILE, EMPTY_INTERACTIONS } from '../data/defaultProfile';
import type { Internship } from '../types/internship';

const SAMPLE_INTERNSHIPS: Internship[] = [
  {
    id: 'int-ai-1',
    title: 'Artificial Intelligence & LLM Intern',
    organization: 'Skill Nexis',
    category: 'MSME',
    location: 'Pan India',
    state: 'Pan India',
    mode: 'Remote',
    duration: '2 Months',
    durationMonths: 2,
    stipend: 15000,
    stipendDisplay: '₹15,000 /month',
    skills: ['Python', 'Large Language Models', 'Natural Language Processing', 'Prompt Engineering'],
    postedDate: '2026-11-18',
    deadline: '2026-11-28',
    openings: 20,
    description: 'Build Python LLM pipelines, retrieval augmented generation chatbots, and NLP tools.',
    responsibilities: ['Develop LLM workflows'],
    eligibility: ['B.Tech CSE'],
    perks: ['6 AICTE credits'],
    verified: true,
    featured: false,
    isOpen: true,
    daysLeft: 10,
  },
  {
    id: 'int-civil-2',
    title: 'Municipal Water Pipeline & GIS Intern',
    organization: 'AMRUT Smart Cities Cell',
    category: 'Government',
    location: 'Jaipur',
    state: 'Rajasthan',
    mode: 'Onsite',
    duration: '6 Months',
    durationMonths: 6,
    stipend: 12000,
    stipendDisplay: '₹12,000 /month',
    skills: ['AutoCAD', 'GIS', 'Civil Surveying', 'Hydrology'],
    postedDate: '2026-11-17',
    deadline: '2026-11-30',
    openings: 10,
    description: 'Support urban water distribution GIS mapping and municipal civil engineering surveys.',
    responsibilities: ['Survey pipeline routes'],
    eligibility: ['B.Tech Civil'],
    perks: ['Certificate'],
    verified: true,
    featured: false,
    isOpen: true,
    daysLeft: 12,
  },
  {
    id: 'int-cyber-3',
    title: 'Cyber Defence & Network Security Intern',
    organization: 'Indian Army Cyber Cell',
    category: 'Army',
    location: 'New Delhi',
    state: 'Delhi',
    mode: 'Onsite',
    duration: '3 Months',
    durationMonths: 3,
    stipend: 20000,
    stipendDisplay: '₹20,000 /month',
    skills: ['Network Security', 'Linux', 'Vulnerability Assessment', 'Python'],
    postedDate: '2026-11-16',
    deadline: '2026-11-25',
    openings: 15,
    description: 'Perform packet analysis, Linux server hardening, and vulnerability testing.',
    responsibilities: ['Audit network endpoints'],
    eligibility: ['B.Tech'],
    perks: ['Certificate'],
    verified: true,
    featured: true,
    isOpen: true,
    daysLeft: 7,
  },
  {
    id: 'int-cyber-4',
    title: 'Enterprise SOC & Threat Intelligence Intern',
    organization: 'C-DAC Security Labs',
    category: 'Government',
    location: 'Pune',
    state: 'Maharashtra',
    mode: 'Hybrid',
    duration: '4 Months',
    durationMonths: 4,
    stipend: 18000,
    stipendDisplay: '₹18,000 /month',
    skills: ['Network Security', 'Linux', 'Threat Intelligence', 'SIEM'],
    postedDate: '2026-11-15',
    deadline: '2026-11-29',
    openings: 8,
    description: 'Monitor SOC alerts, Linux logs, and threat intelligence feeds for public digital infrastructure.',
    responsibilities: ['Analyze SIEM logs'],
    eligibility: ['B.Tech'],
    perks: ['Certificate'],
    verified: true,
    featured: false,
    isOpen: true,
    daysLeft: 11,
  },
];

test('cosineSimilarity computes exact similarity for identical and orthogonal vectors', () => {
  assert.equal(Math.round(cosineSimilarity([1, 0, 1], [1, 0, 1]) * 1000) / 1000, 1);
  assert.equal(cosineSimilarity([1, 0, 0], [0, 1, 0]), 0);
  assert.equal(cosineSimilarity([0, 0, 0], [1, 2, 3]), 0);
});

test('computeWeightedProfileVector shifts centroid toward saved and viewed interaction vectors', () => {
  const profileVec = [1, 0, 0];
  const savedVec = [0, 1, 0];
  const viewedVec = [0, 0, 1];
  const blended = computeWeightedProfileVector(profileVec, [savedVec], [viewedVec]);

  assert.ok(blended[0] > blended[1], 'profile weight should dominate saved weight');
  assert.ok(blended[1] > blended[2], 'saved weight should dominate viewed weight');
});

test('scoreInternships ranks AI/LLM internship highest for AI/ML student profile', () => {
  const results = scoreInternships({
    profile: DEFAULT_CANDIDATE_PROFILE,
    internships: SAMPLE_INTERNSHIPS,
    interactions: EMPTY_INTERACTIONS,
  });

  assert.equal(results.length, 4);
  assert.equal(results[0].internship.id, 'int-ai-1');
  assert.ok(results[0].matchScore > results[1].matchScore);
  assert.ok(results[0].matchedSkills.includes('Python'));
});

test('saving a cybersecurity internship boosts related cybersecurity internship scores', () => {
  const baseline = scoreInternships({
    profile: DEFAULT_CANDIDATE_PROFILE,
    internships: SAMPLE_INTERNSHIPS,
    interactions: EMPTY_INTERACTIONS,
  });
  const baseSoc = baseline.find((r) => r.internship.id === 'int-cyber-4')!;

  const withSavedCyber = scoreInternships({
    profile: DEFAULT_CANDIDATE_PROFILE,
    internships: SAMPLE_INTERNSHIPS,
    interactions: {
      viewedIds: ['int-cyber-3'],
      savedIds: ['int-cyber-3'],
      dismissedIds: [],
      recentEvents: [],
    },
  });
  const boostedSoc = withSavedCyber.find((r) => r.internship.id === 'int-cyber-4')!;

  assert.ok(
    boostedSoc.matchScore > baseSoc.matchScore,
    `expected boosted score (${boostedSoc.matchScore}) > baseline (${baseSoc.matchScore})`,
  );
  assert.ok(boostedSoc.interactionBoost > 0);
});

test('evaluatePmSchemeEligibility enforces age, family income, and govt job rules', () => {
  const eligible = evaluatePmSchemeEligibility(DEFAULT_CANDIDATE_PROFILE);
  assert.equal(eligible.eligible, true);

  const overIncome = evaluatePmSchemeEligibility({
    ...DEFAULT_CANDIDATE_PROFILE,
    annualFamilyIncomeLakhs: 10.5,
    age: 26,
    familyGovtJob: true,
  });
  assert.equal(overIncome.eligible, false);
  assert.equal(overIncome.criteriaChecks.filter((c) => !c.passed).length, 3);
});

test('computeDeterministicReadiness computes valid readiness score and extracts missing skill gaps', () => {
  const ranked = scoreInternships({
    profile: DEFAULT_CANDIDATE_PROFILE,
    internships: SAMPLE_INTERNSHIPS,
    interactions: EMPTY_INTERACTIONS,
  });
  const report = computeDeterministicReadiness(DEFAULT_CANDIDATE_PROFILE, ranked);

  assert.ok(report.overallReadinessScore >= 25 && report.overallReadinessScore <= 98);
  assert.equal(report.competencyBreakdown.length, 4);
  assert.ok(report.topSkillGaps.length > 0);
  assert.equal(report.fourWeekActionPlan.length, 4);
});

test('buildLocalTfidfVectors creates normalized non-empty vectors', () => {
  const { vectors, vocab } = buildLocalTfidfVectors([
    'Python machine learning natural language processing',
    'AutoCAD civil engineering surveying',
  ]);
  assert.equal(vectors.length, 2);
  assert.ok(vocab.length > 4);
  assert.ok(cosineSimilarity(vectors[0], vectors[0]) > 0.99);
});

test('empty-skill unrelated postings do not outrank skill-matched domain postings', () => {
  const emptyUnrelated: Internship = {
    ...SAMPLE_INTERNSHIPS[0],
    id: 'int-empty-finance',
    title: 'Finance & Campus Ambassador Internship',
    skills: [],
    description: 'Assist with campus outreach and accounting ledger entry.',
  };

  const results = scoreInternships({
    profile: DEFAULT_CANDIDATE_PROFILE,
    internships: [emptyUnrelated, SAMPLE_INTERNSHIPS[0]],
    interactions: EMPTY_INTERACTIONS,
    profileEmbedding: [0.8, 0.6],
    embeddingMap: {
      'int-empty-finance': [0.68, 0.73], // ~0.98 raw cosine simulated edge case
      'int-ai-1': [0.8, 0.6],
    },
  });

  assert.equal(results[0].internship.id, 'int-ai-1');
  assert.equal(results[1].skillOverlapScore, 0);
});

test('candidate resumeText influences semantic and skill matching toward resume domain', () => {
  const gisResumeProfile = {
    ...DEFAULT_CANDIDATE_PROFILE,
    skills: ['Documentation'],
    interests: ['Infrastructure'],
    experienceSummary: 'Undergraduate engineering student.',
    resumeText:
      'Civil Engineering Intern at State Water Board. Built AutoCAD pipeline schematics, GIS spatial maps, hydrology runoff models, and municipal civil surveying reports.',
  };

  const results = scoreInternships({
    profile: gisResumeProfile,
    internships: SAMPLE_INTERNSHIPS,
    interactions: EMPTY_INTERACTIONS,
  });

  assert.equal(results[0].internship.id, 'int-civil-2');
  assert.ok(results[0].matchedSkills.includes('AutoCAD'));
  assert.ok(results[0].matchedSkills.includes('GIS'));
});

test('recommendFromResume ranks internships purely from resume text across domains', () => {
  assert.deepEqual(
    recommendFromResume({ resumeText: '   ', internships: SAMPLE_INTERNSHIPS }),
    [],
  );

  const cyberRes = recommendFromResume({
    resumeText:
      'B.Tech CSE candidate. Experienced in Network Security, Linux hardening, SIEM log monitoring, Vulnerability Assessment, and Python automation.',
    internships: SAMPLE_INTERNSHIPS,
  });
  assert.ok(cyberRes[0].internship.id.startsWith('int-cyber'));
  assert.ok(cyberRes[0].matchedSkills.includes('Linux'));

  const civilRes = recommendFromResume({
    resumeText:
      'B.Tech Civil Engineering. Proficient in AutoCAD drafting, GIS spatial mapping, Hydrology, and Civil Surveying.',
    internships: SAMPLE_INTERNSHIPS,
  });
  assert.equal(civilRes[0].internship.id, 'int-civil-2');
  assert.ok(civilRes[0].matchedSkills.includes('AutoCAD'));
});

test('buildResumeReadinessReport generates whatNotToChange and whatToChange grounded in matches', () => {
  const shortResume = 'B.Tech CSE student knowing Python and Linux.';
  const recs = recommendFromResume({ resumeText: shortResume, internships: SAMPLE_INTERNSHIPS });
  const report = buildResumeReadinessReport({ resumeText: shortResume, recommendations: recs });

  assert.ok(report.readinessScore >= 25 && report.readinessScore <= 96);
  assert.ok(report.whatNotToChange.length >= 1);
  assert.ok(report.whatToChange.length >= 2);
  assert.ok(
    report.whatToChange.some((item) => item.toLowerCase().includes('projects') || item.toLowerCase().includes('github')),
  );
  assert.ok(report.topTargetRoles.length > 0);
});
