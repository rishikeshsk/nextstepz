export type WorkMode = 'Remote' | 'Hybrid' | 'Onsite';

export interface Internship {
  id: string;
  sourceKey?: string;
  sourceLabel?: string;
  uid?: string;
  title: string;
  organization: string;
  logo?: string;
  category: string;
  location: string;
  state: string;
  district?: string;
  mode: WorkMode | string;
  duration: string;
  durationMonths: number;
  stipend: number;
  stipendMax?: number;
  stipendDisplay: string;
  skills: string[];
  postedDate: string;
  deadline: string;
  openings: number;
  description: string;
  responsibilities: string[];
  eligibility: string[];
  perks: string[];
  verified: boolean;
  featured: boolean;
  isOpen: boolean;
  daysLeft: number;
  portalUrl?: string;
}

export interface CandidateProfile {
  fullName: string;
  degree: string;
  branch: string;
  graduationYear: number;
  cgpa: number;
  age: number;
  familyGovtJob: boolean;
  annualFamilyIncomeLakhs: number;
  skills: string[];
  interests: string[];
  preferredModes: WorkMode[];
  preferredStates: string[];
  minStipend: number;
  experienceSummary: string;
  resumeText?: string;
  resumeFileName?: string;
}

export interface InteractionEvent {
  internshipId: string;
  title: string;
  organization: string;
  type: 'view' | 'save' | 'unsave';
  timestamp: string;
}

export interface InteractionHistory {
  viewedIds: string[];
  savedIds: string[];
  dismissedIds: string[];
  recentEvents: InteractionEvent[];
}

export interface ScoredRecommendation {
  internship: Internship;
  matchScore: number;
  semanticSimilarity: number;
  skillOverlapScore: number;
  interactionBoost: number;
  matchedSkills: string[];
  missingSkills: string[];
  matchReasons: string[];
  modelMethod: 'gemini-embedding-2' | 'semantic-tfidf-vector';
}

export interface EligibilityCheck {
  label: string;
  passed: boolean;
  detail: string;
}

export interface CompetencyMetric {
  dimension: string;
  score: number;
  benchmark: number;
  note: string;
}

export interface SkillGapItem {
  skill: string;
  priority: 'High' | 'Medium';
  impactCount: number;
  learningAction: string;
}

export interface PrepWeekItem {
  week: string;
  focus: string;
  deliverable: string;
}

export interface ReadinessReport {
  generatedAt: string;
  overallReadinessScore: number;
  readinessTier: 'Deployment Ready' | 'Competitive Candidate' | 'Developing Foundation';
  pmSchemeEligibility: {
    eligible: boolean;
    criteriaChecks: EligibilityCheck[];
  };
  domainFitSummary: string;
  competencyBreakdown: CompetencyMetric[];
  topSkillGaps: SkillGapItem[];
  fourWeekActionPlan: PrepWeekItem[];
  narrativeSummary: string;
  modelUsed: string;
  embeddingModel: string;
}

export interface ScrapeMeta {
  scrapedAt: string;
  sourceUrl: string;
  sourceMode: 'live' | 'cached_snapshot';
  totalOnPortal: number;
  scrapedCount: number;
  portalTitle: string;
  flagshipMissions: string[];
}

export interface ExpertiseQuestion {
  id: string;
  topic: string;
  prompt: string;
  options: string[];
  correctIndex: number;
  explanation: string;
}

export interface ExpertiseQuiz {
  expertiseArea: string;
  detectedSkills: string[];
  source: 'gemini' | 'curated';
  questions: ExpertiseQuestion[];
}

export interface ExpertiseQuestionOutcome {
  id: string;
  topic: string;
  prompt: string;
  selectedIndex: number;
  correctIndex: number;
  isCorrect: boolean;
  selectedOption: string;
  correctOption: string;
  explanation: string;
}

export interface ExpertiseTestResult {
  completedAt: string;
  expertiseArea: string;
  correctCount: number;
  totalCount: number;
  scorePercent: number;
  proficiencyBand: 'Strong Mastery' | 'Competent Foundation' | 'Needs Focused Review';
  strongTopics: string[];
  weakTopics: string[];
  outcomes: ExpertiseQuestionOutcome[];
}

export interface ResumeReadinessReport {
  generatedAt: string;
  readinessScore: number;
  readinessTier: 'Ready to Apply' | 'Competitive' | 'Needs Strengthening';
  summary: string;
  whatNotToChange: string[];
  whatToChange: string[];
  topTargetRoles: {
    title: string;
    organization: string;
    matchScore: number;
    mode: string;
    stipendDisplay: string;
  }[];
  expertiseResult?: ExpertiseTestResult;
}

