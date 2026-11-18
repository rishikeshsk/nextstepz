import type { CandidateProfile, InteractionHistory } from '../types/internship';

export const DEFAULT_CANDIDATE_PROFILE: CandidateProfile = {
  fullName: 'Ananya Deshmukh',
  degree: 'B.Tech',
  branch: 'Computer Science & Engineering',
  graduationYear: 2026,
  cgpa: 8.4,
  age: 22,
  familyGovtJob: false,
  annualFamilyIncomeLakhs: 5.2,
  skills: [
    'Python',
    'Machine Learning',
    'Natural Language Processing',
    'SQL',
    'FastAPI',
    'Data Analytics',
    'Git',
  ],
  interests: [
    'Artificial Intelligence',
    'Large Language Models',
    'Public Digital Infrastructure',
    'Cloud Computing',
  ],
  preferredModes: ['Remote', 'Hybrid'],
  preferredStates: ['Pan India', 'Maharashtra', 'Karnataka', 'Delhi'],
  minStipend: 0,
  experienceSummary:
    'Built a document retrieval RAG prototype in Python using sentence embeddings and FastAPI; completed university coursework in statistical ML, DBMS, and REST APIs.',
  resumeText: '',
  resumeFileName: '',
};

export const EMPTY_INTERACTIONS: InteractionHistory = {
  viewedIds: [],
  savedIds: [],
  dismissedIds: [],
  recentEvents: [],
};

export const DOMAIN_PRESETS: Record<
  string,
  { label: string; skills: string[]; interests: string[]; summary: string }
> = {
  ai_ml: {
    label: 'AI / ML & NLP',
    skills: ['Python', 'Machine Learning', 'Natural Language Processing', 'LLMs', 'PyTorch', 'SQL'],
    interests: ['Artificial Intelligence', 'Generative AI', 'Data Science'],
    summary:
      'Built transformer-based text classifiers and semantic search pipelines in Python; strong foundation in linear algebra and model evaluation.',
  },
  fullstack: {
    label: 'Full-Stack Web',
    skills: ['TypeScript', 'React', 'Node.js', 'REST API', 'PostgreSQL', 'Git', 'Tailwind CSS'],
    interests: ['Web Development', 'Cloud Systems', 'GovTech Portals'],
    summary:
      'Developed full-stack TypeScript and React applications backed by Express and relational databases with automated testing.',
  },
  cyber_cloud: {
    label: 'Cybersecurity & Cloud',
    skills: ['Network Security', 'Linux', 'Python', 'Cloud Computing', 'Vulnerability Assessment', 'Docker'],
    interests: ['Cyber Defence', 'Cloud Infrastructure', 'Digital Forensics'],
    summary:
      'Completed hands-on labs in Linux hardening, packet inspection, OWASP API testing, and containerized cloud deployments.',
  },
  core_infra: {
    label: 'Smart Cities & Core Eng.',
    skills: ['AutoCAD', 'GIS', 'Project Management', 'Data Analytics', 'Excel', 'Urban Planning'],
    interests: ['Smart Cities', 'Infrastructure', 'Water & Sanitation Systems'],
    summary:
      'Trained in GIS spatial mapping, civil site surveying, infrastructure CAD drafting, and municipal data reporting.',
  },
};
