import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildDeterministicExpertiseQuiz,
  detectExpertiseArea,
  gradeExpertiseQuiz,
} from './expertiseTest';

test('detectExpertiseArea identifies Cloud & DevOps from AWS/Docker/Kubernetes resume', () => {
  const resume =
    'ROHAN KULKARNI, B.Tech CSE. Skills: AWS, Docker, Kubernetes, Linux, CI/CD, Python.';
  const result = detectExpertiseArea(resume, ['AWS', 'Docker']);
  assert.equal(result.area, 'Cloud & DevOps');
  assert.ok(result.detectedSkills.length > 0);
});

test('detectExpertiseArea uses word boundaries so "containerized" does not falsely trigger AI', () => {
  const resume = 'Built containerized SIEM deployment pipelines on Linux servers using Docker.';
  const result = detectExpertiseArea(resume, []);
  assert.notEqual(result.area, 'Data Science & AI/ML');
});

test('detectExpertiseArea identifies Data Science & AI/ML accurately', () => {
  const resume =
    'Built NLP sentiment classifier with PyTorch, scikit-learn, pandas, and numpy.';
  const result = detectExpertiseArea(resume, ['Machine Learning', 'Python']);
  assert.equal(result.area, 'Data Science & AI/ML');
});

test('buildDeterministicExpertiseQuiz produces 5 valid 4-option MCQs', () => {
  const quiz = buildDeterministicExpertiseQuiz({
    resumeText: 'React, TypeScript, Express, REST API, MongoDB web developer.',
    matchedSkills: ['React', 'TypeScript'],
  });
  assert.equal(quiz.expertiseArea, 'Full-Stack Web Engineering');
  assert.equal(quiz.questions.length, 5);
  for (const q of quiz.questions) {
    assert.equal(q.options.length, 4);
    assert.ok(q.correctIndex >= 0 && q.correctIndex < 4);
    assert.ok(q.explanation.length > 10);
  }
});

test('gradeExpertiseQuiz computes score, proficiency band, and weak/strong topics', () => {
  const quiz = buildDeterministicExpertiseQuiz({
    resumeText: 'Wazuh SIEM, Nmap, OWASP, Firewall security analyst resume.',
  });
  const answers: Record<string, number> = {};
  quiz.questions.forEach((q, idx) => {
    // Answer 4 out of 5 correctly
    answers[q.id] = idx === 0 ? (q.correctIndex + 1) % 4 : q.correctIndex;
  });

  const graded = gradeExpertiseQuiz(quiz, answers);
  assert.equal(graded.correctCount, 4);
  assert.equal(graded.totalCount, 5);
  assert.equal(graded.scorePercent, 80);
  assert.equal(graded.proficiencyBand, 'Strong Mastery');
  assert.equal(graded.weakTopics.length, 1);
  assert.equal(graded.strongTopics.length, 4);
});
