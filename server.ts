import 'dotenv/config';
import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { createServer as createViteServer } from 'vite';
import { scrapeAicteInternships, scrapeInternshipDetailPage } from './server/scraper';
import {
  embedCandidateAndInternships,
  embedResumeAndInternships,
  generateAiReadinessReport,
  generateExpertiseQuizWithGemini,
  parseResumeWithGemini,
} from './server/aiEngine';
import { recommendFromResume, scoreInternships } from './src/utils/recommender';
import type { CandidateProfile, InteractionHistory } from './src/types/internship';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT || 3000);

function extractErrorInfo(err: unknown): { status: number; message: string; code: string } {
  const anyErr = err as { status?: number; statusCode?: number; code?: string; message?: string };
  let status = anyErr?.statusCode || anyErr?.status || 500;
  let message = anyErr?.message || 'Unexpected server error';
  let code = anyErr?.code || 'INTERNAL_ERROR';

  if (message.trim().startsWith('{')) {
    try {
      const parsed = JSON.parse(message) as {
        error?: { code?: number; message?: string; status?: string };
      };
      if (parsed.error?.code) status = parsed.error.code;
      if (parsed.error?.status) code = parsed.error.status;
      if (parsed.error?.message) {
        const firstLine = parsed.error.message.split('\n')[0];
        message = firstLine || parsed.error.message;
      }
    } catch {
      // keep raw message
    }
  }

  // Redact any API key patterns or internal secret env references
  message = message
    .replace(/AIza[0-9A-Za-z_-]{10,}/g, '[REDACTED]')
    .replace(/GEMINI_API_KEY/gi, 'API credentials');

  if (status === 429 || code === 'RESOURCE_EXHAUSTED') {
    status = 429;
    code = 'RESOURCE_EXHAUSTED';
    message = 'AI service rate limit reached. Please retry in a few seconds.';
  } else if (status === 401 || code === 'MISSING_API_KEY') {
    message = 'AI service credentials are not configured or unauthorized.';
  }

  return { status, message, code };
}

const BLOCKED_SENSITIVE_PATH_RE =
  /(^|\/)(\.(env|git|streamlit|devcontainer|aistudio|vscode|playwright-cli)|server(\/|$)|package(-lock)?\.json$|bun\.lock$|tsconfig\.json$|vite\.config\.ts$|.*\.py$|.*\.(pem|key|p12|pfx|crt|toml|lock)$)/i;

async function startServer() {
  const app = express();
  app.disable('x-powered-by');
  app.use(express.json({ limit: '6mb' }));

  app.use((req, res, next) => {
    if (BLOCKED_SENSITIVE_PATH_RE.test(req.path)) {
      res.status(404).end();
      return;
    }
    next();
  });

  app.get('/api/health', (_req, res) => {
    res.json({
      ok: true,
      timestamp: new Date().toISOString(),
    });
  });

  app.get('/api/internships', async (req, res) => {
    try {
      const refresh = req.query.refresh === '1' || req.query.refresh === 'true';
      const data = await scrapeAicteInternships(refresh);
      res.json(data);
    } catch (err) {
      const info = extractErrorInfo(err);
      res.status(info.status).json({ error: info.message, code: info.code });
    }
  });

  app.get('/api/internships/:id', async (req, res) => {
    try {
      const detail = await scrapeInternshipDetailPage(req.params.id);
      if (!detail.internship) {
        res.status(404).json({ error: 'Internship not found in AICTE catalogue.' });
        return;
      }
      res.json(detail);
    } catch (err) {
      const info = extractErrorInfo(err);
      res.status(info.status).json({ error: info.message, code: info.code });
    }
  });

  app.post('/api/recommend-resume', async (req, res) => {
    try {
      const { resumeText } = req.body as { resumeText?: string };
      if (!resumeText || !resumeText.trim()) {
        res.status(400).json({ error: 'Please paste your resume text or upload a file first.' });
        return;
      }

      const { items, meta } = await scrapeAicteInternships(false);
      const { profileEmbedding, embeddingMap, embeddingModel } =
        await embedResumeAndInternships(resumeText, items);

      const recommendations = recommendFromResume({
        resumeText,
        internships: items,
        embeddingMap,
        resumeEmbedding: profileEmbedding,
      });

      res.json({
        recommendations,
        scrapeMeta: meta,
        embeddingModel,
      });
    } catch (err) {
      const info = extractErrorInfo(err);
      res.status(info.status).json({ error: info.message, code: info.code });
    }
  });

  app.post('/api/recommend', async (req, res) => {
    try {
      const { profile, interactions } = req.body as {
        profile: CandidateProfile;
        interactions: InteractionHistory;
      };
      if (!profile || !Array.isArray(profile.skills)) {
        res.status(400).json({ error: 'Valid candidate profile with skills is required.' });
        return;
      }

      const { items, meta } = await scrapeAicteInternships(false);
      const { profileEmbedding, embeddingMap, embeddingModel } =
        await embedCandidateAndInternships(profile, items);

      const recommendations = scoreInternships({
        profile,
        internships: items,
        interactions: interactions ?? {
          viewedIds: [],
          savedIds: [],
          dismissedIds: [],
          recentEvents: [],
        },
        embeddingMap,
        profileEmbedding,
      });

      res.json({
        recommendations,
        scrapeMeta: meta,
        embeddingModel,
      });
    } catch (err) {
      const info = extractErrorInfo(err);
      res.status(info.status).json({ error: info.message, code: info.code });
    }
  });

  app.post('/api/readiness-report', async (req, res) => {
    try {
      const { profile, interactions } = req.body as {
        profile: CandidateProfile;
        interactions: InteractionHistory;
      };
      if (!profile || !Array.isArray(profile.skills)) {
        res.status(400).json({ error: 'Valid candidate profile is required.' });
        return;
      }

      const { items, meta } = await scrapeAicteInternships(false);
      const { profileEmbedding, embeddingMap } = await embedCandidateAndInternships(
        profile,
        items,
      );

      const safeInteractions: InteractionHistory = interactions ?? {
        viewedIds: [],
        savedIds: [],
        dismissedIds: [],
        recentEvents: [],
      };

      const recommendations = scoreInternships({
        profile,
        internships: items,
        interactions: safeInteractions,
        embeddingMap,
        profileEmbedding,
      });

      const report = await generateAiReadinessReport({
        profile,
        interactions: safeInteractions,
        recommendations,
      });

      res.json({
        report,
        recommendations,
        scrapeMeta: meta,
      });
    } catch (err) {
      const info = extractErrorInfo(err);
      res.status(info.status).json({ error: info.message, code: info.code });
    }
  });

  app.post('/api/parse-resume', async (req, res) => {
    try {
      const { resumeText, fileBase64, mimeType } = req.body as {
        resumeText?: string;
        fileBase64?: string;
        mimeType?: string;
      };
      const parsed = await parseResumeWithGemini({ resumeText, fileBase64, mimeType });
      res.json(parsed);
    } catch (err) {
      const info = extractErrorInfo(err);
      res.status(info.status).json({ error: info.message, code: info.code });
    }
  });

  app.post('/api/expertise-test', async (req, res) => {
    try {
      const { resumeText, matchedSkills } = req.body as {
        resumeText?: string;
        matchedSkills?: string[];
      };
      if (!resumeText || !resumeText.trim()) {
        res.status(400).json({ error: 'Resume text is required to generate an expertise test.' });
        return;
      }
      const quiz = await generateExpertiseQuizWithGemini({
        resumeText,
        matchedSkills: Array.isArray(matchedSkills) ? matchedSkills : [],
      });
      res.json(quiz);
    } catch (err) {
      const info = extractErrorInfo(err);
      res.status(info.status).json({ error: info.message, code: info.code });
    }
  });

  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath, { dotfiles: 'deny' }));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`KaushalSetu AI server listening on http://localhost:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
