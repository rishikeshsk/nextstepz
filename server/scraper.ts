import * as cheerio from 'cheerio';
import fs from 'fs';
import path from 'path';
import type { Internship, ScrapeMeta } from '../src/types/internship';

const AICTE_PAGE_URL = 'https://internship.aicte-india.org/internships?sort=newest';
const AICTE_API_BASE = 'https://internship.aicte-india.org/api/v1/catalogue/internships';
const SNAPSHOT_PATH = path.resolve(process.cwd(), 'src/data/aicteSnapshot.json');

interface ScrapeCache {
  items: Internship[];
  meta: ScrapeMeta;
  fetchedAtMs: number;
}

let memoryCache: ScrapeCache | null = null;
let inFlightScrape: Promise<{ items: Internship[]; meta: ScrapeMeta }> | null = null;
const CACHE_TTL_MS = 10 * 60 * 1000;

function normalizeItem(raw: Record<string, unknown>): Internship {
  const id = String(raw.id ?? raw.uid ?? '');
  const stipend = typeof raw.stipend === 'number' ? raw.stipend : Number(raw.stipend ?? 0) || 0;
  const stipendDisplay =
    typeof raw.stipendDisplay === 'string' && raw.stipendDisplay.trim()
      ? raw.stipendDisplay
      : stipend > 0
        ? `₹${stipend.toLocaleString('en-IN')}`
        : 'Unpaid';

  return {
    id,
    sourceKey: typeof raw.sourceKey === 'string' ? raw.sourceKey : undefined,
    sourceLabel: typeof raw.sourceLabel === 'string' ? raw.sourceLabel : undefined,
    uid: typeof raw.uid === 'string' ? raw.uid : undefined,
    title: String(raw.title ?? 'AICTE Internship'),
    organization: String(raw.organization ?? 'AICTE Partner Organization'),
    logo: typeof raw.logo === 'string' ? raw.logo : undefined,
    category: String(raw.category ?? 'Corporate'),
    location: String(raw.location ?? 'Pan India'),
    state: String(raw.state ?? 'Pan India'),
    district: typeof raw.district === 'string' ? raw.district : undefined,
    mode: String(raw.mode ?? 'Remote'),
    duration: String(raw.duration ?? '2 Months'),
    durationMonths:
      typeof raw.durationMonths === 'number' ? raw.durationMonths : Number(raw.durationMonths ?? 2) || 2,
    stipend,
    stipendMax: typeof raw.stipendMax === 'number' ? raw.stipendMax : stipend,
    stipendDisplay,
    skills: Array.isArray(raw.skills) ? raw.skills.map((s) => String(s)) : [],
    postedDate: String(raw.postedDate ?? ''),
    deadline: String(raw.deadline ?? ''),
    openings: typeof raw.openings === 'number' ? raw.openings : Number(raw.openings ?? 1) || 1,
    description: String(raw.description ?? ''),
    responsibilities: Array.isArray(raw.responsibilities)
      ? raw.responsibilities.map((r) => String(r))
      : [],
    eligibility: Array.isArray(raw.eligibility) ? raw.eligibility.map((e) => String(e)) : [],
    perks: Array.isArray(raw.perks) ? raw.perks.map((p) => String(p)) : [],
    verified: Boolean(raw.verified ?? true),
    featured: Boolean(raw.featured ?? false),
    isOpen: Boolean(raw.isOpen ?? true),
    daysLeft: typeof raw.daysLeft === 'number' ? raw.daysLeft : Number(raw.daysLeft ?? 7) || 7,
    portalUrl: `https://internship.aicte-india.org/internships/${encodeURIComponent(id)}`,
  };
}

async function scrapeHtmlMetadata(): Promise<{
  portalTitle: string;
  flagshipMissions: string[];
}> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 8000);
  try {
    const res = await fetch(AICTE_PAGE_URL, {
      signal: controller.signal,
      headers: {
        'User-Agent': 'Mozilla/5.0 (compatible; KaushalSetuScraper/1.0)',
        Accept: 'text/html',
      },
    });
    if (!res.ok) {
      return {
        portalTitle: 'Browse Internships | National Internship Portal',
        flagshipMissions: [],
      };
    }
    const html = await res.text();
    const $ = cheerio.load(html);
    const portalTitle = $('title').first().text().trim() || 'National Internship Portal';

    const missions = new Set<string>();
    $('a[href*="/dashboards/"]').each((_, el) => {
      const label = $(el).text().replace(/\s+/g, ' ').trim();
      if (label && label.length >= 4 && label.length <= 64) {
        missions.add(label);
      }
    });

    return {
      portalTitle,
      flagshipMissions: Array.from(missions).slice(0, 8),
    };
  } catch {
    return {
      portalTitle: 'Browse Internships | National Internship Portal',
      flagshipMissions: [
        'Indian Army Internship',
        'TULIP Smart Cities Internship',
        'CPWD Internship',
        'NHAI Highways & Infrastructure',
        'AMRUT 2.0 Water Mission',
        'C-DAC Dashboard',
      ],
    };
  } finally {
    clearTimeout(timeout);
  }
}

export async function scrapeAicteInternships(forceRefresh = false): Promise<{
  items: Internship[];
  meta: ScrapeMeta;
}> {
  if (!forceRefresh && memoryCache && Date.now() - memoryCache.fetchedAtMs < CACHE_TTL_MS) {
    return { items: memoryCache.items, meta: memoryCache.meta };
  }

  if (!forceRefresh && inFlightScrape) {
    return inFlightScrape;
  }

  const run = async (): Promise<{ items: Internship[]; meta: ScrapeMeta }> => {
    try {
      const [htmlMeta, page1Res, page2Res, page3Res] = await Promise.all([
        scrapeHtmlMetadata(),
        fetch(`${AICTE_API_BASE}?sort=newest&page=1&pageSize=24`, {
          headers: { 'User-Agent': 'Mozilla/5.0', Accept: 'application/json' },
        }),
        fetch(`${AICTE_API_BASE}?sort=newest&page=2&pageSize=24`, {
          headers: { 'User-Agent': 'Mozilla/5.0', Accept: 'application/json' },
        }),
        fetch(`${AICTE_API_BASE}?sort=newest&page=3&pageSize=24`, {
          headers: { 'User-Agent': 'Mozilla/5.0', Accept: 'application/json' },
        }),
      ]);

      if (!page1Res.ok) {
        throw new Error(`AICTE catalogue responded with HTTP ${page1Res.status}`);
      }

      const p1 = (await page1Res.json()) as { items?: Record<string, unknown>[]; total?: number };
      const p2 = page2Res.ok
        ? ((await page2Res.json()) as { items?: Record<string, unknown>[] })
        : { items: [] };
      const p3 = page3Res.ok
        ? ((await page3Res.json()) as { items?: Record<string, unknown>[] })
        : { items: [] };

      const seenIds = new Set<string>();
      const combinedRaw = [...(p1.items ?? []), ...(p2.items ?? []), ...(p3.items ?? [])];
      const items: Internship[] = [];

      for (const raw of combinedRaw) {
        const normalized = normalizeItem(raw);
        if (normalized.id && !seenIds.has(normalized.id)) {
          seenIds.add(normalized.id);
          items.push(normalized);
        }
      }

      if (items.length === 0) {
        throw new Error('AICTE live scrape returned 0 listings');
      }

      const meta: ScrapeMeta = {
        scrapedAt: new Date().toISOString(),
        sourceUrl: AICTE_PAGE_URL,
        sourceMode: 'live',
        totalOnPortal: typeof p1.total === 'number' ? p1.total : items.length,
        scrapedCount: items.length,
        portalTitle: htmlMeta.portalTitle,
        flagshipMissions: htmlMeta.flagshipMissions,
      };

      memoryCache = { items, meta, fetchedAtMs: Date.now() };
      return { items, meta };
    } catch (err) {
      if (fs.existsSync(SNAPSHOT_PATH)) {
        const snapshotRaw = JSON.parse(fs.readFileSync(SNAPSHOT_PATH, 'utf8')) as {
          scrapedAt?: string;
          sourceUrl?: string;
          totalOnPortal?: number;
          items?: Record<string, unknown>[];
        };
        const items = (snapshotRaw.items ?? []).map(normalizeItem);
        const meta: ScrapeMeta = {
          scrapedAt: snapshotRaw.scrapedAt ?? new Date().toISOString(),
          sourceUrl: snapshotRaw.sourceUrl ?? AICTE_PAGE_URL,
          sourceMode: 'cached_snapshot',
          totalOnPortal: snapshotRaw.totalOnPortal ?? items.length,
          scrapedCount: items.length,
          portalTitle: 'Browse Internships | National Internship Portal',
          flagshipMissions: [
            'Indian Army Internship',
            'TULIP Smart Cities Internship',
            'CPWD Internship',
            'NHAI Highways & Infrastructure',
          ],
        };
        memoryCache = { items, meta, fetchedAtMs: Date.now() - CACHE_TTL_MS + 60_000 };
        return { items, meta };
      }
      throw err;
    }
  };

  inFlightScrape = run();
  try {
    return await inFlightScrape;
  } finally {
    inFlightScrape = null;
  }
}

export async function scrapeInternshipDetailPage(id: string): Promise<{
  internship: Internship | null;
  pageTitle?: string;
  metaDescription?: string;
}> {
  const { items } = await scrapeAicteInternships(false);
  const found = items.find((i) => i.id === id) ?? null;

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);
    const res = await fetch(`https://internship.aicte-india.org/internships/${encodeURIComponent(id)}`, {
      signal: controller.signal,
      headers: { 'User-Agent': 'Mozilla/5.0', Accept: 'text/html' },
    });
    clearTimeout(timeout);
    if (res.ok) {
      const html = await res.text();
      const $ = cheerio.load(html);
      return {
        internship: found,
        pageTitle: $('title').first().text().trim() || undefined,
        metaDescription: $('meta[name="description"]').attr('content') || undefined,
      };
    }
  } catch {
    // Non-fatal; return catalogue record
  }

  return { internship: found };
}
