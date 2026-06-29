import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI, Type } from '@google/genai';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(express.json({ limit: '10mb' })); // support base64 images

// Initialize Gemini client safely
const apiKey = process.env.GEMINI_API_KEY;
let ai: GoogleGenAI | null = null;

if (apiKey) {
  ai = new GoogleGenAI({
    apiKey: apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
} else {
  console.warn('WARNING: GEMINI_API_KEY is missing. AI features will require configuring the key in Settings > Secrets.');
}

// Helper to get safe Gemini Client
function getAIClient() {
  if (!ai) {
    const key = process.env.GEMINI_API_KEY;
    if (!key) {
      throw new Error('GEMINI_API_KEY is not configured in Secrets. Please add your GEMINI_API_KEY to continue.');
    }
    ai = new GoogleGenAI({
      apiKey: key,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });
  }
  return ai;
}

// -------------------------------------------------------------
// API Routes
// -------------------------------------------------------------

// Helper to call Gemini with exponential backoff retries for 503 / High Demand errors,
// 429 / Rate limit errors, and gracefully fall back to alternative models (like gemini-3.1-flash-lite) 
// or local rule-based parsing/generation if it fails completely.
async function callGeminiWithRetry(client: any, params: any, fallbackValue: any, maxRetries = 3, initialDelay = 1000): Promise<any> {
  let attempt = 1;
  let delay = initialDelay;
  const originalModel = params.model;
  while (true) {
    try {
      const response = await client.models.generateContent(params);
      const text = response.text;
      if (!text) {
        throw new Error('Empty response from Gemini.');
      }
      return JSON.parse(text);
    } catch (error: any) {
      const is503 = error?.status === 503 || 
                    (error?.message && error.message.includes('503')) || 
                    (error?.message && error.message.includes('high demand')) || 
                    error?.code === 503 ||
                    error?.status === 'UNAVAILABLE' ||
                    (error?.message && error.message.includes('UNAVAILABLE'));
                    
      const is429 = error?.status === 429 ||
                    error?.code === 429 ||
                    (error?.message && error.message.includes('429')) ||
                    (error?.message && error.message.includes('quota')) ||
                    (error?.message && error.message.includes('RESOURCE_EXHAUSTED')) ||
                    (error?.message && error.message.includes('limit exceeded'));

      if (is503 && attempt <= maxRetries) {
        console.warn(`[Gemini API] 503 High Demand / Spikes. Retry attempt ${attempt}/${maxRetries} after ${delay}ms...`);
        await new Promise((resolve) => setTimeout(resolve, delay));
        attempt++;
        delay *= 2; // exponential backoff
      } else if (is429) {
        // If it's a 429 quota error and we are on the primary model, switch to fallback model immediately without waiting
        if (params.model === 'gemini-2.5-flash') {
          console.warn(`[Gemini API] Primary model ${originalModel} hit 429. Switching immediately to fallback model gemini-3.1-flash-lite...`);
          params.model = 'gemini-3.1-flash-lite';
          attempt = 1;
          delay = initialDelay;
          continue;
        }

        // If we are already on fallback model or another model, see if we should wait
        let waitTime = delay * 2;
        if (error?.message) {
          const match = error.message.match(/retry in ([\d\.]+)s/i);
          if (match && match[1]) {
            waitTime = Math.ceil(parseFloat(match[1]) * 1000) + 500; // Add 500ms buffer
          }
        }

        // If wait time is short, we can wait a little bit
        if (waitTime <= 2500 && attempt <= maxRetries) {
          console.warn(`[Gemini API] 429 Quota Exceeded. Retry attempt ${attempt}/${maxRetries} after ${waitTime}ms...`);
          await new Promise((resolve) => setTimeout(resolve, waitTime));
          attempt++;
          delay *= 2;
        } else {
          // If wait time is too long (like 40 seconds) or max retries exceeded, fail fast
          console.warn(`[Gemini API] 429 Quota Exceeded with long wait time (${waitTime}ms) or limit. Failing fast to smart fallback.`);
          console.error('[Gemini API] Failed completely after retries or non-retryable error, using smart fallback.', error);
          return fallbackValue;
        }
      } else {
        // If the primary model failed completely, let's try the fallback model
        if (params.model === 'gemini-2.5-flash') {
          console.warn(`[Gemini API] Primary model ${originalModel} failed. Trying fallback model gemini-3.1-flash-lite...`);
          params.model = 'gemini-3.1-flash-lite';
          attempt = 1; // Reset retry counter
          delay = initialDelay;
          continue;
        }

        console.error('[Gemini API] Failed completely after retries or non-retryable error, using smart fallback.', error);
        return fallbackValue;
      }
    }
  }
}

// -------------------------------------------------------------
// API Routes
// -------------------------------------------------------------

// Lightweight ping endpoint to keep serverless container warm
app.get('/api/ping', (_req, res) => {
  res.json({ status: 'ok' });
});

// Route 1: Analyze reported issue (classify, assign severity, check duplicates, write auto-reply)
app.post('/api/analyze-issue', async (req, res) => {
  try {
    const { description, location, existingIssues = [], imageBase64 } = req.body;

    if (!description || !location) {
      return res.status(400).json({ error: 'Description and location are required.' });
    }

    const client = getAIClient();

    // Prepare prompt
    const prompt = `
You are the intelligent community issue assistant for CivicAlert.
Analyze the following newly reported community issue:
Location: "${location}"
Description: "${description}"

Below is a list of existing issues reported in the community. Inspect them to check if a highly similar issue has already been reported in the exact same or close-by area within the last 7 days:
Existing Issues: ${JSON.stringify(existingIssues.slice(0, 50))}

Analyze the issue and return structured JSON conforming to the requested schema.

If there is a highly similar issue of the same category (like pothole, leakage) in the same area (e.g. Pune, Kothrud, Koregaon Park) that is still unresolved (or recently resolved in the last few days), set 'isDuplicate' to true and construct a professional warning string for 'duplicateMessage' like:
"A similar [Category] issue was already reported in [Location] [X] days ago. Are you sure you want to submit a new one?"
Otherwise, set 'isDuplicate' to false and 'duplicateMessage' to "".

Categorize the issue as one of: "Pothole", "Water Leakage", "Streetlight", "Garbage", "Other".
Assign a severity level: "Low", "Medium", "High", and provide a one-line explanation for the decision.

Write a warm, supportive, and highly personalized auto-reply for the citizen. Mention their specific location, issue category, and a friendly, encouraging message. Estimate a realistic Pune municipal resolution timeline (e.g., 3-5 days for potholes, 1-2 days for water leakage, 2-3 days for streetlights, etc.).
`;

    // Construct content parts (include image if uploaded)
    const contents: any[] = [];
    if (imageBase64) {
      let mimeType = 'image/jpeg';
      let data = imageBase64;
      if (imageBase64.includes(';base64,')) {
        const parts = imageBase64.split(';base64,');
        mimeType = parts[0].replace(/^data:/, '');
        data = parts[1];
      }
      contents.push({
        inlineData: {
          mimeType,
          data,
        },
      });
    }
    contents.push({ text: prompt });

    // Define smart local fallback to return if Gemini is unavailable
    const fallbackValue = (() => {
      const descLower = description.toLowerCase();
      let category = 'Other';
      if (descLower.includes('pothole') || descLower.includes('road') || descLower.includes('crater') || descLower.includes('street')) category = 'Pothole';
      else if (descLower.includes('leak') || descLower.includes('pipe') || descLower.includes('water') || descLower.includes('drain')) category = 'Water Leakage';
      else if (descLower.includes('light') || descLower.includes('lamp') || descLower.includes('dark')) category = 'Streetlight';
      else if (descLower.includes('garbage') || descLower.includes('trash') || descLower.includes('waste') || descLower.includes('dump')) category = 'Garbage';

      let severity = 'Medium';
      if (descLower.includes('hazard') || descLower.includes('dangerous') || descLower.includes('accident') || descLower.includes('emergency') || descLower.includes('high')) severity = 'High';
      else if (descLower.includes('small') || descLower.includes('minor') || descLower.includes('low')) severity = 'Low';

      const explanation = `Assigned ${severity} severity based on issue description analysis (offline fallback).`;
      const autoReply = `Thank you for reporting this ${category} issue at ${location}. We have registered your complaint and PMC engineers are on it. Estimated resolution time is 2-4 business days.`;
      
      return {
        category,
        severity,
        explanation,
        isDuplicate: false,
        duplicateMessage: '',
        autoReply
      };
    })();

    // Call Gemini with retry and fallback
    const result = await callGeminiWithRetry(client, {
      model: 'gemini-2.5-flash',
      contents: { parts: contents },
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            category: {
              type: Type.STRING,
              description: "Must be exactly one of: 'Pothole', 'Water Leakage', 'Streetlight', 'Garbage', 'Other'",
            },
            severity: {
              type: Type.STRING,
              description: "Must be exactly one of: 'Low', 'Medium', 'High'",
            },
            explanation: {
              type: Type.STRING,
              description: "A short, one-line explanation of why this severity level was assigned.",
            },
            isDuplicate: {
              type: Type.BOOLEAN,
              description: "True if a highly similar issue of the same type exists in the same general area in the provided list.",
            },
            duplicateMessage: {
              type: Type.STRING,
              description: "Detailed warning message if similar issue is found. Otherwise empty string.",
            },
            autoReply: {
              type: Type.STRING,
              description: "Warm, encouraging, highly personalized auto-reply thanking the user and estimating resolution time.",
            },
          },
          required: ['category', 'severity', 'explanation', 'isDuplicate', 'duplicateMessage', 'autoReply'],
        },
      },
    }, fallbackValue);

    res.json(result);
  } catch (error: any) {
    console.error('Error analyzing issue:', error);
    res.status(500).json({ error: error.message || 'Failed to analyze issue' });
  }
});

// Route 1.5: Detect problem details from an uploaded picture
app.post('/api/detect-problem-from-image', async (req, res) => {
  try {
    const { imageBase64 } = req.body;

    if (!imageBase64) {
      return res.status(400).json({ error: 'imageBase64 is required.' });
    }

    const client = getAIClient();

    const prompt = `
You are the intelligent CivicAlert AI computer vision assistant.
Analyze this community incident photo.
Based on the visual evidence, detect:
1. The most likely category (must be exactly one of: "Pothole", "Water Leakage", "Streetlight", "Garbage", or "Other")
2. A clear, highly descriptive, professional citizen description of what the problem is and its hazard level. Keep it objective, detailed, and realistic (1-2 sentences).
3. A suggested Pune neighborhood location (choose from: "Baner, Pune", "Kothrud, Pune", "Shivaji Nagar, Pune", "Viman Nagar, Pune", "Kalyani Nagar, Pune", "Koregaon Park, Pune") based on visual cues if any, or defaulting to "Shivaji Nagar, Pune".
`;

    const contents: any[] = [];
    let mimeType = 'image/jpeg';
    let data = imageBase64;
    if (imageBase64.includes(';base64,')) {
      const parts = imageBase64.split(';base64,');
      mimeType = parts[0].replace(/^data:/, '');
      data = parts[1];
    }
    contents.push({
      inlineData: {
        mimeType,
        data,
      },
    });
    contents.push({ text: prompt });

    const fallbackValue = {
      category: 'Pothole',
      description: 'A street maintenance issue reported via citizen image. Requires physical inspection by PMC municipal engineers.',
      suggestedLocation: 'Shivaji Nagar, Pune'
    };

    const result = await callGeminiWithRetry(client, {
      model: 'gemini-2.5-flash',
      contents: { parts: contents },
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            category: {
              type: Type.STRING,
              description: "Must be exactly one of: 'Pothole', 'Water Leakage', 'Streetlight', 'Garbage', 'Other'",
            },
            description: {
              type: Type.STRING,
              description: "A professional, detailed citizen-like description of the problem (1-2 sentences).",
            },
            suggestedLocation: {
              type: Type.STRING,
              description: "Must be exactly one of: 'Baner, Pune', 'Kothrud, Pune', 'Shivaji Nagar, Pune', 'Viman Nagar, Pune', 'Kalyani Nagar, Pune', 'Koregaon Park, Pune'",
            },
          },
          required: ['category', 'description', 'suggestedLocation'],
        },
      },
    }, fallbackValue);

    res.json(result);
  } catch (error: any) {
    console.error('Error detecting problem from image:', error);
    res.status(500).json({ error: error.message || 'Failed to detect problem' });
  }
});

// Route 2: Generate AI Insights Page (Community Summary, Suggestions)
app.post('/api/ai-insights', async (req, res) => {
  try {
    const { issues = [] } = req.body;
    const client = getAIClient();

    const prompt = `
You are the Chief AI Officer for CivicAlert.
Analyze the following list of reported community issues from Pune:
${JSON.stringify(issues)}

Provide a structured community summary and suggetions:
1. Determine the most common issue category (Pothole / Water Leakage / Streetlight / Garbage / Other).
2. Identify the highest severity unresolved issue (include its description and location if available).
3. Write a professional, encouraging, and detailed summary paragraph about the overall community health and safety based on the reports.
4. Suggest 2-3 specific, actionable priority initiatives for the municipal authorities to address immediately.
`;

    const fallbackValue = (() => {
      const mostCommonCategory = 'Pothole';
      const highestSeverityUnresolved = {
        description: issues[0]?.description || 'A critical pending hazard requires immediate inspection.',
        location: issues[0]?.location || 'Shivaji Nagar, Pune',
        reason: 'Requires physical inspection to prevent pedestrian and vehicular safety hazards.'
      };
      const communityHealthSummary = `CivicAlert registered several public complaints across various neighborhoods in Pune. Pothole repairs, streetlight maintenance, and waste clearance remain top priorities for reinforcing general public safety.`;
      const suggestedActions = [
        'Deploy public maintenance crews to survey and patch major roads.',
        'Address reported high-risk water leakages immediately to prevent local road wear.',
        'Conduct priority public sanitation sweeps in dense commercial zones.'
      ];
      return {
        mostCommonCategory,
        highestSeverityUnresolved,
        communityHealthSummary,
        suggestedActions
      };
    })();

    const result = await callGeminiWithRetry(client, {
      model: 'gemini-2.5-flash',
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            mostCommonCategory: {
              type: Type.STRING,
              description: "The most common reported category.",
            },
            highestSeverityUnresolved: {
              type: Type.OBJECT,
              properties: {
                description: { type: Type.STRING },
                location: { type: Type.STRING },
                reason: { type: Type.STRING, description: "One sentence why this unresolved issue is highly critical." },
              },
              required: ['description', 'location', 'reason'],
            },
            communityHealthSummary: {
              type: Type.STRING,
              description: "A paragraph summarizing the overall safety and health of the neighborhood.",
            },
            suggestedActions: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: "2 to 3 concrete priority action items for the city administration.",
            },
          },
          required: ['mostCommonCategory', 'highestSeverityUnresolved', 'communityHealthSummary', 'suggestedActions'],
        },
      },
    }, fallbackValue);

    res.json(result);
  } catch (error: any) {
    console.error('Error fetching insights:', error);
    res.status(500).json({ error: error.message || 'Failed to fetch insights' });
  }
});

// Route 3: AI Agent Auto-prioritization
app.post('/api/auto-prioritize', async (req, res) => {
  try {
    const { unresolvedIssues = [] } = req.body;

    if (unresolvedIssues.length === 0) {
      return res.json({ rankedIssues: [] });
    }

    const client = getAIClient();
    const prompt = `
You are the Autonomous AI Prioritization Agent for CivicAlert.
Your job is to act like a municipal operations director and rank open community issues in terms of public safety, physical hazard threat, community engagement (upvotes), and age (days open).

Here is the list of ALL unresolved issues:
${JSON.stringify(unresolvedIssues)}

Rank these issues from 1 (most urgent/highest priority) to ${unresolvedIssues.length} (least urgent/lowest priority).
Provide a precise, one-sentence logical justification for why each issue was placed at its specific priority rank.
`;

    const fallbackValue = (() => {
      const sorted = [...unresolvedIssues].sort((a: any, b: any) => {
        const score = (i: any) => {
          let s = 0;
          if (i.severity === 'High') s += 100;
          if (i.severity === 'Medium') s += 50;
          s += (i.upvotes || 0) * 10;
          return s;
        };
        return score(b) - score(a);
      });
      const rankedIssues = sorted.map((issue: any, index: number) => ({
        rank: index + 1,
        id: issue.id,
        title: issue.description ? (issue.description.substring(0, 45) + '...') : `${issue.category} Issue`,
        location: issue.location || 'Pune',
        category: issue.category || 'Other',
        severity: issue.severity || 'Medium',
        reason: `Ranked at #${index + 1} based on its ${issue.severity} severity classification and community upvotes.`
      }));
      return { rankedIssues };
    })();

    const result = await callGeminiWithRetry(client, {
      model: 'gemini-2.5-flash',
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            rankedIssues: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  rank: { type: Type.INTEGER, description: "1-based order of importance." },
                  id: { type: Type.STRING, description: "The matching ID of the issue." },
                  title: { type: Type.STRING, description: "Brief title or summary." },
                  location: { type: Type.STRING },
                  category: { type: Type.STRING },
                  severity: { type: Type.STRING },
                  reason: { type: Type.STRING, description: "A detailed one-sentence reasoning for this exact rank." },
                },
                required: ['rank', 'id', 'title', 'location', 'category', 'severity', 'reason'],
              },
            },
          },
          required: ['rankedIssues'],
        },
      },
    }, fallbackValue);

    res.json(result);
  } catch (error: any) {
    console.error('Error auto-prioritizing:', error);
    res.status(500).json({ error: error.message || 'Failed to auto-prioritize' });
  }
});

// Route 4: Weekly AI Digest Report Generation
app.post('/api/weekly-digest', async (req, res) => {
  try {
    const defaultDate = new Date().toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' });
    const { issues = [], currentDate = defaultDate } = req.body;
    const client = getAIClient();

    const prompt = `
You are the Civic Intelligence Officer for CivicAlert.
Generate a structured weekly briefing report formatted like a real government/municipal document.
Current Date: ${currentDate}

Analyze these issues:
${JSON.stringify(issues)}

Provide:
1. An official report title and date.
2. The total count of issues reported in the past week.
3. A detailed breakdown count by category.
4. Top 3 urgent unresolved issues (if there are fewer than 3, list the available ones; if there are none, note that).
5. Number of issues resolved this week.
6. A professional closing paragraph summarizing neighborhood progress and safety, along with a mock official reference number / seal text.
`;

    const fallbackValue = (() => {
      const cats = ['Pothole', 'Water Leakage', 'Streetlight', 'Garbage', 'Other'];
      const categoryBreakdown = cats.map(cat => ({
        category: cat,
        count: issues.filter((i: any) => i.category === cat).length
      }));
      const topUrgentUnresolved = issues
        .filter((i: any) => i.status !== 'Resolved')
        .slice(0, 3)
        .map((i: any) => ({
          title: i.description ? (i.description.substring(0, 45) + '...') : `${i.category} Issue`,
          location: i.location || 'Pune',
          severity: i.severity || 'Medium',
          reason: 'Open physical hazard posing threat to motorists and pedestrians.'
        }));
      return {
        digestTitle: 'CivicAlert Weekly Digest - Pune',
        digestDate: currentDate,
        totalReportedThisWeek: issues.length,
        categoryBreakdown,
        topUrgentUnresolved,
        totalResolvedThisWeek: issues.filter((i: any) => i.status === 'Resolved').length,
        communityHealthAssessment: 'Municipal public response and resolution indices have improved this cycle. Community reports are helping PMC target structural repairs dynamically.',
        officialSealText: 'PMC-CA-2026-BAK'
      };
    })();

    const result = await callGeminiWithRetry(client, {
      model: 'gemini-2.5-flash',
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            digestTitle: { type: Type.STRING, description: "Official sounding title like 'CivicAlert Weekly Digest - Pune'" },
            digestDate: { type: Type.STRING },
            totalReportedThisWeek: { type: Type.INTEGER },
            categoryBreakdown: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  category: { type: Type.STRING },
                  count: { type: Type.INTEGER },
                },
                required: ['category', 'count'],
              },
            },
            topUrgentUnresolved: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  title: { type: Type.STRING },
                  location: { type: Type.STRING },
                  severity: { type: Type.STRING },
                  reason: { type: Type.STRING },
                },
                required: ['title', 'location', 'severity', 'reason'],
              },
            },
            totalResolvedThisWeek: { type: Type.INTEGER },
            communityHealthAssessment: { type: Type.STRING, description: "Comprehensive neighborhood progress report paragraph." },
            officialSealText: { type: Type.STRING, description: "Official reference ID or validation string like 'PMC-CA-2026-X12'" },
          },
          required: [
            'digestTitle',
            'digestDate',
            'totalReportedThisWeek',
            'categoryBreakdown',
            'topUrgentUnresolved',
            'totalResolvedThisWeek',
            'communityHealthAssessment',
            'officialSealText',
          ],
        },
      },
    }, fallbackValue);

    res.json(result);
  } catch (error: any) {
    console.error('Error generating weekly digest:', error);
    res.status(500).json({ error: error.message || 'Failed to generate weekly digest' });
  }
});

// Route 5: Autonomous AI Agents (Auto-escalation, Smart duplicate merger, Patrol, and Predictive forecasting)
app.post('/api/run-autonomous-agents', async (req, res) => {
  try {
    const { issues = [] } = req.body;
    const client = getAIClient();

    const now = Date.now();
    const sevenDaysAgoMs = now - 7 * 24 * 60 * 60 * 1000;
    
    const currentDateStr = new Date(now).toLocaleDateString('en-US', { dateStyle: 'medium' });
    const sevenDaysAgoStr = new Date(sevenDaysAgoMs).toLocaleDateString('en-US', { dateStyle: 'medium' });

    const prompt = `
You are the Supreme Autonomous AI Operations Director for Pune Municipal Corporation (PMC) and CivicAlert.
You run in the background and perform structural data decisions and proactive intelligence briefings for Pune.

Current Date of analysis is ${currentDateStr} (Unix timestamp: ${now}).

Analyze the full set of Pune city reports:
${JSON.stringify(issues)}

You must perform three major autonomous operations:

TASK 1: Auto-escalation of Unresolved Issues is DEPRECATED and DISABLED
Do not escalate any issues under any circumstances. Always return an empty array [] for escalations in the JSON response.

TASK 2: Smart Duplicate Merger
Find any clusters of 2 or more (preferably 3 or more) unresolved reports about the same physical problem in the same immediate area (e.g. within same neighborhood like Koregaon Park, Kothrud, Senapati Bapat Road, Shivaji Nagar, Viman Nagar).
If you identify duplicates, group them into a single "master issue" card:
- Select the oldest reported issue ID as the 'masterId'.
- List the other issue IDs as 'duplicateIds'.
- Generate a consolidated 'masterTitle' (e.g., "Multiple Reports: Garbage pile near West End") and a comprehensive 'masterDescription' summarizing details of all.
- Bump the 'bumpedSeverity' level to reflect cumulative urgency (e.g. set to High if 3+ reports, or raise it by one level).
- Provide a 'mergerExplanation' explaining why these were grouped together.
- Generate a 'notificationMessageToReporters' notifying reporters that their issue has been merged into a master card and collectively prioritized.

TASK 3: Weekly Autonomous Patrol Briefing & Predictive Forecasting
Conduct an audit of the city reports and determine:
1. Inactive Issues: Flag issues older than 7 days (reported on or before ${sevenDaysAgoStr}, which corresponds to a "createdAt" timestamp <= ${sevenDaysAgoMs}) with no updates.
2. High-Growth Area: Identify which Pune neighborhood has the fastest-growing number of complaints.
3. Category Spike: Detect any category that has spiked this week.
4. Issue of the Week: Surface exactly one highly critical issue needing immediate attention.
5. Predictive Issue Forecasting: Look at patterns of past issues and predict what is likely to go wrong next (e.g. drainage reports indicating flooding, power issues indicating transformer failures). Provide a detailed prediction paragraph with recommended action.
6. A beautiful briefingMarkdown summarizing the entire state of Pune's workstation and these findings.

Conform strictly to the response schema and output valid JSON.
`;

    const fallbackValue = (() => {
      const escalations: any[] = [];
      const merges: any[] = [];
      const unresolved = issues.filter((i: any) => i.status !== 'Resolved');

      return {
        escalations,
        merges,
        patrolBriefing: {
          flaggedIssueIds: unresolved.slice(0, 2).map((i: any) => i.id),
          fastestGrowingArea: 'Shivaji Nagar',
          categorySpike: 'Potholes',
          issueOfTheWeek: {
            id: unresolved[0]?.id || 'none',
            title: unresolved[0]?.description ? unresolved[0].description.substring(0, 40) + '...' : 'Pothole Incident',
            description: unresolved[0]?.description || 'A reported local community issue.',
            urgencyReason: 'Requires urgent physical inspection by PMC engineering crew.'
          },
          predictiveForecast: 'Water accumulation in low-lying roads is highly likely to trigger pavement surface degradation and pothole propagation in the Baner and Kothrud sectors over the next 48 hours.',
          patrolBriefingTitle: 'PMC Autonomous Patrol Briefing (Backup System)',
          briefingMarkdown: '### Pune City Patrol Briefing\n\nOur system detected open citizen issues requiring attention. Weekly routine maintenance schedules have been dispatched.'
        }
      };
    })();

    const result = await callGeminiWithRetry(client, {
      model: 'gemini-2.5-flash',
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            escalations: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  id: { type: Type.STRING },
                  originalSeverity: { type: Type.STRING },
                  escalatedSeverity: { type: Type.STRING },
                  escalationNotice: { type: Type.STRING },
                },
                required: ['id', 'originalSeverity', 'escalatedSeverity', 'escalationNotice'],
              },
            },
            merges: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  masterId: { type: Type.STRING },
                  duplicateIds: {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                  },
                  masterTitle: { type: Type.STRING },
                  masterDescription: { type: Type.STRING },
                  bumpedSeverity: { type: Type.STRING },
                  mergerExplanation: { type: Type.STRING },
                  notificationMessageToReporters: { type: Type.STRING },
                },
                required: ['masterId', 'duplicateIds', 'masterTitle', 'masterDescription', 'bumpedSeverity', 'mergerExplanation', 'notificationMessageToReporters'],
              },
            },
            patrolBriefing: {
              type: Type.OBJECT,
              properties: {
                flaggedIssueIds: {
                  type: Type.ARRAY,
                  items: { type: Type.STRING },
                },
                fastestGrowingArea: { type: Type.STRING },
                categorySpike: { type: Type.STRING },
                issueOfTheWeek: {
                  type: Type.OBJECT,
                  properties: {
                    id: { type: Type.STRING },
                    title: { type: Type.STRING },
                    description: { type: Type.STRING },
                    urgencyReason: { type: Type.STRING },
                  },
                  required: ['id', 'title', 'description', 'urgencyReason'],
                },
                predictiveForecast: { type: Type.STRING },
                patrolBriefingTitle: { type: Type.STRING },
                briefingMarkdown: { type: Type.STRING },
              },
              required: ['flaggedIssueIds', 'fastestGrowingArea', 'categorySpike', 'issueOfTheWeek', 'predictiveForecast', 'patrolBriefingTitle', 'briefingMarkdown'],
            },
          },
          required: ['escalations', 'merges', 'patrolBriefing'],
        },
      },
    }, fallbackValue);

    res.json(result);
  } catch (error: any) {
    console.error('Error running autonomous agents:', error);
    res.status(500).json({ error: error.message || 'Autonomous agents failed' });
  }
});

// -------------------------------------------------------------
// Serve static frontend files in production or run Vite in dev mode
// -------------------------------------------------------------
const isProd = process.env.NODE_ENV === 'production';

if (isProd) {
  const distPath = path.resolve(__dirname, 'dist');
  app.use(express.static(distPath));
  app.get('*', (_req, res) => {
    res.sendFile(path.resolve(distPath, 'index.html'));
  });
} else {
  // ESM dynamic import to prevent Vite from bundling server files when building
  const { createServer } = await import('vite');
  const viteServer = await createServer({
    server: { middlewareMode: true },
    appType: 'spa',
  });
  app.use(viteServer.middlewares);
}

const port = 3000;
app.listen(port, '0.0.0.0', () => {
  console.log(`[CivicAlert Server] running on http://localhost:${port}`);
});
