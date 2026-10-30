import express from 'express';
import dotenv from 'dotenv';
import path from 'path';
import fs from 'fs';
import os from 'os';
import { fileURLToPath } from 'url';
import { GoogleGenAI } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;
const IS_VERCEL = Boolean(process.env.VERCEL);

// Vercel serverless giới hạn body ~4.5MB; giữ 10mb khi chạy máy chủ riêng.
app.use(express.json({ limit: IS_VERCEL ? '4mb' : '10mb' }));

// ===== Gemini: khóa CHỈ đọc từ biến môi trường phía máy chủ, không bao giờ gửi xuống trình duyệt =====
const apiKey = process.env.GEMINI_API_KEY || '';
const hasGeminiKey = Boolean(apiKey && apiKey !== 'MY_GEMINI_API_KEY' && apiKey.trim().length > 0);
const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-3.8-flash';

let ai: GoogleGenAI | null = null;
if (hasGeminiKey) {
  ai = new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

/**
 * Lấy đối tượng AI client linh hoạt:
 * 1. Nếu client gửi khóa cá nhân (header x-gemini-api-key hợp lệ) -> ưu tiên sử dụng khóa đó.
 * 2. Nếu không -> sử dụng khóa server GEMINI_API_KEY (nếu có).
 */
function getAIInstance(req?: express.Request): { client: GoogleGenAI | null; isUserKey: boolean } {
  const userKey = (req?.headers['x-gemini-api-key'] as string)?.trim() || (req?.body?.userApiKey as string)?.trim();
  if (userKey && userKey.startsWith('AIza') && userKey.length >= 25) {
    try {
      const userClient = new GoogleGenAI({
        apiKey: userKey,
        httpOptions: {
          headers: {
            'User-Agent': 'eduverse-user-key',
          },
        },
      });
      return { client: userClient, isUserKey: true };
    } catch {}
  }
  return { client: ai, isUserKey: false };
}


// ===== Kiểm soát nguồn gọi (tùy chọn): ALLOWED_ORIGINS="https://ten-mien.vercel.app,https://..." =====
const allowedOrigins = (process.env.ALLOWED_ORIGINS || '')
  .split(',')
  .map(s => s.trim())
  .filter(Boolean);

function originGuard(req: express.Request, res: express.Response, next: express.NextFunction) {
  if (allowedOrigins.length === 0) return next();
  const origin = req.headers.origin || '';
  const referer = req.headers.referer || '';
  const ok = allowedOrigins.some(o => origin === o || referer.startsWith(o));
  if (!ok && (origin || referer)) {
    return res.status(403).json({ error: 'Nguồn gọi không được phép.', code: 'ORIGIN_FORBIDDEN' });
  }
  return next();
}

// ===== Giới hạn tần suất đơn giản theo IP cho các API tốn chi phí AI =====
const RATE_LIMIT_PER_MIN = parseInt(process.env.RATE_LIMIT_PER_MIN || '20', 10);
const rateBuckets = new Map<string, { count: number; resetAt: number }>();

function aiRateLimit(req: express.Request, res: express.Response, next: express.NextFunction) {
  const ip = (req.headers['x-forwarded-for'] as string)?.split(',')[0]?.trim() || req.socket.remoteAddress || 'unknown';
  const now = Date.now();
  const bucket = rateBuckets.get(ip);
  if (!bucket || bucket.resetAt < now) {
    rateBuckets.set(ip, { count: 1, resetAt: now + 60_000 });
    return next();
  }
  if (bucket.count >= RATE_LIMIT_PER_MIN) {
    res.setHeader('Retry-After', Math.ceil((bucket.resetAt - now) / 1000).toString());
    return res.status(429).json({ error: 'Quá nhiều yêu cầu AI. Vui lòng thử lại sau ít phút.', code: 'RATE_LIMITED' });
  }
  bucket.count += 1;
  return next();
}

app.use('/api/gemini', originGuard, aiRateLimit);
app.use('/api/vr/analyze-360-image', originGuard, aiRateLimit);
app.use('/api/vr/publish', originGuard);
app.use('/api/vr/submit-attempt', originGuard);

// ===== Kiểm thử Gemini thật (có cache 5 phút để không tốn hạn mức) =====
type GeminiProbe = { status: 'ok' | 'no_key' | 'error'; errorCode?: string; message?: string; checkedAt: string };
let probeCache: { at: number; result: GeminiProbe } | null = null;

function classifyGeminiError(err: any): { errorCode: string; message: string } {
  const raw = String(err?.message || err || '');
  const status = err?.status || err?.code;
  if (status === 401 || status === 403 || /API key|PERMISSION_DENIED|UNAUTHENTICATED/i.test(raw)) {
    return { errorCode: 'INVALID_KEY', message: 'Khóa Gemini không hợp lệ hoặc không có quyền.' };
  }
  if (status === 429 || /RESOURCE_EXHAUSTED|quota/i.test(raw)) {
    return { errorCode: 'QUOTA', message: 'Đã hết hạn mức Gemini hoặc bị giới hạn tần suất.' };
  }
  if (status === 404 || /not found|model/i.test(raw)) {
    return { errorCode: 'MODEL', message: `Mô hình "${GEMINI_MODEL}" không khả dụng với khóa này.` };
  }
  if (/fetch failed|ENOTFOUND|ECONN|ETIMEDOUT|network/i.test(raw)) {
    return { errorCode: 'NETWORK', message: 'Máy chủ không kết nối được tới dịch vụ Gemini.' };
  }
  return { errorCode: 'UNKNOWN', message: 'Dịch vụ Gemini trả về lỗi.' };
}

async function probeGemini(force = false): Promise<GeminiProbe> {
  if (!ai) return { status: 'no_key', checkedAt: new Date().toISOString() };
  const age = probeCache ? Date.now() - probeCache.at : Infinity;
  // Không cho ép kiểm thử dồn dập (tối thiểu 30 giây) để bảo vệ hạn mức
  if (probeCache && (age < 30_000 || (!force && age < 5 * 60_000))) return probeCache.result;
  let result: GeminiProbe;
  try {
    const r = await ai.models.generateContent({ model: GEMINI_MODEL, contents: 'Trả lời đúng một từ: OK' });
    result = (r.text || '').trim().length > 0
      ? { status: 'ok', checkedAt: new Date().toISOString() }
      : { status: 'error', errorCode: 'EMPTY', message: 'Gemini trả về phản hồi rỗng.', checkedAt: new Date().toISOString() };
  } catch (err: any) {
    result = { status: 'error', ...classifyGeminiError(err), checkedAt: new Date().toISOString() };
  }
  probeCache = { at: Date.now(), result };
  return result;
}

// Health check endpoint
// - /api/health          : nhanh, chỉ kiểm tra cấu hình
// - /api/health?probe=1  : gọi Gemini thật (cache 5 phút) để xác nhận AI hoạt động
app.get('/api/health', async (req, res) => {
  const { client: userAI, isUserKey } = getAIInstance(req);
  const wantProbe = req.query.probe === '1';
  let gemini = wantProbe ? await probeGemini(req.query.force === '1') : (probeCache?.result || null);

  // Nếu client gửi khóa người dùng, probe trực tiếp với khóa người dùng
  if (isUserKey && userAI && wantProbe) {
    try {
      const r = await userAI.models.generateContent({ model: GEMINI_MODEL, contents: 'OK' });
      gemini = (r.text || '').trim().length > 0
        ? { status: 'ok', checkedAt: new Date().toISOString() }
        : { status: 'error', errorCode: 'EMPTY', message: 'Gemini trả về phản hồi rỗng.', checkedAt: new Date().toISOString() };
    } catch (err: any) {
      gemini = { status: 'error', ...classifyGeminiError(err), checkedAt: new Date().toISOString() };
    }
  }

  res.json({
    status: 'ok',
    hasGeminiKey: hasGeminiKey || isUserKey,
    hasUserKey: isUserKey,
    model: GEMINI_MODEL,
    gemini, // null = chưa kiểm thử
    storagePersistent: !IS_VERCEL,
    runtime: IS_VERCEL ? 'vercel' : 'node',
    platform: 'EduLife AI Studio 2026',
    timestamp: new Date().toISOString(),
  });
});

// Endpoint: Xác thực khóa Gemini API từ người dùng (an toàn, không lưu trữ trên máy chủ)
app.post('/api/gemini/verify-key', async (req, res) => {
  const userKey = ((req.body?.apiKey || req.headers['x-gemini-api-key']) as string || '').trim();
  if (!userKey || !userKey.startsWith('AIza') || userKey.length < 25) {
    return res.status(400).json({ 
      ok: false, 
      message: 'Định dạng khóa không hợp lệ. Khóa Google Gemini bắt đầu bằng "AIza..." và có độ dài khoảng 39 ký tự.' 
    });
  }

  try {
    const testAI = new GoogleGenAI({
      apiKey: userKey,
      httpOptions: {
        headers: {
          'User-Agent': 'eduverse-verify-key',
        },
      },
    });

    const resp = await testAI.models.generateContent({
      model: GEMINI_MODEL,
      contents: 'Ping: trả lời đúng một từ: OK',
    });

    if ((resp.text || '').trim().length > 0) {
      return res.json({ 
        ok: true, 
        message: 'Xác thực thành công! Khóa Gemini API hoạt động hoàn hảo và sẵn sàng phục vụ.',
        model: GEMINI_MODEL
      });
    }
    return res.status(500).json({ ok: false, message: 'Google Gemini phản hồi rỗng.' });
  } catch (err: any) {
    const errInfo = classifyGeminiError(err);
    return res.status(400).json({ 
      ok: false, 
      message: errInfo.message || 'Không thể xác thực khóa với Google Gemini.',
      errorCode: errInfo.errorCode 
    });
  }
});


// Helper: safe JSON parsing from AI text
function cleanAndParseJSON(rawText: string) {
  let cleaned = rawText.trim();
  if (cleaned.startsWith('```json')) {
    cleaned = cleaned.replace(/^```json\s*/, '').replace(/\s*```$/, '');
  } else if (cleaned.startsWith('```')) {
    cleaned = cleaned.replace(/^```\s*/, '').replace(/\s*```$/, '');
  }
  return JSON.parse(cleaned);
}

// 1. Generate Scenario endpoint (Upgraded for Multi-Scene, Duration, 8 Styles, and Roles)
app.post('/api/gemini/generate-scenario', async (req, res) => {
  const { 
    grade, 
    topic, 
    activityTitle, 
    sourceType,
    customTopic,
    durationOption,
    durationMinutes,
    style,
    characterCount,
    customCharacters,
    participationFormat,
    situationCount,
    promptNotes 
  } = req.body;

  const targetSituationCount = Math.max(1, Math.min(30, parseInt(situationCount, 10) || 3));

  const durationMin = durationMinutes || (
    durationOption === '3min' ? 3 :
    durationOption === '5min' ? 5 :
    durationOption === '10min' ? 10 :
    durationOption === '15min' ? 15 :
    durationOption === '20min' ? 20 : 10
  );

  const sceneCount = durationMin <= 3 ? 1 : durationMin <= 5 ? 2 : durationMin <= 10 ? 3 : durationMin <= 15 ? 4 : 5;
  const dialoguePerScene = durationMin <= 3 ? 4 : durationMin <= 5 ? 4 : 5;

  const styleDescriptions: Record<string, string> = {
    everyday: 'Đời thường, gần gũi: ngôn ngữ học sinh tự nhiên, chân thật, tránh giáo điều.',
    humorous: 'Hài hước, dí dỏm: tình huống hiểu lầm vui nhộn, tiếng cười nhẹ nhàng, nhân văn, tuyệt đối không chế giễu.',
    detective: 'Thám tử, điều tra: có manh mối, tang chứng, lời khai nhân chứng, suy luận logic tìm ra sự thật.',
    emotional: 'Cảm xúc, nhân văn: chạm đến rung động tâm lý sâu lắng, tình cảm gia đình, thầy cô, bạn bè xúc động.',
    debate: 'Tranh biện, đa chiều: hai phe bảo vệ quan điểm đối lập với luận cứ sắc bén, phân tích thiệt hơn.',
    adventure: 'Phiêu lưu, khám phá: thử thách thực địa, dã ngoại, sinh tồn, kỹ năng phối hợp vượt khó.',
    career_interview: 'Phỏng vấn nghề nghiệp: nhà tuyển dụng / chuyên gia phỏng vấn ứng viên học sinh, xử lý tình huống nghiệp vụ.',
    interactive_choice: 'Kịch tương tác nhiều lựa chọn: các quyết định phân nhánh rõ ràng định đoạt kết cục.'
  };

  const selectedStyleDesc = styleDescriptions[style] || styleDescriptions['everyday'];
  const { client: activeAI } = getAIInstance(req);

  if (!activeAI) {
    return res.status(200).json({
      fallback: true,
      message: 'Sử dụng kịch bản sư phạm chất lượng cao từ cơ sở dữ liệu học liệu (Chế độ offline)',
    });
  }

  try {
    const prompt = `
Bạn là chuyên gia sư phạm cao cấp bộ môn Hoạt động trải nghiệm, hướng nghiệp THPT (Chương trình GDPT 2018 - Kết nối tri thức với cuộc sống).
Hãy thiết kế kịch bản sân khấu tình huống học đường đa cảnh (Multi-scene Roleplay) chất lượng xuất sắc với các thông số bắt buộc:

1. THÔNG TIN BÀI DẠY:
- Khối lớp: Lớp ${grade || 10}
- Chủ đề: ${customTopic?.title || topic || 'Kỹ năng làm chủ bản thân và thích ứng'}
- Hoạt động: ${activityTitle || 'Xử lý tình huống tương tác'}
- Nguồn: ${sourceType === 'custom' ? 'Giáo viên tự biên soạn đề bài riêng' : 'Theo phân phối chương trình SGK'}
${customTopic ? `- Bối cảnh GV cung cấp: ${customTopic.context}
- Vấn đề trọng tâm: ${customTopic.problem}
- Nhân vật GV yêu cầu: ${customTopic.characters}
- Mục tiêu giáo dục: ${customTopic.goal}
- Thông điệp mong muốn: ${customTopic.message}` : ''}

2. THỜI LƯỢNG & QUY MÔ KỊCH BẢN:
- Thời lượng: ${durationMin} phút
- Số cảnh bắt buộc: ${sceneCount} cảnh riêng biệt (Scene 1 đến Scene ${sceneCount})
- Số lượt thoại mỗi cảnh: ít nhất ${dialoguePerScene} lượt thoại qua lại giữa các nhân vật
- Số nhân vật: ${characterCount || 3} nhân vật. ${customCharacters ? `Danh sách nhân vật: ${JSON.stringify(customCharacters)}` : ''}

3. SỐ LƯỢNG TÌNH HUỐNG THỬ THÁCH (BẮT BUỘC ĐÚNG ${targetSituationCount} TÌNH HUỐNG):
- Bạn PHẢI tạo đúng chính xác ${targetSituationCount} tình huống riêng biệt (từ 1 đến ${targetSituationCount}) trong mảng "situations".
- Độ khó tăng dần:
  + Tình huống 1 đến ${Math.ceil(targetSituationCount / 3)}: "Cơ bản"
  + Tình huống ${Math.ceil(targetSituationCount / 3) + 1} đến ${Math.ceil((targetSituationCount * 2) / 3)}: "Nâng cao"
  + Tình huống còn lại: "Thử thách chuyên sâu"
- Tuyệt đối không lặp lại nội dung hoặc xung đột. Mỗi tình huống có 1 phương án tối ưu (isOptimal: true) và 2-3 phương án còn lại (isOptimal: false).

4. PHONG CÁCH KỊCH BẢN:
- Phong cách: "${style || 'everyday'}"
- Đặc trưng phong cách: ${selectedStyleDesc}

5. HÌNH THỨC THAM GIA:
- Hình thức: ${participationFormat || 'branching'}
- Ghi chú bổ sung của GV: ${promptNotes || 'Tình huống kịch tính, lôi cuốn, đúng tâm lý học sinh THPT'}

Yêu cầu xuất ra định dạng JSON DUY NHẤT (không bọc trong markdown code fence lạ, đúng cấu trúc sau):
{
  "title": "Tên kịch bản sân khấu hấp dẫn",
  "category": "Cá nhân" | "Gia đình" | "Nhà trường" | "Xã hội" | "Nghề nghiệp",
  "context": "Tóm tắt bối cảnh bao quát toàn bộ vở kịch",
  "conflict": "Mâu thuẫn hoặc thách thức cốt lõi cần giải quyết",
  "situationCount": ${targetSituationCount},
  "situations": [
    {
      "id": "sit-1",
      "orderNumber": 1,
      "difficulty": "Cơ bản",
      "title": "Tên tình huống 1",
      "context": "Bối cảnh tình huống 1",
      "problem": "Vấn đề nan giải",
      "question": "Câu hỏi xử lý tình huống đặt ra cho học sinh",
      "options": [
        {
          "id": "A",
          "text": "Phương án A",
          "consequence": "Hệ quả A",
          "isOptimal": true,
          "impactType": "positive",
          "feedback": "Giải thích vì sao là phương án tối ưu"
        },
        {
          "id": "B",
          "text": "Phương án B",
          "consequence": "Hệ quả B",
          "isOptimal": false,
          "impactType": "caution",
          "feedback": "Giải thích hạn chế và rủi ro"
        },
        {
          "id": "C",
          "text": "Phương án C",
          "consequence": "Hệ quả C",
          "isOptimal": false,
          "impactType": "neutral",
          "feedback": "Giải thích tính thụ động"
        }
      ],
      "educationalTakeaway": "Bài học rút ra từ tình huống này",
      "reflectionPrompt": "Câu hỏi tự vấn cá nhân"
    }
  ],
  "characters": [
    {"name": "Tên nhân vật", "role": "Vai trò", "description": "Nét tính cách, tâm lý đặc trưng"}
  ],
  "scenes": [
    {
      "sceneNumber": 1,
      "sceneTitle": "Tên phân cảnh 1",
      "sceneContext": "Bối cảnh cụ thể của cảnh này",
      "dialogues": [
        {"speaker": "Tên nhân vật", "role": "Vai trò", "emotion": "neutral" | "confused" | "worried" | "enthusiastic" | "thoughtful" | "calm", "text": "Lời thoại tự nhiên"}
      ],
      "climaxOrConflict": "Điểm thắt nút / mâu thuẫn của cảnh này",
      "studentTask": "Nhiệm vụ hoặc câu hỏi đặt ra cho học sinh tại cảnh này",
      "options": [
        {
          "id": "A",
          "text": "Phương án xử lý 1",
          "consequence": "Hệ quả tức thì",
          "impactType": "positive" | "caution" | "neutral",
          "scoreAdvice": "Lời nhận xét phản tư",
          "growthFeedback": "Bài học phát triển bản thân"
        },
        {
          "id": "B",
          "text": "Phương án xử lý 2",
          "consequence": "Hệ quả tương ứng",
          "impactType": "positive" | "caution" | "neutral",
          "scoreAdvice": "Lời nhận xét phản tư",
          "growthFeedback": "Bài học phát triển bản thân"
        }
      ],
      "sceneTakeaway": "Gợi mở sau cảnh này"
    }
  ],
  "dialogue": [
    {"speaker": "Nhân vật", "role": "Vai trò", "text": "Tất cả lời thoại tổng hợp liên tục"}
  ],
  "options": [
    {"id": "A", "text": "Phương án tổng thể 1", "consequence": "Hệ quả tổng thể", "impactType": "positive", "scoreAdvice": "Nhận xét", "growthFeedback": "Bài học"}
  ],
  "reflectionQuestions": [
    "Câu hỏi phản tư 1",
    "Câu hỏi phản tư 2"
  ],
  "educationalMessage": "Thông điệp giáo dục ý nghĩa và bài học sâu sắc",
  "evaluationCriteria": [
    "Khả năng nhận diện vấn đề và làm chủ cảm xúc",
    "Kỹ năng lắng nghe thấu cảm và giao tiếp hòa nhã",
    "Tính trách nhiệm và khả thi của giải pháp đưa ra"
  ]
}
`;

    const response = await activeAI.models.generateContent({
      model: GEMINI_MODEL,
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        temperature: 0.75,
      },
    });

    const parsed = cleanAndParseJSON(response.text || '{}');
    if (!Array.isArray(parsed.situations)) {
      parsed.situations = [];
    }

    // Ensure exact targetSituationCount without duplicates or missing items (1 to 30)
    if (parsed.situations.length < targetSituationCount) {
      const needed = targetSituationCount - parsed.situations.length;
      for (let i = 0; i < needed; i++) {
        const orderNum = parsed.situations.length + 1;
        const diff = orderNum <= Math.ceil(targetSituationCount / 3) ? 'Cơ bản' : 
                     orderNum <= Math.ceil((targetSituationCount * 2) / 3) ? 'Nâng cao' : 'Thử thách chuyên sâu';
        parsed.situations.push({
          id: `sit-auto-${orderNum}-${Date.now()}`,
          orderNumber: orderNum,
          difficulty: diff,
          title: `Tình huống ${orderNum}: Xử lý thách thức trải nghiệm học đường số ${orderNum}`,
          context: `Tại môi trường THPT, học sinh đối diện với tình huống cần ra quyết định và thể hiện trách nhiệm cá nhân trong tập thể.`,
          problem: `Mâu thuẫn nhận thức hoặc áp lực cần giải quyết thấu cảm và khoa học.`,
          question: `Ở tình huống ${orderNum}, phương án hành động nào thể hiện sự chủ động, tôn trọng và đạt kết quả bền vững nhất?`,
          options: [
            {
              id: 'A',
              text: 'Bình tĩnh lắng nghe, phân tích nguyên nhân khách quan và đối thoại tìm giải pháp chung.',
              consequence: 'Bảo vệ mối quan hệ, nâng cao kỹ năng làm việc nhóm và đạt đồng thuận cao.',
              isOptimal: true,
              impactType: 'positive',
              feedback: 'Rất chính xác! Đối thoại xây dựng và tôn trọng sự khác biệt là giải pháp tối ưu.'
            },
            {
              id: 'B',
              text: 'Phản ứng gay gắt và bảo vệ ý kiến chủ quan bất chấp ý kiến số đông.',
              consequence: 'Gây bất hòa và làm chậm tiến độ tập thể.',
              isOptimal: false,
              impactType: 'caution',
              feedback: 'Nóng nảy dễ dẫn đến xung đột không đáng có.'
            },
            {
              id: 'C',
              text: 'Im lặng, thoái thác trách nhiệm và để mặc tình huống tự trôi qua.',
              consequence: 'Vấn đề tồn đọng và làm giảm sút uy tín cá nhân.',
              isOptimal: false,
              impactType: 'neutral',
              feedback: 'Tránh né không giúp phát triển năng lực giải quyết vấn đề.'
            }
          ],
          educationalTakeaway: 'Tôn trọng sự khác biệt và cùng nhau kiến tạo giải pháp tích cực.',
          reflectionPrompt: 'Em rút ra bài học gì về việc làm chủ cảm xúc qua tình huống này?'
        });
      }
    }
    if (parsed.situations.length > targetSituationCount) {
      parsed.situations = parsed.situations.slice(0, targetSituationCount);
    }
    parsed.situationCount = targetSituationCount;

    return res.json({ success: true, scenario: parsed });
  } catch (err: any) {
    console.error('Error in /api/gemini/generate-scenario:', err);
    return res.status(500).json({ error: err.message || 'Lỗi khi tạo tình huống từ Gemini' });
  }
});

// Endpoint: Regenerate a SINGLE scene independently without altering other scenes
app.post('/api/gemini/regenerate-scene', async (req, res) => {
  const { sceneNumber, overallContext, style, characters, teacherNote } = req.body;
  const { client: activeAI } = getAIInstance(req);

  if (!activeAI) {
    return res.json({
      fallback: true,
      message: 'Chế độ offline',
    });
  }

  try {
    const prompt = `
Bạn là biên kịch sư phạm kịch tương tác học đường.
Hãy viết lại RIÊNG CẢNH SỐ ${sceneNumber} cho vở kịch học đường với các thông tin:
- Bối cảnh chung: ${overallContext}
- Phong cách: ${style || 'everyday'}
- Nhân vật tham gia: ${JSON.stringify(characters || [])}
- Góp ý điều chỉnh của giáo viên: ${teacherNote || 'Làm sâu sắc hơn mâu thuẫn và tăng tính kịch tính nhưng văn minh'}

Xuất ra JSON duy nhất định dạng RoleplayScene:
{
  "sceneNumber": ${sceneNumber},
  "sceneTitle": "Tên cảnh điều chỉnh",
  "sceneContext": "Bối cảnh cảnh",
  "dialogues": [
    {"speaker": "Tên", "role": "Vai trò", "emotion": "calm", "text": "Lời thoại"}
  ],
  "climaxOrConflict": "Điểm xung đột mới",
  "studentTask": "Nhiệm vụ cho học sinh",
  "options": [
    {"id": "A", "text": "Lựa chọn A", "consequence": "Hệ quả A", "impactType": "positive", "scoreAdvice": "Lời khuyên", "growthFeedback": "Bài học"}
  ],
  "sceneTakeaway": "Gợi ý rút ra"
}
`;

    const response = await activeAI.models.generateContent({
      model: GEMINI_MODEL,
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        temperature: 0.8,
      },
    });

    const parsed = cleanAndParseJSON(response.text || '{}');
    return res.json({ success: true, scene: parsed });
  } catch (err: any) {
    console.error('Error in /api/gemini/regenerate-scene:', err);
    return res.status(500).json({ error: err.message || 'Lỗi khi tái tạo cảnh' });
  }
});

// 2. Analyze Student Free Response
app.post('/api/gemini/analyze-response', async (req, res) => {
  const { scenarioContext, conflict, studentAnswer, studentName } = req.body;
  const { client: activeAI } = getAIInstance(req);

  if (!activeAI) {
    return res.json({
      fallback: true,
      analysis: {
        score: 85,
        criteriaScores: {
          problemIdentification: 85,
          respectForOthers: 90,
          communicationSkills: 80,
          responsibility: 85,
          consequenceAwareness: 80,
          feasibility: 85,
        },
        strengths: 'Bạn đã nhìn nhận đúng bản chất mâu thuẫn và giữ được thái độ tích cực.',
        suggestions: 'Có thể lắng nghe sâu hơn góc nhìn của người khác trước khi đưa ra quyết định cuối cùng.',
        reflectivePrompt: 'Nếu người đối diện có phản ứng không như kỳ vọng, bạn sẽ điều chỉnh cách tiếp cận thế nào?',
        constructiveComment: 'Cách xử lý thể hiện tinh thần chủ động và tinh thần trách nhiệm của học sinh THPT.',
      },
    });
  }

  try {
    const prompt = `
Bạn là chuyên gia đánh giá năng lực giáo dục trải nghiệm THPT.
Hãy phân tích câu trả lời của học sinh "${studentName || 'Học sinh'}" đối với tình huống sau:
Bối cảnh: ${scenarioContext}
Vấn đề cần giải quyết: ${conflict}
Cách xử lý của học sinh: "${studentAnswer}"

QUY TẮC ĐẶC BIỆT:
- Phân tích mang tính xây dựng, tuyệt đối KHÔNG gắn nhãn nhân cách, đạo đức học sinh.
- Đánh giá theo 6 tiêu chí (điểm từ 0 đến 100):
  1. problemIdentification (Nhận diện vấn đề)
  2. respectForOthers (Sự tôn trọng người khác)
  3. communicationSkills (Khả năng giao tiếp)
  4. responsibility (Tính trách nhiệm)
  5. consequenceAwareness (Khả năng cân nhắc hệ quả)
  6. feasibility (Tính khả thi của giải pháp)

Yêu cầu xuất ra JSON:
{
  "score": Điểm tổng quan trung bình (số nguyên 0-100),
  "criteriaScores": {
    "problemIdentification": điểm số,
    "respectForOthers": điểm số,
    "communicationSkills": điểm số,
    "responsibility": điểm số,
    "consequenceAwareness": điểm số,
    "feasibility": điểm số
  },
  "strengths": "Điểm tích cực, điểm sáng trong cách giải quyết của bạn",
  "suggestions": "Gợi ý hoàn thiện giải pháp theo hướng trưởng thành và nhân văn hơn",
  "reflectivePrompt": "Một câu hỏi gợi mở để học sinh tự suy ngẫm sâu hơn",
  "constructiveComment": "Nhận xét tổng kết khích lệ tinh thần người học"
}
`;

    const response = await activeAI.models.generateContent({
      model: GEMINI_MODEL,
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        temperature: 0.5,
      },
    });

    const parsed = cleanAndParseJSON(response.text || '{}');
    return res.json({ success: true, analysis: parsed });
  } catch (err: any) {
    console.error('Error in /api/gemini/analyze-response:', err);
    return res.status(500).json({ error: err.message || 'Lỗi khi phân tích phản hồi' });
  }
});

// Helper: Normalize Lesson Plan structure ensuring all 5512 sections are present and typed correctly
function normalizeLessonPlan(data: any, fallbackInfo: any) {
  const root = data.lessonPlan || data.lesson_plan || data.keHoachBaiDay || data;
  
  // Normalize objectives
  const objectives = {
    knowledge: root.objectives?.knowledge || root.lessonObjectives?.knowledge || root.knowledge || `Học sinh nắm vững các khái niệm, biểu hiện và phương pháp cốt lõi về "${fallbackInfo.topic}".`,
    competence: root.objectives?.competence || root.lessonObjectives?.competence || root.competence || 'Năng lực tự chủ và tự học; năng lực giao tiếp và hợp tác; năng lực thích ứng với cuộc sống và giải quyết vấn đề với các biểu hiện quan sát được qua hành vi thảo luận, đóng vai.',
    qualities: root.objectives?.qualities || root.lessonObjectives?.qualities || root.qualities || 'Chăm chỉ, trung thực, tinh thần trách nhiệm và lòng nhân ái, tôn trọng kỷ luật lớp học.'
  };

  // Normalize equipment (Mục II)
  const equipment = {
    teacherPreparation: Array.isArray(root.equipment?.teacherPreparation) 
      ? root.equipment.teacherPreparation 
      : (typeof root.equipment?.teacherPreparation === 'string' ? [root.equipment.teacherPreparation] : [
          'Kế hoạch bài dạy (Giáo án) theo chuẩn Công văn 5512/BGDĐT',
          'Slide trình chiếu đa phương tiện (PowerPoint/Canva), video tình huống thực tế hoặc ảnh 360° VR Studio',
          'Phiếu học tập nhóm, Bảng tiêu chí đánh giá Rubric và kịch bản phân vai'
        ]),
    studentPreparation: Array.isArray(root.equipment?.studentPreparation)
      ? root.equipment.studentPreparation
      : (typeof root.equipment?.studentPreparation === 'string' ? [root.equipment.studentPreparation] : [
          `Sách giáo khoa Hoạt động trải nghiệm, hướng nghiệp Lớp ${fallbackInfo.grade} (Kết nối tri thức)`,
          'Vở ghi chép, bút viết, giấy A0/A3 hoặc giấy ghi chú sticky notes',
          'Điện thoại thông minh/máy tính bảng quét mã QR bài học (nếu điều kiện cho phép)'
        ]),
    digitalToolsAndWorksheets: Array.isArray(root.equipment?.digitalToolsAndWorksheets)
      ? root.equipment.digitalToolsAndWorksheets
      : (typeof root.equipment?.digitalToolsAndWorksheets === 'string' ? [root.equipment.digitalToolsAndWorksheets] : [
          'Phiếu học tập số 1: Nhận diện xung đột nhận thức và bài toán lựa chọn',
          'Công cụ số hỗ trợ: Sân khấu Động AI Pro, VR Studio 360°, Trò chơi tương tác',
          'Bảng kiểm tự đánh giá và phiếu đánh giá đồng đẳng (Peer-assessment)'
        ])
  };

  // Normalize activities (Mục III)
  const rawActs = Array.isArray(root.activities) ? root.activities : 
                 Array.isArray(root.phases) ? root.phases :
                 Array.isArray(root.cac_hoat_dong) ? root.cac_hoat_dong : [];

  const activities = rawActs.map((act: any, idx: number) => ({
    phaseNumber: act.phaseNumber || (idx + 1),
    phaseName: act.phaseName || `Hoạt động ${idx + 1}`,
    name: act.name || act.title || `Hoạt động ${idx + 1}: Trải nghiệm thực tế`,
    objective: act.objective || act.muc_tieu || 'Phát triển năng lực học sinh qua hoạt động tương tác.',
    duration: typeof act.duration === 'number' ? act.duration : (parseInt(act.duration || '10', 10) || 10),
    teacherTask: Array.isArray(act.teacherTask) 
      ? act.teacherTask 
      : (typeof act.teacherTask === 'string' ? [act.teacherTask] : ['Giáo viên phổ biến nhiệm vụ, hướng dẫn và quan sát học sinh.']),
    studentTask: Array.isArray(act.studentTask) 
      ? act.studentTask 
      : (typeof act.studentTask === 'string' ? [act.studentTask] : ['Học sinh thảo luận nhóm, thực hành và trình bày sản phẩm.']),
    materials: Array.isArray(act.materials) 
      ? act.materials 
      : (typeof act.materials === 'string' ? [act.materials] : ['SGK, phiếu học tập, bảng nhóm']),
    expectedProduct: act.expectedProduct || act.san_pham || 'Sản phẩm phiếu học tập hoặc phương án xử lý tình huống nhóm.',
    organizationSteps: {
      step1Assign: act.organizationSteps?.step1Assign || 'Bước 1: Chuyển giao nhiệm vụ - GV giải thích mục tiêu, phát phiếu học tập và phân công nhiệm vụ cho các nhóm.',
      step2Execute: act.organizationSteps?.step2Execute || 'Bước 2: Thực hiện nhiệm vụ - HS tiếp nhận nhiệm vụ, thảo luận đội nhóm, phân vai và xây dựng phương án giải quyết.',
      step3Discuss: act.organizationSteps?.step3Discuss || 'Bước 3: Báo cáo, thảo luận - Đại diện nhóm báo cáo trước lớp; các nhóm khác chú ý lắng nghe, phản biện và bổ sung.',
      step4Conclude: act.organizationSteps?.step4Conclude || 'Bước 4: Kết luận, nhận định - GV nhận xét tinh thần làm việc, chuẩn hóa kiến thức cốt lõi và rút ra thông điệp giáo dục.'
    },
    guidingQuestions: Array.isArray(act.guidingQuestions) 
      ? act.guidingQuestions 
      : [
          'Theo các em, đâu là mấu chốt xung đột cần tháo gỡ trong tình huống này?',
          'Nếu là người trong cuộc, em sẽ ưu tiên phương án nào để vừa giải quyết việc vừa giữ được hòa khí?'
        ],
    assessmentCriteria: Array.isArray(act.assessmentCriteria) 
      ? act.assessmentCriteria 
      : [
          'Mức độ hợp tác và trách nhiệm của từng thành viên trong nhóm',
          'Tính nhân văn, thuyết phục và khả thi của giải pháp đưa ra'
        ]
  }));

  return {
    title: root.title || (fallbackInfo.periodNumber ? `Kế hoạch bài dạy (Tiết ${fallbackInfo.periodNumber}/${fallbackInfo.totalPeriodsInTopic || 1}): ${fallbackInfo.periodTitle || fallbackInfo.activity}` : `Kế hoạch bài dạy: ${fallbackInfo.activity}`),
    grade: fallbackInfo.grade,
    topic: fallbackInfo.topic,
    duration: fallbackInfo.duration,
    durationMinutes: fallbackInfo.durationMinutes,
    periodNumber: fallbackInfo.periodNumber,
    totalPeriodsInTopic: fallbackInfo.totalPeriodsInTopic,
    periodTitle: fallbackInfo.periodTitle,
    frameworkType: fallbackInfo.frameworkType || '5512_4phases',
    supportedTools: fallbackInfo.supportedTools || [],
    objectives,
    equipment,
    classConditions: root.classConditions || fallbackInfo.classConditions || 'Lớp học tiêu chuẩn có máy chiếu',
    organizationFormat: root.organizationFormat || fallbackInfo.organizationFormat || 'Hoạt động nhóm, thảo luận và sân khấu hóa tình huống',
    activities,
    pedagogicalAdvice: root.pedagogicalAdvice || 'Tạo môi trường cởi mở, tôn trọng sự khác biệt và khích lệ học sinh tự tin bày tỏ chính kiến cá nhân.',
    anonymousRubricNote: root.anonymousRubricNote || 'Sử dụng mã số định danh học sinh ẩn danh (VD: HS-10A1-09) khi tổ chức đánh giá đồng đẳng để bảo đảm khách quan và bảo mật.'
  };
}

// 3. Generate Experiential Lesson Plan (Kế hoạch bài dạy GDPT 2018 theo CV 5512)
app.post('/api/gemini/generate-lesson-plan', async (req, res) => {
  const { 
    grade, 
    topic, 
    activity, 
    duration, 
    durationMinutes,
    frameworkType, 
    supportedTools, 
    objectives, 
    organizationFormat, 
    classConditions,
    periodNumber,
    totalPeriods,
    periodTitle,
    periodContent
  } = req.body;

  const targetMinutes = durationMinutes || (duration === '90 phút' ? 90 : 45);
  const selectedFramework = frameworkType || '5512_4phases';
  const { client: activeAI } = getAIInstance(req);

  if (!activeAI) {
    return res.json({
      fallback: true,
      message: 'Sử dụng kế hoạch bài dạy chuẩn mẫu bộ Kết nối tri thức',
    });
  }

  try {
    const is4Phases = selectedFramework === '5512_4phases';
    const activitiesDescription = is4Phases
      ? `Đúng 4 hoạt động theo chuẩn khung Công văn 5512/BGDĐT:
1. Hoạt động 1: Xác định vấn đề / Nhiệm vụ học tập / Khởi động (Warm-up)
2. Hoạt động 2: Hình thành kiến thức mới / Khám phá và giải quyết tình huống (Exploration)
3. Hoạt động 3: Luyện tập, thực hành kỹ năng (Practice)
4. Hoạt động 4: Vận dụng, mở rộng vào thực tiễn đời sống (Application)`
      : `Đúng 6 hoạt động theo chu trình học tập trải nghiệm Kolb:
1. Hoạt động 1: Khởi động (Tạo tâm thế và kết nối cảm xúc)
2. Hoạt động 2: Khám phá (Trải nghiệm thực tế và nhận diện vấn đề)
3. Hoạt động 3: Thực hành (Đóng vai, xử lý tình huống, tranh biện)
4. Hoạt động 4: Chia sẻ, phản tư (Tự vấn và soi chiếu cảm xúc)
5. Hoạt động 5: Vận dụng (Chuyển hóa nhận thức thành kế hoạch hành động)
6. Hoạt động 6: Đánh giá cuối tiết (Phiếu tự đánh giá và bài học rút ra)`;

    const prompt = `
Bạn là giáo viên chuyên gia đầu ngành Hoạt động trải nghiệm, hướng nghiệp THPT (Bộ sách Kết nối tri thức với cuộc sống).
Hãy thiết kế Kế hoạch bài dạy (Giáo án chi tiết) theo chuẩn Công văn 5512/BGDĐT và chương trình GDPT 2018.

THÔNG SỐ BÀI DẠY:
- Khối lớp: Lớp ${grade}
- Chủ đề SGK: ${topic}
- Tên hoạt động/bài dạy: ${activity}
${periodNumber ? `- Cấu hình theo tiết: Tiết ${periodNumber}/${totalPeriods || 1} - ${periodTitle || activity}
- Trọng tâm của tiết này: ${periodContent || activity}` : ''}
- Tổng thời lượng: ${duration || `${targetMinutes} phút`} (Đúng ${targetMinutes} phút)
- Khung bài dạy: ${is4Phases ? 'Chuẩn 4 hoạt động Công văn 5512' : 'Chuẩn chu trình trải nghiệm 6 pha'}
- Công cụ bổ trợ: ${(supportedTools && supportedTools.length > 0) ? supportedTools.join(', ') : 'Sân khấu Động AI, Trò chơi tương tác, VR Studio 360°'}
- Mục tiêu cần đạt của giáo viên: ${objectives || 'Phát triển năng lực tự chủ, giao tiếp hợp tác và thích ứng nghề nghiệp'}
- Hình thức tổ chức: ${organizationFormat || 'Hoạt động nhóm, sân khấu tương tác và thảo luận bàn tròn'}
- Điều kiện lớp học: ${classConditions || 'Lớp học tiêu chuẩn có máy chiếu'}

QUY TẮC ĐẶC BIỆT BẮT BUỘC:
1. Tổng thời lượng các hoạt động trong phần III CỘNG LẠI PHẢI BẰNG ĐÚNG ${targetMinutes} PHÚT.
2. Mục I (Mục tiêu): Phải viết cụ thể, đo lường được theo chuẩn GDPT 2018 (Kiến thức, Năng lực chung + Năng lực đặc thù HĐTN-HN, Phẩm chất gắn với hành vi cụ thể).
3. Mục II (Thiết bị và học liệu): Phải có đủ 3 phần: Giáo viên chuẩn bị, Học sinh chuẩn bị, Học liệu số & Phiếu học tập.
4. Mục III (Tiến trình dạy học): Có ${activitiesDescription}.
Mỗi hoạt động BẮT BUỘC có đủ:
- phaseNumber (số 1..N)
- phaseName (tên pha chuẩn 5512)
- name (tên hoạt động cụ thể hấp dẫn)
- duration (số phút)
- objective (mục tiêu hoạt động)
- teacherTask (mảng các bước cụ thể của giáo viên)
- studentTask (mảng các bước cụ thể của học sinh)
- materials (học liệu, đạo cụ sử dụng)
- expectedProduct (sản phẩm học sinh làm ra)
- organizationSteps (đủ 4 bước: step1Assign, step2Execute, step3Discuss, step4Conclude)
- guidingQuestions (mảng 2-3 câu hỏi gợi mở, lời dẫn sư phạm)
- assessmentCriteria (mảng 2-3 tiêu chí đánh giá rõ ràng)

XUẤT RA DUY NHẤT ĐỊNH DẠNG JSON THEO SCHEMA NÀY:
{
  "title": "Kế hoạch bài dạy${periodNumber ? ` (Tiết ${periodNumber}/${totalPeriods || 1})` : ''}: ${periodTitle || activity}",
  "grade": ${grade},
  "topic": "${topic}",
  "duration": "${duration || `${targetMinutes} phút`}",
  "durationMinutes": ${targetMinutes},
  "periodNumber": ${periodNumber || 'null'},
  "totalPeriodsInTopic": ${totalPeriods || 'null'},
  "periodTitle": "${periodTitle || ''}",
  "frameworkType": "${selectedFramework}",
  "objectives": {
    "knowledge": "Kiến thức cụ thể...",
    "competence": "Năng lực chung và năng lực đặc thù môn HĐTN-HN...",
    "qualities": "Phẩm chất gắn với hành vi cụ thể..."
  },
  "equipment": {
    "teacherPreparation": ["Giáo án chuẩn 5512...", "Slide bài giảng điện tử...", "Phiếu học tập số 1..."],
    "studentPreparation": ["Sách giáo khoa Lớp ${grade}...", "Vở ghi, bút...", "Giấy A0/A3..."],
    "digitalToolsAndWorksheets": ["Phiếu học tập 1...", "Công cụ số: Sân khấu AI...", "Bảng tiêu chí rubric..."]
  },
  "classConditions": "${classConditions || 'Lớp học tiêu chuẩn có máy chiếu'}",
  "organizationFormat": "${organizationFormat || 'Hoạt động nhóm, sân khấu tương tác'}",
  "activities": [
    {
      "phaseNumber": 1,
      "phaseName": "Hoạt động 1: ...",
      "name": "Tên hoạt động 1",
      "duration": 7,
      "objective": "Mục tiêu...",
      "teacherTask": ["GV chia nhóm...", "GV giao phiếu..."],
      "studentTask": ["HS thảo luận...", "HS phân vai..."],
      "materials": ["Máy chiếu, phiếu học tập..."],
      "expectedProduct": "Sản phẩm cụ thể...",
      "organizationSteps": {
        "step1Assign": "Bước 1: Chuyển giao nhiệm vụ - GV...",
        "step2Execute": "Bước 2: Thực hiện nhiệm vụ - HS...",
        "step3Discuss": "Bước 3: Báo cáo, thảo luận - Đại diện...",
        "step4Conclude": "Bước 4: Kết luận, nhận định - GV chuẩn hóa..."
      },
      "guidingQuestions": ["Câu hỏi gợi mở 1...", "Câu hỏi gợi mở 2..."],
      "assessmentCriteria": ["Tiêu chí 1...", "Tiêu chí 2..."]
    }
  ],
  "pedagogicalAdvice": "Lời khuyên sư phạm cho giáo viên...",
  "anonymousRubricNote": "Hướng dẫn bảo mật mã ẩn danh học sinh..."
}
`;

    const response = await activeAI.models.generateContent({
      model: GEMINI_MODEL,
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        temperature: 0.65,
      },
    });

    const parsed = cleanAndParseJSON(response.text || '{}');
    const normalized = normalizeLessonPlan(parsed, {
      grade,
      topic,
      activity: periodTitle || activity,
      duration: duration || `${targetMinutes} phút`,
      durationMinutes: targetMinutes,
      periodNumber,
      totalPeriodsInTopic: totalPeriods,
      periodTitle,
      frameworkType: selectedFramework,
      supportedTools,
      organizationFormat,
      classConditions
    });

    return res.json({ success: true, lessonPlan: normalized });
  } catch (err: any) {
    console.error('Error in /api/gemini/generate-lesson-plan:', err);
    return res.status(500).json({ error: err.message || 'Lỗi khi tạo giáo án' });
  }
});

// Endpoint: AI suggests period distribution for a topic across N periods
app.post('/api/gemini/suggest-period-distribution', async (req, res) => {
  const { 
    grade, 
    topic, 
    totalPeriods, 
    minutesPerPeriod, 
    learningObjectives, 
    focusContent, 
    requiredActivities, 
    departmentNotes 
  } = req.body;

  const periodsCount = Math.max(1, Math.min(20, parseInt(totalPeriods, 10) || 3));
  const minPerPeriod = parseInt(minutesPerPeriod, 10) || 45;
  const { client: activeAI } = getAIInstance(req);

  if (!activeAI) {
    return res.json({
      fallback: true,
      message: 'Chế độ offline',
    });
  }

  try {
    const prompt = `
Bạn là chuyên gia thẩm định chương trình giáo dục Hoạt động trải nghiệm, hướng nghiệp THPT (Chương trình GDPT 2018).
Hãy xây dựng BẢNG PHÂN PHỐI NỘI DUNG DẠY HỌC THEO SỐ TIẾT cho chủ đề sau:
- Khối lớp: Lớp ${grade || 10}
- Tên chủ đề: ${topic}
- Tổng số tiết: ${periodsCount} tiết
- Thời lượng mỗi tiết: ${minPerPeriod} phút (Tổng thời lượng: ${periodsCount * minPerPeriod} phút)
- Mục tiêu cần đạt: ${learningObjectives || 'Nắm vững kiến thức, phát triển năng lực tự chủ, giao tiếp và thích ứng'}
- Nội dung trọng tâm: ${focusContent || 'Nội dung cốt lõi của chủ đề'}
- Hoạt động bắt buộc: ${Array.isArray(requiredActivities) ? requiredActivities.join('; ') : (requiredActivities || 'Trải nghiệm thực tế, sân khấu tình huống, thảo luận nhóm')}
- Ghi chú tổ chuyên môn / nhà trường: ${departmentNotes || 'Bám sát kế hoạch giáo dục nhà trường'}

YÊU CẦU BẮT BUỘC:
1. Phân bổ hợp lý thành đúng ${periodsCount} tiết (Tiết 1 đến Tiết ${periodsCount}).
2. Mỗi tiết PHẢI có:
   - periodNumber: số thứ tự tiết (1..${periodsCount})
   - periodTitle: tên trọng tâm ngắn gọn của tiết đó
   - durationMinutes: đúng ${minPerPeriod} phút
   - objectives: mục tiêu riêng cần đạt của tiết đó
   - focusContent: nội dung trọng tâm của tiết
   - learningProduct: sản phẩm học tập cụ thể học sinh hoàn thành trong tiết
   - activities: danh sách 3-4 hoạt động trong tiết, TỔNG THỜI LƯỢNG CÁC HOẠT ĐỘNG PHẢI BẰNG ĐÚNG ${minPerPeriod} PHÚT.
     Mỗi hoạt động có: id, name, duration (số phút), type ("KhoiDong" | "KhamPha" | "LuyenTap" | "VanDung" | "DanhGia"), description.

XUẤT RA DUY NHẤT ĐỊNH DẠNG JSON:
{
  "totalPeriods": ${periodsCount},
  "minutesPerPeriod": ${minPerPeriod},
  "periods": [
    {
      "periodNumber": 1,
      "periodTitle": "Tên trọng tâm tiết 1",
      "durationMinutes": ${minPerPeriod},
      "objectives": "Mục tiêu cụ thể của tiết 1",
      "focusContent": "Nội dung trọng tâm tiết 1",
      "learningProduct": "Sản phẩm của tiết 1",
      "activities": [
        {"id": "act-1-1", "name": "Khởi động: ...", "duration": 8, "type": "KhoiDong", "description": "Tạo tâm thế..."},
        {"id": "act-1-2", "name": "Khám phá: ...", "duration": 15, "type": "KhamPha", "description": "Nhận diện vấn đề..."},
        {"id": "act-1-3", "name": "Luyện tập: ...", "duration": 14, "type": "LuyenTap", "description": "Thực hành..."},
        {"id": "act-1-4", "name": "Vận dụng: ...", "duration": 8, "type": "VanDung", "description": "Giao nhiệm vụ..."}
      ]
    }
  ]
}
`;

    const response = await activeAI.models.generateContent({
      model: GEMINI_MODEL,
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        temperature: 0.6,
      },
    });

    const parsed = cleanAndParseJSON(response.text || '{}');
    return res.json({ success: true, distribution: parsed });
  } catch (err: any) {
    console.error('Error in /api/gemini/suggest-period-distribution:', err);
    return res.status(500).json({ error: err.message || 'Lỗi khi đề xuất phân phối chương trình' });
  }
});

// Endpoint: Regenerate a SINGLE activity in lesson plan
app.post('/api/gemini/regenerate-lesson-activity', async (req, res) => {
  const { 
    grade, 
    topic, 
    activityName, 
    phaseNumber, 
    phaseName, 
    duration, 
    frameworkType, 
    teacherPrompt 
  } = req.body;
  const { client: activeAI } = getAIInstance(req);

  if (!activeAI) {
    return res.json({
      fallback: true,
      message: 'Chế độ offline',
    });
  }

  try {
    const prompt = `
Bạn là chuyên gia giáo dục THPT môn Hoạt động trải nghiệm, hướng nghiệp.
Hãy viết lại RIÊNG một hoạt động dạy học theo chuẩn Công văn 5512/BGDĐT:
- Khối lớp: Lớp ${grade}
- Chủ đề: ${topic}
- Pha/Hoạt động số: ${phaseNumber} (${phaseName})
- Tên hoạt động: ${activityName}
- Thời lượng: ${duration || 10} phút
- Góp ý điều chỉnh của giáo viên: ${teacherPrompt || 'Làm sâu sắc hơn nhiệm vụ học sinh và tăng tính thực hành'}

Yêu cầu xuất ra JSON duy nhất định dạng:
{
  "phaseNumber": ${phaseNumber},
  "phaseName": "${phaseName}",
  "name": "Tên hoạt động điều chỉnh",
  "duration": ${duration || 10},
  "objective": "Mục tiêu cụ thể",
  "teacherTask": ["Nhiệm vụ GV 1", "Nhiệm vụ GV 2"],
  "studentTask": ["Nhiệm vụ HS 1", "Nhiệm vụ HS 2"],
  "materials": ["Học liệu, đạo cụ"],
  "expectedProduct": "Sản phẩm mong đợi",
  "organizationSteps": {
    "step1Assign": "Bước 1: Chuyển giao nhiệm vụ - ...",
    "step2Execute": "Bước 2: Thực hiện nhiệm vụ - ...",
    "step3Discuss": "Bước 3: Báo cáo, thảo luận - ...",
    "step4Conclude": "Bước 4: Kết luận, nhận định - ..."
  },
  "guidingQuestions": ["Câu hỏi gợi ý 1", "Câu hỏi gợi ý 2"],
  "assessmentCriteria": ["Tiêu chí đánh giá 1", "Tiêu chí đánh giá 2"]
}
`;

    const response = await activeAI.models.generateContent({
      model: GEMINI_MODEL,
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        temperature: 0.7,
      },
    });

    const parsed = cleanAndParseJSON(response.text || '{}');
    return res.json({ success: true, activity: parsed });
  } catch (err: any) {
    console.error('Error in /api/gemini/regenerate-lesson-activity:', err);
    return res.status(500).json({ error: err.message || 'Lỗi khi tái tạo hoạt động' });
  }
});

// 4. Teacher AI Assistant (10 pedagogical modes)
app.post('/api/gemini/teacher-assistant', async (req, res) => {
  const { mode, grade, topic, userQuery, studentCount } = req.body;
  const { client: activeAI } = getAIInstance(req);

  if (!activeAI) {
    return res.json({
      fallback: true,
      message: 'Chế độ Trợ lý AI ngoại tuyến: Đã tổng hợp gợi ý sư phạm dựa trên cẩm nang GV HĐTN-HN THPT.',
    });
  }

  try {
    const prompt = `
Bạn là trợ lý AI chuyên biệt về Hoạt động trải nghiệm, hướng nghiệp THPT (Bộ sách Kết nối tri thức với cuộc sống).
Chế độ hỗ trợ giáo viên: ${mode}
(1. Thiết kế hoạt động học tập / 2. Tạo tình huống giáo dục / 3. Tạo kịch bản đóng vai / 4. Tạo câu hỏi thảo luận / 5. Thiết kế trò chơi / 6. Xây dựng phiếu đánh giá / 7. Gợi ý phương pháp tổ chức / 8. Điều chỉnh theo số lượng học sinh / 9. Tạo bài tập vận dụng / 10. Gợi ý cách xử lý tình huống sư phạm)

Khối lớp: Lớp ${grade || 11}
Chủ đề: ${topic || 'Chung'}
Sĩ số học sinh: ${studentCount || 40} học sinh
Yêu cầu cụ thể của giáo viên: "${userQuery}"

Hãy trả lời chuyên sâu, thiết thực, có cấu trúc sư phạm rõ ràng bằng tiếng Việt.
Xuất ra JSON:
{
  "title": "Tiêu đề giải pháp / gợi ý",
  "summary": "Tóm tắt ngắn gọn",
  "steps": ["Bước 1 / Điểm 1", "Bước 2 / Điểm 2", "..."],
  "sampleContent": "Nội dung chi tiết hoặc kịch bản / câu hỏi / rubric mẫu",
  "pedagogicalNote": "Lưu ý sư phạm quan trọng (tâm lý lứa tuổi 15-18, tôn trọng học sinh)"
}
`;

    const response = await activeAI.models.generateContent({
      model: GEMINI_MODEL,
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        temperature: 0.6,
      },
    });

    const parsed = cleanAndParseJSON(response.text || '{}');
    return res.json({ success: true, response: parsed });
  } catch (err: any) {
    console.error('Error in /api/gemini/teacher-assistant:', err);
    return res.status(500).json({ error: err.message || 'Lỗi trợ lý giáo viên' });
  }
});

// 5. VR Studio Task Generator (AI TẠO NHIỆM VỤ VR)
app.post('/api/gemini/vr-generate-task', async (req, res) => {
  const { grade, topic, requirements, roomContext, difficulty, duration, taskType } = req.body;
  const { client: activeAI } = getAIInstance(req);

  if (!activeAI) {
    return res.json({
      fallback: true,
      message: 'Chế độ ngoại tuyến: Nhiệm vụ VR được chuẩn bị từ bộ tiêu chuẩn GDPT 2018.',
    });
  }

  try {
    const prompt = `
Bạn là chuyên gia sư phạm Hoạt động trải nghiệm, hướng nghiệp THPT và thiết kế trò chơi giáo dục thực tế ảo VR 360°.
Hãy thiết kế một nhiệm vụ học tập tương tác đặt trong bối cảnh không gian thực tế ảo 360° với các thông số:
- Khối lớp: Lớp ${grade || 10}
- Chủ đề: ${topic || 'Khám phá bản thân và môi trường học đường'}
- Yêu cầu cần đạt: ${requirements || 'Rèn luyện năng lực tự chủ, giao tiếp, hợp tác và giải quyết vấn đề'}
- Bối cảnh phòng VR 360°: ${roomContext || 'Lớp học THPT / Sân trường / Phòng tư vấn tâm lý'}
- Mức độ khó: ${difficulty || 'Vừa sức'}
- Thời lượng hoạt động: ${duration || '10-15 phút'}
- Loại nhiệm vụ mong muốn: ${taskType || 'Tình huống phân nhánh / Trắc nghiệm / Đóng vai'}

Yêu cầu xuất ra JSON hợp lệ:
{
  "taskTitle": "Tên nhiệm vụ ngắn gọn, kích thích tò mò",
  "taskType": "quiz" | "true_false" | "short_answer" | "dilemma" | "roleplay" | "exploration",
  "storyContext": "Bối cảnh câu chuyện ngắn gọn khi học sinh nhìn quanh phòng VR (2-3 câu)",
  "situation": "Tình huống hoặc thách thức học sinh cần giải quyết tại điểm tương tác này",
  "dilemmaOptions": [
    {
      "id": "A",
      "text": "Lựa chọn 1",
      "consequence": "Hệ quả tương ứng",
      "isOptimal": boolean,
      "feedback": "Lời khuyên phản tư"
    },
    {
      "id": "B",
      "text": "Lựa chọn 2",
      "consequence": "Hệ quả tương ứng",
      "isOptimal": boolean,
      "feedback": "Lời khuyên phản tư"
    }
  ],
  "quizQuestion": "Câu hỏi nếu là trắc nghiệm",
  "quizOptions": ["Lựa chọn 1", "Lựa chọn 2", "Lựa chọn 3", "Lựa chọn 4"],
  "correctAnswerIndex": 0,
  "explanation": "Lời giải thích khoa học và nhân văn",
  "roleplayCharacter": {
    "name": "Tên nhân vật",
    "role": "Thầy cô / Bạn thân / Phụ huynh / Nhà tuyển dụng",
    "dialogue": "Lời thoại nhân vật thách thức học sinh"
  },
  "pedagogicalSuggestions": "Gợi ý cho giáo viên cách điều phối hoạt động trên lớp",
  "applicationQuestion": "Câu hỏi vận dụng vào đời sống sau khi rời phòng VR",
  "completionCriteria": "Tiêu chí đánh giá hoàn thành nhiệm vụ"
}
`;

    const response = await activeAI.models.generateContent({
      model: GEMINI_MODEL,
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        temperature: 0.7,
      },
    });

    const parsed = cleanAndParseJSON(response.text || '{}');
    return res.json({ success: true, task: parsed });
  } catch (err: any) {
    console.error('Error in /api/gemini/vr-generate-task:', err);
    return res.status(500).json({ error: err.message || 'Lỗi khi tạo nhiệm vụ VR bằng AI' });
  }
});

// Helper: Get local network IPv4 addresses
function getLocalIpAddresses(): string[] {
  const interfaces = os.networkInterfaces();
  const addresses: string[] = [];
  for (const name of Object.keys(interfaces)) {
    for (const iface of interfaces[name] || []) {
      if (iface.family === 'IPv4' && !iface.internal) {
        addresses.push(iface.address);
      }
    }
  }
  return addresses;
}

// Storage directories for published lessons and student attempts
// LƯU Ý: Trên Vercel chỉ /tmp ghi được và dữ liệu KHÔNG bền vững (mất khi hàm khởi động lại).
// Cần cơ sở dữ liệu bên ngoài (Vercel Blob/KV, Supabase...) để lưu bài xuất bản & kết quả lâu dài.
const DATA_DIR = IS_VERCEL ? path.join(os.tmpdir(), 'eduverse-data') : path.join(__dirname, 'data');
const PUBLISHED_VR_DIR = path.join(DATA_DIR, 'published_vr');
const ATTEMPTS_VR_DIR = path.join(DATA_DIR, 'vr_attempts');

if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
if (!fs.existsSync(PUBLISHED_VR_DIR)) fs.mkdirSync(PUBLISHED_VR_DIR, { recursive: true });
if (!fs.existsSync(ATTEMPTS_VR_DIR)) fs.mkdirSync(ATTEMPTS_VR_DIR, { recursive: true });

// 10. Network info endpoint (Detect LAN IP for Classroom Wi-Fi QR access)
app.get('/api/network-info', (_req, res) => {
  const ips = getLocalIpAddresses();
  const primaryIp = ips[0] || '127.0.0.1';
  res.json({
    ips,
    primaryIp,
    port,
    recommendedUrl: `http://${primaryIp}:${port}`,
    hostname: os.hostname(),
    platform: process.platform,
  });
});

// 11. Publish VR Lesson
app.post('/api/vr/publish', (req, res) => {
  try {
    const { project } = req.body;
    if (!project || !project.id) {
      return res.status(400).json({ error: 'Dữ liệu bài học không hợp lệ' });
    }
    const filePath = path.join(PUBLISHED_VR_DIR, `${project.id}.json`);
    fs.writeFileSync(filePath, JSON.stringify(project, null, 2), 'utf-8');
    
    // Save index mapping shareCode to id
    const shareCode = project.publication?.shareCode || project.id;
    const indexFile = path.join(DATA_DIR, 'vr_code_index.json');
    let codeIndex: Record<string, string> = {};
    if (fs.existsSync(indexFile)) {
      try {
        codeIndex = JSON.parse(fs.readFileSync(indexFile, 'utf-8'));
      } catch {}
    }
    codeIndex[shareCode.toUpperCase()] = project.id;
    fs.writeFileSync(indexFile, JSON.stringify(codeIndex, null, 2), 'utf-8');

    return res.json({ success: true, id: project.id, shareCode });
  } catch (err: any) {
    console.error('Error publishing VR project:', err);
    return res.status(500).json({ error: err.message || 'Lỗi khi lưu bài học lên máy chủ' });
  }
});

// 12. Get Published VR Lesson by ID or Share Code
app.get('/api/vr/published/:idOrCode', (req, res) => {
  try {
    const { idOrCode } = req.params;
    let targetId = idOrCode;

    // Check code index first
    const indexFile = path.join(DATA_DIR, 'vr_code_index.json');
    if (fs.existsSync(indexFile)) {
      try {
        const codeIndex = JSON.parse(fs.readFileSync(indexFile, 'utf-8'));
        if (codeIndex[idOrCode.toUpperCase()]) {
          targetId = codeIndex[idOrCode.toUpperCase()];
        }
      } catch {}
    }

    const filePath = path.join(PUBLISHED_VR_DIR, `${targetId}.json`);
    if (fs.existsSync(filePath)) {
      const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
      return res.json({ success: true, project: data });
    }

    // Try scanning files if not matched
    const files = fs.readdirSync(PUBLISHED_VR_DIR);
    for (const f of files) {
      if (f.endsWith('.json')) {
        try {
          const item = JSON.parse(fs.readFileSync(path.join(PUBLISHED_VR_DIR, f), 'utf-8'));
          if (item.id === idOrCode || item.publication?.shareCode?.toUpperCase() === idOrCode.toUpperCase()) {
            return res.json({ success: true, project: item });
          }
        } catch {}
      }
    }

    return res.status(404).json({ error: 'Không tìm thấy bài học với mã này' });
  } catch (err: any) {
    return res.status(500).json({ error: err.message });
  }
});

// 13. List all published projects
app.get('/api/vr/published', (_req, res) => {
  try {
    const files = fs.readdirSync(PUBLISHED_VR_DIR);
    const list: any[] = [];
    for (const f of files) {
      if (f.endsWith('.json')) {
        try {
          const item = JSON.parse(fs.readFileSync(path.join(PUBLISHED_VR_DIR, f), 'utf-8'));
          list.push(item);
        } catch {}
      }
    }
    return res.json({ success: true, projects: list });
  } catch (err: any) {
    return res.status(500).json({ error: err.message });
  }
});

// 14. Student submit attempt
app.post('/api/vr/submit-attempt', (req, res) => {
  try {
    const { attempt } = req.body;
    if (!attempt || !attempt.id || !attempt.projectId) {
      return res.status(400).json({ error: 'Dữ liệu kết quả không hợp lệ' });
    }
    const filePath = path.join(ATTEMPTS_VR_DIR, `${attempt.id}.json`);
    fs.writeFileSync(filePath, JSON.stringify(attempt, null, 2), 'utf-8');
    return res.json({ success: true, attemptId: attempt.id });
  } catch (err: any) {
    return res.status(500).json({ error: err.message });
  }
});

// 15. Get attempts for a project (for teacher review)
app.get('/api/vr/attempts/:projectId', (req, res) => {
  try {
    const { projectId } = req.params;
    const files = fs.readdirSync(ATTEMPTS_VR_DIR);
    const results: any[] = [];
    for (const f of files) {
      if (f.endsWith('.json')) {
        try {
          const item = JSON.parse(fs.readFileSync(path.join(ATTEMPTS_VR_DIR, f), 'utf-8'));
          if (item.projectId === projectId) {
            results.push(item);
          }
        } catch {}
      }
    }
    return res.json({ success: true, attempts: results });
  } catch (err: any) {
    return res.status(500).json({ error: err.message });
  }
});

// 16. AI Vision: Analyze Reference Photos for 360 Panorama Reconstruction
app.post('/api/vr/analyze-360-image', async (req, res) => {
  try {
    const { images, userNotes } = req.body;
    if (!images || !Array.isArray(images) || images.length === 0) {
      return res.status(400).json({ error: 'Chưa có ảnh nào được tải lên.' });
    }

    const mainImg = images[0];
    const isMulti = images.length > 1;
    let visionError: { errorCode: string; message: string } | null = null;
    const { client: activeAI } = getAIInstance(req);

    // If Gemini AI Vision is active with key
    if (activeAI) {
      try {
        const parts: any[] = [];
        for (const img of images.slice(0, 3)) { // analyze up to 3 images
          if (img.dataUrl && img.dataUrl.includes(',')) {
            const matches = img.dataUrl.match(/^data:([A-Za-z-+\/]+);base64,(.+)$/);
            if (matches && matches.length === 3) {
              parts.push({
                inlineData: {
                  mimeType: matches[1],
                  data: matches[2]
                }
              });
            }
          }
        }

        const visionPrompt = `
Bạn là chuyên gia thị giác máy tính AI (Computer Vision) và chuyên gia tái dựng hình ảnh toàn cảnh thực tế ảo Panorama 360° Equirectangular (Tỷ lệ chuẩn 2:1).
Nhiệm vụ của bạn là phân tích sâu sắc bức ảnh do giáo viên cung cấp${userNotes ? ` kết hợp với ghi chú mục tiêu của giáo viên: "${userNotes}"` : ''}.

YÊU CẦU QUAN TRỌNG:
1. KHÔNG GIỚI HẠN VÀO BẤT KỲ CHỦ ĐỀ CỐ ĐỊNH NÀO. Bức ảnh có thể thuộc bất kỳ lĩnh vực nào:
   - Thiên nhiên & Địa lý (biển đảo, rừng núi, hang động, sông hồ, núi lửa, sa mạc...)
   - Lịch sử & Di sản văn hóa (di tích, đình chùa, bảo tàng, chiến trường xưa, làng nghề truyền thống...)
   - Nghề nghiệp & Công nghiệp (nhà máy, phòng phẫu thuật y tế, xưởng cơ khí, phòng thiết kế, công trường, cảng biển...)
   - Nông nghiệp hiện đại (nhà kính, trang trại, vườn ươm, cánh đồng công nghệ cao...)
   - Khoa học, Công nghệ & Vũ trụ (phòng lab, đài thiên văn, trạm không gian, mô hình sinh học...)
   - Nghệ thuật, Văn hóa & Đời sống (nhà hát, phòng tranh, triển lãm, lễ hội, khu phố cổ, chợ truyền thống...)
   - Giáo dục & Học đường (khuôn viên, thư viện, giảng đường, sân trường...).
2. Phân tích chính xác những gì thực sự có trong ảnh, nhận diện phong cách mỹ thuật, vật liệu, điều kiện ánh sáng và cấu trúc không gian.
3. Đề xuất kế hoạch bù đắp và tái dựng thông minh cho toàn bộ không gian hình cầu 360 độ (bán cầu sau 180°, cực đỉnh bầu trời/trần nhà zenith, cực đáy mặt đất/sàn nhà nadir) sao cho hòa quyện liền mạch với ảnh gốc.

Hãy trả về định dạng JSON thuần túy theo schema sau:
{
  "primarySubject": "Tên chủ thể chính hoặc bối cảnh cốt lõi trong bức ảnh (ngắn gọn, chính xác)",
  "keyObjectsAndDetails": ["Vật thể/chi tiết nhận diện 1", "Vật thể/chi tiết nhận diện 2", "Vật thể/chi tiết nhận diện 3", "Vật thể/chi tiết nhận diện 4"],
  "environment": "Mô tả chi tiết môi trường và không gian bao quanh",
  "architecturalStyle": "Phong cách kiến trúc, địa hình hoặc phong cách thẩm mỹ trực quan",
  "colorAndLighting": "Tông màu chủ đạo, nhiệt độ màu và đặc điểm nguồn sáng tự nhiên/nhân tạo",
  "spatialLayout": "Bố cục không gian, chiều sâu phối cảnh và vị trí tương đối giữa các phần trong ảnh",
  "cameraPerspective": "Góc chụp và tiêu cự của ảnh gốc (ví dụ: góc nhìn ngang tầm mắt, góc rộng, toàn cảnh, cận cảnh...)",
  "elementsToPreserve": ["Đặc điểm nhận diện độc đáo cần giữ nguyên 1", "Chi tiết vật thể quan trọng cần bảo tồn 2"],
  "reconstructedAreas": [
    "Khu vực đối diện 180° phía sau lưng người quan sát",
    "Khu vực cực bắc Zenith (+90° trên đỉnh đầu / trần nhà / vòm trời)",
    "Khu vực cực nam Nadir (-90° dưới mặt đất / sàn nhà / thảm thực vật)",
    "Hai sườn mở rộng 90° trái và 90° phải"
  ],
  "isSameSpaceMultiImage": ${isMulti ? 'true hoặc false (nếu có nhiều ảnh: xác định có thuộc cùng 1 không gian không)' : 'null'},
  "consistencyNote": "Nhận xét về tính đồng nhất không gian giữa các ảnh tải lên",
  "uncertaintyDisclaimer": "Lưu ý trung thực: Do ảnh gốc là hình ảnh phẳng đơn lẻ, toàn bộ vùng không gian 360° bao quanh bên ngoài góc nhìn của ảnh sẽ được AI suy đoán và tái dựng một cách khoa học, hài hòa theo ngữ cảnh của ảnh gốc, không phải là dữ liệu hiện trạng thực tế."
}
`;
        parts.push(visionPrompt);

        const aiResponse = await activeAI.models.generateContent({
          model: GEMINI_MODEL,
          contents: parts,
          config: {
            responseMimeType: 'application/json',
            temperature: 0.3,
          }
        });

        const text = aiResponse.text || '';
        const parsed = cleanAndParseJSON(text);
        return res.json({ success: true, analysis: parsed, isAI: true, model: GEMINI_MODEL });
      } catch (geminiErr: any) {
        console.warn('Gemini vision API error, falling back to server heuristics:', geminiErr.message);
        visionError = classifyGeminiError(geminiErr);
      }
    }

    // Server-side fallback heuristics (KHÔNG phải AI — được gắn cờ isAI:false)
    const cleanName = (mainImg.name || 'khong_gian').replace(/[-_]/g, ' ');
    const fallbackAnalysis = {
      primarySubject: `Không gian ${cleanName}`,
      keyObjectsAndDetails: [
        'Vật thể trung tâm trong tầm nhìn chính',
        'Bề mặt sàn / nền nhà với kết cấu vật liệu rõ ràng',
        'Vách tường hoặc đường chân trời định hình không gian',
        'Các chi tiết phụ trợ và thiết bị xung quanh'
      ],
      environment: userNotes ? `Không gian theo yêu cầu: "${userNotes}"` : 'Môi trường thực tế tương thích với chủ đề',
      architecturalStyle: 'Ảnh chụp chân thực, phối cảnh không gian sâu',
      colorAndLighting: 'Ánh sáng tự nhiên ban ngày, chiếu sáng đồng đều',
      spatialLayout: 'Phối cảnh chiều sâu tập trung ở hướng chính diện của ảnh',
      cameraPerspective: 'Góc nhìn ngang tầm mắt người quan sát (Eye-level FOV ~70°)',
      elementsToPreserve: [
        'Bố cục của chủ thể chính trong bức ảnh gốc',
        'Tông màu chủ đạo và nhiệt độ ánh sáng hiện tại',
        'Đặc điểm nhận diện kiến trúc hoặc cảnh quan đặc trưng'
      ],
      reconstructedAreas: [
        'Góc nhìn phía sau lưng người quan sát (180 độ đối diện)',
        'Khu vực trần nhà / bầu trời đỉnh đầu (Cực bắc Zenith 90°)',
        'Khu vực sàn nhà / mặt đất dưới chân (Cực nam Nadir -90°)',
        'Hai bên sườn góc nhìn ngoại biên mở rộng'
      ],
      isSameSpaceMultiImage: isMulti ? true : null,
      consistencyNote: isMulti ? `Đã phân tích ${images.length} ảnh: Các ảnh có tính chất không gian tương đồng và có thể tổng hợp thành chuỗi phòng học 360°.` : '',
      uncertaintyDisclaimer: 'Lưu ý trung thực: Do ảnh gốc là hình ảnh phẳng chụp ở một góc nhất định, toàn bộ vùng không gian 360° bao quanh còn lại (phía sau lưng, trên trần và dưới sàn) sẽ được AI suy đoán và tái dựng sáng tạo theo ngữ cảnh của ảnh gốc, không phải là dữ liệu hiện trạng thực tế.'
    };

    return res.json({
      success: true,
      analysis: fallbackAnalysis,
      isAI: false,
      fallbackReason: activeAI ? (visionError?.errorCode || 'GEMINI_ERROR') : 'NO_KEY',
      fallbackMessage: activeAI
        ? (visionError?.message || 'Dịch vụ Gemini lỗi.') + ' Đang dùng phân tích cơ bản (không phải AI).'
        : 'Chưa cấu hình GEMINI_API_KEY. Thầy/Cô có thể dán API Key cá nhân trong cài đặt hoặc dùng phân tích cơ bản (không phải AI).',
    });
  } catch (err: any) {
    return res.status(500).json({ error: err.message });
  }
});

// 17. AI 360 Image Generation Status / Bridge
app.post('/api/vr/generate-image-360', async (_req, res) => {
  // Master Prompt Constraint: "Nếu mô hình không hỗ trợ tạo ảnh Equirectangular thực sự, phải thông báo rõ giới hạn, không tự tuyên bố ảnh đã đạt chuẩn VR 360°. Không làm giả chức năng tạo ảnh nếu API chưa được cấu hình."
  return res.json({
    success: false,
    canGenerateDirectly: false,
    provider: 'Blockade Labs Skybox AI / Midjourney v6 / Stable Diffusion LORAs',
    message: 'Dịch vụ sinh ảnh AI trực tiếp yêu cầu kết nối chuyên biệt đến mô hình tạo ảnh toàn cảnh Equirectangular 360° (như Blockade Labs Skybox AI API hoặc Midjourney API). Để đảm bảo ảnh đạt chuẩn tỷ lệ 2:1 bọc kín hình cầu không đường nối, Thầy/Cô vui lòng: (1) Sao chép Prompt tiếng Anh chuẩn kỹ thuật đã được tạo; (2) Dán vào Skybox AI hoặc Midjourney để tải ảnh về; (3) Tải ảnh đó vào EDUVERSE VR Studio để gắn Hotspot và bài học.',
    proceduralFallbackAvailable: true
  });
});

async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.join(__dirname, 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(port, '0.0.0.0', () => {
    console.log(`EduLife AI Studio Server running on http://0.0.0.0:${port}`);
  });
}

// Trên Vercel: không listen, Vercel gọi trực tiếp `app` qua api/index.ts; frontend trả về tĩnh từ dist/.
if (!IS_VERCEL) {
  startServer().catch((err) => {
    console.error('Failed to start server:', err);
    process.exit(1);
  });
}

export default app;
