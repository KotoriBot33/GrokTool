import type { GrokSettings } from "../settings";
import { getDynamicHeaders } from "./headers";
import { getModelInfo, toGrokModel } from "./models";

export interface OpenAIChatMessage {
  role: string;
  content: string | Array<{ type: string; text?: string; image_url?: { url?: string } }>;
}

export interface OpenAIChatRequestBody {
  model: string;
  messages: OpenAIChatMessage[];
  stream?: boolean;
  video_config?: {
    aspect_ratio?: string;
    video_length?: number;
    resolution?: string;
    preset?: string;
  };
}

export function parseVideoLengthStrict(raw: unknown):
  | { ok: true; value: number }
  | { ok: false; reason: "missing" | "not_number" | "out_of_range" } {
  if (raw === undefined || raw === null || raw === "") {
    return { ok: false, reason: "missing" };
  }
  const n = typeof raw === "number" ? raw : Number(raw);
  if (!Number.isFinite(n)) return { ok: false, reason: "not_number" };
  const normalized = Math.floor(n);
  if (normalized < 1 || normalized > 15) return { ok: false, reason: "out_of_range" };
  return { ok: true, value: normalized };
}

export const CONVERSATION_API = "https://grok.com/rest/app-chat/conversations/new";

export function extractContent(messages: OpenAIChatMessage[]): { content: string; images: string[] } {
  const images: string[] = [];
  const extracted: Array<{ role: string; text: string }> = [];

  for (const msg of messages) {
    const role = msg.role ?? "user";
    const content = msg.content ?? "";

    const parts: string[] = [];
    if (Array.isArray(content)) {
      for (const item of content) {
        if (item?.type === "text") {
          const t = item.text ?? "";
          if (t.trim()) parts.push(t);
        }
        if (item?.type === "image_url") {
          const url = item.image_url?.url;
          if (url) images.push(url);
        }
      }
    } else {
      const t = String(content);
      if (t.trim()) parts.push(t);
    }

    if (parts.length) extracted.push({ role, text: parts.join("\n") });
  }

  let lastUserIndex: number | null = null;
  for (let i = extracted.length - 1; i >= 0; i--) {
    if (extracted[i]!.role === "user") {
      lastUserIndex = i;
      break;
    }
  }

  const out: string[] = [];
  for (let i = 0; i < extracted.length; i++) {
    const role = extracted[i]!.role || "user";
    const text = extracted[i]!.text;
    if (i === lastUserIndex) out.push(text);
    else out.push(`${role}: ${text}`);
  }

  return { content: out.join("\n\n"), images };
}

export function buildConversationPayload(args: {
  requestModel: string;
  content: string;
  imgIds: string[];
  imgUris: string[];
  postId?: string;
  videoConfig?: {
    aspect_ratio?: string;
    video_length?: number;
    resolution?: string;
    preset?: string;
  };
  settings: GrokSettings;
}): { payload: Record<string, unknown>; referer?: string; isVideoModel: boolean } {
  const { requestModel, content, imgIds, imgUris, postId, settings } = args;
  const cfg = getModelInfo(requestModel);
  const { grokModel, mode, isVideoModel } = toGrokModel(requestModel);

  if (cfg?.is_video_model) {
    if (!postId) throw new Error("视频模型缺少 postId（需要先创建 media post）");

    const aspectRatio = (args.videoConfig?.aspect_ratio ?? "").trim() || "3:2";
    const parsedVideoLength = parseVideoLengthStrict(args.videoConfig?.video_length);
    if (!parsedVideoLength.ok) {
      throw new Error("invalid_video_length");
    }
    const videoLength = parsedVideoLength.value;
    const resolution = (args.videoConfig?.resolution ?? "SD") === "HD" ? "HD" : "SD";
    const preset = (args.videoConfig?.preset ?? "normal").trim();

    let modeFlag = "--mode=custom";
    if (preset === "fun") modeFlag = "--mode=extremely-crazy";
    else if (preset === "normal") modeFlag = "--mode=normal";
    else if (preset === "spicy") modeFlag = "--mode=extremely-spicy-or-crazy";

    const prompt = `${String(content || "").trim()} ${modeFlag}`.trim();

    return {
      isVideoModel: true,
      referer: "https://grok.com/imagine",
      payload: {
        temporary: true,
        modelName: "grok-3",
        message: prompt,
        toolOverrides: { videoGen: true },
        enableSideBySide: true,
        responseMetadata: {
          experiments: [],
          modelConfigOverride: {
            modelMap: {
              videoGenModelConfig: {
                parentPostId: postId,
                aspectRatio,
                videoLength,
                videoResolution: resolution,
              },
            },
          },
        },
      },
    };
  }

  return {
    isVideoModel,
    payload: {
      temporary: settings.temporary ?? true,
      modelName: grokModel,
      message: content,
      fileAttachments: imgIds,
      imageAttachments: [],
      disableSearch: false,
      enableImageGeneration: true,
      returnImageBytes: false,
      returnRawGrokInXaiRequest: false,
      enableImageStreaming: true,
      imageGenerationCount: 2,
      forceConcise: false,
      toolOverrides: {},
      enableSideBySide: true,
      sendFinalMetadata: true,
      isReasoning: false,
      webpageUrls: [],
      disableTextFollowUps: true,
      responseMetadata: { requestModelDetails: { modelId: grokModel } },
      disableMemory: false,
      forceSideBySide: false,
      modelMode: mode,
      isAsyncChat: false,
    },
  };
}

export async function sendConversationRequest(args: {
  payload: Record<string, unknown>;
  cookie: string;
  settings: GrokSettings;
  referer?: string;
}): Promise<Response> {
  const { payload, cookie, settings, referer } = args;
  const headers = getDynamicHeaders(settings, "/rest/app-chat/conversations/new");
  headers.Cookie = cookie;
  if (referer) headers.Referer = referer;
  const body = JSON.stringify(payload);

  return fetch(CONVERSATION_API, { method: "POST", headers, body });
}
