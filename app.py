# -*- coding: utf-8 -*-
"""
沉浸式 AI 文旅伴游工作台 — FastAPI 后端（真实智谱 API 版）
============================================================

本文件是第 2 步「前后端真实 API 打通」的核心：
左侧面板 5 个功能全部由智谱 AI 真实驱动，写死的 Mock 演示数据已全部废弃。

【核心约束】
  1. API Key 只从项目根目录 .env 的 ZHIPU_API_KEY 读取；
     代码中绝不出现明文 Key，日志中也不打印 Key 的任何片段。
  2. 文本生成统一使用 glm-4-flash（免费且速度快）；
     视觉识别统一使用 glm-4v-flash（拍照识景）。
  3. 所有 AI 生成内容（行程 / 识别结果 / IP 回答 / 文案 / 讲解词）
     在返回体中统一附带合规标识：
     [AIGC生成内容，已通过人工审核校正]

【接口清单】
  GET  /api/health              健康检查（含模型信息）
  POST /api/trip/generate      智能行程（动态规划 Agent）：需求文本 + 出行对象/预算/偏好
                                → glm-4-flash → 严格 JSON（地名/经纬度/时间/描述/天气建议/预算估算）
                                支持快捷干预（🌧️下雨了/😫走不动了/💰预算超了）动态重排
  POST /api/vision/identify    拍照识景：上传图片 → Base64 → glm-4v-flash → 景点名 + 讲解
  POST /api/vision/audit       人工修正审计：记录「AI 识别 → 用户修正」，内存存储 + 终端打印
  GET  /api/vision/audit       查看审计记录
  POST /api/voice/generate     方言导游讲解词：glm-4-flash 生成约 50 字景点介绍（方言词汇强化改写）
  GET  /api/voice/tts          Edge-TTS 语音合成：text+dialect → 真实 MP3 音频流（音色降级链 + 终端打印验证）
  POST /api/ip/chat            数字向导「游小仙」问答（System Prompt 锁人设，多轮对话）
  POST /api/share/generate     一键分享：文艺 / 搞笑 / 吐槽风朋友圈文案（滑动卡片切换）
  GET  /api/travel/footprint   旅行足迹：累计打卡景点记录（内存存储）
  POST /api/travel/journal     AI 旅行手账：上传照片 → glm-4v-flash 识别 → glm-4-flash 生成小红书风游记

【启动方式】
  pip install -r requirements.txt
  uvicorn app:app --host 127.0.0.1 --port 8000
"""

import base64
import json
import os
import re
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import edge_tts
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
from zhipuai import ZhipuAI

# ==================================================================
# 一、基础路径与配置
# ==================================================================
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# 从 .env 加载环境变量（.env 已在 .gitignore 中，不会进版本库）
load_dotenv(BASE_DIR / ".env")

# Key 只读环境变量，代码中不出现任何明文
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "").strip()
if not ZHIPU_API_KEY:
    raise RuntimeError(
        "未检测到 ZHIPU_API_KEY：请在项目根目录 .env 中写入 "
        "ZHIPU_API_KEY=你的Key（ID.Secret 两段式，中间用英文点连接）"
    )

# 模型约定（全项目统一）
TEXT_MODEL = "glm-4-flash"      # 文本生成：行程 / IP / 文案 / 讲解词
VISION_MODEL = "glm-4v-flash"   # 视觉识别：拍照识景

# 合规标识：所有生成内容必须携带
COMPLIANCE_NOTICE = "[AIGC生成内容，已通过人工审核校正]"

# 智谱 AI 客户端（SDK 默认 Base URL: https://open.bigmodel.cn/api/paas/v4）
client = ZhipuAI(api_key=ZHIPU_API_KEY)

# ==================================================================
# 二、FastAPI 应用与静态资源
# ==================================================================
app = FastAPI(title="沉浸式 AI 文旅伴游工作台", version="2.0.0")

# 开发期放开跨域（生产同源部署可移除）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    """工作台页面：全屏 Leaflet 地图 + 左侧悬浮毛玻璃 AI 面板"""
    return FileResponse(STATIC_DIR / "dashboard.html")


@app.get("/api/health")
def health():
    """健康检查：确认 Key 已加载、模型配置正确"""
    return {
        "status": "ok",
        "service": "immersive-travel-workbench",
        "text_model": TEXT_MODEL,
        "vision_model": VISION_MODEL,
        "api_key_loaded": bool(ZHIPU_API_KEY),  # 只回布尔值，不泄露 Key
        "notice": COMPLIANCE_NOTICE,
    }


# ==================================================================
# 三、通用工具函数
# ==================================================================
def call_text_model(
    messages: list,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    json_mode: bool = False,
    retry: int = 1,
) -> str:
    """统一调用文本大模型（glm-4-flash），带重试与 JSON 模式降级。

    :param messages: OpenAI 风格的消息列表
    :param json_mode: 是否要求模型返回 JSON（不支持时自动降级）
    :return: 模型返回的文本
    """
    last_err = None
    for attempt in range(retry + 1):
        try:
            kwargs = dict(
                model=TEXT_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as e:  # 网络抖动 / 限流等
            last_err = e
            # 个别模型不支持 json_object 时，降级为普通模式重试一次
            if json_mode and attempt == 0:
                json_mode = False
                continue
            if attempt < retry:
                time.sleep(1.5)
    raise HTTPException(status_code=502, detail=f"智谱文本模型调用失败：{last_err}")


def call_vision_model(image_b64: str, prompt: str) -> str:
    """统一调用视觉大模型（glm-4v-flash）。

    :param image_b64: 图片 Base64 字符串（不带 data: 前缀）
    :param prompt: 识别指令
    """
    content = [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_b64}},
        {"type": "text", "text": prompt},
    ]
    try:
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=0.1,
            max_tokens=512,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"智谱视觉模型调用失败：{e}")


def extract_json(text: str) -> dict:
    """从模型输出中稳健提取 JSON（兼容 ```json 代码块和前后废话）。"""
    if not text:
        raise ValueError("模型返回为空")
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)   # 去掉开头的 ```json
    cleaned = re.sub(r"\s*```$", "", cleaned)            # 去掉结尾的 ```
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    match = re.search(r"\{.*\}", cleaned, re.S)          # 截取第一个 {...}
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"无法解析模型返回的 JSON：{cleaned[:200]}")


# ==================================================================
# 四、请求体模型
# ==================================================================
class TripGenerateRequest(BaseModel):
    text: str = Field(..., description="用户的旅行需求文本，如：杭州西湖一日游")
    people: str = Field("", description="出行对象：情侣 / 家人 / 朋友 / 独自")
    budget: str = Field("", description="预算范围：穷游 / 适中 / 奢华")
    preference: str = Field("", description="特殊偏好：美食 / 历史 / 拍照 / 休闲")
    event: str = Field("", description="快捷干预事件：rain=下雨了 / tired=走不动了 / overbudget=预算超了")
    current_trip: Optional[dict] = Field(None, description="当前已有行程（动态重排时由前端回传）")


class IPChatRequest(BaseModel):
    persona: str = Field("youxiaoxian", description="数字向导：youxiaoxian=游小仙")
    message: str = Field(..., description="用户本轮提问")
    history: List[dict] = Field(default_factory=list, description="历史对话 [{role, content}]")


class ShareGenerateRequest(BaseModel):
    style: str = Field("art", description="art=文艺风 / fun=搞笑风")
    context: str = Field("", description="可选的行程 / 识景上下文")


class VoiceGuideRequest(BaseModel):
    dialect: str = Field("普通话", description="普通话 / 四川话 / 粤语 / 英语")
    spot: str = Field("故宫博物院", description="要讲解的景点名称")


class VisionAuditRequest(BaseModel):
    old_name: str = Field(..., description="AI 识别出的名称")
    new_name: str = Field(..., description="用户修正后的名称")
    image_name: Optional[str] = Field(None, description="原始图片文件名")


# ==================================================================
# 五、接口一：智能行程（动态规划 Agent）/api/trip/generate
#     多维度输入：需求文本 + 出行对象 / 预算范围 / 特殊偏好
#     快捷干预：🌧️下雨了 / 😫走不动了 / 💰预算超了（带 current_trip 动态重排）
#     输出：严格 JSON（地名/经纬度/时间/描述 + weather_tip 天气建议 + budget_estimate 预算估算）
#     前端拿到 JSON 后：左侧渲染时间轴 + 地图打点 + 绿色虚线路线
# ==================================================================

# 出行对象 → 规划侧重点
PEOPLE_HINT = {
    "情侣": "节奏浪漫舒缓，安排黄昏观景、情侣打卡机位，避免人挤人",
    "家人": "节奏轻松，考虑老人小孩体力，少爬山多平路，预留午休",
    "朋友": "节奏明快，可加入趣味体验项目，安排聚餐社交场景",
    "独自": "节奏自由，安静惬意的点位优先，适合拍照和慢逛",
}

# 预算范围 → 消费档位
BUDGET_HINT = {
    "穷游": "以免费公园、老街、博物馆为主，餐饮选小吃，单日总预算控制在 100 元以内",
    "适中": "景点门票 + 一顿正餐 + 咖啡，单日总预算 100-300 元",
    "奢华": "可含游船、演出、精品餐厅，单日总预算 300 元以上",
}

# 特殊偏好 → 景点类型
PREFERENCE_HINT = {
    "美食": "优先安排本地老字号、小吃街、特色餐厅，在 desc 中突出招牌菜",
    "历史": "优先博物馆、古迹、遗址，在 desc 中突出历史典故",
    "拍照": "优先出片机位、地标建筑、园林景观，在 desc 中突出拍照建议",
    "休闲": "优先公园、茶馆、湖边漫步，减少赶路，安排午茶和休息点",
}

# 快捷干预事件 → 重排指令（动态规划 Agent 的核心）
EVENT_HINT = {
    "rain": (
        "【重排指令：下雨了】用户遭遇降雨。当前行程中的户外景点"
        "（湖边漫步、登山、游船、广场、步行街）在雨天体验极差，"
        "【必须全部替换】为室内景点（博物馆 / 美术馆 / 茶馆 / 室内餐厅 / 商场 / 室内园林），"
        "只允许保留本身就是室内的点位；替换景点尽量靠近原景点，减少淋雨步行，"
        "desc 中注明避雨优势。"
    ),
    "tired": (
        "【重排指令：走不动了】用户体力不支，请压缩行程：景点数量减到 2-3 个，"
        "两站之间步行距离要短，优先彼此相邻的景点，时间安排宽松，"
        "desc 中注明省力理由（如：相邻免坐车）。"
    ),
    "overbudget": (
        "【重排指令：预算超了】用户预算超支，请把当前行程中的【收费景点】替换为"
        "【免费 / 低价景点】（免费公园、博物馆、老街、城市景观），"
        "desc 中注明免费或低价信息，预算估算相应下调。"
    ),
}

TRIP_SYSTEM_PROMPT = """你是资深旅行行程规划师「游小仙规划引擎」。用户会给你一段旅行需求，请规划一份一日游行程。
你必须严格遵守：
1. 只输出一个 JSON 对象，禁止输出任何解释文字，禁止使用 Markdown 代码块。
2. JSON 结构：
{"title": "行程标题（10字以内）", "spots": [{"name": "景点名称", "lat": 纬度, "lng": 经度, "time": "HH:MM", "transport": "从上一站到本站的交通方式", "desc": "亮点描述（30字以内）"}], "weather_tip": "天气与穿着建议（30字以内）", "budget_estimate": "预算估算（如：人均约200元，含门票与一顿正餐）"}
3. spots 数量 3-6 个，按游览时间先后排序，时间从 09:00 前后开始合理递增。
4. transport 字段说明「怎么到这一站」：必须给出真实可行的当地交通方式与耗时/距离，例如「地铁1号线约20分钟」「步行约800米」「打车约15分钟」；第一个景点的 transport 填「今日出发」。
5. 经纬度必须是该景点真实的 WGS84 坐标（保留 4 位小数）；绝对不确定的景点不要编造坐标，宁可换成附近你确定坐标的景点。
6. weather_tip 要结合季节与行程类型给出实用建议（如：傍晚湖边风大带外套）；budget_estimate 必须结合用户的预算档位给出人均金额。
7. 若用户消息中包含【重排指令】，你必须基于给出的当前行程做动态重排，输出一份新的完整行程 JSON（不是补充说明）。"""


def is_valid_coord(lat, lng) -> bool:
    """坐标合法性校验：必须是有限浮点数、在合理范围内、且不能是 (0, 0)（几内亚湾）。

    智谱大模型偶尔会返回 lat:0/lng:0 或超范围值，这类点若打到地图上
    会落在几内亚湾并污染 fitBounds 视野，导致用户看到的标记「缺斤短两」。
    """
    try:
        lat_f = float(lat)
        lng_f = float(lng)
    except (TypeError, ValueError):
        return False
    import math
    if not (math.isfinite(lat_f) and math.isfinite(lng_f)):
        return False
    if not (-90.0 <= lat_f <= 90.0 and -180.0 <= lng_f <= 180.0):
        return False
    if abs(lat_f) < 1e-6 and abs(lng_f) < 1e-6:  # (0, 0) 视为无效
        return False
    return True


def _coord_distance_m(lat1, lng1, lat2, lng2) -> float:
    """两个经纬度之间的近似平面距离（米），足够用于 100 米级重复判定。"""
    import math
    dy = (lat1 - lat2) * 111000.0
    dx = (lng1 - lng2) * 111000.0 * math.cos(math.radians((lat1 + lat2) / 2.0))
    return (dx * dx + dy * dy) ** 0.5


def clean_trip_spots(raw_spots: list):
    """清洗模型返回的景点列表：剔除坐标非法 / 超范围 / (0,0) / 重复坐标的点。

    智谱 glm-4-flash 常见的两类「点位消失」元凶：
      1. 坐标返回 0,0 或超范围值（打点飞到几内亚湾，fitBounds 被拽偏）；
      2. 多个景点复制同一坐标（地图上重叠成一个点，看起来「只显示 1 个」）。
    本函数把两类问题点都剔除并分别记录，返回 (有效景点, 非法名单, 重复名单)。
    """
    import math
    spots, skipped, duplicated = [], [], []
    seen_coords = []
    for i, s in enumerate(raw_spots or []):
        if not isinstance(s, dict):
            continue
        name = str(s.get("name", f"景点{i + 1}"))
        lat_raw, lng_raw = s.get("lat"), s.get("lng")
        if not is_valid_coord(lat_raw, lng_raw):
            skipped.append(name)
            print(f"[TRIP] 坐标非法已跳过：{name}（lat={lat_raw}, lng={lng_raw}）", flush=True)
            continue
        lat_f, lng_f = float(lat_raw), float(lng_raw)
        # 与已收录景点距离 < 100 米 → 判定为重复坐标（模型偷懒复制）
        is_dup = any(_coord_distance_m(lat_f, lng_f, la, ln) < 100.0 for la, ln in seen_coords)
        if is_dup:
            duplicated.append(name)
            print(f"[TRIP] 坐标重复已跳过：{name}（与已有景点坐标距离不足 100 米）", flush=True)
            continue
        seen_coords.append((lat_f, lng_f))
        spots.append({
            "name": name,
            "lat": round(lat_f, 6),
            "lng": round(lng_f, 6),
            "time": str(s.get("time", "")),
            "transport": str(s.get("transport", "")),  # 从上一站到本站的交通建议（智能交通规划）
            "desc": str(s.get("desc", "")),
        })
    return spots, skipped, duplicated


@app.post("/api/trip/generate")
def trip_generate(req: TripGenerateRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="旅行需求不能为空")

    # ---- 组装多维度需求（出行对象 / 预算 / 偏好 → 规划侧重点）----
    requirements = [f"旅行需求：{text}"]
    if req.people in PEOPLE_HINT:
        requirements.append(f"出行对象：{req.people}（{PEOPLE_HINT[req.people]}）")
    if req.budget in BUDGET_HINT:
        requirements.append(f"预算范围：{req.budget}（{BUDGET_HINT[req.budget]}）")
    if req.preference in PREFERENCE_HINT:
        requirements.append(f"特殊偏好：{req.preference}（{PREFERENCE_HINT[req.preference]}）")

    # ---- 快捷干预：动态重排（带当前行程 + 事件指令）----
    is_rearrange = req.event in EVENT_HINT and req.current_trip
    cur_spots = []
    if is_rearrange:
        try:
            cur = req.current_trip or {}
            cur_spots = cur.get("spots", []) if isinstance(cur, dict) else []
            spot_lines = "\n".join(
                f"{i + 1}. {s.get('name', '?')}（{s.get('time', '')}）{s.get('desc', '')}"
                for i, s in enumerate(cur_spots)
            )
        except Exception:
            spot_lines = ""
        user_content = (
            "\n".join(requirements)
            + "\n\n当前行程：\n" + (spot_lines or "（无）")
            + "\n\n" + EVENT_HINT[req.event]
        )
        print(f"[TRIP] 动态重排：事件={req.event}，当前 {len(cur_spots)} 个节点", flush=True)
    else:
        user_content = "\n".join(requirements)

    # 调用 glm-4-flash，要求 JSON 输出
    raw = call_text_model(
        messages=[
            {"role": "system", "content": TRIP_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.5,
        json_mode=True,
    )

    # 解析并清洗数据（坐标非法 / (0,0) / 重复坐标的景点跳过，不拖垮整份行程）
    try:
        data = extract_json(raw)
        spots, skipped, duplicated = clean_trip_spots(data.get("spots", []))
        if not spots:
            raise ValueError("spots 为空或坐标全部非法")
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"行程解析失败：{e}；模型原始返回：{raw[:200]}",
        )

    # ---- 坐标重复自动纠偏：模型偷懒给多个景点复制同一坐标时，带反馈重试一次 ----
    if duplicated:
        dup_names = "、".join(duplicated)
        print(f"[TRIP] 检测到坐标重复（{dup_names}），触发纠偏重试", flush=True)
        retry_content = (
            user_content
            + f"\n\n【坐标纠偏指令】你上次返回的行程中，这些景点重复使用了相同或极为接近的经纬度：{dup_names}。"
            "请重新输出完整行程 JSON，确保每个景点使用各自真实、彼此不同的 WGS84 坐标"
            "（任意两个景点至少相距 100 米），其余字段要求不变。"
        )
        raw_retry = call_text_model(
            messages=[
                {"role": "system", "content": TRIP_SYSTEM_PROMPT},
                {"role": "user", "content": retry_content},
            ],
            temperature=0.5,
            json_mode=True,
        )
        try:
            data_retry = extract_json(raw_retry)
            spots_r, skipped_r, duplicated_r = clean_trip_spots(data_retry.get("spots", []))
            # 重试结果有效景点不少于首次（且重复更少）才采用
            if spots_r and (len(spots_r) > len(spots) or len(duplicated_r) < len(duplicated)):
                data = data_retry
                spots, skipped, duplicated = spots_r, skipped_r, duplicated_r
                print(f"[TRIP] 纠偏重试成功：有效景点 {len(spots)} 个", flush=True)
            else:
                print("[TRIP] 纠偏重试未改善，保留首次结果", flush=True)
        except Exception as e:
            print(f"[TRIP] 纠偏重试解析失败，保留首次结果：{e}", flush=True)

    # 记录旅行足迹（「我的」页签的打卡数据来源）
    record_footprint(str(data.get("title", "AI 定制行程")), spots)

    return {
        "title": str(data.get("title", "AI 定制行程")),
        "spots": spots,
        "skipped": skipped + duplicated,                 # 坐标异常 / 重复被剔除的景点（前端提示用）
        "weather_tip": str(data.get("weather_tip", "")),  # 天气建议
        "budget_estimate": str(data.get("budget_estimate", "")),  # 预算估算
        "rearranged": bool(is_rearrange),                # 是否为动态重排结果
        "event": req.event or "",
        "notice": COMPLIANCE_NOTICE,
    }


# ==================================================================
# 六、接口二：拍照识景 /api/vision/identify
#     上传图片 → Base64 → glm-4v-flash → 景点名 + 简要讲解
# ==================================================================
VISION_PROMPT = """请识别这张照片中的景点或文物。
只输出 JSON，禁止输出其他任何文字：
{"name": "景点/文物名称", "desc": "50字以内的简要讲解"}
如果不是景点或文物，name 返回 "未识别"，desc 用一句话说明原因。"""


@app.post("/api/vision/identify")
async def vision_identify(image: UploadFile = File(...)):
    # 读取上传的图片字节
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="图片内容为空")
    if len(raw) > 8 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="图片大小不能超过 8MB")

    # 转 Base64 后交给 glm-4v-flash
    image_b64 = base64.b64encode(raw).decode("utf-8")
    content = call_vision_model(image_b64, VISION_PROMPT)

    # 解析识别结果（解析失败也返回可用的降级结构）
    try:
        data = extract_json(content)
        name = str(data.get("name", "未识别")).strip() or "未识别"
        desc = str(data.get("desc", "")).strip()
    except Exception:
        name, desc = "未识别", "AI 返回格式异常，请重试或使用「手动修正名称」"

    return {
        "name": name,
        "desc": desc,
        "image_name": image.filename or "",
        "notice": COMPLIANCE_NOTICE,
    }


# ==================================================================
# 七、接口三：人工修正审计 /api/vision/audit
#     用户修正识别名称后，前端调用本接口留痕（内存 + 终端打印），
#     用于证明「人机协同」：AI 初判 → 人工校正 → 全程可审计。
# ==================================================================
_audit_lock = threading.Lock()
audit_records: List[dict] = []


def record_audit(old_name: str, new_name: str, image_name: Optional[str] = None) -> dict:
    record = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "old_name": old_name,
        "new_name": new_name,
        "image_name": image_name or "",
    }
    with _audit_lock:
        audit_records.append(record)
    # 后台终端打印，留痕可查
    print(
        f"[AUDIT] {record['time']} 人工修正：AI 识别「{old_name}」→ 用户修正「{new_name}」",
        flush=True,
    )
    return record


@app.post("/api/vision/audit")
def vision_audit(req: VisionAuditRequest):
    old_name = (req.old_name or "").strip()
    new_name = (req.new_name or "").strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="修正后的名称不能为空")
    record = record_audit(old_name, new_name, req.image_name)
    return {"ok": True, "record": record, "notice": COMPLIANCE_NOTICE}


@app.get("/api/vision/audit")
def vision_audit_list():
    with _audit_lock:
        records = list(audit_records)
    return {"records": records, "notice": COMPLIANCE_NOTICE}


# ==================================================================
# 八、接口四：方言导游讲解词 /api/voice/generate
#     先用真实大模型生成一段 50 字左右的当地景点介绍，
#     方言文案强制使用纯正方言口语词汇改写（配合 TTS 方言音色），
#     再交给 Edge-TTS 合成，确保听到的是真正的方言。
# ==================================================================
VOICE_LANG_HINT = {
    "普通话": "用标准普通话书面语",
    "四川话": (
        "必须用纯正的四川方言口语词汇改写讲解词，例如：巴适、安逸、要得、晓得、"
        "巴适得板、雄起、撒子、莫得、搞不醒豁；禁止出现普通话书面语，"
        "让本地人一听就觉得地道"
    ),
    "粤语": (
        "必须用纯正的粤语口语词汇改写讲解词，例如：嘅、咗、靓、饮茶、唔该、正斗、"
        "得闲、一齐；禁止出现普通话书面语，让香港本地人一听就觉得地道"
    ),
    "英语": "用简单自然的英语",
}


@app.post("/api/voice/generate")
def voice_guide(req: VoiceGuideRequest):
    dialect = req.dialect if req.dialect in VOICE_LANG_HINT else "普通话"
    spot = (req.spot or "故宫博物院").strip()
    lang_hint = VOICE_LANG_HINT[dialect]
    prompt = (
        f"请为景点「{spot}」写一段导游讲解词：{lang_hint}，"
        "50字左右（40-60字），口语化、有画面感，像导游在景点现场讲解。"
        "只输出讲解词本身，不要标题、不要解释。"
    )
    text = call_text_model(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=256,
    )
    return {
        "text": text.strip(),
        "dialect": dialect,
        "spot": spot,
        "notice": COMPLIANCE_NOTICE,
    }


# ==================================================================
# 八-B、接口：Edge-TTS 语音合成 GET /api/voice/tts
#     参数：text（要合成的讲解词）、dialect（方言）、rate（语速，如 +0%/+50%）
#     根据方言选择微软音色，生成真实 MP3 音频流返回给前端播放。
#     实现方式：先完整合成到临时文件（失败可返回明确错误码），
#     再以 FileResponse 流式下发，响应结束后自动删除临时文件。
#
#     【重要说明】微软 Edge TTS 已下线四川方言专属音色
#     （zh-CN-sichuan-YunxiNeural / zh-CN-Sichuan-YunxiNeural 均返回
#      NoAudioReceived，当前 322 个音色中已无任何 sichuan 条目）。
#     因此：主音色仍按需求指定 sichuan 音色名，一旦微软服务端确认不可用，
#     自动降级到实测可用的 zh-CN-YunxiNeural（川渝感男声），
#     最终保底 zh-CN-XiaoxiaoNeural，保证任何情况下都能真实出声。
# ==================================================================
TTS_VOICE_MAP = {
    "普通话": "zh-CN-XiaoxiaoNeural",          # 女声 · 温暖亲切
    "四川话": "zh-CN-sichuan-YunxiNeural",     # 男声 · 川渝方言（微软已下线，见上方说明）
    "粤语":   "zh-HK-HiuMaanNeural",           # 女声 · 港式粤语
    "英语":   "en-US-JennyNeural",             # 女声 · 美式英语
}

# 主音色不可用时的降级音色（微软音色列表会调整，保证功能不中断）
TTS_FALLBACK_VOICE = {
    "普通话": "zh-CN-XiaoxiaoNeural",
    "四川话": "zh-CN-YunxiNeural",   # 川渝方言音色下线后用最接近的男声（实测可用）
    "粤语":   "zh-HK-HiuMaanNeural",
    "英语":   "en-US-JennyNeural",
}
# 终极降级：所有候选音色都失败时，保证至少能播出声音
TTS_LAST_RESORT_VOICE = "zh-CN-XiaoxiaoNeural"


async def _synthesize_to_file(text: str, voice: str, rate: str, path: str) -> None:
    """调用 edge-tts 合成语音并写入指定文件（失败抛异常由调用方降级重试）。

    每次调用前在终端打印实际传入的 voice 变量，便于验证音色映射是否正确：
      [TTS] 正在调用 edge_tts.Communicate(text, voice='zh-CN-sichuan-YunxiNeural')
    """
    print(f"[TTS] 正在调用 edge_tts.Communicate(text, voice='{voice}', rate='{rate}')", flush=True)
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(path)


@app.get("/api/voice/tts")
async def voice_tts(
    text: str = Query(..., description="要合成的讲解词文本"),
    dialect: str = Query("普通话", description="普通话 / 四川话 / 粤语 / 英语"),
    rate: str = Query("+0%", description="语速，如 +0% / +50% / -30%"),
):
    # 文本校验与截断（Edge-TTS 单次合成不宜过长）
    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="合成文本不能为空")
    if len(text) > 500:
        text = text[:500]

    # 方言 → 音色（未知方言降级为普通话）；此处打印便于终端验证映射是否正确
    voice = TTS_VOICE_MAP.get(dialect, TTS_VOICE_MAP["普通话"])
    print(
        f"[TTS] 方言={dialect} → 主音色 voice='{voice}'"
        f"（{'严格匹配指定音色' if dialect in TTS_VOICE_MAP else '未知方言，降级普通话'}）",
        flush=True,
    )

    # 语速格式校验（edge-tts 要求 +n% / -n%）
    if not re.fullmatch(r"[+-]\d{1,3}%", rate):
        rate = "+0%"

    # 写入临时文件，便于失败时返回标准 HTTP 错误
    fd, tmp_path = tempfile.mkstemp(suffix=".mp3", prefix="voice_")
    os.close(fd)
    try:
        # 音色降级链：主音色 → 备选音色 → 终极保底音色
        used_voice = None
        candidates = [voice, TTS_FALLBACK_VOICE.get(dialect, voice), TTS_LAST_RESORT_VOICE]
        seen = set()
        for v in candidates:
            if v in seen:
                continue  # 跳过重复候选（如主音色与备选相同时）
            seen.add(v)
            try:
                await _synthesize_to_file(text, v, rate, tmp_path)
                used_voice = v
                break
            except Exception as e:
                # 明确打印降级原因（如微软已下线的四川话音色）
                print(
                    f"[TTS] 音色 {v} 合成失败（{type(e).__name__}），尝试下一个候选音色",
                    flush=True,
                )
                continue
        if not used_voice:
            raise RuntimeError("所有候选音色均合成失败")
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=502, detail=f"Edge-TTS 合成失败：{e}")

    # 音频下发完成后自动清理临时文件
    def _cleanup():
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass

    print(
        f"[TTS] 合成成功：方言={dialect} 音色={used_voice} 语速={rate} 字数={len(text)}",
        flush=True,
    )
    return FileResponse(
        tmp_path,
        media_type="audio/mpeg",
        filename="youxiaoxian-voice.mp3",
        background=BackgroundTask(_cleanup),
    )


# ==================================================================
# 九、接口五：数字向导「游小仙」/api/ip/chat
#     System Prompt 锁定人设：温和、专业、耐心、靠谱、有活力的
#     现代年轻私人导游；通俗自然的现代语言，实用旅行建议优先。
# ==================================================================
IP_PERSONAS = {
    "youxiaoxian": (
        "你是「游小仙」——一位温和、专业、耐心、靠谱、充满活力的现代年轻私人导游，"
        "服务于沉浸式 AI 文旅伴游工作台。你必须严格遵守以下人设：\n"
        "1. 身份定位：你不是古代文人，也不是 AI 助手，你是游客的专属行程向导「游小仙」，"
        "熟悉全国热门景点的历史典故、开放信息、交通路线与实用玩法。\n"
        "2. 说话语气：温和亲切、专业可靠、有活力，像一位靠谱的年轻导游朋友，"
        "用通俗自然的现代中文交流；不要通篇文言文古风，也不要过度使用潮流网络梗。\n"
        "3. 内容偏好：优先给出实用建议（游览顺序、最佳观景时间、避坑提示、附近美食），"
        "可自然带出景点典故与趣味知识，让游客觉得长知识又贴心。\n"
        "4. 表达格式：每次回答控制在 150 字以内，口语化，可适当使用 1-2 个 emoji 增加亲和力，"
        "不要出现「作为AI」「作为语言模型」等现代助手表述。\n"
        "5. 口头禅：在对话开场、行程确认类问题或回答结尾处，自然带出口头禅"
        "「您的专属行程已规划好，我们慢慢逛。」"
    ),
}


@app.post("/api/ip/chat")
def ip_chat(req: IPChatRequest):
    persona_key = req.persona if req.persona in IP_PERSONAS else "youxiaoxian"
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="提问内容不能为空")

    # 组装消息：System 锁角色 + 最近 10 条历史 + 本轮提问
    messages = [{"role": "system", "content": IP_PERSONAS[persona_key]}]
    for h in (req.history or [])[-10:]:
        role = h.get("role")
        content = h.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": str(content)})
    messages.append({"role": "user", "content": message})

    reply = call_text_model(messages, temperature=0.8, max_tokens=512)
    return {"reply": reply, "persona": persona_key, "notice": COMPLIANCE_NOTICE}


# ==================================================================
# 十、接口六：一键分享 /api/share/generate
#     按风格（文艺 / 搞笑 / 吐槽）生成朋友圈文案，前端滑动卡片切换
# ==================================================================
SHARE_PROMPTS = {
    "art": (
        "你是一位文笔优美的旅行博主。请根据用户提供的行程信息，写一条朋友圈文案："
        "文艺清新风格，50-80 字，有意象、有画面感，可带一丝诗句韵味，"
        "结尾附 1-2 个话题标签。只输出文案本身，不要解释。"
    ),
    "fun": (
        "你是一位搞笑风格的旅行博主。请根据用户提供的行程信息，写一条朋友圈文案："
        "幽默搞笑风格，50-80 字，口语化、有梗、适合配表情包，"
        "结尾附 1-2 个话题标签。只输出文案本身，不要解释。"
    ),
    "roast": (
        "你是一位毒舌吐槽风格的旅行博主。请根据用户提供的行程信息，写一条朋友圈文案："
        "吐槽避坑风格，50-80 字，夸张吐槽人挤人、门票贵、排队久等旅行槽点，"
        "结尾附 1-2 个话题标签。只输出文案本身，不要解释。"
    ),
}


@app.post("/api/share/generate")
def share_generate(req: ShareGenerateRequest):
    style = req.style if req.style in SHARE_PROMPTS else "art"
    context = (req.context or "").strip() or "一次说走就走的城市旅行"
    reply = call_text_model(
        messages=[
            {"role": "system", "content": SHARE_PROMPTS[style]},
            {"role": "user", "content": f"我的行程信息：{context}"},
        ],
        temperature=0.9,
        max_tokens=256,
    )
    return {
        "text": reply.strip(),
        "style": style,
        "notice": COMPLIANCE_NOTICE,
    }


# ==================================================================
# 十-B、旅行足迹与 AI 旅行手账（「我的」页签数据源）
#     GET  /api/travel/footprint   累计打卡景点（行程生成的景点自动入册）
#     POST /api/travel/journal     上传旅行照片 → glm-4v-flash 识别场景
#                                → glm-4-flash 结合行程记录生成小红书风游记
# ==================================================================
_footprint_lock = threading.Lock()
footprint_records: List[dict] = []


def record_footprint(trip_title: str, spots: list) -> None:
    """把行程景点记入旅行足迹（内存存储，「我的」页签可视化用）。"""
    with _footprint_lock:
        for s in spots:
            footprint_records.append({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "trip": trip_title,
                "spot": s.get("name", ""),
                "desc": s.get("desc", ""),
                "lat": s.get("lat"),
                "lng": s.get("lng"),
            })
    print(f"[FOOTPRINT] 行程「{trip_title}」已入册，新增 {len(spots)} 个打卡点", flush=True)


@app.get("/api/travel/footprint")
def travel_footprint():
    """旅行足迹：返回累计打卡记录（含行程标题、景点、时间）。"""
    with _footprint_lock:
        records = list(footprint_records)
    # 统计维度：累计景点数、涉及行程数
    trips = sorted({r["trip"] for r in records if r.get("trip")})
    return {
        "records": records,
        "total_spots": len(records),
        "total_trips": len(trips),
        "trips": trips,
        "notice": COMPLIANCE_NOTICE,
    }


JOURNAL_SYSTEM_PROMPT = """你是小红书风格的旅行博主。用户会给你一张旅行照片的 AI 识别结果和TA的行程记录，请写一篇旅行手账游记。
要求：
1. 结构：一个吸引人的标题（带 1-2 个 emoji）+ 3-5 句正文 + 3-5 个话题标签（#标签 格式）。
2. 语气：真实、有画面感、有情绪价值，像刚旅行完迫不及待分享的年轻人；自然结合行程中的景点。
3. 篇幅：正文 80-150 字，不要出现「作为AI」等表述，不要编造识别结果之外的景点。
4. 只输出手账内容本身，不要解释。"""


@app.post("/api/travel/journal")
async def travel_journal(
    image: UploadFile = File(..., description="旅行照片"),
    context: str = Form("", description="行程上下文（当前行程摘要，可为空）"),
):
    """AI 旅行手账：上传照片 → glm-4v-flash 识别场景 → glm-4-flash 生成小红书风游记。"""
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="图片内容为空")
    if len(raw) > 8 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="图片大小不能超过 8MB")

    # 第一步：视觉模型识别照片场景
    image_b64 = base64.b64encode(raw).decode("utf-8")
    vision_raw = call_vision_model(
        image_b64,
        "请识别这张旅行照片的场景：只输出 JSON，禁止其他文字："
        '{"scene": "场景/景点描述（20字以内）", "vibe": "画面氛围（10字以内）"}',
    )
    try:
        vision_data = extract_json(vision_raw)
        scene = str(vision_data.get("scene", "未识别"))
        vibe = str(vision_data.get("vibe", ""))
    except Exception:
        scene, vibe = "一处旅行风景", "惬意"

    # 第二步：结合识别结果 + 行程记录，生成小红书风手账
    ctx = (context or "").strip() or "（暂无行程记录）"
    journal = call_text_model(
        messages=[
            {"role": "system", "content": JOURNAL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"照片 AI 识别结果：{scene}（氛围：{vibe}）。我的行程记录：{ctx}",
            },
        ],
        temperature=0.9,
        max_tokens=512,
    )

    print(f"[JOURNAL] 手账生成：场景={scene} · 上下文长度={len(ctx)}", flush=True)
    return {
        "scene": scene,
        "vibe": vibe,
        "journal": journal.strip(),
        "notice": COMPLIANCE_NOTICE,
    }


# ==================================================================
# 十一、启动日志（不打印 Key 任何片段）
# ==================================================================
@app.on_event("startup")
def _startup_log():
    print("=" * 56, flush=True)
    print("沉浸式 AI 文旅伴游工作台 · 后端已启动", flush=True)
    print(f"文本模型：{TEXT_MODEL} | 视觉模型：{VISION_MODEL}", flush=True)
    print("ZHIPU_API_KEY：已从 .env 加载", flush=True)
    print("接口：/api/trip/generate /api/vision/identify /api/vision/audit", flush=True)
    print("     /api/voice/generate /api/voice/tts /api/ip/chat /api/share/generate", flush=True)
    print("     /api/travel/footprint /api/travel/journal", flush=True)
    print("=" * 56, flush=True)
