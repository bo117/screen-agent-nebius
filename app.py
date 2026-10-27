# -*- coding: utf-8 -*-
"""
Screen Agent — 用 NVIDIA Nemotron 在 Nebius Token Factory 上做"读屏 + 推理"

双模型分工（这是本项目的核心设计）：
  · 视觉：Nemotron 3.5 Super VL —— 负责"看"：识别界面/场景、读文字、定位元素
  · 推理：Nemotron 3 Ultra      —— 负责"想"：判断用户意图、给下一步建议、指出问题

两者串行：先看，再想。用户纠正时，视觉重看 + 推理重判，模型承认并改正。

启动：
  export NEBIUS_API_KEY=...
  uvicorn app:app --host 0.0.0.0 --port 8000
"""
import base64
import json
import os
import time
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

BASE_URL = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
MODEL_VL = os.environ.get("MODEL_VL", "nvidia/nemotron-3.5-super-vl")
MODEL_REASON = os.environ.get("MODEL_REASON", "nvidia/nemotron-3-ultra-550b-a55b")
API_KEY = os.environ.get("NEBIUS_API_KEY", "")

app = FastAPI(title="Screen Agent", version="1.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

_client: Optional[OpenAI] = None


def client() -> OpenAI:
    global _client
    if not API_KEY:
        raise HTTPException(500, "NEBIUS_API_KEY 未设置。请在环境变量里配置后再启动。")
    if _client is None:
        _client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=120)
    return _client


VL_PROMPT = """你是一个屏幕与图像理解助手。请只做"看"的工作，不要推测用户意图。

输出严格的 JSON（不要 markdown 代码块，不要多余文字）：
{
  "scene": "一张截图 / 一张照片 / 一个网页 / 一个应用界面",
  "app_name": "如果是软件界面，写出应用或网站名；否则填 null",
  "layout": "一句话描述整体布局",
  "texts": ["画面上可读的主要文字，最多 12 条"],
  "elements": [{"name": "元素名", "role": "按钮/输入框/图标/图片/文本", "box": [x1,y1,x2,y2]}],
  "colors": ["主要颜色"],
  "uncertain": ["你不确定的地方，最多 3 条"]
}

box 用 0-1 归一化坐标。如果画面上没有明确的界面元素，elements 留空数组。"""

REASON_PROMPT = """你是屏幕使用助手。下面是一个视觉模型对画面的结构化描述。
请基于它做"想"的工作，不要编造描述里没有的内容。

视觉描述：
{vl_json}

用户的问题：{question}

请用中文回答，结构如下（不要用 markdown 标题，用短段落）：
1. 这是什么、当前处于什么状态
2. 用户最可能想做什么（给 1-2 种可能，说明依据）
3. 建议的下一步操作（具体到点哪里、输入什么）
4. 如果有风险或异常，指出来

如果视觉描述里的 uncertain 有内容，在回答里说明哪些判断可能不准。"""


class Turn(BaseModel):
    role: str
    content: str


class AskIn(BaseModel):
    image_id: str
    question: str
    history: List[Turn] = []


_state = {}


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _call_vl(image_b64: str, mime: str, extra: str = "") -> dict:
    msg = [{"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (mime, image_b64)}},
           {"type": "text", "text": VL_PROMPT + (("\n\n补充要求：" + extra) if extra else "")}]
    r = client().chat.completions.create(
        model=MODEL_VL,
        messages=[{"role": "user", "content": msg}],
        temperature=0.1,
        max_tokens=1200,
    )
    txt = r.choices[0].message.content or ""
    txt = txt.strip()
    if txt.startswith("```"):
        txt = txt.split("```")[1]
        if txt.startswith("json"):
            txt = txt[4:]
    try:
        return json.loads(txt)
    except Exception:
        return {"raw": txt, "parse_error": True}


def _call_reason(vl: dict, question: str, history: List[Turn]) -> str:
    sys = REASON_PROMPT.format(vl_json=json.dumps(vl, ensure_ascii=False)[:4000],
                               question=question)
    msgs = [{"role": "system", "content": sys}]
    for t in history[-6:]:
        msgs.append({"role": t.role, "content": t.content})
    msgs.append({"role": "user", "content": question})
    r = client().chat.completions.create(
        model=MODEL_REASON, messages=msgs, temperature=0.3, max_tokens=1500)
    return (r.choices[0].message.content or "").strip()


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "has_key": bool(API_KEY), "model_vl": MODEL_VL,
            "model_reason": MODEL_REASON, "images_cached": len(_state)}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > 12 * 1024 * 1024:
        raise HTTPException(413, "图片超过 12MB")
    iid = "img_%d" % int(time.time() * 1000)
    mime = file.content_type or "image/png"
    _state[iid] = {"b64": _b64(data), "mime": mime, "vl": None, "turns": []}
    return {"image_id": iid}


@app.post("/api/look")
async def look(payload: dict):
    """只看：返回结构化视觉描述"""
    iid = payload.get("image_id")
    if iid not in _state:
        raise HTTPException(404, "image_id 不存在，请重新上传")
    st = _state[iid]
    vl = _call_vl(st["b64"], st["mime"], payload.get("extra", ""))
    st["vl"] = vl
    return {"vl": vl}


@app.post("/api/ask")
def ask(inp: AskIn):
    """看 + 想"""
    if inp.image_id not in _state:
        raise HTTPException(404, "image_id 不存在，请重新上传")
    st = _state[inp.image_id]
    if st["vl"] is None:
        st["vl"] = _call_vl(st["b64"], st["mime"])
    ans = _call_reason(st["vl"], inp.question, inp.history)
    st["turns"].append({"q": inp.question, "a": ans})
    return {"answer": ans, "vl": st["vl"], "vl_model": MODEL_VL, "reason_model": MODEL_REASON}


@app.post("/api/correct")
def correct(payload: dict):
    """用户纠正：视觉重看 + 推理重判，模型必须承认并改正"""
    iid = payload.get("image_id")
    correction = payload.get("correction", "")
    if iid not in _state:
        raise HTTPException(404, "image_id 不存在")
    st = _state[iid]
    hint = "用户指出上一轮判断有误：%s。请重新仔细观察，特别关注与该说法相关的区域。" % correction
    vl2 = _call_vl(st["b64"], st["mime"], hint)
    st["vl"] = vl2
    q = ("用户纠正：%s\n请先明确承认之前看错了，再基于新的观察重新回答。"
         "格式：第一句必须是「好的，是我看错了」这类承认，然后给出修正后的判断。" % correction)
    ans = _call_reason(vl2, q, [])
    return {"answer": ans, "vl": vl2}


@app.exception_handler(Exception)
async def on_error(request, exc):
    return JSONResponse(status_code=500, content={"error": str(exc)[:400]})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
