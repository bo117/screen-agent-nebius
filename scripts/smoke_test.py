# -*- coding: utf-8 -*-
"""smoke_test.py —— 不用浏览器，直接验证 Nebius Token Factory 的接线

用法：
  export NEBIUS_API_KEY=...
  python scripts/smoke_test.py --image screenshot.png

会做三件事：
  1. 调 VL 模型看一张图，打印结构化 JSON
  2. 调推理模型回答一句"我该做什么"
  3. 调一次纠正流程，检查它是否真的承认错误
"""
import argparse
import base64
import json
import os
import sys

from openai import OpenAI

BASE = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
VL = os.environ.get("MODEL_VL", "nvidia/nemotron-3.5-super-vl")
RS = os.environ.get("MODEL_REASON", "nvidia/nemotron-3-ultra-550b-a55b")
KEY = os.environ.get("NEBIUS_API_KEY", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--question", default="这张图里是什么？我接下来该做什么？")
    args = ap.parse_args()

    if not KEY:
        print("FAIL: 没有 NEBIUS_API_KEY。先 export 再跑。")
        return 1
    if not os.path.exists(args.image):
        print("FAIL: 图片不存在 ->", args.image)
        return 1

    c = OpenAI(base_url=BASE, api_key=KEY, timeout=120)
    b64 = base64.b64encode(open(args.image, "rb").read()).decode()
    ext = os.path.splitext(args.image)[1].lstrip(".").lower() or "png"
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"

    print("[1/3] VL 模型:", VL)
    try:
        r = c.chat.completions.create(
            model=VL,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (mime, b64)}},
                {"type": "text", "text": '只描述你看到的，输出 JSON：{"scene":..,"texts":[..],"elements":[..],"uncertain":[..]}'},
            ]}],
            temperature=0.1, max_tokens=800)
        vl_text = r.choices[0].message.content
        print(vl_text[:800])
    except Exception as e:
        print("FAIL: VL 调用失败 ->", str(e)[:300])
        return 1

    print("\n[2/3] 推理模型:", RS)
    try:
        r2 = c.chat.completions.create(
            model=RS,
            messages=[
                {"role": "system", "content": "你是屏幕使用助手，只能基于给定的视觉描述回答，不要编造。"},
                {"role": "user", "content": "视觉描述：\n%s\n\n问题：%s" % (vl_text, args.question)},
            ], temperature=0.3, max_tokens=800)
        ans = r2.choices[0].message.content
        print(ans[:800])
    except Exception as e:
        print("FAIL: 推理调用失败 ->", str(e)[:300])
        return 1

    print("\n[3/3] 纠正流程")
    try:
        r3 = c.chat.completions.create(
            model=RS,
            messages=[
                {"role": "system", "content": "用户会纠正你。第一句必须明确承认之前看错了，然后给出修正判断。"},
                {"role": "user", "content": "之前你说这是香蕉。用户说：不不，它是个苹果。请回应。"},
            ], temperature=0.3, max_tokens=300)
        fix = r3.choices[0].message.content
        print(fix[:400])
        ok = any(w in fix for w in ("看错", "抱歉", "说得对", "的确", "有误", "认错"))
        print("\n纠正行为检查:", "PASS（有承认）" if ok else "WARN（没看到承认措辞）")
    except Exception as e:
        print("FAIL: 纠正调用失败 ->", str(e)[:300])
        return 1

    print("\nSMOKE TEST DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
