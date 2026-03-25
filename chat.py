"""终端对话测试 — 和艾莉丝聊天。

用法: python chat.py

需要:
  1. RAG 服务器运行中 (python -m uvicorn src.server:app --port 8787)
  2. config.yaml 中配置了 LLM provider (用于生成回复)
"""

import sys
import httpx

RAG_URL = "http://localhost:8787"
SENDER_ID = "0000000000"
SENDER_NICKNAME = "鲁迪乌斯"

# 用哪个 provider 生成艾莉丝的回复
# 从 config 读取，或直接指定
LLM_PROVIDER = "deepseek"


def get_reply(system_prompt: str, user_message: str, history: list[dict]) -> str:
    """调用 LLM 生成艾莉丝的回复。"""
    from src.config import Config
    config = Config()
    provider_cfg = config.get_provider_config(LLM_PROVIDER)
    if not provider_cfg:
        return f"[错误] 未找到 provider: {LLM_PROVIDER}"

    api_key = provider_cfg["api_key"]
    model = provider_cfg["model"]
    base_url = provider_cfg["base_url"]

    messages = [{"role": "system", "content": system_prompt}]
    for h in history[-10:]:  # 最近 10 轮
        messages.append(h)
    messages.append({"role": "user", "content": user_message})

    resp = httpx.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": 0.7, "max_tokens": 512},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    msg = data["choices"][0]["message"]
    return msg.get("content") or msg.get("reasoning_content") or ""


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")

    # 检查服务器
    try:
        r = httpx.get(f"{RAG_URL}/health", timeout=5)
        info = r.json()
        print(f"RAG 服务器已连接 (chunks: {info['scene_count']})")
    except Exception:
        print("错误: RAG 服务器未启动，请先运行:")
        print("  python -m uvicorn src.server:app --host 0.0.0.0 --port 8787")
        return

    print(f"对话模型: {LLM_PROVIDER}")
    print("输入消息和艾莉丝聊天，输入 quit 退出，输入 /clear 重置会话")
    print("=" * 50)

    history: list[dict] = []
    context: list[str] = []

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("再见。")
            break
        if user_input == "/clear":
            history.clear()
            context.clear()
            # 重置服务器端 L4 状态
            httpx.post(f"{RAG_URL}/retrieve", json={
                "user_message": "", "sender_id": SENDER_ID,
            }, timeout=10)
            print("[会话已重置]")
            continue

        # 1. 调 RAG 获取 system_prompt
        try:
            rag_resp = httpx.post(f"{RAG_URL}/retrieve", json={
                "user_message": user_input,
                "conversation_context": context[-6:],
                "sender_id": SENDER_ID,
                "sender_nickname": SENDER_NICKNAME,
            }, timeout=120)
            rag_data = rag_resp.json()
            system_prompt = rag_data.get("enhanced_system_prompt", "")
            meta = rag_data.get("metadata", {})
        except Exception as e:
            print(f"[RAG 错误: {e}]")
            system_prompt = ""
            meta = {}

        if not system_prompt:
            print("[警告: RAG 返回空 prompt，使用默认]")
            system_prompt = "你是艾莉丝·格雷拉特，用符合角色的方式回应。"

        # Debug: 输入 /debug 切换显示 system_prompt
        if user_input == "/debug":
            debug_mode = not globals().get("_debug", False)
            globals()["_debug"] = debug_mode
            print(f"[debug {'开启' if debug_mode else '关闭'}]")
            continue

        if globals().get("_debug"):
            print(f"\n{'='*40} SYSTEM PROMPT {'='*40}")
            print(system_prompt)
            print(f"{'='*95}")

        # 2. 调 LLM 生成回复
        try:
            reply = get_reply(system_prompt, user_input, history)
        except Exception as e:
            print(f"[LLM 错误: {e}]")
            continue

        print(f"\n艾莉丝: {reply}")

        # 显示元数据
        if meta:
            tokens = meta.get("total_tokens", "?")
            l3 = meta.get("l3_scenes_used", [])
            print(f"  [tokens={tokens}, scenes={len(l3)}]")

        # 更新历史
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})
        context.append(user_input)
        context.append(reply)


if __name__ == "__main__":
    main()
