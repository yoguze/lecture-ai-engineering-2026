"""第1回: Gemini APIに質問するスクリプト（連続対話にも対応）。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"  # 従来の1回だけ質問
    uv run python -m app.ask                      # 従来の既定の質問
    uv run python -m app.ask --chat               # 会話履歴を保持して連続対話

--chat では exit または quit で終了する。
会話履歴は実行中のチャットセッション内に保持され、プログラム終了時に消える。
"""

import argparse

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def show_token_usage(response) -> None:
    """APIのレスポンスに含まれる入力・出力トークン数を表示する。"""
    usage = response.usage_metadata
    if usage is not None:
        print(f"[入力トークン数] {usage.prompt_token_count}")
        print(f"[出力トークン数] {usage.candidates_token_count}")


def ask(question: str) -> str:
    """従来どおり、会話履歴を使わずに1回だけ質問する。"""
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=MODEL, contents=question)
    show_token_usage(response)
    return response.text or ""


def chat() -> None:
    """会話履歴を保持して対話する。"""
    client = genai.Client(api_key=get_api_key())
    session = client.chats.create(model=MODEL)
    print("チャット開始（終了するには exit または quit と入力）")

    while True:
        try:
            question = input("\nあなた: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nチャットを終了します。")
            break

        if question.lower() in ("exit", "quit"):
            print("チャットを終了します。")
            break
        if not question:
            continue

        try:
            response = session.send_message(question)
        except errors.APIError as e:
            print(api_error_message(e.code, e.message))
            continue

        print(f"\nGemini: {response.text or '（回答テキストなし）'}")
        show_token_usage(response)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLMへの単発質問・連続対話")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--chat", action="store_true", help="会話履歴を保持して連続対話する")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        if args.chat:
            chat()
        else:
            print(f"[質問] {args.question}\n")
            print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
