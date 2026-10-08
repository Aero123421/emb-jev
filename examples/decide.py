"""Run a local text, image, audio, or mixed-input decision."""

import argparse
import json

from emb_jev import Engine


def make_tasks(engine, media):
    if media == "image":
        candidates = {
            "food": "料理や食べ物の写真",
            "landscape": "自然の風景",
            "document": "書類や文書",
        }
    elif media == "audio":
        candidates = {
            "speech": "人が話している音声",
            "music": "楽器や歌による音楽",
            "rain": "雨の音",
        }
    else:
        candidates = {
            "billing": "請求、支払い、返金",
            "technical": "アプリの不具合、ログイン障害",
            "sales": "購入、見積もり",
        }
    return {"category": engine.prepare(type="choice", options=candidates)}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text")
    parser.add_argument("--image")
    parser.add_argument("--audio")
    parser.add_argument("--device")
    parser.add_argument("--offline", action="store_true")
    return parser


def setup(args):
    modalities = ["text"] + (["image"] if args.image else []) + (["audio"] if args.audio else [])
    engine = Engine(
        modalities=modalities,
        cache_path=".cache/candidates.sqlite3",
        device=args.device,
        local_files_only=args.offline,
    )
    tasks = make_tasks(engine, "image" if args.image else "audio" if args.audio else "text")
    return engine, tasks


if __name__ == "__main__":
    parser = arguments()
    args = parser.parse_args()
    if not any((args.text, args.image, args.audio)):
        parser.error("provide --text, --image, or --audio")
    engine, tasks = setup(args)
    with engine:
        results = engine.decide(tasks, text=args.text, image=args.image, audio=args.audio)
        print(
            json.dumps(
                {name: r.to_dict() for name, r in results.items()}, ensure_ascii=False, indent=2
            )
        )
