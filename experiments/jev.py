"""Resumable Jev labeling with a persistent, conservative spending ledger."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
from decimal import Decimal
from pathlib import Path

import httpx

MODEL = "jev-1.13.0"
NANODOLLARS_PER_TOKEN = 42  # $0.042 per million input tokens, verified 2026-10-08.
MAX_INPUT_TOKENS = 64_000
RESERVATION = MAX_INPUT_TOKENS * NANODOLLARS_PER_TOKEN


def load_key(path: Path) -> str:
    for name in ("TYPESAFE_API_KEY", "JEV_API_KEY"):
        if os.environ.get(name):
            return os.environ[name].strip()
    content = path.read_text().strip()
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) == 1 and "=" not in lines[0] and not lines[0].startswith("#"):
        return lines[0]
    for line in lines:
        if line.startswith("export "):
            line = line[7:].strip()
        name, sep, value = line.partition("=")
        if sep and name.strip() in ("TYPESAFE_API_KEY", "JEV_API_KEY"):
            value = value.strip().strip("\"'")
            if value:
                return value
    raise ValueError("Set TYPESAFE_API_KEY in .env (a single raw token is also accepted)")


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    """Reserve the API's entire context cost *before* each HTTP attempt.

    Successful responses settle to reported usage. Failed/unknown attempts retain
    the full reservation, including after restart. Concurrent processes share the
    same SQLite transaction lock. This counts retries, not just successful calls.
    """

    def __init__(self, path: Path, budget_usd: str = "4"):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.limit = int(Decimal(budget_usd) * 1_000_000_000)
        if self.limit <= 0:
            raise ValueError("budget_usd must be positive")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY, fingerprint TEXT NOT NULL,
                charge INTEGER NOT NULL, tokens INTEGER, status TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS responses (
                fingerprint TEXT PRIMARY KEY, body TEXT NOT NULL
            );
            """
        )

    def reserve(self, fingerprint: str) -> int:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            total = self.db.execute("SELECT COALESCE(SUM(charge), 0) FROM attempts").fetchone()[0]
            if total + RESERVATION > self.limit:
                raise BudgetExceeded("Jev spending cap reached; no request was sent")
            cursor = self.db.execute(
                "INSERT INTO attempts(fingerprint,charge,status) VALUES(?,?,?)",
                (fingerprint, RESERVATION, "reserved"),
            )
            self.db.commit()
            return cursor.lastrowid
        except BaseException:
            self.db.rollback()
            raise

    def cached(self, fingerprint: str) -> dict | None:
        row = self.db.execute(
            "SELECT body FROM responses WHERE fingerprint=?", (fingerprint,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def success(self, attempt: int, fingerprint: str, body: dict) -> None:
        tokens = body["usage"]["input_tokens"]
        if type(tokens) is not int or not 0 <= tokens <= MAX_INPUT_TOKENS:
            raise ValueError("Unexpected usage: reservation retained; inspect API contract")
        with self.db:
            self.db.execute(
                "UPDATE attempts SET charge=?,tokens=?,status='success' WHERE id=?",
                (tokens * NANODOLLARS_PER_TOKEN, tokens, attempt),
            )
            self.db.execute(
                "INSERT OR REPLACE INTO responses VALUES(?,?)",
                (fingerprint, json.dumps(body)),
            )

    def failure(self, attempt: int, status: str) -> None:
        with self.db:
            self.db.execute("UPDATE attempts SET status=? WHERE id=?", (status, attempt))

    def summary(self) -> dict:
        attempts, charged, tokens, successes = self.db.execute(
            "SELECT COUNT(*),COALESCE(SUM(charge),0),COALESCE(SUM(tokens),0),"
            "COALESCE(SUM(status='success'),0) FROM attempts"
        ).fetchone()
        return {
            "attempts": attempts,
            "successful_calls": successes,
            "input_tokens": tokens,
            "successful_usage_estimated_usd": tokens * NANODOLLARS_PER_TOKEN / 1e9,
            "conservative_spend_including_unknown_usd": charged / 1e9,
            "cap_usd": self.limit / 1e9,
            "price_per_million_input_tokens_usd": 0.042,
            "pricing_verified_on": "2026-10-08",
        }

    def close(self) -> None:
        self.db.close()


def fingerprint(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


async def classify(
    client: httpx.AsyncClient, ledger: Ledger, text: str, criteria: dict, instructions: str
) -> dict:
    payload = {
        "model": MODEL,
        "state": text,
        "questions": {
            "intent": {"type": "choice", "instructions": instructions, "criteria": criteria}
        },
    }
    digest = fingerprint(payload)
    if cached := ledger.cached(digest):
        return cached
    for retry in range(3):
        attempt = ledger.reserve(digest)
        try:
            response = await client.post("/v1/systemone", json=payload)
        except httpx.HTTPError:
            ledger.failure(attempt, "transport_error")
            if retry == 2:
                raise RuntimeError("Jev transport failure; full reservation retained") from None
            await asyncio.sleep(2**retry)
            continue
        if response.status_code in (429, 500, 502, 503, 504):
            ledger.failure(attempt, f"http_{response.status_code}")
            if retry == 2:
                raise RuntimeError(f"Jev HTTP {response.status_code}; retries exhausted")
            try:
                delay = float(response.headers.get("retry-after", 2**retry))
            except ValueError:
                delay = 2**retry
            await asyncio.sleep(min(30, max(0, delay)))
            continue
        if response.status_code != 200:
            ledger.failure(attempt, f"http_{response.status_code}")
            # Never include response bodies, request headers, or keys in errors.
            raise RuntimeError(f"Jev HTTP {response.status_code}; reservation retained")
        try:
            body = response.json()
            answer = body["answers"]["intent"]
            if answer["choice"] not in criteria:
                raise ValueError("Unknown choice")
            if set(answer["probabilities"]) != set(criteria):
                raise ValueError("Incomplete probability distribution")
            ledger.success(attempt, digest, body)
        except (ValueError, KeyError, TypeError):
            ledger.failure(attempt, "invalid_response")
            raise RuntimeError("Invalid Jev response; full reservation retained") from None
        return body
    raise AssertionError("unreachable")


async def label_rows(
    rows: list[dict],
    ledger: Ledger,
    criteria: dict,
    instructions: str,
    key: str,
    concurrency: int = 8,
) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    async with httpx.AsyncClient(
        base_url="https://api.typesafe.ai",
        headers={"Authorization": f"Bearer {key}"},
        timeout=60,
        limits=httpx.Limits(max_connections=concurrency),
    ) as client:

        async def one(row: dict) -> dict:
            nonlocal completed
            async with semaphore:
                body = await classify(client, ledger, row["text"], criteria, instructions)
            completed += 1
            if completed % 100 == 0 or completed == len(rows):
                print(
                    json.dumps({"labeled": completed, "total": len(rows), **ledger.summary()}),
                    flush=True,
                )
            return {"id": row["id"], "response": body}

        tasks = [asyncio.create_task(one(row)) for row in rows]
        try:
            return await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
