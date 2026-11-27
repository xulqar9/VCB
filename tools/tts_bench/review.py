"""Blind listening test: output/review.html

One page, fully local (no external URLs, fonts or scripts). For every sentence it
shows one clip per engine variant, side by side, in an order that is shuffled
separately for each sentence, with the engine names hidden. Each clip gets 1-5
ratings for naturalness, emotion and human sounds. One button reveals the engine
names and saves the ratings to a JSON file.

The clips on the page are copies in output/review/clips/ under random file
names (a file name must not give the engine away), and their loudness is
matched so that "louder" does not read as "better". The original WAVs in
output/<engine>/ are never touched.
"""

from __future__ import annotations

import array
import base64
import hashlib
import html
import json
import math
import random
import re
import shutil
import wave
from datetime import datetime
from pathlib import Path

from .registry import ENGINES
from .report import load_variants
from .sentences import ANGLE_TAG, Sentence, sentences_fingerprint

REVIEW_HTML = "review.html"
CLIPS_DIR = Path("review") / "clips"
_CLIP_NAME = re.compile(r"^[0-9a-f]{12}\.wav$")

TARGET_RMS = 0.07          # about -23 dBFS of speech level
PEAK_LIMIT = 0.97          # never amplify into clipping
_GATE = 0.1                # windows quieter than 10% of the loudest count as pauses


# -- loudness matching (stdlib only) -------------------------------------------

def _speech_rms(samples: array.array, rate: int) -> float:
    """RMS of the parts of the clip that carry speech: 50 ms windows quieter
    than 10% of the loudest window (pauses, breaths) are ignored."""
    win = max(1, int(0.05 * rate))
    levels = []
    for i in range(0, max(len(samples) - win, 1), win):
        chunk = samples[i:i + win]
        levels.append(math.sqrt(sum(x * x for x in chunk) / len(chunk)) / 32768.0)
    if not levels:
        return 0.0
    gate = _GATE * max(levels)
    active = [v for v in levels if v >= gate]
    return math.sqrt(sum(v * v for v in active) / len(active))


def match_loudness(src: Path, dst: Path) -> bool:
    """Write a copy of a 16-bit mono WAV at a common speech level. Returns False
    (and writes nothing) for any other format, so the caller can copy it as is."""
    try:
        with wave.open(str(src), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                return False
            rate = wav.getframerate()
            samples = array.array("h")
            samples.frombytes(wav.readframes(wav.getnframes()))
    except (wave.Error, EOFError, OSError):
        return False
    if not samples:
        return False
    level = _speech_rms(samples, rate)
    peak = max(max(samples), -min(samples)) / 32768.0
    if level <= 0 or peak <= 0:
        return False
    gain = min(TARGET_RMS / level, PEAK_LIMIT / peak)
    scaled = array.array("h", (max(-32768, min(32767, int(round(x * gain)))) for x in samples))
    with wave.open(str(dst), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(scaled.tobytes())
    return True


# -- picking and blinding the clips --------------------------------------------

def pick_clip(rows: list[dict], sentence_id: str) -> dict | None:
    """The clip to review: run 1 of the sentence, or the first run that succeeded."""
    good = [r for r in rows if r["counted"] and r["status"] == "ok"
            and r["sentence_id"] == sentence_id and r.get("wav")]
    return min(good, key=lambda r: r["run"], default=None)


def _token(seed: int, variant: str, sentence_id: str) -> str:
    """Random-looking, but reproducible, file name for a clip."""
    return hashlib.sha256(f"{seed}:{variant}:{sentence_id}".encode()).hexdigest()[:12]


def _mode_label(mode: str) -> str:
    return "reference clip" if mode == "clone" else "built-in voice"


def _title(engine: str) -> str:
    spec = ENGINES.get(engine)
    return spec.title if spec else engine


def plan_review(variants: list[dict], sentences: list[Sentence], seed: int) -> list[dict]:
    """For each sentence: its clips in a shuffled order, with a letter and a
    token each. Pure data, so it can be tested without any audio."""
    plan = []
    for s in sentences:
        entries = []
        for v in variants:
            row = pick_clip(v["rows"], s.id)
            entries.append({"variant": v["variant"], "engine": v["engine"], "mode": v["mode"],
                            "title": _title(v["engine"]), "run": row["run"] if row else None,
                            "wav": row["wav"] if row else None, "token": _token(seed, v["variant"], s.id)})
        random.Random(f"review:{seed}:{s.id}").shuffle(entries)     # a fresh order for each sentence
        for letter, entry in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", entries):
            entry["letter"] = letter
        plan.append({"sentence": s, "clips": entries})
    return plan


# -- page ----------------------------------------------------------------------

_FIELDS = (("naturalness", "Naturalness"), ("emotion", "Emotion"), ("human_sounds", "Human sounds"))


def _sentence_html(text: str) -> str:
    """Escape the text and show the canonical <tags> as small chips."""
    out, last = [], 0
    for m in ANGLE_TAG.finditer(text):
        out.append(html.escape(text[last:m.start()]))
        out.append(f'<span class="tag">{html.escape(m.group(1).lower())}</span>')
        last = m.end()
    out.append(html.escape(text[last:]))
    return "".join(out)


def _rating_row(token: str, field: str, label: str, enabled: bool) -> str:
    if not enabled:
        return (f'<div class="row na"><span class="label">{label}</span>'
                '<span class="na-note">only for sentences with sounds</span></div>')
    pills = "".join(
        f'<input type="radio" id="{token}-{field}-{n}" name="{token}-{field}" value="{n}" '
        f'data-token="{token}" data-field="{field}">'
        f'<label for="{token}-{field}-{n}" title="{n}">{n}</label>' for n in range(1, 6))
    return (f'<div class="row" role="radiogroup" aria-label="{label}"><span class="label">{label}</span>'
            f'<span class="pills">{pills}</span>'
            f'<button type="button" class="clear" data-token="{token}" data-field="{field}" '
            f'aria-label="Clear {label.lower()} rating" title="Clear">&times;</button></div>')


def _clip_html(entry: dict, sentence: Sentence) -> str:
    token = entry["token"]
    head = (f'<div class="head"><span class="letter">{entry["letter"]}</span>'
            '<span class="who" hidden></span></div>')
    if not entry["wav"]:
        return (f'<article class="clip missing" data-token="{token}">{head}'
                '<p class="muted">No audio: this engine failed on this sentence.</p></article>')
    rows = "".join(_rating_row(token, f, label, f != "human_sounds" or bool(sentence.tags))
                   for f, label in _FIELDS)
    src = (CLIPS_DIR / f"{token}.wav").as_posix()
    return (f'<article class="clip" data-token="{token}">{head}'
            f'<audio controls preload="none" src="{src}"></audio><div class="rate">{rows}</div></article>')


def _section_html(item: dict, n: int) -> str:
    s: Sentence = item["sentence"]
    clips = "".join(_clip_html(c, s) for c in item["clips"])
    return (f'<section class="sentence" id="s-{s.id}"><header><span class="num">{n}</span>'
            f'<span class="chip cat-{s.category}">{html.escape(s.category)}</span>'
            f'<span class="chip">{html.escape(s.intent)}</span></header>'
            f'<p class="text">{_sentence_html(s.text)}</p><div class="grid">{clips}</div></section>')


def render_page(plan: list[dict], meta: dict) -> str:
    """The whole HTML document. `meta` is embedded as JSON; the engine names
    inside it are base64 so that viewing the source does not spoil the test."""
    sections = "".join(_section_html(item, i) for i, item in enumerate(plan, 1))
    values = {"SECTIONS": sections,
              "META": json.dumps(meta).replace("<", "\\u003c"),
              "N_SENTENCES": str(len(plan)),
              "N_VARIANTS": str(len({c["variant"] for item in plan for c in item["clips"]}))}
    return re.sub(r"%%([A-Z_]+)%%", lambda m: values[m.group(1)], _PAGE)   # one pass: text is never re-scanned


def build_review(output_dir: Path, sentences: list[Sentence], seed: int = 1234,
                 normalize: bool = True) -> Path | None:
    """Write output/review.html and the clip copies. None if there is nothing to review."""
    output_dir = Path(output_dir)
    variants, _ = load_variants(output_dir)
    variants = [v for v in variants if any(r["counted"] and r["status"] == "ok" for r in v["rows"])]
    if not variants:
        return None
    plan = plan_review(variants, sentences, seed)

    clips_dir = output_dir / CLIPS_DIR
    clips_dir.mkdir(parents=True, exist_ok=True)
    for old in clips_dir.iterdir():                       # only files this tool wrote
        if old.is_file() and _CLIP_NAME.match(old.name):
            old.unlink()
    mapping = {}
    fingerprint = hashlib.sha256()                        # identifies this exact set of clips
    for item in plan:
        for c in item["clips"]:
            if c["wav"]:
                src, dst = output_dir / c["wav"], clips_dir / f"{c['token']}.wav"
                fingerprint.update(src.read_bytes())
                if not (normalize and match_loudness(src, dst)):
                    shutil.copyfile(src, dst)
            mapping[c["token"]] = {"variant": c["variant"], "engine": c["engine"], "title": c["title"],
                                   "mode": c["mode"], "mode_label": _mode_label(c["mode"]),
                                   "sentence": item["sentence"].id, "category": item["sentence"].category,
                                   "intent": item["sentence"].intent, "letter": c["letter"],
                                   "run": c["run"], "wav": c["wav"]}
    fingerprint.update(json.dumps(mapping, sort_keys=True).encode())
    # Ratings are kept in the browser under this id: new audio gets a new id, so
    # old ratings can never be applied to clips they were not given for.
    meta = {
        "page_id": fingerprint.hexdigest()[:12],
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "sentences_sha": sentences_fingerprint(sentences), "loudness_matched": normalize,
        "mapping": base64.b64encode(json.dumps(mapping).encode("ascii")).decode("ascii"),
    }
    path = output_dir / REVIEW_HTML
    path.write_text(render_page(plan, meta), encoding="utf-8")
    return path


_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TTS blind listening test</title>
<style>
:root { color-scheme: light dark; --bg:#f5f6f8; --panel:#fff; --text:#1b2330; --muted:#667085; --line:#e2e6ed;
  --accent:#2f6fed; --on-accent:#fff; --chip:#eceff4; --ok:#18794e; --radius:12px; }
@media (prefers-color-scheme: dark) { :root { --bg:#0f1319; --panel:#171d26; --text:#e8ecf2; --muted:#98a4b6;
  --line:#2a3342; --accent:#6b97ff; --on-accent:#0b1226; --chip:#222b38; --ok:#4cc38a; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
.wrap { max-width:1180px; margin:0 auto; padding:0 20px; }
.top { padding:28px 0 12px; }
h1 { margin:0 0 6px; font-size:26px; letter-spacing:-.01em; }
.lead { margin:0 0 10px; color:var(--muted); max-width:70ch; }
details.legend { background:var(--panel); border:1px solid var(--line); border-radius:var(--radius); padding:10px 14px; max-width:70ch; }
details.legend summary { cursor:pointer; font-weight:600; }
details.legend dl { margin:10px 0 2px; } details.legend dt { font-weight:600; margin-top:8px; } details.legend dd { margin:0; color:var(--muted); }
.bar { position:sticky; top:0; z-index:5; background:color-mix(in srgb, var(--bg) 88%, transparent); backdrop-filter:blur(8px);
  border-bottom:1px solid var(--line); }
.bar .wrap { display:flex; align-items:center; gap:18px; padding-top:10px; padding-bottom:10px; flex-wrap:wrap; }
.progress { flex:1 1 240px; min-width:200px; }
.progress .text { font-size:13px; color:var(--muted); margin-bottom:4px; display:flex; justify-content:space-between; gap:10px; }
.meter { height:6px; border-radius:6px; background:var(--chip); overflow:hidden; }
.meter > span { display:block; height:100%; width:0; background:var(--accent); transition:width .2s; }
button.primary { appearance:none; border:0; background:var(--accent); color:var(--on-accent); font:inherit; font-weight:600;
  padding:10px 18px; border-radius:10px; cursor:pointer; }
button.primary:hover { filter:brightness(1.07); } button.primary:focus-visible, .pills input:focus-visible + label { outline:2px solid var(--accent); outline-offset:2px; }
.status { font-size:13px; color:var(--muted); flex-basis:100%; min-height:1em; } .status:empty { display:none; }
main { padding:18px 0 60px; }
#results { background:var(--panel); border:1px solid var(--line); border-radius:var(--radius); padding:16px 18px; margin-bottom:22px; }
#results h2 { margin:0 0 10px; font-size:18px; }
#results table { border-collapse:collapse; width:100%; } #results th, #results td { padding:7px 10px; text-align:right; border-bottom:1px solid var(--line); }
#results th:first-child, #results td:first-child { text-align:left; } #results th { font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }
.sentence { background:var(--panel); border:1px solid var(--line); border-radius:var(--radius); padding:16px 18px 18px; margin-bottom:18px; }
.sentence > header { display:flex; align-items:center; gap:8px; margin-bottom:6px; }
.num { font-weight:700; color:var(--muted); min-width:1.6em; }
.chip { background:var(--chip); border-radius:999px; padding:2px 10px; font-size:12.5px; color:var(--muted); }
.chip.cat-emotion { color:#b4531a; } .chip.cat-human { color:#7a4bd8; } @media (prefers-color-scheme: dark) { .chip.cat-emotion { color:#f2a36b; } .chip.cat-human { color:#b79bff; } }
.text { font-size:18px; margin:2px 0 14px; max-width:75ch; }
.tag { background:var(--chip); border:1px solid var(--line); border-radius:6px; padding:0 6px; font-size:13px; color:var(--muted); margin:0 1px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(300px, 1fr)); gap:12px; }
.clip { border:1px solid var(--line); border-radius:10px; padding:10px 12px 12px; background:var(--bg); }
.clip .head { display:flex; align-items:center; gap:8px; margin-bottom:8px; min-height:26px; }
.letter { display:inline-grid; place-items:center; width:26px; height:26px; border-radius:50%; background:var(--accent); color:var(--on-accent); font-weight:700; font-size:13px; }
.who { font-weight:600; } .who small { color:var(--muted); font-weight:400; }
.revealed .clip .letter { background:var(--chip); color:var(--muted); }
audio { width:100%; height:40px; }
.clip.broken audio { outline:2px solid #d97706; outline-offset:2px; border-radius:8px; }
.notice { flex-basis:100%; font-size:13.5px; padding:8px 12px; border:1px solid #d97706; border-radius:8px;
  background:color-mix(in srgb, #d97706 14%, var(--panel)); }
.notice code { background:var(--chip); border-radius:5px; padding:0 5px; }
.rate { margin-top:10px; display:grid; grid-template-columns:minmax(0, 1fr); gap:6px; }
.row { display:grid; grid-template-columns:88px minmax(0, 1fr) 20px; align-items:center; gap:8px; }
.row .label { font-size:13px; color:var(--muted); }
.row.na { opacity:.55; grid-template-columns:88px minmax(0, 1fr); } .na-note { font-size:12px; }
.pills { display:flex; gap:4px; min-width:0; }
.pills input { position:absolute; opacity:0; pointer-events:none; }
.pills label { display:grid; place-items:center; flex:1 1 0; min-width:0; max-width:34px; height:30px; border:1px solid var(--line); border-radius:8px; cursor:pointer;
  background:var(--panel); font-size:13px; user-select:none; }
.pills label:hover { border-color:var(--accent); }
.pills input:checked + label { background:var(--accent); color:var(--on-accent); border-color:var(--accent); font-weight:700; }
button.clear { appearance:none; border:0; background:none; color:var(--muted); cursor:pointer; font-size:16px; line-height:1; padding:0; visibility:hidden; }
.row:hover button.clear, .row:focus-within button.clear { visibility:visible; }
.missing .muted, .muted { color:var(--muted); margin:0; }
footer { color:var(--muted); font-size:13px; padding:0 0 40px; } footer button { all:unset; color:var(--accent); cursor:pointer; text-decoration:underline; }
[hidden] { display:none !important; }
</style>
</head>
<body>
<header class="top"><div class="wrap">
  <h1>TTS blind listening test</h1>
  <p class="lead">%%N_SENTENCES%% sentences, %%N_VARIANTS%% clips each. Engine names are hidden and the order is shuffled for every sentence.
  Listen, then rate from 1 (poor) to 5 (excellent). Ratings are optional, saved in this browser as you go, and never leave this computer.</p>
  <details class="legend"><summary>What the scores mean</summary><dl>
    <dt>Naturalness</dt><dd>Does it sound like a real person talking? 1 = robotic or glitchy, 5 = indistinguishable from a human.</dd>
    <dt>Emotion</dt><dd>Does the voice carry the feeling shown above the sentence (excited, sad, whispering...)? 1 = flat or wrong, 5 = exactly right.</dd>
    <dt>Human sounds</dt><dd>Only for sentences with a sound tag: are the laugh, sigh, hesitation or breath there and believable? 1 = missing or garbled, 5 = convincing.</dd>
  </dl></details>
</div></header>
<div class="bar"><div class="wrap">
  <div class="progress"><div class="text"><span id="progress-text">0 of 0 ratings</span><span id="progress-pct"></span></div>
    <div class="meter"><span id="meter"></span></div></div>
  <button class="primary" id="reveal" type="button">Reveal engines &amp; save ratings (JSON)</button>
  <div class="status" id="status" role="status" aria-live="polite"></div>
  <div class="notice" id="audio-problem" role="alert" hidden><strong>A clip could not be loaded.</strong>
    <span id="problem-disk">VS Code's built-in browser blocks audio files that are opened from disk: run
      <code>python tools\tts_benchmark.py serve</code> and open the address it prints (any other browser can open
      this file directly). If you moved this page, keep the <code>review</code> folder next to it.</span>
    <span id="problem-served" hidden>Check that <code>serve</code> is still running, then reload the page. If a
      clip is still missing, rebuild the page with <code>python tools\tts_benchmark.py review</code> and reload it.</span></div>
</div></div>
<main class="wrap">
  <section id="results" hidden></section>
%%SECTIONS%%
</main>
<footer><div class="wrap">Clips are loudness-matched copies; the originals are in the engine folders. <button type="button" id="restart">Clear all ratings and start over</button></div></footer>
<script type="application/json" id="meta">%%META%%</script>
<script>
(() => {
  "use strict";
  const meta = JSON.parse(document.getElementById("meta").textContent);
  const mapping = JSON.parse(atob(meta.mapping));            // clip token -> engine (hidden until reveal)
  const FIELDS = [["naturalness", "Naturalness"], ["emotion", "Emotion"], ["human_sounds", "Human sounds"]];
  const KEY = "vcb-tts-review:" + meta.page_id;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

  // ---- state: kept in localStorage so a refresh or a crash loses nothing
  let state = { ratings: {}, revealed: false };
  try { Object.assign(state, JSON.parse(localStorage.getItem(KEY) || "{}")); } catch (e) { /* private mode */ }
  const persist = () => { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) { /* ignore */ } };
  const rating = (token, field) => (state.ratings[token] || {})[field] || null;

  function setRating(token, field, value) {
    const r = state.ratings[token] || (state.ratings[token] = {});
    if (value === null) delete r[field]; else r[field] = value;
    persist(); updateProgress();
  }

  // ---- progress
  const groups = () => Array.from(new Set($$('input[type="radio"]').map(i => i.name)));
  function counts() {
    const names = groups();
    return { total: names.length, done: names.filter(n => $(`input[name="${n}"]:checked`)).length };
  }
  function updateProgress() {
    const { total, done } = counts();
    $("#progress-text").textContent = `${done} of ${total} ratings`;
    $("#progress-pct").textContent = total ? Math.round(100 * done / total) + "%" : "";
    $("#meter").style.width = total ? (100 * done / total) + "%" : "0";
  }

  // ---- restore saved ratings, wire up the inputs
  for (const [token, fields] of Object.entries(state.ratings)) {
    for (const [field, value] of Object.entries(fields)) {
      const input = $(`input[name="${token}-${field}"][value="${value}"]`);
      if (input) input.checked = true;
    }
  }
  document.addEventListener("change", e => {
    const t = e.target;
    if (t.matches('input[type="radio"][data-field]')) setRating(t.dataset.token, t.dataset.field, Number(t.value));
  });
  document.addEventListener("click", e => {
    const b = e.target.closest("button.clear");
    if (!b) return;
    $$(`input[name="${b.dataset.token}-${b.dataset.field}"]`).forEach(i => { i.checked = false; });
    setRating(b.dataset.token, b.dataset.field, null);
  });
  // one clip plays at a time, so clips can be compared by ear
  document.addEventListener("play", e => {
    $$("audio").forEach(a => { if (a !== e.target) a.pause(); });
  }, true);

  // A browser that cannot load the clips (VS Code's built-in one blocks local audio files) would leave every
  // player dead without a word, so say what is wrong and what to do.
  const problem = $("#audio-problem");
  const fromDisk = location.protocol === "file:";
  $("#problem-disk").hidden = !fromDisk;
  $("#problem-served").hidden = fromDisk;
  function clipFailed(audio) {
    problem.hidden = false;
    const clip = audio.closest(".clip");
    if (clip) clip.classList.add("broken");
  }
  document.addEventListener("error", e => { if (e.target instanceof HTMLMediaElement) clipFailed(e.target); }, true);
  const firstClip = $("audio");
  if (firstClip) {      // clips load only when played, so try one now: the notice then comes before the first click
    const probe = new Audio();
    probe.preload = "metadata";
    probe.addEventListener("error", () => clipFailed(firstClip));
    probe.src = firstClip.src;
  }

  // ---- reveal
  const mean = xs => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
  const fmt = x => x === null ? "-" : x.toFixed(2);

  function averages() {
    const out = {};
    for (const [token, info] of Object.entries(mapping)) {
      const a = out[info.variant] || (out[info.variant] = { title: info.title, mode: info.mode_label, values: {}, rated: 0, silent: 0 });
      if (!info.wav) a.silent += 1;
      let any = false;
      for (const [f] of FIELDS) {
        const v = rating(token, f);
        if (v) { (a.values[f] = a.values[f] || []).push(v); any = true; }
      }
      if (any) a.rated += 1;
    }
    return Object.entries(out).map(([variant, a]) => ({
      variant, title: a.title, mode: a.mode, clips_rated: a.rated, no_audio: a.silent,
      naturalness: mean(a.values.naturalness || []), emotion: mean(a.values.emotion || []),
      human_sounds: mean(a.values.human_sounds || []),
    }));
  }

  function renderResults() {
    const rows = averages().sort((a, b) => (b.naturalness || 0) - (a.naturalness || 0));
    const box = $("#results");
    box.innerHTML = "<h2>Average ratings by engine</h2>";
    const table = document.createElement("table");
    table.innerHTML = "<thead><tr><th>Engine</th><th>Naturalness</th><th>Emotion</th><th>Human sounds</th><th>Clips rated</th><th>No audio</th></tr></thead>";
    const body = document.createElement("tbody");
    for (const r of rows) {
      const tr = document.createElement("tr");
      const name = document.createElement("td");
      name.textContent = r.title + " (" + r.mode + ")";
      tr.append(name);
      for (const v of [fmt(r.naturalness), fmt(r.emotion), fmt(r.human_sounds), String(r.clips_rated), String(r.no_audio)]) {
        const td = document.createElement("td"); td.textContent = v; tr.append(td);
      }
      body.append(tr);
    }
    table.append(body);
    box.append(table);
    box.hidden = false;
  }

  function reveal() {
    for (const clip of $$(".clip[data-token]")) {
      const info = mapping[clip.dataset.token];
      const who = $(".who", clip);
      who.innerHTML = "";
      who.append(info.title + " ");
      const small = document.createElement("small");
      small.textContent = info.mode_label;
      who.append(small);
      who.hidden = false;
    }
    document.body.classList.add("revealed");
    renderResults();
  }

  // ---- the JSON file
  function report() {
    const ratings = Object.entries(mapping).map(([token, info]) => ({
      sentence_id: info.sentence, category: info.category, intent: info.intent,
      engine: info.engine, variant: info.variant, voice: info.mode, run: info.run, clip_label: info.letter,
      wav: info.wav, audio: Boolean(info.wav), naturalness: rating(token, "naturalness"),
      emotion: rating(token, "emotion"), human_sounds: rating(token, "human_sounds"),
    }));
    return { page_id: meta.page_id, sentences_sha: meta.sentences_sha, generated_at: meta.generated_at,
             saved_at: new Date().toISOString(), scale: "1 (poor) to 5 (excellent); null = not rated",
             averages: averages(), ratings };
  }

  async function save(text, name) {
    if (window.showSaveFilePicker) {                          // Chromium: choose where to save
      try {
        const handle = await window.showSaveFilePicker({ suggestedName: name,
          types: [{ description: "JSON", accept: { "application/json": [".json"] } }] });
        const w = await handle.createWritable(); await w.write(text); await w.close();
        return "Saved " + handle.name + ".";
      } catch (err) {
        if (err && err.name === "AbortError") return "Save cancelled. Click the button again to save.";
      }
    }
    const a = document.createElement("a");                    // everywhere else: a normal download
    a.href = URL.createObjectURL(new Blob([text], { type: "application/json" }));
    a.download = name; document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    return "Downloaded " + name + " (look in your Downloads folder).";
  }

  const button = $("#reveal");
  function markRevealed() { button.textContent = "Save ratings again (JSON)"; }
  button.addEventListener("click", async () => {
    if (!state.revealed) {
      const { total, done } = counts();
      if (done < total && !confirm(`${total - done} of ${total} ratings are still empty.\nReveal the engine names anyway? You will no longer be able to rate blind.`)) return;
      state.revealed = true; persist(); reveal(); markRevealed();
    } else {
      renderResults();                                        // ratings may have changed since the reveal
    }
    $("#status").textContent = await save(JSON.stringify(report(), null, 2), "ratings.json");
  });
  $("#restart").addEventListener("click", () => {
    if (!confirm("Clear every rating on this page and hide the engine names again?")) return;
    try { localStorage.removeItem(KEY); } catch (e) { /* ignore */ }
    location.reload();
  });

  updateProgress();
  if (state.revealed) { reveal(); markRevealed(); }
})();
</script>
</body>
</html>
"""
