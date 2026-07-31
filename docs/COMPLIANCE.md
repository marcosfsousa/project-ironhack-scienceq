# Compliance notes — EU AI Act (Regulation (EU) 2024/1689)

ScienceQ is an AI system that interacts directly with users and generates text, so the
transparency obligations of **Article 50** apply to it. This file records the dates, the
standing rules, and — most importantly — the **feature tripwires** that must trigger
re-assessment before certain product changes ship.

A detailed internal assessment (2026-07) sits behind this summary; it is maintained
locally and not tracked in this repository.

## Status

| Obligation | Status | Tracking |
|---|---|---|
| Art. 50(1)/(5) — inform users they are interacting with an AI system, before first interaction, accessibly | UI surfaces shipped (landing-page disclosure + per-answer AI badge) and satisfy first-interaction on their own; the chat-path identity intent is implemented, unit-tested, and confirmed by the [live probe on the regex-miss tail](https://github.com/marcosfsousa/project-ironhack-scienceq/pull/31#issuecomment-5096475667) (2026-07-27), which also showed non-English identity questions answered honestly in the user's own language | [#15](https://github.com/marcosfsousa/project-ironhack-scienceq/issues/15) |
| Art. 50(2) — machine-readable marking of AI-generated output | Provenance substrate first, full marking to follow | [#18](https://github.com/marcosfsousa/project-ironhack-scienceq/issues/18) |
| Art. 50(3), 50(4) — emotion recognition / deepfakes / published public-interest text | Not triggered by the current product (see tripwires); tripwire 2 considered and cleared for committed eval artifacts on 2026-07-31 (see recorded assessments) | — |
| High-risk classification (Art. 6 / Annex III) | Not applicable | — |

The React SPA is the only user-facing surface. The legacy Streamlit frontend, which
carried the Art. 50(1) disclosure only on its landing screen and lost it once a
conversation started, was retired in
[#13](https://github.com/marcosfsousa/project-ironhack-scienceq/issues/13) rather than
patched. Any new surface has to carry the disclosure in its own right.

## Key dates

- **2 August 2026** — Article 50 applies. The Art. 50(1) disclosure duty has **no
  transitional relief**.
- **2 December 2026** — end of the transitional window for Art. 50(2) marking
  obligations for generative AI systems already on the market (per the adopted
  Digital Omnibus text; not yet published in the Official Journal).
- **On Omnibus OJ publication** — re-verify the transitional provision's exact wording,
  in particular whether it covers systems "placed on the market" only or also those
  "put into service" before 2 August 2026. The covered population matters here.

Authoritative texts: Regulation (EU) 2024/1689 (OJ L, 2024/1689); Commission
Guidelines on Article 50, **C(2026) 5054 final, 20 July 2026** (cite the final text —
point numbering changed from the consultation draft); Code of Practice on marking and
labelling of AI-generated content, final 10 June 2026.

## Feature tripwires — re-assess BEFORE shipping any of these

The following features would change which Article 50 paragraphs apply. None of them
may ship without re-running the Article 50 assessment:

1. **Shareable answer permalinks** — public URLs to generated answers.
2. **A public answer archive or history** — generated answers viewable outside the
   authoring session.
3. **SEO-indexed Q&A pages** — generated answers served to crawlers.
4. **Auto-posting answers** to social or any external channel.

Why: each of these makes generated text **published**, which triggers Art. 50(4)'s
regime for AI-generated text published to inform the public on matters of public
interest. Its exemption requires human editorial review, which this product does not
have.

Also re-assess on: adding emotion recognition or any biometric feature (Art. 50(3));
generating audio, image, or video output (Art. 50(4) deepfake regime — its modality
list is image/audio/video); any feature resembling education assessment or scoring
(Annex III area 3, high-risk).

## Recorded assessments

Conclusions reached when a tripwire was considered and cleared. From 2026-07-31, a
tripwire cleared without an entry here has not been cleared — the point of this section
is that the reasoning is auditable rather than implicit in whoever last thought about
it.

The rule runs forward only. Two earlier conclusions sit outside it rather than being
unmade by it: the detailed 2026-07 assessment named at the top of this file, which is
maintained off-repo, and the standing "not triggered by the current product" position in
the Art. 50(3)/(4) status row. Each is revisited when its own tripwire fires, not on
this section's silence.

### 2026-07-31 — committed eval artifacts (tripwire 2)

**What was assessed.** Every tracked file under `eval/results/` that holds generated
answers — 562 of them across eight files as of this assessment, in two shapes:

- **Five `run_*.json` checkpoints, 112 answers.** One per scored eval case. Three added
  2026-03-08/09, a fourth on 2026-04-10, and `run_20260730_191842.json` committed by
  [#117](https://github.com/marcosfsousa/project-ironhack-scienceq/pull/117) itself.
- **Three sweep files, 450 answers.** `reranker_sweep_20260409_133416.json` (2 combos),
  `sweep_retrieval_stage1_20260410_100224.json` (11) and
  `sweep_retrieval_stage2_20260410_114731.json` (5) — the Phase 5 tuning runs, which
  answer the same 25 cases once per configuration combination. Four times the volume of
  the checkpoint files, and outside the first version of this entry.

The two `.csv` files beside them carry aggregate scores only and no generated text.

#117's `.gitignore` rule tracks each future `run_*.json`, which makes that half
systematic rather than incidental; the three sweep files are a closed set predating the
rule, and no future sweep output is tracked by it. Raised in review of #117 against
tripwire 2, "a public answer archive or history", and widened to the sweep artifacts in
review of [#121](https://github.com/marcosfsousa/project-ironhack-scienceq/pull/121).

**Conclusion: Art. 50(4) does not attach.** Its second subparagraph applies to text
"published with the purpose of informing the public on matters of public interest".
These artifacts are engineering records of system behaviour under measurement — the
generated text is the object being scored, not content offered to any reader as
information. The purpose test is not met, so the obligation does not attach and the
human-editorial-review exemption is not reached.

Both shapes qualify in exactly the same sense, and the sweeps make the point plainer
than the checkpoints do: one question answered eleven times under eleven retrieval
configurations is not addressed to a reader at all. No one of those answers is offered
as the answer — only the comparison between them carries meaning, which is what a
measurement is. Volume does not change the purpose test, so the 450 do not sit
differently from the 112.

**This conclusion rests entirely on purpose, so it does not transfer.** It covers eval
artifacts only. Surfacing the same generated answers as content — a rendered results
page, a docs page quoting answers as examples, anything crawlable — is a different
purpose and re-fires tripwires 1–3. Re-assess rather than citing this entry.

**Not the reason.** Per the standing rule below, the repository being public and
open-source is not a basis for this conclusion and must never be recorded as one. The
conclusion would be identical in a private repository; publication is what raised the
question, not what answers it.

**Left open.** These artifacts are AI-generated text without machine-readable marking.
Art. 50(2) marking is tracked separately in
[#18](https://github.com/marcosfsousa/project-ironhack-scienceq/issues/18); whether
committed eval outputs fall within that work is a question for it, not for this entry.

## Standing rules

- **Open-source licensing does not affect these obligations.** Article 50 is a named
  exception to the AI Act's free-and-open-source exclusion (Art. 2(12)). The public
  repository must never be recorded as a reason the obligations don't apply.
- **Using third-party models does not shift the duties upstream.** ScienceQ is the
  provider of the AI *system*; Article 50 attaches in that capacity regardless of who
  trained or hosts the underlying models (Art. 50(2) covers systems "including
  general-purpose AI systems").
- **Compliance with Article 50 does not make a use lawful under other law** (Art.
  50(6)). Copyright, platform terms, and data protection are separate surfaces with
  their own tracking: [#16](https://github.com/marcosfsousa/project-ironhack-scienceq/issues/16),
  [#17](https://github.com/marcosfsousa/project-ironhack-scienceq/issues/17).

---

*This file is an engineering record, not legal advice.*
