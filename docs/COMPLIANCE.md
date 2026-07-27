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
| Art. 50(1)/(5) — inform users they are interacting with an AI system, before first interaction, accessibly | Shipped — landing-page disclosure + per-answer AI badge cover first interaction; the chat path now answers identity questions deterministically | [#15](https://github.com/marcosfsousa/project-ironhack-scienceq/issues/15) |
| Art. 50(2) — machine-readable marking of AI-generated output | Provenance substrate first, full marking to follow | [#18](https://github.com/marcosfsousa/project-ironhack-scienceq/issues/18) |
| Art. 50(3), 50(4) — emotion recognition / deepfakes / published public-interest text | Not triggered by the current product (see tripwires) | — |
| High-risk classification (Art. 6 / Annex III) | Not applicable | — |

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
