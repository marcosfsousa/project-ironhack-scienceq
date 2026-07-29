// Guards the browser floor that vite.config.ts pins in `build.target` against
// the two ways tailwind v4 has been found to breach it. Both were shipped and
// caught in review during #76, so this is a record of real failures, not theory.
//
// SCOPE — read this before citing the check as evidence of anything. It proves
// exactly two properties of the built CSS, and it is not a certificate that the
// app renders correctly at `build.target`:
//
//   1. No `var()`-valued logical shorthand. v4 builds spacing utilities on a
//      single `--spacing` base, so `px-3` emits
//      `padding-inline: calc(var(--spacing) * 3)`. Lightning CSS downlevels
//      `padding-inline` / `padding-block` to their `-start`/`-end` longhands for
//      the pinned safari14 target — Safari shipped the longhands in 12.1 but the
//      shorthands only in 14.1 — but only when it can split the value
//      statically. A `var()` inside is opaque to it, so the shorthand survives
//      and Safari 14.0 drops the declaration. index.css pins the spacing steps
//      in use inside `@theme inline` so the literal is substituted and Lightning
//      CSS can expand it; this check is what keeps that pin honest.
//
//   2. No individual `translate` / `rotate` / `scale` property. v4's transform
//      utilities emit these instead of v3's universally-supported
//      `transform: translate(…)`. They are Chrome 104 / Edge 104 / Safari 14.1,
//      above the pin on every engine, and Lightning CSS has no downlevel path
//      for them at all — static values are left alone too, so there is no
//      `@theme inline` trick here. Write the offset as a `transform:` value
//      instead; `transition-transform` covers `transform` as well. This one is
//      not Safari-only, so raising the pin to safari14.1 would not fix it.
//
// Known gaps this does NOT cover, all pre-dating the v4 port: `aspect-ratio`
// (Safari 15) and flexbox `gap` (Safari 14.1) are both emitted and both
// unsupported at the floor, in v3 as much as v4, and neither is polyfillable.
// Whether `safari14` is still worth pinning given those is a separate decision.
//
// It reads the built CSS rather than the running app on purpose — the dev server
// serves CSS without the `build.target` transform, so the Playwright suite
// cannot see either class of regression.

import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

const FRONTEND = join(import.meta.dirname, "..");
const ASSETS = join(FRONTEND, "dist", "assets");

// Longest names first so a match reports `scroll-padding-inline` rather than the
// `padding-inline` inside it. No lookbehind: an earlier attempt used `(?<!-)` to
// avoid matching mid-name, which instead blinded the check to the whole
// `scroll-*` family. It was never needed — `\s*:` already cannot match the
// `-start` / `-end` longhands, which are the forms that are actually fine.
const SHORTHANDS = [
  "scroll-padding-inline",
  "scroll-padding-block",
  "scroll-margin-inline",
  "scroll-margin-block",
  "padding-inline",
  "padding-block",
  "margin-inline",
  "margin-block",
  "inset-inline",
  "inset-block",
];
const VAR_SHORTHAND = new RegExp(`\\b(${SHORTHANDS.join("|")})\\s*:\\s*([^;}]*var\\([^;}]*)`, "g");

// Individual transform properties, at a declaration boundary so `transition-
// property: transform,translate,…` (a value, harmless) does not trip it.
const TRANSFORM_PROP = /[{;]\s*(translate|rotate|scale)\s*:\s*([^;}]+)/g;

let files;
try {
  files = readdirSync(ASSETS).filter((f) => f.endsWith(".css"));
} catch {
  console.error(`check-css-floor: no build output at ${ASSETS} — run \`npm run build\` first.`);
  process.exit(1);
}

if (files.length === 0) {
  console.error(`check-css-floor: no .css in ${ASSETS} — run \`npm run build\` first.`);
  process.exit(1);
}

// Run standalone against a `dist` older than the sources and every assertion
// below is vacuous. CI orders Build before this, so this only bites locally.
const newestCss = Math.max(...files.map((f) => statSync(join(ASSETS, f)).mtimeMs));
const sources = [join(FRONTEND, "src"), join(FRONTEND, "vite.config.ts")];
const newestSource = (function walk(paths) {
  let newest = 0;
  for (const p of paths) {
    let st;
    try {
      st = statSync(p);
    } catch {
      continue;
    }
    if (st.isDirectory()) {
      newest = Math.max(newest, walk(readdirSync(p).map((c) => join(p, c))));
    } else {
      newest = Math.max(newest, st.mtimeMs);
    }
  }
  return newest;
})(sources);

if (newestSource > newestCss) {
  console.error(
    `check-css-floor: dist/assets is older than src/ — it does not reflect the current\n` +
      `source, so checking it would pass or fail for the wrong reasons. Run \`npm run build\`.\n`
  );
  process.exit(1);
}

const shorthandHits = [];
const transformHits = [];
for (const file of files) {
  const css = readFileSync(join(ASSETS, file), "utf8");
  for (const [, prop, value] of css.matchAll(VAR_SHORTHAND)) {
    shorthandHits.push(`  ${prop}: ${value.trim()}`);
  }
  for (const [, prop, value] of css.matchAll(TRANSFORM_PROP)) {
    transformHits.push(`  ${prop}: ${value.trim()}`);
  }
}

const problems = [];
if (shorthandHits.length > 0) {
  problems.push(
    `${shorthandHits.length} logical shorthand(s) survived downleveling for the pinned\n` +
      `build.target. Safari 14.0 drops these declarations and renders without that spacing:\n\n` +
      [...new Set(shorthandHits)].sort().join("\n") +
      `\n\nFix: pin the spacing step(s) involved in the \`@theme inline\` block in src/index.css,\n` +
      `so the literal is inlined and Lightning CSS can expand the shorthand.`
  );
}
if (transformHits.length > 0) {
  problems.push(
    `${transformHits.length} individual transform propert(y/ies) emitted. These are Chrome 104 /\n` +
      `Edge 104 / Safari 14.1 — above the pin on every engine — and cannot be downleveled:\n\n` +
      [...new Set(transformHits)].sort().join("\n") +
      `\n\nFix: write the offset as a \`transform:\` value instead of a translate-*/scale-*/rotate-*\n` +
      `utility, e.g. \`[transform:translateX(-100%)]\`. \`transition-transform\` covers it.\n` +
      `If no element uses such a utility, check your comments: v4 scans raw text, so naming a\n` +
      `real utility in prose regenerates it. Write it with a \`*\` instead.`
  );
}

if (problems.length > 0) {
  console.error(`check-css-floor:\n\n${problems.join("\n\n")}\n\nSee the comment at the top of this script.\n`);
  process.exit(1);
}

console.log(
  `check-css-floor: ok — ${files.length} stylesheet(s), no var()-valued logical shorthands and no ` +
    `individual translate/rotate/scale properties.`
);
