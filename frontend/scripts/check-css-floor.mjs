// Guards the browser floor that vite.config.ts pins in `build.target`.
//
// Tailwind v4 emits spacing utilities as logical shorthands built on a single
// `--spacing` base: `px-3` becomes `padding-inline: calc(var(--spacing) * 3)`.
// Lightning CSS downlevels `padding-inline` / `padding-block` to their
// `-start`/`-end` longhands for the pinned safari14 target — Safari shipped the
// longhands in 12.1 but the shorthands only in 14.1 — but it can only do that
// when it can split the value statically. A `var()` inside is opaque to it, so
// the shorthand survives into the bundle and Safari 14.0 drops the declaration
// entirely, taking the padding with it.
//
// index.css pins the spacing steps in use inside `@theme inline` so the literal
// is substituted into the utility and Lightning CSS can expand it. This check
// is what keeps that pin honest: use a spacing step that isn't pinned there and
// the build output regains a `var()`-valued shorthand, and this fails.
//
// It reads the built CSS rather than the running app on purpose — the dev
// server serves CSS without the `build.target` transform, so the Playwright
// suite cannot see this class of regression at all.

import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

const ASSETS = join(import.meta.dirname, "..", "dist", "assets");

// The shorthands Safari gained only in 14.1. The longhand forms are fine.
const SHORTHANDS = ["padding-inline", "padding-block", "margin-inline", "margin-block", "inset-inline", "inset-block"];
const PATTERN = new RegExp(`(?<!-)\\b(${SHORTHANDS.join("|")})\\s*:\\s*([^;}]*var\\([^;}]*)`, "g");

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

const offenders = [];
for (const file of files) {
  const css = readFileSync(join(ASSETS, file), "utf8");
  for (const [, prop, value] of css.matchAll(PATTERN)) {
    offenders.push({ file, decl: `${prop}: ${value.trim()}` });
  }
}

if (offenders.length > 0) {
  const unique = [...new Set(offenders.map((o) => `  ${o.decl}`))].sort();
  console.error(
    `check-css-floor: ${offenders.length} logical shorthand(s) survived downleveling for the pinned\n` +
      `build.target. Safari 14.0 will drop these declarations and render without that spacing:\n\n` +
      unique.join("\n") +
      `\n\nFix: pin the spacing step(s) involved in the \`@theme inline\` block in src/index.css,\n` +
      `so the literal is inlined and Lightning CSS can expand the shorthand. See the comment there.\n`
  );
  process.exit(1);
}

console.log(`check-css-floor: ok — no var()-valued logical shorthands in ${files.length} stylesheet(s).`);
