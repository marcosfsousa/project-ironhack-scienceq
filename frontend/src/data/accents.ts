import type { Accent } from "@/types";

// The accent order is load-bearing: ChatView's radiogroup walks this array with
// arrow keys and wraps at both ends, so the sequence *is* the keyboard contract.
// It lives here rather than in ChatView.tsx so the Playwright specs can import
// it — tsconfig.node.json (which typechecks tests/) has no `jsx` setting, so a
// spec cannot import a .tsx module.
export const ACCENTS: Accent[] = ["Indigo", "Blue", "Amber", "Cyan"];
