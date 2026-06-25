import type { Config } from "tailwindcss";

/**
 * Accent colors resolve through CSS variables (--acc / --acc-soft / --acc-bd)
 * set in index.css per [data-accent]. This keeps runtime accent-switching
 * working with static Tailwind classes (text-accent, bg-accent-soft, …).
 */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          DEFAULT: "#0a0c0f", // page
          sidebar: "#0d1014",
          card: "#101419", // source cards / video card
          panel: "#14181e", // composer, chips, ingest url box
          raised: "#12161c", // suggestion buttons
          hover: "#131820",
          chip: "#1b2129",
        },
        line: {
          DEFAULT: "rgba(255,255,255,0.06)",
          strong: "rgba(255,255,255,0.10)",
        },
        accent: {
          DEFAULT: "var(--acc)",
          soft: "var(--acc-soft)",
          bd: "var(--acc-bd)",
        },
        ok: { DEFAULT: "#2bb673", soft: "#6ee7a8" },
        // Muted text ramp lifted verbatim from the prototype.
        mut: {
          100: "#e7e9ec",
          200: "#d2d6dc",
          250: "#d6d9de",
          300: "#b9bdc5",
          400: "#9aa0a8",
          500: "#8a9099",
          600: "#767c85",
          700: "#5d636c",
          800: "#565c64",
        },
      },
      fontFamily: {
        sans: ["IBM Plex Sans", "system-ui", "sans-serif"],
        mono: ["IBM Plex Mono", "ui-monospace", "monospace"],
      },
      keyframes: {
        blink: { "0%,49%": { opacity: "1" }, "50%,100%": { opacity: "0" } },
        pulse: {
          "0%,100%": { transform: "scale(.7)", opacity: ".5" },
          "50%": { transform: "scale(1.1)", opacity: "1" },
        },
        fadeUp: {
          from: { opacity: "0", transform: "translateY(7px)" },
          to: { opacity: "1", transform: "none" },
        },
        slideIn: {
          from: { transform: "translateX(-100%)" },
          to: { transform: "translateX(0)" },
        },
        barGrow: { from: { transform: "scaleX(0)" }, to: { transform: "scaleX(1)" } },
      },
      animation: {
        blink: "blink 1.05s steps(1) infinite",
        pulse2: "pulse 1s ease-in-out infinite",
        spin2: "spin .7s linear infinite",
        fadeUp: "fadeUp .32s ease both",
        slideIn: "slideIn .32s cubic-bezier(.22,.61,.36,1)",
        barGrow: "barGrow .5s ease both",
      },
    },
  },
  plugins: [],
} satisfies Config;
