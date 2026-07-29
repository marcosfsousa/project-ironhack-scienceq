// Tailwind v4 ships its PostCSS plugin as its own package; the `tailwindcss`
// key that worked in v3 is not a plugin any more. Vendor prefixing is handled
// inside Tailwind now, so `autoprefixer` is gone from here and from
// devDependencies rather than left in to do nothing.
export default {
  plugins: {
    "@tailwindcss/postcss": {},
  },
};
