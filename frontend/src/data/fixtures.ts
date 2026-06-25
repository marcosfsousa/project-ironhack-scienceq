import type { CatalogVideo } from "@/types";

/**
 * Offline dev fixture mirroring GET /api/catalog (flat list, 50 videos).
 * useCatalog() fetches the real endpoint and falls back to this if it fails,
 * so the UI renders without a backend running.
 */
const yt = (id: string) => `https://www.youtube.com/watch?v=${id}`;

export const CATALOG_FIXTURE: CatalogVideo[] = [
  // Biology
  { video_id: "b1", topic: "Biology", title: "Genetic Engineering Will Change Everything Forever – CRISPR", channel: "Kurzgesagt – In a Nutshell", duration: "16:03", url: yt("jAhjPd4uNFY"), source: "corpus" },
  { video_id: "b2", topic: "Biology", title: "Your Gut Microbiome: The Most Important Organ You Never Knew", channel: "TED", duration: "12:18", url: yt("9DxF4cFXBdc"), source: "corpus" },
  { video_id: "b3", topic: "Biology", title: "The Making of a Theory: Darwin, Wallace, and Natural Selection", channel: "HHMI BioInteractive", duration: "9:41", url: yt("xQ9np2P3Mhc"), source: "corpus" },
  { video_id: "b4", topic: "Biology", title: "The #1 Antidote to Aging", channel: "Big Think", duration: "8:55", url: yt("Qk2rdwf0wAk"), source: "corpus" },
  { video_id: "b5", topic: "Biology", title: "Myths and Misconceptions About Evolution", channel: "TED-Ed", duration: "4:12", url: yt("mZt1Gn0R22Q"), source: "corpus" },
  { video_id: "b6", topic: "Biology", title: "How Your Body Builds Proteins", channel: "SciShow", duration: "6:30", url: yt("oefAI2x2CQM"), source: "corpus" },
  // Cognitive Science
  { video_id: "cs1", topic: "Cognitive Science", title: "How Your Brain Decides What Is Beautiful", channel: "TED", duration: "10:02", url: yt("9SeF1f6itEo"), source: "corpus" },
  { video_id: "cs2", topic: "Cognitive Science", title: "The Bias That Divides Us", channel: "Big Think", duration: "7:48", url: yt("Fbmg_K20Pp4"), source: "corpus" },
  { video_id: "cs3", topic: "Cognitive Science", title: "Predictive Coding: How the Brain Hallucinates Reality", channel: "Big Think", duration: "9:12", url: yt("lyu7v7nWzfo"), source: "corpus" },
  // Cosmology
  { video_id: "co1", topic: "Cosmology", title: "What Lies Beyond the Observable Universe?", channel: "PBS Space Time", duration: "14:35", url: yt("XBr4GkRnY04"), source: "corpus" },
  { video_id: "co2", topic: "Cosmology", title: "The Edge of the Universe", channel: "Kurzgesagt – In a Nutshell", duration: "11:20", url: yt("uD4izuDMUQA"), source: "corpus" },
  { video_id: "co3", topic: "Cosmology", title: "A Brief History of the Universe", channel: "Veritasium", duration: "18:44", url: yt("kr9Pj7-Vep0"), source: "corpus" },
  { video_id: "co4", topic: "Cosmology", title: "What Happened Before the Big Bang?", channel: "PBS Space Time", duration: "13:09", url: yt("k4SmnFCfMz0"), source: "corpus" },
  { video_id: "co5", topic: "Cosmology", title: "Dark Matter: The Situation Has Changed", channel: "Sabine Hossenfelder", duration: "9:58", url: yt("op2WfQNNbsU"), source: "corpus" },
  // Education
  { video_id: "ed1", topic: "Education", title: "The First Secret of Great Design", channel: "TED", duration: "9:30", url: yt("9gQzZsq3XzE"), source: "corpus" },
  { video_id: "ed2", topic: "Education", title: "How to Learn Anything Faster", channel: "Big Think", duration: "7:14", url: yt("Y_Q7Ks2Nm1Q"), source: "corpus" },
  // Mathematics
  { video_id: "m1", topic: "Mathematics", title: "The Essence of Calculus", channel: "3Blue1Brown", duration: "17:04", url: yt("WUvTyaaNkzM"), source: "corpus" },
  { video_id: "m2", topic: "Mathematics", title: "But What Is a Fourier Transform?", channel: "3Blue1Brown", duration: "20:57", url: yt("spUNpyF58BY"), source: "corpus" },
  { video_id: "m3", topic: "Mathematics", title: "The Map of Mathematics", channel: "Domain of Science", duration: "11:06", url: yt("OmJ-4B-mS-Y"), source: "corpus" },
  { video_id: "m4", topic: "Mathematics", title: "How Imaginary Numbers Were Invented", channel: "Veritasium", duration: "23:29", url: yt("cUzklzVXJwo"), source: "corpus" },
  { video_id: "m5", topic: "Mathematics", title: "The Riemann Hypothesis, Explained", channel: "Quanta Magazine", duration: "16:18", url: yt("zlm1aajH6gY"), source: "corpus" },
  // Neuroscience
  { video_id: "n1", topic: "Neuroscience", title: "You Can Grow New Brain Cells", channel: "TED", duration: "10:48", url: yt("B_tjKYvEziI"), source: "corpus" },
  { video_id: "n2", topic: "Neuroscience", title: "How the Brain Tells Time", channel: "SciShow", duration: "8:02", url: yt("M2_aQk9Yb6Y"), source: "corpus" },
  { video_id: "n3", topic: "Neuroscience", title: "Your Brain Wasn't Built to Hold This Much", channel: "Big Think", duration: "6:39", url: yt("ZauHo7eU1bM"), source: "corpus" },
  { video_id: "n4", topic: "Neuroscience", title: "The Neuroscience of Sleep", channel: "Veritasium", duration: "12:51", url: yt("nm1TxQj9IsQ"), source: "corpus" },
  // Philosophy
  { video_id: "p1", topic: "Philosophy", title: "The Trolley Problem in Real Life", channel: "TED-Ed", duration: "5:08", url: yt("yg16u_bzjPE"), source: "corpus" },
  { video_id: "p2", topic: "Philosophy", title: "Why Anything at All Exists", channel: "Big Think", duration: "8:20", url: yt("Bn0Ph_evyOE"), source: "corpus" },
  { video_id: "p3", topic: "Philosophy", title: "The Paradox of Free Will", channel: "Closer To Truth", duration: "11:42", url: yt("0HXmFTk0e8c"), source: "corpus" },
  { video_id: "p4", topic: "Philosophy", title: "What Is Consciousness?", channel: "Kurzgesagt – In a Nutshell", duration: "9:15", url: yt("H6u0VBqNBQ8"), source: "corpus" },
  { video_id: "p5", topic: "Philosophy", title: "Stoicism: A Practical Guide", channel: "Einzelgänger", duration: "13:33", url: yt("R9OCA6UFE-0"), source: "corpus" },
  { video_id: "p6", topic: "Philosophy", title: "The Meaning of Life", channel: "The School of Life", duration: "7:55", url: yt("M_pIK7ghGw4"), source: "corpus" },
  // Physics
  { video_id: "ph1", topic: "Physics", title: "Why No One Has Measured The Speed Of Light", channel: "Veritasium", duration: "19:42", url: yt("pTn6Ewhb27k"), source: "corpus" },
  { video_id: "ph2", topic: "Physics", title: "The Speed of Light", channel: "3Blue1Brown", duration: "12:11", url: yt("MO_Q_f1WgQI"), source: "corpus" },
  { video_id: "ph3", topic: "Physics", title: "Measuring the Speed of Light", channel: "Big Think", duration: "6:48", url: yt("Q1lL-hXO27Q"), source: "corpus" },
  { video_id: "ph4", topic: "Physics", title: "The Biggest Misconception About Gravity", channel: "Veritasium", duration: "14:20", url: yt("XRr1kaXKBsU"), source: "corpus" },
  { video_id: "ph5", topic: "Physics", title: "Quantum Entanglement, Explained", channel: "PBS Space Time", duration: "15:37", url: yt("JFozGfxmi8A"), source: "corpus" },
  { video_id: "ph6", topic: "Physics", title: "The Map of Physics", channel: "Domain of Science", duration: "8:24", url: yt("ZihywtixUYo"), source: "corpus" },
  { video_id: "ph7", topic: "Physics", title: "How Particle Accelerators Work", channel: "MinutePhysics", duration: "4:55", url: yt("nogVdcjGgdM"), source: "corpus" },
  // Psychology
  { video_id: "ps1", topic: "Psychology", title: "Your Brain on Stress and Anxiety", channel: "TED-Ed", duration: "4:31", url: yt("WuyPuH9ojCE"), source: "corpus" },
  { video_id: "ps2", topic: "Psychology", title: "The Psychology of Your Future Self", channel: "TED", duration: "6:50", url: yt("XNbaR54Gpj4"), source: "corpus" },
  { video_id: "ps3", topic: "Psychology", title: "Why We Procrastinate", channel: "Big Think", duration: "9:11", url: yt("Qvcx7Y4caQE"), source: "corpus" },
  { video_id: "ps4", topic: "Psychology", title: "The Backfire Effect", channel: "Veritasium", duration: "8:38", url: yt("ZdM5Y0gC0vE"), source: "corpus" },
  { video_id: "ps5", topic: "Psychology", title: "How Memory Works", channel: "SciShow Psych", duration: "7:22", url: yt("bSycdIx-C48"), source: "corpus" },
  { video_id: "ps6", topic: "Psychology", title: "The Marshmallow Test, Revisited", channel: "Big Think", duration: "10:05", url: yt("M0yhHKWUa0g"), source: "corpus" },
  { video_id: "ps7", topic: "Psychology", title: "Cognitive Dissonance", channel: "Sprouts", duration: "5:14", url: yt("dDQGFqNK-Ig"), source: "corpus" },
  // Technology
  { video_id: "t1", topic: "Technology", title: "How Does ChatGPT Actually Work?", channel: "Computerphile", duration: "12:40", url: yt("bSvTVREwSNw"), source: "live" },
  { video_id: "t2", topic: "Technology", title: "The Race to Build Quantum Computers", channel: "Veritasium", duration: "16:55", url: yt("-UlxHPIEVqA"), source: "corpus" },
  { video_id: "t3", topic: "Technology", title: "Why Moore's Law Is Ending", channel: "Two Minute Papers", duration: "8:17", url: yt("_9Og1J8Ra0w"), source: "corpus" },
  { video_id: "t4", topic: "Technology", title: "How GPS Works", channel: "Wendover Productions", duration: "11:48", url: yt("XEjpHmVfVOk"), source: "corpus" },
  { video_id: "t5", topic: "Technology", title: "The Internet's Hidden Infrastructure", channel: "Kurzgesagt – In a Nutshell", duration: "10:33", url: yt("-Rpc4egEUVQ"), source: "corpus" },
];

export const SUGGESTIONS: string[] = [
  "Why has no one measured the speed of light?",
  "How does natural selection actually work?",
  "What happens at the edge of the observable universe?",
  "What videos do you have on mathematics?",
];

/** Topic display order in the sidebar. */
export const TOPIC_ORDER = [
  "Biology",
  "Cognitive Science",
  "Cosmology",
  "Education",
  "Mathematics",
  "Neuroscience",
  "Philosophy",
  "Physics",
  "Psychology",
  "Technology",
];
