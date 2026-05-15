const PptxGenJS = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const pptx = new PptxGenJS();

// ─── Configuration ───────────────────────────────────────────────────────────
pptx.layout = "LAYOUT_WIDE"; // 13.33" x 7.5"
pptx.author = "RADAR Team";
pptx.title = "RADAR - Real-Time Vehicle Analysis System";

// Color palette - Midnight Executive with warm accents
const C = {
  navy:       "1E2761",
  darkNavy:   "0F1535",
  steelBlue:  "2E86AB",
  coral:      "E8553D",
  gold:       "F0A500",
  white:      "FFFFFF",
  offWhite:   "F4F6F9",
  lightGray:  "E8ECF1",
  medGray:    "8B95A5",
  darkGray:   "3A3F47",
  green:      "28A745",
  lightGreen: "D4EDDA",
  lightCoral: "F8D7DA",
  lightBlue:  "D1ECF1",
  accent2:    "6C63FF",
};

// Fonts
const F = {
  title:  "Arial Black",
  header: "Calibri",
  body:   "Calibri",
  mono:   "Consolas",
};

// Dimensions
const W = 13.33;
const H = 7.5;

// ─── Helper Functions ────────────────────────────────────────────────────────

function addBg(slide, color) {
  slide.background = { color: color };
}

function addGradientBg(slide) {
  slide.background = { fill: { type: "solid", color: C.darkNavy } };
  // Overlay shape for gradient effect
  slide.addShape(pptx.shapes.RECTANGLE, {
    x: 0, y: 0, w: W, h: H,
    fill: { type: "solid", color: C.darkNavy },
  });
  // Top accent bar
  slide.addShape(pptx.shapes.RECTANGLE, {
    x: 0, y: 0, w: W, h: 0.06,
    fill: { type: "solid", color: C.coral },
  });
}

function addDarkBg(slide) {
  slide.addShape(pptx.shapes.RECTANGLE, {
    x: 0, y: 0, w: W, h: H,
    fill: { type: "solid", color: C.darkNavy },
  });
  // Subtle gradient bar at bottom
  slide.addShape(pptx.shapes.RECTANGLE, {
    x: 0, y: H - 0.06, w: W, h: 0.06,
    fill: { type: "solid", color: C.steelBlue },
  });
}

function addLightBg(slide) {
  slide.addShape(pptx.shapes.RECTANGLE, {
    x: 0, y: 0, w: W, h: H,
    fill: { type: "solid", color: C.offWhite },
  });
  // Top accent bar
  slide.addShape(pptx.shapes.RECTANGLE, {
    x: 0, y: 0, w: W, h: 0.05,
    fill: { type: "solid", color: C.navy },
  });
  // Footer bar
  slide.addShape(pptx.shapes.RECTANGLE, {
    x: 0, y: H - 0.35, w: W, h: 0.35,
    fill: { type: "solid", color: C.navy },
  });
  // Footer text
  slide.addText("RADAR — Real-Time Vehicle Analysis System", {
    x: 0.5, y: H - 0.32, w: 8, h: 0.3,
    fontSize: 9, color: C.medGray, fontFace: F.body,
  });
}

function addSectionHeader(slide, number, title, subtitle) {
  addDarkBg(slide);
  // Large section number
  slide.addText(number, {
    x: 0.8, y: 1.5, w: 2.5, h: 2.5,
    fontSize: 96, fontFace: F.title, color: C.coral,
    bold: true, transparency: 20,
  });
  // Title
  slide.addText(title, {
    x: 3.2, y: 2.0, w: 9, h: 1.2,
    fontSize: 40, fontFace: F.title, color: C.white,
    bold: true,
  });
  // Subtitle
  if (subtitle) {
    slide.addText(subtitle, {
      x: 3.2, y: 3.3, w: 9, h: 0.8,
      fontSize: 18, fontFace: F.body, color: C.steelBlue,
    });
  }
  // Decorative line
  slide.addShape(pptx.shapes.RECTANGLE, {
    x: 3.2, y: 3.1, w: 2.5, h: 0.04,
    fill: { type: "solid", color: C.coral },
  });
}

function addStatCard(slide, x, y, value, label, color) {
  // Card background
  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x: x, y: y, w: 2.5, h: 1.5,
    fill: { type: "solid", color: C.white },
    shadow: { type: "outer", blur: 6, offset: 2, color: "000000", opacity: 0.15 },
    rectRadius: 0.1,
  });
  // Value
  slide.addText(value, {
    x: x, y: y + 0.15, w: 2.5, h: 0.8,
    fontSize: 32, fontFace: F.title, color: color || C.navy,
    bold: true, align: "center",
  });
  // Label
  slide.addText(label, {
    x: x, y: y + 0.9, w: 2.5, h: 0.4,
    fontSize: 11, fontFace: F.body, color: C.medGray,
    align: "center",
  });
}

// ─── SLIDE 1: Title ─────────────────────────────────────────────────────────
let slide = pptx.addSlide();
addDarkBg(slide);

// Decorative shapes
slide.addShape(pptx.shapes.RECTANGLE, {
  x: 0, y: 0, w: 0.15, h: H,
  fill: { type: "solid", color: C.coral },
});

// Main title
slide.addText("RADAR", {
  x: 1, y: 1.2, w: 11, h: 1.8,
  fontSize: 80, fontFace: F.title, color: C.white,
  bold: true,
});
// Subtitle
slide.addText("Real-Time Vehicle Analysis System", {
  x: 1, y: 3.0, w: 11, h: 0.8,
  fontSize: 28, fontFace: F.body, color: C.steelBlue,
});
// Tagline
slide.addText("Egyptian License Plate Detection  ·  OCR  ·  Seatbelt & Mobile Detection", {
  x: 1, y: 4.0, w: 11, h: 0.6,
  fontSize: 16, fontFace: F.body, color: C.medGray,
});
// Decorative line
slide.addShape(pptx.shapes.RECTANGLE, {
  x: 1, y: 3.85, w: 4, h: 0.04,
  fill: { type: "solid", color: C.coral },
});

// Tech stack pills
const techs = ["YOLOv11m", "YOLO26m", "YOLO26s", "Streamlit", "LapSRN", "Real-ESRGAN"];
techs.forEach((tech, i) => {
  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x: 1 + i * 1.85, y: 5.0, w: 1.7, h: 0.4,
    fill: { type: "solid", color: C.navy },
    rectRadius: 0.05,
    line: { color: C.steelBlue, width: 1 },
  });
  slide.addText(tech, {
    x: 1 + i * 1.85, y: 5.0, w: 1.7, h: 0.4,
    fontSize: 10, fontFace: F.mono, color: C.steelBlue,
    align: "center", valign: "middle",
  });
});

// ─── SLIDE 2: The Problem ────────────────────────────────────────────────────
slide = pptx.addSlide();
addLightBg(slide);

slide.addText("The Problem", {
  x: 0.7, y: 0.4, w: 5, h: 0.7,
  fontSize: 32, fontFace: F.title, color: C.navy, bold: true,
});

// Problem cards
const problems = [
  { icon: "🚗", title: "Traffic Enforcement", desc: "Manual monitoring of vehicles, license plates, and driver behavior is slow, error-prone, and requires significant human resources." },
  { icon: "🔍", title: "License Plate OCR", desc: "Egyptian plates have unique Arabic characters and dual-language format (Arabic + numbers). Traditional OCR struggles with this complexity." },
  { icon: "📱", title: "Driver Safety", desc: "Seatbelt violations and mobile phone usage while driving are leading causes of accidents. Manual detection is inconsistent." },
  { icon: "⚡", title: "Real-Time Processing", desc: "Traffic systems need fast, automated analysis that can process live feeds and images efficiently." },
];

problems.forEach((p, i) => {
  const col = i % 2;
  const row = Math.floor(i / 2);
  const x = 0.7 + col * 6.2;
  const y = 1.5 + row * 2.7;

  // Card
  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x: x, y: y, w: 5.8, h: 2.3,
    fill: { type: "solid", color: C.white },
    shadow: { type: "outer", blur: 6, offset: 2, color: "000000", opacity: 0.1 },
    rectRadius: 0.1,
  });
  // Left accent bar
  slide.addShape(pptx.shapes.RECTANGLE, {
    x: x, y: y + 0.2, w: 0.06, h: 1.9,
    fill: { type: "solid", color: C.coral },
  });
  // Icon
  slide.addText(p.icon, {
    x: x + 0.3, y: y + 0.2, w: 0.6, h: 0.5,
    fontSize: 24,
  });
  // Title
  slide.addText(p.title, {
    x: x + 0.9, y: y + 0.25, w: 4.5, h: 0.4,
    fontSize: 18, fontFace: F.header, color: C.navy, bold: true,
  });
  // Description
  slide.addText(p.desc, {
    x: x + 0.3, y: y + 0.8, w: 5.2, h: 1.3,
    fontSize: 13, fontFace: F.body, color: C.darkGray,
    lineSpacingMultiple: 1.3,
  });
});

// ─── SLIDE 3: Our Solution - System Architecture ───────────────────────────
slide = pptx.addSlide();
addSectionHeader(slide, "01", "System Architecture", "How RADAR processes images end-to-end");

// ─── SLIDE 4: Pipeline Flow ──────────────────────────────────────────────────
slide = pptx.addSlide();
addLightBg(slide);

slide.addText("End-to-End Pipeline", {
  x: 0.7, y: 0.4, w: 6, h: 0.7,
  fontSize: 32, fontFace: F.title, color: C.navy, bold: true,
});

// Pipeline steps
const steps = [
  { num: "1", title: "Image Input", desc: "Upload photo or capture from video", color: C.steelBlue },
  { num: "2", title: "Car Detection", desc: "YOLO26s detects vehicles (car, bus, truck)", color: C.navy },
  { num: "3", title: "Interior Analysis", desc: "Seatbelt & mobile phone detection", color: C.accent2 },
  { num: "4", title: "Plate Detection", desc: "YOLOv11m locates license plates", color: C.coral },
  { num: "5", title: "Super Resolution", desc: "LapSRN & Real-ESRGAN enhance crops", color: C.gold },
  { num: "6", title: "OCR", desc: "3 models read characters (Arabic + numbers)", color: C.green },
];

steps.forEach((s, i) => {
  const x = 0.5 + i * 2.1;
  const y = 1.5;

  // Connector arrow
  if (i > 0) {
    slide.addShape(pptx.shapes.RIGHT_ARROW, {
      x: x - 0.35, y: y + 0.7, w: 0.35, h: 0.35,
      fill: { type: "solid", color: C.lightGray },
    });
  }

  // Step circle
  slide.addShape(pptx.shapes.OVAL, {
    x: x + 0.35, y: y, w: 1.1, h: 1.1,
    fill: { type: "solid", color: s.color },
    shadow: { type: "outer", blur: 4, offset: 1, color: "000000", opacity: 0.2 },
  });
  slide.addText(s.num, {
    x: x + 0.35, y: y + 0.1, w: 1.1, h: 0.9,
    fontSize: 28, fontFace: F.title, color: C.white,
    bold: true, align: "center", valign: "middle",
  });

  // Title
  slide.addText(s.title, {
    x: x - 0.1, y: y + 1.3, w: 2, h: 0.4,
    fontSize: 13, fontFace: F.header, color: C.navy,
    bold: true, align: "center",
  });
  // Description
  slide.addText(s.desc, {
    x: x - 0.1, y: y + 1.7, w: 2, h: 0.7,
    fontSize: 10, fontFace: F.body, color: C.darkGray,
    align: "center", lineSpacingMultiple: 1.2,
  });
});

// Bottom info box
slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
  x: 0.7, y: 4.5, w: 11.9, h: 2.2,
  fill: { type: "solid", color: C.white },
  shadow: { type: "outer", blur: 4, offset: 2, color: "000000", opacity: 0.08 },
  rectRadius: 0.1,
});

slide.addText("Roboflow-Style Live Filtering", {
  x: 1.0, y: 4.7, w: 5, h: 0.4,
  fontSize: 18, fontFace: F.header, color: C.navy, bold: true,
});
slide.addText(
  "All models run ONCE at minimum confidence (0.01) and cache results. Sidebar sliders filter cached results client-side — no model re-run on slider change. This gives real-time interactive filtering without GPU overhead.",
  {
    x: 1.0, y: 5.2, w: 11, h: 1.2,
    fontSize: 13, fontFace: F.body, color: C.darkGray,
    lineSpacingMultiple: 1.4,
  }
);

// ─── SLIDE 5: Model Overview ─────────────────────────────────────────────────
slide = pptx.addSlide();
addSectionHeader(slide, "02", "Our Models", "Six specialized models working together");

// ─── SLIDE 6: Model Overview Cards ──────────────────────────────────────────
slide = pptx.addSlide();
addLightBg(slide);

slide.addText("Model Arsenal", {
  x: 0.7, y: 0.4, w: 6, h: 0.7,
  fontSize: 32, fontFace: F.title, color: C.navy, bold: true,
});

const models = [
  { name: "Plate Detector", arch: "YOLOv11m", task: "Locate license plates in images", classes: "1", color: C.coral },
  { name: "OCR V1", arch: "YOLOv11m", task: "Character recognition (baseline)", classes: "38", color: C.steelBlue },
  { name: "OCR V2", arch: "YOLO26m", task: "Improved character recognition", classes: "38", color: C.navy },
  { name: "OCR V2 Weighted", arch: "YOLO26m", task: "Weighted loss for better precision", classes: "38", color: C.accent2 },
  { name: "Car Detector", arch: "YOLO26s", task: "Vehicle detection (car/bus/truck)", classes: "3", color: C.green },
  { name: "Seatbelt+Mobile", arch: "YOLOv11m", task: "Driver safety violation detection", classes: "5", color: C.gold },
];

models.forEach((m, i) => {
  const col = i % 3;
  const row = Math.floor(i / 3);
  const x = 0.7 + col * 4.1;
  const y = 1.4 + row * 2.9;

  // Card
  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x: x, y: y, w: 3.8, h: 2.5,
    fill: { type: "solid", color: C.white },
    shadow: { type: "outer", blur: 5, offset: 2, color: "000000", opacity: 0.12 },
    rectRadius: 0.1,
  });
  // Top accent
  slide.addShape(pptx.shapes.RECTANGLE, {
    x: x, y: y, w: 3.8, h: 0.06,
    fill: { type: "solid", color: m.color },
  });
  // Model name
  slide.addText(m.name, {
    x: x + 0.3, y: y + 0.25, w: 3.2, h: 0.4,
    fontSize: 18, fontFace: F.header, color: C.navy, bold: true,
  });
  // Architecture badge
  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x: x + 0.3, y: y + 0.75, w: 1.6, h: 0.35,
    fill: { type: "solid", color: m.color },
    rectRadius: 0.05,
  });
  slide.addText(m.arch, {
    x: x + 0.3, y: y + 0.75, w: 1.6, h: 0.35,
    fontSize: 11, fontFace: F.mono, color: C.white,
    bold: true, align: "center", valign: "middle",
  });
  // Classes badge
  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x: x + 2.1, y: y + 0.75, w: 1.3, h: 0.35,
    fill: { type: "solid", color: C.lightGray },
    rectRadius: 0.05,
  });
  slide.addText(m.classes + " classes", {
    x: x + 2.1, y: y + 0.75, w: 1.3, h: 0.35,
    fontSize: 10, fontFace: F.body, color: C.darkGray,
    align: "center", valign: "middle",
  });
  // Task description
  slide.addText(m.task, {
    x: x + 0.3, y: y + 1.3, w: 3.2, h: 0.9,
    fontSize: 13, fontFace: F.body, color: C.darkGray,
    lineSpacingMultiple: 1.3,
  });
});

// ─── SLIDE 7: Plate Detection Model ─────────────────────────────────────────
slide = pptx.addSlide();
addSectionHeader(slide, "03", "Plate Detection", "YOLOv11m — Finding license plates in the wild");

// ─── SLIDE 8: Plate Detection Details ────────────────────────────────────────
slide = pptx.addSlide();
addLightBg(slide);

slide.addText("License Plate Detection", {
  x: 0.7, y: 0.4, w: 8, h: 0.7,
  fontSize: 30, fontFace: F.title, color: C.navy, bold: true,
});
slide.addText("YOLOv11m  |  Fine-tuned on Egyptian car plates", {
  x: 0.7, y: 1.0, w: 8, h: 0.4,
  fontSize: 14, fontFace: F.body, color: C.steelBlue,
});

// Left column - details
slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
  x: 0.7, y: 1.7, w: 5.5, h: 4.8,
  fill: { type: "solid", color: C.white },
  shadow: { type: "outer", blur: 5, offset: 2, color: "000000", opacity: 0.1 },
  rectRadius: 0.1,
});

const plateDetails = [
  { label: "Architecture", value: "YOLOv11m (medium)" },
  { label: "Input Size", value: "640 × 640 pixels" },
  { label: "Detection Target", value: "Egyptian license plates" },
  { label: "Confidence Threshold", value: "0.25" },
  { label: "Output", value: "Bounding box + confidence" },
  { label: "Post-processing", value: "Sorted widest-first for best OCR" },
];

plateDetails.forEach((d, i) => {
  const y = 2.0 + i * 0.7;
  slide.addText(d.label, {
    x: 1.1, y: y, w: 2.2, h: 0.35,
    fontSize: 12, fontFace: F.body, color: C.medGray,
    bold: true,
  });
  slide.addText(d.value, {
    x: 3.3, y: y, w: 2.5, h: 0.35,
    fontSize: 13, fontFace: F.body, color: C.navy,
  });
  if (i < plateDetails.length - 1) {
    slide.addShape(pptx.shapes.RECTANGLE, {
      x: 1.1, y: y + 0.45, w: 4.7, h: 0.01,
      fill: { type: "solid", color: C.lightGray },
    });
  }
});

// Right column - key insight
slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
  x: 6.7, y: 1.7, w: 5.9, h: 4.8,
  fill: { type: "solid", color: C.navy },
  rectRadius: 0.1,
});

slide.addText("Why Plate-First Detection?", {
  x: 7.2, y: 2.0, w: 5, h: 0.5,
  fontSize: 20, fontFace: F.header, color: C.white, bold: true,
});

const insights = [
  "Detects plates in the full image before cropping — avoids missing small plates",
  "Plates sorted by width (widest first) — best quality crop for OCR",
  "Runs at cache-min confidence (0.01) then filtered client-side at 0.25",
  "Single model handles all Egyptian plate styles (white, blue, old format)",
];

insights.forEach((ins, i) => {
  slide.addText("▸  " + ins, {
    x: 7.2, y: 2.7 + i * 0.8, w: 5, h: 0.7,
    fontSize: 12, fontFace: F.body, color: C.lightGray,
    lineSpacingMultiple: 1.3,
  });
});

// ─── SLIDE 9: OCR Section Header ─────────────────────────────────────────────
slide = pptx.addSlide();
addSectionHeader(slide, "04", "OCR Models", "Reading characters from Egyptian license plates");

// ─── SLIDE 10: OCR Challenge ─────────────────────────────────────────────────
slide = pptx.addSlide();
addLightBg(slide);

slide.addText("The OCR Challenge", {
  x: 0.7, y: 0.4, w: 8, h: 0.7,
  fontSize: 30, fontFace: F.title, color: C.navy, bold: true,
});
slide.addText("Egyptian plates have unique challenges", {
  x: 0.7, y: 1.0, w: 8, h: 0.4,
  fontSize: 14, fontFace: F.body, color: C.steelBlue,
});

// Challenge cards
const ocrChallenges = [
  { title: "Dual Language", desc: "Plates contain both Arabic letters and Latin/Arabic numerals — OCR must handle bidirectional text", icon: "🔤" },
  { title: "38 Character Classes", desc: "10 digits + 28 Arabic letters with visually similar characters (e.g., ع vs غ, ص vs ض)", icon: "📊" },
  { title: "Low Resolution", desc: "Cropped plates from far vehicles are blurry — super-resolution enhancement is critical", icon: "🔍" },
  { title: "Varied Conditions", desc: "Different lighting, angles, plate styles (white, blue), and degradation levels", icon: "🌤" },
];

ocrChallenges.forEach((ch, i) => {
  const col = i % 2;
  const row = Math.floor(i / 2);
  const x = 0.7 + col * 6.2;
  const y = 1.7 + row * 2.5;

  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x: x, y: y, w: 5.8, h: 2.1,
    fill: { type: "solid", color: C.white },
    shadow: { type: "outer", blur: 4, offset: 2, color: "000000", opacity: 0.1 },
    rectRadius: 0.1,
  });
  slide.addText(ch.icon, { x: x + 0.3, y: y + 0.2, w: 0.5, h: 0.5, fontSize: 24 });
  slide.addText(ch.title, {
    x: x + 1.0, y: y + 0.25, w: 4.3, h: 0.4,
    fontSize: 17, fontFace: F.header, color: C.navy, bold: true,
  });
  slide.addText(ch.desc, {
    x: x + 0.3, y: y + 0.8, w: 5.2, h: 1.0,
    fontSize: 13, fontFace: F.body, color: C.darkGray,
    lineSpacingMultiple: 1.3,
  });
});

// ─── SLIDE 11: OCR V2 Training Results ───────────────────────────────────────
slide = pptx.addSlide();
addLightBg(slide);

slide.addText("OCR V2 — Training Results", {
  x: 0.7, y: 0.4, w: 8, h: 0.7,
  fontSize: 30, fontFace: F.title, color: C.navy, bold: true,
});
slide.addText("YOLO26m  |  100 epochs  |  batch=16  |  AdamW  |  characters_final dataset (9,324 images)", {
  x: 0.7, y: 1.0, w: 12, h: 0.4,
  fontSize: 12, fontFace: F.body, color: C.steelBlue,
});

// Stat cards
addStatCard(slide, 0.7, 1.6, "99.2%", "mAP@50", C.coral);
addStatCard(slide, 3.3, 1.6, "77.4%", "mAP@50-95", C.steelBlue);
addStatCard(slide, 5.9, 1.6, "98.9%", "Precision", C.navy);
addStatCard(slide, 8.5, 1.6, "99.0%", "Recall", C.green);

// Training charts - use overview image
const ocrChartPath = path.join(__dirname, "charts", "ocr_v2_overview.png");
if (fs.existsSync(ocrChartPath)) {
  slide.addImage({
    path: ocrChartPath,
    x: 0.5, y: 3.4, w: 6.0, h: 3.8,
  });
}

// Right side - training details
slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
  x: 7.0, y: 3.4, w: 5.8, h: 3.8,
  fill: { type: "solid", color: C.white },
  shadow: { type: "outer", blur: 4, offset: 2, color: "000000", opacity: 0.1 },
  rectRadius: 0.1,
});

slide.addText("Training Configuration", {
  x: 7.4, y: 3.6, w: 5, h: 0.4,
  fontSize: 16, fontFace: F.header, color: C.navy, bold: true,
});

const ocrTrainDetails = [
  "Architecture: YOLO26m (medium)",
  "Dataset: characters_final — 9,324 images",
  "Image Size: 640 × 640",
  "Optimizer: AdamW, lr=0.01",
  "Augmentation: RandAugment + MixUp(0.1)",
  "Warmup: 3 epochs (removed from charts)",
  "Best epoch: 78 / 100",
  "Platform: Kaggle T4 GPU",
];

ocrTrainDetails.forEach((d, i) => {
  slide.addText("▸  " + d, {
    x: 7.4, y: 4.2 + i * 0.36, w: 5, h: 0.35,
    fontSize: 11, fontFace: F.body, color: C.darkGray,
  });
});

// ─── SLIDE 12: OCR V2 Loss & Precision Charts ───────────────────────────────
slide = pptx.addSlide();
addLightBg(slide);

slide.addText("OCR V2 — Training Curves", {
  x: 0.7, y: 0.4, w: 8, h: 0.7,
  fontSize: 30, fontFace: F.title, color: C.navy, bold: true,
});

const ocrLossPath = path.join(__dirname, "charts", "ocr_v2_loss.png");
const ocrPrPath = path.join(__dirname, "charts", "ocr_v2_precision_recall.png");

if (fs.existsSync(ocrLossPath)) {
  slide.addImage({ path: ocrLossPath, x: 0.5, y: 1.3, w: 6.0, h: 3.0 });
}
if (fs.existsSync(ocrPrPath)) {
  slide.addImage({ path: ocrPrPath, x: 6.8, y: 1.3, w: 6.0, h: 3.0 });
}

// Bottom - mAP chart
const ocrMapPath = path.join(__dirname, "charts", "ocr_v2_map.png");
if (fs.existsSync(ocrMapPath)) {
  slide.addImage({ path: ocrMapPath, x: 3.0, y: 4.5, w: 7.0, h: 2.7 });
}

// ─── SLIDE 13: OCR Model Comparison ──────────────────────────────────────────
slide = pptx.addSlide();
addLightBg(slide);

slide.addText("OCR Model Comparison", {
  x: 0.7, y: 0.4, w: 8, h: 0.7,
  fontSize: 30, fontFace: F.title, color: C.navy, bold: true,
});
slide.addText("Three OCR models run side-by-side for best accuracy", {
  x: 0.7, y: 1.0, w: 10, h: 0.4,
  fontSize: 14, fontFace: F.body, color: C.steelBlue,
});

// Comparison table header
const compModels = [
  { name: "OCR V1", arch: "YOLOv11m", desc: "Baseline character model", trained: "Egyptian plates dataset", badge: C.medGray },
  { name: "OCR V2", arch: "YOLO26m", desc: "Improved with characters_final dataset", trained: "9,324 images, 100 epochs", badge: C.green },
  { name: "OCR V2 Weighted-3", arch: "YOLO26m", desc: "Weighted BCE loss for class balance", trained: "Same data, weighted training", badge: C.steelBlue },
];

compModels.forEach((m, i) => {
  const x = 0.7 + i * 4.15;
  const y = 1.7;

  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x: x, y: y, w: 3.9, h: 4.5,
    fill: { type: "solid", color: C.white },
    shadow: { type: "outer", blur: 5, offset: 2, color: "000000", opacity: 0.12 },
    rectRadius: 0.1,
  });
  // Top accent
  slide.addShape(pptx.shapes.RECTANGLE, {
    x: x, y: y, w: 3.9, h: 0.06,
    fill: { type: "solid", color: m.badge },
  });

  slide.addText(m.name, {
    x: x + 0.3, y: y + 0.25, w: 3.3, h: 0.4,
    fontSize: 20, fontFace: F.header, color: C.navy, bold: true,
  });

  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x: x + 0.3, y: y + 0.8, w: 1.5, h: 0.3,
    fill: { type: "solid", color: m.badge },
    rectRadius: 0.05,
  });
  slide.addText(m.arch, {
    x: x + 0.3, y: y + 0.8, w: 1.5, h: 0.3,
    fontSize: 10, fontFace: F.mono, color: C.white,
    bold: true, align: "center", valign: "middle",
  });

  slide.addText(m.desc, {
    x: x + 0.3, y: y + 1.3, w: 3.3, h: 0.8,
    fontSize: 13, fontFace: F.body, color: C.darkGray,
    lineSpacingMultiple: 1.3,
  });

  // Training info
  slide.addShape(pptx.shapes.RECTANGLE, {
    x: x + 0.3, y: y + 2.2, w: 3.3, h: 0.01,
    fill: { type: "solid", color: C.lightGray },
  });
  slide.addText("Training Data", {
    x: x + 0.3, y: y + 2.4, w: 3.3, h: 0.3,
    fontSize: 10, fontFace: F.body, color: C.medGray, bold: true,
  });
  slide.addText(m.trained, {
    x: x + 0.3, y: y + 2.7, w: 3.3, h: 0.5,
    fontSize: 12, fontFace: F.body, color: C.darkGray,
  });

  // Enhancement info
  slide.addShape(pptx.shapes.RECTANGLE, {
    x: x + 0.3, y: y + 3.3, w: 3.3, h: 0.01,
    fill: { type: "solid", color: C.lightGray },
  });
  slide.addText("Each model runs on 3 versions:", {
    x: x + 0.3, y: y + 3.5, w: 3.3, h: 0.3,
    fontSize: 11, fontFace: F.body, color: C.medGray,
  });

  const versions = ["Original crop", "LapSRN 2× enhanced", "Real-ESRGAN enhanced"];
  versions.forEach((v, vi) => {
    slide.addText("●  " + v, {
      x: x + 0.3, y: y + 3.85 + vi * 0.28, w: 3.3, h: 0.25,
      fontSize: 11, fontFace: F.body, color: C.darkGray,
    });
  });
});

// Bottom note
slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
  x: 0.7, y: 6.4, w: 11.9, h: 0.5,
  fill: { type: "solid", color: C.lightBlue },
  rectRadius: 0.05,
});
slide.addText("💡  Total: 3 models × 3 enhancements = 9 OCR results per plate for maximum accuracy", {
  x: 1.0, y: 6.4, w: 11, h: 0.5,
  fontSize: 13, fontFace: F.body, color: C.navy, bold: true,
});

// ─── SLIDE 14: Enhancement Pipeline ──────────────────────────────────────────
slide = pptx.addSlide();
addSectionHeader(slide, "05", "Super Resolution", "AI-powered image enhancement for better OCR");

// ─── SLIDE 15: Enhancement Details ───────────────────────────────────────────
slide = pptx.addSlide();
addLightBg(slide);

slide.addText("AI Super-Resolution Pipeline", {
  x: 0.7, y: 0.4, w: 8, h: 0.7,
  fontSize: 30, fontFace: F.title, color: C.navy, bold: true,
});

// Enhancement models
const enhancements = [
  {
    name: "Original Crop",
    desc: "Raw plate crop from plate detector — may be blurry or pixelated, especially for distant vehicles",
    color: C.medGray,
    detail: "Baseline — no processing applied"
  },
  {
    name: "LapSRN 2×",
    desc: "Laplacian Pyramid Super-Resolution Network — upscales 2× using deep learning. Fast inference, good for mild blur.",
    color: C.steelBlue,
    detail: "LapSRN_x2.pb — can crash on certain shapes, wrapped in try/except fallback"
  },
  {
    name: "Real-ESRGAN",
    desc: "Real-World Super-Resolution GAN — handles complex degradations (blur, noise, compression artifacts). Higher quality but slower.",
    color: C.coral,
    detail: "Excels at recovering text details from degraded plate images"
  },
];

enhancements.forEach((e, i) => {
  const y = 1.5 + i * 1.85;

  // Step number circle
  slide.addShape(pptx.shapes.OVAL, {
    x: 0.9, y: y + 0.2, w: 0.7, h: 0.7,
    fill: { type: "solid", color: e.color },
  });
  slide.addText(String(i + 1), {
    x: 0.9, y: y + 0.2, w: 0.7, h: 0.7,
    fontSize: 22, fontFace: F.title, color: C.white,
    bold: true, align: "center", valign: "middle",
  });

  // Arrow connector
  if (i < 2) {
    slide.addShape(pptx.shapes.RECTANGLE, {
      x: 1.22, y: y + 0.95, w: 0.04, h: 0.9,
      fill: { type: "solid", color: C.lightGray },
    });
  }

  // Content
  slide.addText(e.name, {
    x: 2.0, y: y + 0.15, w: 5, h: 0.4,
    fontSize: 20, fontFace: F.header, color: C.navy, bold: true,
  });
  slide.addText(e.desc, {
    x: 2.0, y: y + 0.6, w: 10, h: 0.6,
    fontSize: 13, fontFace: F.body, color: C.darkGray,
    lineSpacingMultiple: 1.3,
  });
  slide.addText(e.detail, {
    x: 2.0, y: y + 1.2, w: 10, h: 0.3,
    fontSize: 11, fontFace: F.body, color: C.medGray, italic: true,
  });
});

// ─── SLIDE 16: Seatbelt Section Header ────────────────────────────────────────
slide = pptx.addSlide();
addSectionHeader(slide, "06", "Seatbelt & Mobile Detection", "Driver safety violation detection");

// ─── SLIDE 17: Seatbelt Model Details ────────────────────────────────────────
slide = pptx.addSlide();
addLightBg(slide);

slide.addText("Seatbelt & Mobile Detection", {
  x: 0.7, y: 0.4, w: 8, h: 0.7,
  fontSize: 30, fontFace: F.title, color: C.navy, bold: true,
});
slide.addText("YOLOv11m  |  80 epochs  |  batch=32  |  6,424 images  |  5 classes", {
  x: 0.7, y: 1.0, w: 12, h: 0.4,
  fontSize: 12, fontFace: F.body, color: C.steelBlue,
});

// Class breakdown
const seatbeltClasses = [
  { name: "person-noseatbelt", desc: "Driver/passenger without seatbelt", color: C.coral },
  { name: "person-seatbelt", desc: "Driver/passenger wearing seatbelt", color: C.green },
  { name: "seatbelt", desc: "Seatbelt strap visible", color: C.steelBlue },
  { name: "windshield", desc: "Windshield detection (context)", color: C.medGray },
  { name: "mobile", desc: "Mobile phone in use", color: C.gold },
];

slide.addText("Detection Classes", {
  x: 0.7, y: 1.6, w: 5, h: 0.4,
  fontSize: 18, fontFace: F.header, color: C.navy, bold: true,
});

seatbeltClasses.forEach((cl, i) => {
  const y = 2.2 + i * 0.55;
  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x: 0.9, y: y, w: 0.3, h: 0.3,
    fill: { type: "solid", color: cl.color },
    rectRadius: 0.03,
  });
  slide.addText(cl.name, {
    x: 1.4, y: y, w: 2.5, h: 0.35,
    fontSize: 12, fontFace: F.mono, color: C.navy, bold: true,
  });
  slide.addText(cl.desc, {
    x: 4.0, y: y, w: 2.5, h: 0.35,
    fontSize: 12, fontFace: F.body, color: C.darkGray,
  });
});

// Right side - dataset info
slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
  x: 7.0, y: 1.6, w: 5.8, h: 2.8,
  fill: { type: "solid", color: C.navy },
  rectRadius: 0.1,
});

slide.addText("Dataset Construction", {
  x: 7.4, y: 1.85, w: 5, h: 0.4,
  fontSize: 18, fontFace: F.header, color: C.white, bold: true,
});

const datasetInfo = [
  "Merged from 3 sources: v3 base + v2 mobile + v5 unique",
  "6,424 images at 640 × 640 resolution",
  "484 mobile phone labels",
  "Train/Valid/Test split with balanced distribution",
  "Built from v3 base (NOT v4 which was 320×320)",
];

datasetInfo.forEach((d, i) => {
  slide.addText("▸  " + d, {
    x: 7.4, y: 2.5 + i * 0.38, w: 5, h: 0.35,
    fontSize: 12, fontFace: F.body, color: C.lightGray,
  });
});

// Stat cards for seatbelt
addStatCard(slide, 0.7, 5.0, "90.0%", "mAP@50", C.coral);
addStatCard(slide, 3.3, 5.0, "54.3%", "mAP@50-95", C.steelBlue);
addStatCard(slide, 5.9, 5.0, "89.8%", "Precision", C.navy);
addStatCard(slide, 8.5, 5.0, "86.6%", "Recall", C.green);

// ─── SLIDE 18: Seatbelt Training Results ──────────────────────────────────────
slide = pptx.addSlide();
addLightBg(slide);

slide.addText("Seatbelt Model — Training Curves", {
  x: 0.7, y: 0.4, w: 8, h: 0.7,
  fontSize: 30, fontFace: F.title, color: C.navy, bold: true,
});

const seatbeltChartPath = path.join(__dirname, "charts", "seatbelt_overview.png");
if (fs.existsSync(seatbeltChartPath)) {
  slide.addImage({ path: seatbeltChartPath, x: 0.5, y: 1.2, w: 12.3, h: 5.8 });
}

// ─── SLIDE 19: Seatbelt Loss & Precision Charts ──────────────────────────────
slide = pptx.addSlide();
addLightBg(slide);

slide.addText("Seatbelt Model — Detailed Metrics", {
  x: 0.7, y: 0.4, w: 8, h: 0.7,
  fontSize: 30, fontFace: F.title, color: C.navy, bold: true,
});

const seatLossPath = path.join(__dirname, "charts", "seatbelt_loss.png");
const seatPrPath = path.join(__dirname, "charts", "seatbelt_precision_recall.png");

if (fs.existsSync(seatLossPath)) {
  slide.addImage({ path: seatLossPath, x: 0.3, y: 1.2, w: 6.3, h: 3.0 });
}
if (fs.existsSync(seatPrPath)) {
  slide.addImage({ path: seatPrPath, x: 6.8, y: 1.2, w: 6.3, h: 3.0 });
}

const seatMapPath = path.join(__dirname, "charts", "seatbelt_map.png");
if (fs.existsSync(seatMapPath)) {
  slide.addImage({ path: seatMapPath, x: 3.0, y: 4.4, w: 7.0, h: 2.7 });
}

// ─── SLIDE 20: Car Detection ──────────────────────────────────────────────────
slide = pptx.addSlide();
addSectionHeader(slide, "07", "Car Detection", "YOLO26s — Vehicle detection & tracking");

// ─── SLIDE 21: Car Detection Details ─────────────────────────────────────────
slide = pptx.addSlide();
addLightBg(slide);

slide.addText("Vehicle Detection & Tracking", {
  x: 0.7, y: 0.4, w: 8, h: 0.7,
  fontSize: 30, fontFace: F.title, color: C.navy, bold: true,
});

// Two columns
// Left - Model info
slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
  x: 0.7, y: 1.5, w: 5.8, h: 5.0,
  fill: { type: "solid", color: C.white },
  shadow: { type: "outer", blur: 5, offset: 2, color: "000000", opacity: 0.1 },
  rectRadius: 0.1,
});

slide.addText("YOLO26s Car Detector", {
  x: 1.1, y: 1.7, w: 5, h: 0.4,
  fontSize: 20, fontFace: F.header, color: C.navy, bold: true,
});

const carDetails = [
  { label: "Architecture", value: "YOLO26s (small — fast)" },
  { label: "Input Size", value: "640 × 640" },
  { label: "Detection Classes", value: "car (2), bus (5), truck (7)" },
  { label: "Tracking", value: "ByteTrack for video frame tracking" },
  { label: "Confidence", value: "0.4 threshold" },
  { label: "Purpose", value: "Locate vehicles before interior/plate analysis" },
];

carDetails.forEach((d, i) => {
  const y = 2.3 + i * 0.65;
  slide.addText(d.label, {
    x: 1.1, y: y, w: 2.2, h: 0.3,
    fontSize: 12, fontFace: F.body, color: C.medGray, bold: true,
  });
  slide.addText(d.value, {
    x: 3.3, y: y, w: 2.8, h: 0.3,
    fontSize: 13, fontFace: F.body, color: C.navy,
  });
  if (i < carDetails.length - 1) {
    slide.addShape(pptx.shapes.RECTANGLE, {
      x: 1.1, y: y + 0.4, w: 4.9, h: 0.01,
      fill: { type: "solid", color: C.lightGray },
    });
  }
});

// Right - Video pipeline info
slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
  x: 6.9, y: 1.5, w: 5.7, h: 5.0,
  fill: { type: "solid", color: C.navy },
  rectRadius: 0.1,
});

slide.addText("Video Pipeline", {
  x: 7.3, y: 1.7, w: 5, h: 0.4,
  fontSize: 20, fontFace: F.header, color: C.white, bold: true,
});

const videoPipeline = [
  "Scan all video frames for cars",
  "ByteTrack assigns track IDs per pass",
  "Deduplication: IoU > 0.5 + frame proximity < 30",
  "Match cars by spatial proximity (NOT track ID across passes)",
  "Find plate crops for each unique car",
  "Rank plate candidates by crop area",
];

videoPipeline.forEach((d, i) => {
  const num = String(i + 1);
  slide.addShape(pptx.shapes.OVAL, {
    x: 7.3, y: 2.3 + i * 0.65, w: 0.35, h: 0.35,
    fill: { type: "solid", color: C.coral },
  });
  slide.addText(num, {
    x: 7.3, y: 2.3 + i * 0.65, w: 0.35, h: 0.35,
    fontSize: 11, fontFace: F.title, color: C.white,
    bold: true, align: "center", valign: "middle",
  });
  slide.addText(d, {
    x: 7.9, y: 2.3 + i * 0.65, w: 4.5, h: 0.4,
    fontSize: 12, fontFace: F.body, color: C.lightGray,
  });
});

// ─── SLIDE 22: Training Infrastructure ────────────────────────────────────────
slide = pptx.addSlide();
addSectionHeader(slide, "08", "Training Infrastructure", "Kaggle T4 GPU — Training at scale");

// ─── SLIDE 23: Training Infrastructure Details ───────────────────────────────
slide = pptx.addSlide();
addLightBg(slide);

slide.addText("Training Infrastructure & Config", {
  x: 0.7, y: 0.4, w: 10, h: 0.7,
  fontSize: 30, fontFace: F.title, color: C.navy, bold: true,
});

// Two main columns
// Left - Hardware
slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
  x: 0.7, y: 1.5, w: 5.8, h: 5.0,
  fill: { type: "solid", color: C.white },
  shadow: { type: "outer", blur: 5, offset: 2, color: "000000", opacity: 0.1 },
  rectRadius: 0.1,
});

slide.addText("Hardware & Platform", {
  x: 1.1, y: 1.7, w: 5, h: 0.4,
  fontSize: 18, fontFace: F.header, color: C.navy, bold: true,
});

const hwInfo = [
  { label: "Platform", value: "Kaggle Notebooks (free T4 GPU)" },
  { label: "GPU", value: "NVIDIA Tesla T4 (16GB VRAM)" },
  { label: "OCR V2 batch", value: "16 (batch=32 OOMs for YOLOv11m)" },
  { label: "Seatbelt batch", value: "32 (fits for YOLOv11m at 640)" },
  { label: "OCR V2 Time", value: "~7 min/epoch × 100 = ~12 hours" },
  { label: "Seatbelt Time", value: "~4 min/epoch × 80 = ~5 hours" },
];

hwInfo.forEach((d, i) => {
  const y = 2.3 + i * 0.65;
  slide.addText(d.label, {
    x: 1.1, y: y, w: 2.5, h: 0.3,
    fontSize: 12, fontFace: F.body, color: C.medGray, bold: true,
  });
  slide.addText(d.value, {
    x: 3.6, y: y, w: 2.5, h: 0.3,
    fontSize: 13, fontFace: F.body, color: C.navy,
  });
  if (i < hwInfo.length - 1) {
    slide.addShape(pptx.shapes.RECTANGLE, {
      x: 1.1, y: y + 0.4, w: 4.9, h: 0.01,
      fill: { type: "solid", color: C.lightGray },
    });
  }
});

// Right - Key training lessons
slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
  x: 6.9, y: 1.5, w: 5.7, h: 5.0,
  fill: { type: "solid", color: C.white },
  shadow: { type: "outer", blur: 5, offset: 2, color: "000000", opacity: 0.1 },
  rectRadius: 0.1,
});

slide.addText("Lessons Learned", {
  x: 7.3, y: 1.7, w: 5, h: 0.4,
  fontSize: 18, fontFace: F.header, color: C.navy, bold: true,
});

const lessons = [
  { title: "Batch Size Matters", desc: "YOLOv11m batch=32 OOMs on T4 — use batch=16" },
  { title: "Class Weighting", desc: "Weighted BCE loss didn't improve results — normal training with more epochs was better" },
  { title: "Augmentation", desc: "RandAugment + MixUp(0.1) gave best results" },
  { title: "Warmup Removal", desc: "First 3 epochs are warmup — exclude from analysis charts" },
  { title: "Image Size", desc: "Must match training imgsz=640 or predictions break" },
  { title: "Roboflow Labels", desc: "Some exports have polygon data (>5 values/line) — truncate to first 5" },
];

lessons.forEach((l, i) => {
  const y = 2.3 + i * 0.75;
  slide.addText(l.title, {
    x: 7.3, y: y, w: 5, h: 0.3,
    fontSize: 13, fontFace: F.header, color: C.navy, bold: true,
  });
  slide.addText(l.desc, {
    x: 7.3, y: y + 0.3, w: 5, h: 0.3,
    fontSize: 11, fontFace: F.body, color: C.darkGray,
  });
});

// ─── SLIDE 24: GUI Demo section ──────────────────────────────────────────────
slide = pptx.addSlide();
addSectionHeader(slide, "09", "Live Demo", "Streamlit Web Interface");

// ─── SLIDE 25: GUI Features ──────────────────────────────────────────────────
slide = pptx.addSlide();
addLightBg(slide);

slide.addText("Streamlit Web Interface", {
  x: 0.7, y: 0.4, w: 8, h: 0.7,
  fontSize: 30, fontFace: F.title, color: C.navy, bold: true,
});

// Features grid
const guiFeatures = [
  { title: "Smart Sidebar", desc: "Real-time confidence sliders for all models (step=0.01). Model status indicators. Cache-min detection for instant filtering.", icon: "⚙️" },
  { title: "Photo Mode", desc: "Upload PNG images → detect cars as clickable cards → interior analysis (seatbelt/mobile) → plate OCR with enhancement comparison.", icon: "📷" },
  { title: "Clickable Car Grid", desc: "Detected vehicles shown as image cards with green/red plate badge. Click to analyze. Filtered by confidence slider in real-time.", icon: "🚗" },
  { title: "9-Way OCR Comparison", desc: "3 OCR models × 3 enhancement levels = 9 results per plate. Annotated images with per-class colored boxes and Franco labels.", icon: "🔤" },
  { title: "Sample Gallery", desc: "Built-in sample images gallery with compact expandable grid. Click any thumbnail to process.", icon: "🖼️" },
  { title: "Violation Badges", desc: "Colored Streamlit badges: red 'No Belt', green 'Belt', red 'Phone'. Instant visual feedback for safety violations.", icon: "🏷️" },
];

guiFeatures.forEach((f, i) => {
  const col = i % 2;
  const row = Math.floor(i / 2);
  const x = 0.7 + col * 6.2;
  const y = 1.4 + row * 1.9;

  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x: x, y: y, w: 5.8, h: 1.6,
    fill: { type: "solid", color: C.white },
    shadow: { type: "outer", blur: 4, offset: 2, color: "000000", opacity: 0.1 },
    rectRadius: 0.1,
  });

  slide.addText(f.icon, {
    x: x + 0.2, y: y + 0.15, w: 0.5, h: 0.4, fontSize: 20,
  });
  slide.addText(f.title, {
    x: x + 0.7, y: y + 0.15, w: 4.8, h: 0.35,
    fontSize: 16, fontFace: F.header, color: C.navy, bold: true,
  });
  slide.addText(f.desc, {
    x: x + 0.2, y: y + 0.6, w: 5.3, h: 0.9,
    fontSize: 11, fontFace: F.body, color: C.darkGray,
    lineSpacingMultiple: 1.3,
  });
});

// ─── SLIDE 26: Key Results Summary ────────────────────────────────────────────
slide = pptx.addSlide();
addDarkBg(slide);

slide.addText("Key Results", {
  x: 0.7, y: 0.4, w: 6, h: 0.8,
  fontSize: 36, fontFace: F.title, color: C.white, bold: true,
});
slide.addShape(pptx.shapes.RECTANGLE, {
  x: 0.7, y: 1.2, w: 3, h: 0.04,
  fill: { type: "solid", color: C.coral },
});

// Big stat boxes
const stats = [
  { value: "99.2%", label: "OCR V2\nmAP@50", color: C.coral },
  { value: "90.0%", label: "Seatbelt\nmAP@50", color: C.steelBlue },
  { value: "9", label: "OCR Results\nPer Plate", color: C.gold },
  { value: "6", label: "Specialized\nModels", color: C.green },
  { value: "38", label: "Character\nClasses", color: C.accent2 },
  { value: "<1s", label: "Per Image\nInference", color: C.white },
];

stats.forEach((s, i) => {
  const col = i % 3;
  const row = Math.floor(i / 3);
  const x = 0.7 + col * 4.15;
  const y = 1.8 + row * 2.7;

  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x: x, y: y, w: 3.8, h: 2.3,
    fill: { type: "solid", color: C.navy },
    line: { color: s.color, width: 2 },
    rectRadius: 0.1,
  });

  slide.addText(s.value, {
    x: x, y: y + 0.2, w: 3.8, h: 1.0,
    fontSize: 44, fontFace: F.title, color: s.color,
    bold: true, align: "center",
  });
  slide.addText(s.label, {
    x: x, y: y + 1.3, w: 3.8, h: 0.8,
    fontSize: 14, fontFace: F.body, color: C.medGray,
    align: "center", lineSpacingMultiple: 1.3,
  });
});

// ─── SLIDE 27: Challenges ─────────────────────────────────────────────────────
slide = pptx.addSlide();
addLightBg(slide);

slide.addText("Challenges We Faced", {
  x: 0.7, y: 0.4, w: 8, h: 0.7,
  fontSize: 30, fontFace: F.title, color: C.navy, bold: true,
});

const challenges = [
  { title: "Video Tracking Unreliability", desc: "ByteTrack assigns fresh track IDs on each video pass. Cross-pass car matching requires spatial proximity (IoU), not track IDs. Video mode was removed pending a better approach.", color: C.coral },
  { title: "Class Imbalance in OCR", desc: "Some Arabic characters appear far more frequently than others. Attempted weighted BCE loss — but normal training with more epochs performed better.", color: C.gold },
  { title: "LapSRN Instability", desc: "Super-resolution model crashes on certain image shapes (cv2.error in merge). Required try/except fallback to original image.", color: C.steelBlue },
  { title: "GPU Memory Constraints", desc: "Kaggle T4 has 16GB VRAM. YOLOv11m batch=32 OOMs — had to use batch=16. This doubles training time.", color: C.accent2 },
  { title: "Roboflow Label Format", desc: "Some Roboflow YOLO exports include polygon data (>5 values per line). Had to truncate to standard format (class x y w h).", color: C.medGray },
  { title: "Arabic Text Rendering", desc: "Bidirectional text (numbers LTR, Arabic RTL) requires careful handling. Used separate_chars() to split and reverse correctly.", color: C.navy },
];

challenges.forEach((ch, i) => {
  const col = i % 2;
  const row = Math.floor(i / 2);
  const x = 0.7 + col * 6.2;
  const y = 1.4 + row * 1.9;

  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x: x, y: y, w: 5.8, h: 1.6,
    fill: { type: "solid", color: C.white },
    shadow: { type: "outer", blur: 4, offset: 2, color: "000000", opacity: 0.1 },
    rectRadius: 0.1,
  });
  // Left accent
  slide.addShape(pptx.shapes.RECTANGLE, {
    x: x, y: y + 0.15, w: 0.06, h: 1.3,
    fill: { type: "solid", color: ch.color },
  });

  slide.addText(ch.title, {
    x: x + 0.3, y: y + 0.15, w: 5.2, h: 0.35,
    fontSize: 15, fontFace: F.header, color: C.navy, bold: true,
  });
  slide.addText(ch.desc, {
    x: x + 0.3, y: y + 0.55, w: 5.2, h: 0.9,
    fontSize: 11, fontFace: F.body, color: C.darkGray,
    lineSpacingMultiple: 1.3,
  });
});

// ─── SLIDE 28: Future Work ────────────────────────────────────────────────────
slide = pptx.addSlide();
addLightBg(slide);

slide.addText("What's Next?", {
  x: 0.7, y: 0.4, w: 8, h: 0.7,
  fontSize: 30, fontFace: F.title, color: C.navy, bold: true,
});

const futureWork = [
  { title: "Video Mode Revival", desc: "Revisit video processing with better tracking approach. Current plate-first method was too slow.", icon: "🎬", priority: "High" },
  { title: "Auto Mode", desc: "Automatically select best plate crop using sharpness scoring, not just area ranking.", icon: "🤖", priority: "High" },
  { title: "Color Classification", desc: "K-means clustering for vehicle color detection. Config ready (COLOR_KMEANS_CLUSTERS), implementation pending.", icon: "🎨", priority: "Medium" },
  { title: "Speed Estimation", desc: "Use car tracking between frames to estimate vehicle speed from video feeds.", icon: "⚡", priority: "Medium" },
  { title: "Improved Deduplication", desc: "Current IoU-based dedup works, but visual similarity (feature matching) would be more robust.", icon: "🔄", priority: "Medium" },
  { title: "Model Optimization", desc: "Quantization and TensorRT for faster inference. Deploy on edge devices (Jetson, Coral).", icon: "🚀", priority: "Low" },
];

futureWork.forEach((f, i) => {
  const col = i % 2;
  const row = Math.floor(i / 2);
  const x = 0.7 + col * 6.2;
  const y = 1.4 + row * 1.9;

  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x: x, y: y, w: 5.8, h: 1.6,
    fill: { type: "solid", color: C.white },
    shadow: { type: "outer", blur: 4, offset: 2, color: "000000", opacity: 0.1 },
    rectRadius: 0.1,
  });

  slide.addText(f.icon, { x: x + 0.2, y: y + 0.1, w: 0.5, h: 0.4, fontSize: 20 });
  slide.addText(f.title, {
    x: x + 0.7, y: y + 0.1, w: 3.5, h: 0.35,
    fontSize: 16, fontFace: F.header, color: C.navy, bold: true,
  });
  // Priority badge
  const pColor = f.priority === "High" ? C.coral : f.priority === "Medium" ? C.gold : C.green;
  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x: x + 4.6, y: y + 0.15, w: 0.9, h: 0.28,
    fill: { type: "solid", color: pColor },
    rectRadius: 0.05,
  });
  slide.addText(f.priority, {
    x: x + 4.6, y: y + 0.15, w: 0.9, h: 0.28,
    fontSize: 9, fontFace: F.body, color: C.white,
    bold: true, align: "center", valign: "middle",
  });

  slide.addText(f.desc, {
    x: x + 0.2, y: y + 0.6, w: 5.3, h: 0.8,
    fontSize: 12, fontFace: F.body, color: C.darkGray,
    lineSpacingMultiple: 1.3,
  });
});

// ─── SLIDE 29: Thank You ──────────────────────────────────────────────────────
slide = pptx.addSlide();
addDarkBg(slide);

// Decorative bar
slide.addShape(pptx.shapes.RECTANGLE, {
  x: 0, y: 0, w: 0.15, h: H,
  fill: { type: "solid", color: C.coral },
});

slide.addText("Thank You", {
  x: 1, y: 1.5, w: 11, h: 1.5,
  fontSize: 64, fontFace: F.title, color: C.white,
  bold: true,
});
slide.addShape(pptx.shapes.RECTANGLE, {
  x: 1, y: 3.1, w: 3, h: 0.04,
  fill: { type: "solid", color: C.coral },
});
slide.addText("Questions?", {
  x: 1, y: 3.5, w: 11, h: 0.8,
  fontSize: 28, fontFace: F.body, color: C.steelBlue,
});
slide.addText("RADAR — Real-Time Vehicle Analysis System\nEgyptian License Plate Detection  ·  OCR  ·  Seatbelt & Mobile Detection", {
  x: 1, y: 4.8, w: 11, h: 1.0,
  fontSize: 16, fontFace: F.body, color: C.medGray,
  lineSpacingMultiple: 1.5,
});

// Tech stack pills again
const techsEnd = ["YOLOv11m", "YOLO26m", "YOLO26s", "Streamlit", "LapSRN", "Real-ESRGAN"];
techsEnd.forEach((tech, i) => {
  slide.addShape(pptx.shapes.ROUNDED_RECTANGLE, {
    x: 1 + i * 1.85, y: 6.0, w: 1.7, h: 0.4,
    fill: { type: "solid", color: C.navy },
    rectRadius: 0.05,
    line: { color: C.steelBlue, width: 1 },
  });
  slide.addText(tech, {
    x: 1 + i * 1.85, y: 6.0, w: 1.7, h: 0.4,
    fontSize: 10, fontFace: F.mono, color: C.steelBlue,
    align: "center", valign: "middle",
  });
});

// ─── Save ─────────────────────────────────────────────────────────────────────
const outPath = path.join(__dirname, "RADAR_Presentation.pptx");
pptx.writeFile({ fileName: outPath }).then(() => {
  console.log("Presentation saved to:", outPath);
  console.log("Total slides:", pptx.slides.length);
}).catch(err => {
  console.error("Error:", err);
});
