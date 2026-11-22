// LesNet — check a skin spot.
//
// Rules this file keeps:
//   * A person never sees an exception, a status code or a stack trace. Every failure path ends in
//     one of the plain-language messages in MESSAGES, with a way forward.
//   * Photos are shrunk in the browser before upload (phone photos are 3–12 MB), with EXIF rotation
//     applied, so uploads are fast on mobile data and the server never has to decode a huge image.
//   * The free host sleeps when idle; the first request can take ~a minute. That is retried
//     automatically and explained, not reported as an error.
(() => {
  "use strict";

  const UPLOAD_MAX_SIDE = 1024;       // server reduces further to the model's 256 px input
  const REQUEST_TIMEOUT_MS = 75000;
  const TOTAL_BUDGET_MS = 150000;     // give a cold start room before giving up
  const SLOW_HINT_MS = 7000;

  const VERDICTS = {
    reassure: {
      tone: "low", title: "Low concern",
      body: "Nothing in this photo looked like the higher-risk spots LesNet has learned from. Keep an eye on it. " +
            "If it grows, changes shape or colour, bleeds, itches or doesn't heal, see a doctor.",
    },
    refer: {
      tone: "refer", title: "Worth getting checked",
      body: "This spot shares some features with spots that a dermatologist would want to look at. Book a routine " +
            "appointment with your GP or a dermatologist and show them this spot.",
    },
    urgent: {
      tone: "urgent", title: "Get it checked soon",
      body: "This spot shares features with spots that need prompt assessment. Arrange to see a GP or dermatologist " +
            "soon rather than waiting to see whether it changes.",
    },
  };

  const MESSAGES = {
    out_of_distribution: {
      title: "This doesn't look like a skin spot",
      body: "LesNet only works on close-up photos of a single spot on the skin. Try again with the spot centred and " +
            "filling most of the frame.",
    },
    low_quality: {
      title: "The photo is too blurry or dark",
      body: "Try again in daylight or bright light, hold the phone steady, and tap the spot to focus.",
    },
    unreadable: {
      title: "That file couldn't be opened",
      body: "Try a JPEG or PNG photo. On an iPhone, use “Take a photo”, or set Settings › Camera › Formats to " +
            "“Most Compatible”.",
    },
    too_large: {
      title: "That image is too large",
      body: "Try a normal photo straight from your phone or camera.",
    },
    not_image: {
      title: "That isn't a photo",
      body: "Choose an image file, such as a JPEG or PNG from your camera roll.",
    },
    unavailable: {
      title: "The checker isn't responding right now",
      body: "Your photo wasn't checked. This is usually temporary, so please try again in a minute.",
    },
    offline: {
      title: "You seem to be offline",
      body: "Your photo wasn't checked. Reconnect to the internet and try again.",
    },
  };
  // server `reason` -> message key (anything unknown becomes "unavailable")
  const REASON_TO_MESSAGE = {
    out_of_distribution: "out_of_distribution", low_quality: "low_quality", unreadable: "unreadable",
    no_image: "unreadable", too_large: "too_large",
  };

  const $ = (id) => document.getElementById(id);
  const root = $("check");
  const stage = $("stage");
  const lens = $("lens");
  const lensPhoto = $("lens-photo");
  const status = $("status");
  const statusLine = $("status-line");
  const statusSub = $("status-sub");
  const result = $("result");
  const filePick = $("file-pick");
  const fileCamera = $("file-camera");
  const dropveil = $("dropveil");

  let lastBlob = null;
  let photoUrl = null;
  let runId = 0;

  // ------------------------------------------------------------------ helpers
  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else node.setAttribute(key, value);
    }
    for (const child of children) if (child) node.append(child);
    return node;
  }

  const prettyLabel = (label) => {
    const text = String(label || "").replace(/_/g, " ").trim();
    return text.charAt(0).toUpperCase() + text.slice(1);
  };

  function setState(state) {
    root.dataset.state = state;
    stage.setAttribute("aria-busy", state === "working" ? "true" : "false");
  }

  function looksLikeImage(file) {
    if (!file) return false;
    if (file.type && file.type.startsWith("image/")) return true;
    return /\.(jpe?g|png|webp|heic|heif|bmp|gif|tiff?)$/i.test(file.name || "");
  }

  // Shrink + apply EXIF orientation in the browser. Any failure (e.g. HEIC on desktop Chrome) falls back
  // to sending the original file, and the server decides whether it can read it.
  async function prepare(file) {
    try {
      const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
      const scale = Math.min(1, UPLOAD_MAX_SIDE / Math.max(bitmap.width, bitmap.height));
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(bitmap.width * scale));
      canvas.height = Math.max(1, Math.round(bitmap.height * scale));
      canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
      if (bitmap.close) bitmap.close();
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
      return blob || file;
    } catch (_) {
      return file;
    }
  }

  function showPhoto(blob) {
    if (photoUrl) URL.revokeObjectURL(photoUrl);
    photoUrl = URL.createObjectURL(blob);
    lensPhoto.onerror = () => { lensPhoto.hidden = true; };   // e.g. HEIC preview unsupported: keep the lens
    lensPhoto.src = photoUrl;
    lensPhoto.hidden = false;
  }

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  // One POST. Resolves to {kind: "ok"|"reason"|"retry", data}. Never throws.
  async function postOnce(blob) {
    const body = new FormData();
    body.append("image", blob, "photo.jpg");
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    try {
      const response = await fetch("/predict", { method: "POST", body, signal: controller.signal });
      let data = null;
      try { data = await response.json(); } catch (_) { data = null; }
      if (data && data.triage && data.triage !== "abstain") return { kind: "ok", data };
      if (data && data.reason && REASON_TO_MESSAGE[data.reason]) return { kind: "reason", data };
      return { kind: "retry" };              // 502/503/504, model warming up, empty or non-JSON body
    } catch (_) {
      return { kind: "retry" };              // network drop or timeout
    } finally {
      clearTimeout(timer);
    }
  }

  // ------------------------------------------------------------------ flow
  async function check(file) {
    if (!looksLikeImage(file)) { showMessage("not_image", false); return; }
    const myRun = ++runId;
    setState("working");
    result.hidden = true;
    status.hidden = false;
    statusLine.textContent = "Checking your photo…";
    statusSub.textContent = "";

    const blob = await prepare(file);
    if (myRun !== runId) return;
    lastBlob = blob;
    showPhoto(blob);
    await analyse(myRun);
  }

  async function analyse(myRun) {
    setState("working");
    result.hidden = true;
    status.hidden = false;
    statusLine.textContent = "Checking your photo…";
    statusSub.textContent = "";
    const started = Date.now();
    const slowHint = setTimeout(() => {
      if (myRun !== runId) return;
      statusLine.textContent = "Still checking…";
      statusSub.textContent = "If nobody has used LesNet for a while, the server takes up to a minute to wake up. " +
                              "Keep this page open.";
    }, SLOW_HINT_MS);

    let outcome = { kind: "retry" };
    let attempt = 0;
    while (outcome.kind === "retry" && Date.now() - started < TOTAL_BUDGET_MS) {
      if (!navigator.onLine) break;
      if (attempt > 0) await sleep(Math.min(8000, 2000 * attempt));
      if (myRun !== runId) { clearTimeout(slowHint); return; }
      outcome = await postOnce(lastBlob);
      attempt += 1;
    }
    clearTimeout(slowHint);
    if (myRun !== runId) return;

    status.hidden = true;
    if (outcome.kind === "ok") showVerdict(outcome.data);
    else if (outcome.kind === "reason") showMessage(REASON_TO_MESSAGE[outcome.data.reason], false);
    else showMessage(navigator.onLine ? "unavailable" : "offline", true);
  }

  // ------------------------------------------------------------------ rendering
  function buttons(retryable, againLabel) {
    const row = el("div", { class: "result-actions" });
    if (retryable) {
      const retry = el("button", { type: "button", class: "btn primary", text: "Try again" });
      retry.addEventListener("click", () => analyse(++runId));
      row.append(retry);
    }
    const again = el("button", { type: "button", class: retryable ? "btn" : "btn primary",
                                 text: retryable ? "Choose a different photo" : (againLabel || "Check another spot") });
    again.addEventListener("click", reset);
    row.append(again);
    return row;
  }

  function finish(tone, node) {
    lens.dataset.tone = tone;
    result.replaceChildren(node);
    result.hidden = false;
    setState("done");
    result.focus({ preventScroll: true });
    result.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
                            block: "nearest" });
  }

  function showMessage(key, retryable) {
    const message = MESSAGES[key] || MESSAGES.unavailable;
    if (!lastBlob) lensPhoto.hidden = true;
    finish("unknown", el("div", { class: "verdict tone-unknown" },
      el("h2", { class: "verdict-title", text: message.title }),
      el("p", { class: "verdict-body", text: message.body }),
      buttons(retryable, "Try another photo")));
  }

  function showVerdict(data) {
    const verdict = VERDICTS[data.triage];
    if (!verdict) { showMessage("unavailable", true); return; }

    const block = el("div", { class: `verdict tone-${verdict.tone}` },
      el("h2", { class: "verdict-title", text: verdict.title }),
      el("p", { class: "verdict-body", text: verdict.body }));

    const matches = (data.fine_predictions || []).filter((p) => p && p.label);
    if (matches.length) {
      const list = el("ol", { class: "matches" });
      for (const match of matches) {
        const pct = Math.max(0, Math.min(100, Number(match.probability) || 0));
        const bar = el("span", { class: "match-bar" });
        bar.style.setProperty("--w", `${Math.max(2, pct)}%`);
        list.append(el("li", {},
          el("span", { class: "match-name", text: prettyLabel(match.label) }),
          el("span", { class: "match-pct", text: pct < 1 ? "<1%" : `${Math.round(pct)}%` }),
          bar));
      }
      block.append(el("section", { class: "similar", "aria-label": "Closest matches" },
        el("h3", { text: "Looks most like" }),
        list,
        el("p", { class: "note", text: "These are the kinds of spots in LesNet's training images that this photo " +
                                       "resembles most. They are not a diagnosis." })));
    }

    const details = el("details", { class: "details" }, el("summary", { text: "How this was worked out" }));
    const score = Number(data.p_malignant);
    details.append(el("p", { text: "LesNet turns the photo into a numeric description using a general-purpose " +
      "image model, then compares it with labelled dermatology images. It deliberately leans towards “get it " +
      "checked”: missing a cancer is worse than an unnecessary appointment." }));
    if (Number.isFinite(score)) {
      details.append(el("p", { text: `Similarity to malignant examples: ${Math.round(score * 100)} out of 100.` }));
    }
    if (data.model) details.append(el("p", { class: "note", text: `Model: ${data.model}` }));
    block.append(details, buttons(false));
    finish(verdict.tone, block);
  }

  function reset() {
    runId += 1;
    lastBlob = null;
    if (photoUrl) URL.revokeObjectURL(photoUrl);
    photoUrl = null;
    lensPhoto.hidden = true;
    lensPhoto.removeAttribute("src");
    delete lens.dataset.tone;
    result.hidden = true;
    status.hidden = true;
    filePick.value = "";
    fileCamera.value = "";
    setState("idle");
    $("btn-camera").focus();
  }

  // ------------------------------------------------------------------ wiring
  const pick = () => filePick.click();
  $("lens-pick").addEventListener("click", pick);
  $("btn-pick").addEventListener("click", pick);
  $("btn-camera").addEventListener("click", () => fileCamera.click());
  for (const input of [filePick, fileCamera]) {
    input.addEventListener("change", () => input.files && input.files[0] && check(input.files[0]));
  }

  let dragDepth = 0;
  const hasFiles = (event) => Array.from(event.dataTransfer ? event.dataTransfer.types : []).includes("Files");
  window.addEventListener("dragenter", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    if (dragDepth++ === 0) dropveil.classList.add("show");
  });
  window.addEventListener("dragover", (event) => { if (hasFiles(event)) event.preventDefault(); });
  window.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; dropveil.classList.remove("show"); } });
  window.addEventListener("drop", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    dragDepth = 0;
    dropveil.classList.remove("show");
    const file = event.dataTransfer.files[0];
    if (file) check(file);
  });
  document.addEventListener("paste", (event) => {
    const item = Array.from(event.clipboardData ? event.clipboardData.files : [])[0];
    if (item) check(item);
  });

  // Wake a sleeping server while the person is still framing their photo.
  fetch("/labels", { method: "GET" }).catch(() => {});
})();
