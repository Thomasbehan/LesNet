// LesNet — fullscreen, boxless, elegant. Cursor spotlight, drop-anywhere, choreographed result reveal.
document.addEventListener("DOMContentLoaded", () => {
  const input = document.getElementById("image");
  const camera = document.getElementById("camera");
  const previewImage = document.getElementById("preview-image");
  const previewContainer = document.getElementById("preview-container");
  const loading = document.getElementById("loading");
  const analysisResult = document.getElementById("analysis-result");
  const viewUpload = document.getElementById("view-upload");
  const viewResult = document.getElementById("view-result");
  const dropveil = document.getElementById("dropveil");
  const root = document.documentElement;

  const TRIAGE_VIEW = {
    reassure: { title: "Low concern", cls: "t-reassure",
      advice: "No high-risk features were detected. Keep monitoring and see a clinician if it changes in size, colour, or shape." },
    refer: { title: "See a clinician", cls: "t-refer",
      advice: "Some features warrant assessment. Please book a dermatology review." },
    urgent: { title: "Seek prompt review", cls: "t-urgent",
      advice: "Concerning features were detected. Please seek prompt clinical review." },
    abstain: { title: "Inconclusive", cls: "t-abstain",
      advice: "This image could not be confidently assessed. Retake a clear, close, well-lit photo, or consult a dermatologist." },
  };
  const REASON_TEXT = {
    low_quality: "The image looked blurry or low quality.",
    out_of_distribution: "The image didn't look like a typical skin-lesion photo.",
    model_unavailable: "The triage model is not loaded yet.",
  };
  const buzz = (ms) => { try { navigator.vibrate && navigator.vibrate(ms); } catch (e) { /* no-op */ } };
  const prettyLabel = (s) => { const t = String(s).replace(/_/g, " ").trim(); return t.charAt(0).toUpperCase() + t.slice(1); };

  // cursor-follow spotlight
  window.addEventListener("pointermove", (e) => {
    root.style.setProperty("--mx", e.clientX + "px");
    root.style.setProperty("--my", e.clientY + "px");
  });
  // magnetic glow on the primary CTA
  const cta = document.getElementById("browse-btn");
  if (cta) cta.addEventListener("pointermove", (e) => {
    const r = cta.getBoundingClientRect();
    cta.style.setProperty("--bx", (e.clientX - r.left) + "px");
    cta.style.setProperty("--by", (e.clientY - r.top) + "px");
  });

  cta && cta.addEventListener("click", () => input.click());
  const cam = document.getElementById("camera-btn");
  if (cam) cam.addEventListener("click", () => (camera || input).click());
  input.addEventListener("change", (e) => e.target.files[0] && handleFile(e.target.files[0]));
  if (camera) camera.addEventListener("change", (e) => e.target.files[0] && handleFile(e.target.files[0]));
  document.getElementById("back-btn").addEventListener("click", () => { buzz(8); analysisResult.innerHTML = ""; showView(viewUpload); });

  // drop anywhere on the window
  let dragDepth = 0;
  window.addEventListener("dragenter", (e) => { e.preventDefault(); if (dragDepth++ === 0) dropveil.classList.add("show"); });
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("dragleave", (e) => { e.preventDefault(); if (--dragDepth <= 0) { dragDepth = 0; dropveil.classList.remove("show"); } });
  window.addEventListener("drop", (e) => { e.preventDefault(); dragDepth = 0; dropveil.classList.remove("show");
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]); });

  function showView(el) {
    [viewUpload, viewResult].forEach((v) => v.classList.remove("active"));
    void el.offsetWidth; el.classList.add("active");
    window.scrollTo(0, 0);
  }

  function handleFile(file) {
    if (!file || !file.type.startsWith("image/")) return;
    buzz(8);
    const reader = new FileReader();
    reader.onload = (e) => { previewImage.src = e.target.result; showView(viewResult); startAnalysis(file); };
    reader.readAsDataURL(file);
  }

  function startAnalysis(file) {
    const formData = new FormData();
    formData.append("image", file);
    loading.style.display = "flex";
    analysisResult.innerHTML = "";
    previewContainer.classList.add("scanning");
    fetch("predict", { method: "POST", body: formData, headers: { "X-Requested-With": "XMLHttpRequest" } })
      .then((r) => r.json())
      .then((data) => { loading.style.display = "none"; previewContainer.classList.remove("scanning"); render(data); })
      .catch((err) => { loading.style.display = "none"; previewContainer.classList.remove("scanning");
        analysisResult.innerHTML = `<p class="vadvice">Something went wrong: ${err.message}</p>`; });
  }

  function render(data) {
    const view = TRIAGE_VIEW[data.triage] || TRIAGE_VIEW.abstain;
    buzz(data.triage === "urgent" ? [35, 55, 35] : 14);
    const fine = data.fine_predictions || [];

    // top-3 diagnoses are the main content
    let dx = "";
    if (data.valid_image && fine.length) {
      dx = `<div class="dx"><span class="vlabel">Most likely diagnoses</span>` + fine.map((p, i) => `
        <div class="dxrow ${i === 0 ? "top" : ""}" style="animation-delay:${0.35 + i * 0.1}s">
          <span class="dxname"><span class="rank">${i + 1}</span>${prettyLabel(p.label)}</span>
          <span class="dxpct">${p.probability.toFixed(1)}%</span>
          <i class="dxbar" data-w="${Math.max(3, p.probability).toFixed(1)}"></i>
        </div>`).join("") + `</div>`;
    }

    // benign / suspicious / malignant as small secondary metrics
    let metrics = "";
    if (data.valid_image && data.probabilities) {
      const seg = (k) => `<div class="seg"><div class="k">${k}</div><div class="v">${(data.probabilities[k] * 100).toFixed(0)}%</div></div>`;
      metrics = `<div class="metrics">${seg("benign")}${seg("suspicious")}${seg("malignant")}</div>`;
    } else if (data.reason) {
      metrics = `<p class="vadvice">${REASON_TEXT[data.reason] || data.reason}</p>`;
    }

    analysisResult.innerHTML = `
      <div class="verdict ${view.cls}">
        <span class="vlabel">Assessment</span>
        <div class="vword"><span class="vdot"></span>${view.title}</div>
        <p class="vadvice">${view.advice}</p>
      </div>
      ${dx}
      ${metrics}
      <p class="foot">Triage guidance, not a diagnosis.${data.model ? " · " + data.model : ""}</p>`;

    requestAnimationFrame(() => analysisResult.querySelectorAll(".dxbar").forEach((el) => { el.style.width = el.dataset.w + "%"; }));
  }
});
