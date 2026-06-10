document.addEventListener("DOMContentLoaded", () => {
  const dropArea = document.getElementById("drop-area");
  const input = document.getElementById("image");
  const previewContainer = document.getElementById("preview-container");
  const previewImage = document.getElementById("preview-image");
  const loading = document.getElementById("loading");
  const responseData = document.getElementById("response-data");

  // Referral-biased presentation of each triage action. Never a definitive diagnosis.
  const TRIAGE_VIEW = {
    reassure: {
      title: "Low concern",
      badge: "bg-success",
      advice: "No high-risk features were detected. Keep monitoring this lesion and see a clinician if it changes in size, colour, or shape.",
    },
    refer: {
      title: "See a clinician",
      badge: "bg-warning text-dark",
      advice: "Some features warrant assessment. Please book a dermatology review.",
    },
    urgent: {
      title: "Seek prompt review",
      badge: "bg-danger",
      advice: "Concerning features were detected. Please seek prompt clinical review.",
    },
    abstain: {
      title: "Inconclusive — see a dermatologist",
      badge: "bg-secondary",
      advice: "This image could not be confidently assessed. Retake a clear, close, well-lit photo of the lesion, or consult a dermatologist.",
    },
  };

  const REASON_TEXT = {
    low_quality: "The image looked blurry or low quality.",
    out_of_distribution: "The image didn't look like a typical skin-lesion photo.",
    model_unavailable: "The triage model is not loaded yet.",
  };

  dropArea.addEventListener("click", () => input.click());
  dropArea.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropArea.classList.add("hover");
  });
  dropArea.addEventListener("dragleave", () => dropArea.classList.remove("hover"));
  dropArea.addEventListener("drop", (event) => {
    event.preventDefault();
    dropArea.classList.remove("hover");
    if (event.dataTransfer.files.length > 0) {
      handleFile(event.dataTransfer.files[0]);
    }
  });
  input.addEventListener("change", (event) => handleFile(event.target.files[0]));

  function handleFile(file) {
    if (file && file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = (event) => {
        previewImage.src = event.target.result;
        previewContainer.style.display = "block";
        startAnalysis(file);
      };
      reader.readAsDataURL(file);
    }
  }

  function startAnalysis(file) {
    const formData = new FormData();
    formData.append("image", file);
    loading.style.display = "block";

    fetch("predict", {
      method: "POST",
      body: formData,
      headers: { "X-Requested-With": "XMLHttpRequest" },
    })
      .then((response) => response.json())
      .then((data) => {
        loading.style.display = "none";
        renderTriage(data);
      })
      .catch((error) => {
        console.error(error);
        loading.style.display = "none";
        responseData.style.display = "block";
        responseData.innerHTML = `<span style="color:red;">An error occurred: ${error.message}</span>`;
      });
  }

  function renderTriage(data) {
    const analysisResult = document.getElementById("analysis-result");
    const view = TRIAGE_VIEW[data.triage] || TRIAGE_VIEW.abstain;
    const date = new Date().toLocaleString();

    let detail = "";
    if (data.valid_image && data.probabilities) {
      const malignant = (data.p_malignant * 100).toFixed(1);
      const rows = ["benign", "suspicious", "malignant"]
        .map((name) => `${name}: ${(data.probabilities[name] * 100).toFixed(1)}%`)
        .join(" · ");
      const conformal = (data.conformal_set || []).join(", ") || "none";
      let lesion = "";
      if (data.lesion_type) {
        const fine = (data.fine_predictions || [])
          .map((prediction) => `${prediction.label} (${prediction.probability.toFixed(1)}%)`)
          .join(", ");
        lesion = `<p class="mt-2">Most likely lesion type: <strong>${data.lesion_type}</strong></p>
        <p class="text-muted small mb-1">Other possibilities: ${fine}</p>`;
      }
      detail = `
        ${lesion}
        <p class="mt-2">Estimated malignancy probability: <strong>${malignant}%</strong></p>
        <p class="text-muted small mb-1">Triage probabilities: ${rows}</p>
        <p class="text-muted small">Plausible categories (conformal set): ${conformal}</p>`;
    } else if (data.reason) {
      detail = `<p class="text-muted small mt-2">${REASON_TEXT[data.reason] || data.reason}</p>`;
    }

    analysisResult.className = "text-center py-4";
    analysisResult.innerHTML = `
      <div><span class="badge ${view.badge}">${view.title}</span></div>
      <p class="mt-3">${view.advice}</p>
      ${detail}
      <p class="text-muted small mt-3"><em>This is triage guidance, not a diagnosis.</em></p>
      <small class="text-muted">Analysed: ${date}</small>`;

    const diagnosisDescription = document.getElementById("diagnosis-description");
    const diagnosisLastUpdated = document.getElementById("diagnosis-last-updated");
    if (diagnosisDescription) {
      diagnosisDescription.textContent = view.advice + " Always confirm with a qualified clinician.";
    }
    if (diagnosisLastUpdated) {
      diagnosisLastUpdated.textContent = `Last updated: ${date}`;
    }
  }
});
