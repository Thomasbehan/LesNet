document.addEventListener("DOMContentLoaded", () => {
  const dropArea = document.getElementById("drop-area");
  const input = document.getElementById("image");
  const previewContainer = document.getElementById("preview-container");
  const previewImage = document.getElementById("preview-image");
  const loading = document.getElementById("loading");
  const responseData = document.getElementById("response-data");
  const diagnosisImage = document.getElementById("diagnosis-image");

  // Handle drag-and-drop functionality
  dropArea.addEventListener("click", () => input.click());

  dropArea.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropArea.classList.add("hover");
  });

  dropArea.addEventListener("dragleave", () => {
    dropArea.classList.remove("hover");
  });

  dropArea.addEventListener("drop", (event) => {
    event.preventDefault();
    dropArea.classList.remove("hover");
    if (event.dataTransfer.files.length > 0) {
      handleFile(event.dataTransfer.files[0]);
    }
  });

  input.addEventListener("change", (event) => {
    handleFile(event.target.files[0]);
  });

  // Handle file and preview
  function handleFile(file) {
    if (file && file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewContainer.style.display = "block";

        // Start analysis automatically
        startAnalysis(file);
      };
      reader.readAsDataURL(file);
    }
  }

  // Start analysis function
  function startAnalysis(file) {
    const formData = new FormData();
    formData.append("image", file);

    // Show loading indicator
    loading.style.display = "block";

    // AJAX request
    fetch("predict", {
      method: "POST",
      body: formData,
      headers: {
        "X-Requested-With": "XMLHttpRequest",
      },
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        return response.json();
      })
      .then((data) => {
        loading.style.display = "none";
        if (data.prediction && data.confidence) {
          updateAnalysisBox(data.prediction, data.confidence, new Date().toLocaleDateString());
          updateDiagnosisInfo(data.prediction);
        } else {
          responseData.style.display = "block"
          responseData.innerHTML = `<span>No valid result returned. Please try again.</span>`;
        }
      })
      .catch((error) => {
        loading.style.display = "none";
        responseData.style.display = "block"
        responseData.innerHTML = `<span style="color:red;">An error occurred: ${error.message}</span>`;
      });
  }

  async function updateDiagnosisInfo(prediction) {
      const diagnosisInfo = document.getElementById("diagnosis-info");
      const diagnosisDescription = document.getElementById("diagnosis-description");
      const diagnosisLastUpdated = document.getElementById("diagnosis-last-updated");

      // Default placeholder if no prediction
      if (!prediction) {
        diagnosisDescription.textContent = "Upload an image to get started.";
        diagnosisLastUpdated.textContent = "Last updated: N/A";
        return;
      }

      // Fetch information from Wikipedia
      try {
        const response = await fetch(`https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(prediction)}`);
        if (!response.ok) throw new Error("Failed to fetch diagnosis information.");

        const data = await response.json();

        // Update diagnosis information dynamically
        diagnosisDescription.innerHTML = data.extract || "Information not available.";
        diagnosisLastUpdated.textContent = `Last updated: ${new Date().toLocaleTimeString()}`;

        // Update image if available
        if (data.thumbnail && data.thumbnail.source) {
          diagnosisImage.src = data.thumbnail.source;
        }

      } catch (error) {
        console.error(error);
        diagnosisDescription.textContent = "Failed to load diagnosis information. Please try again.";
        diagnosisLastUpdated.textContent = `Last updated: ${new Date().toLocaleTimeString()}`;
      }
    }

  function updateAnalysisBox(prediction, confidence, date) {
      const analysisResult = document.getElementById("analysis-result");
      const analysisActions = document.getElementById("analysis-actions");

      // Clear previous result styles
      analysisResult.className = "text-center py-4";

      if (prediction && confidence) {
        // Format the confidence level
        const confidenceText = `${confidence.toFixed(2)}% confidence`;

        // Set result content
        analysisResult.innerHTML = `
          <strong>Result:</strong> ${confidenceText} it is <strong>${prediction}</strong>.
          <div class="mt-3">
            <span class="badge ${
              confidence > 80 ? "bg-success" : "bg-warning text-dark"
            }">
              ${confidence > 80 ? "High Confidence" : "Moderate Confidence"}
            </span>
          </div>
        `;

        // Style the result box based on confidence
        analysisResult.classList.add(
          confidence > 80 ? "success" : "warning"
        );

        // Show additional actions
        analysisActions.classList.remove("d-none");
      } else {
        // Display default message for missing analysis
        analysisResult.innerHTML = `<p>No analysis available yet.</p>`;
        analysisResult.classList.add("error");

        // Hide additional actions
        analysisActions.classList.add("d-none");
      }
    }
});
