document.addEventListener("DOMContentLoaded", () => {
  const dropArea = document.getElementById("drop-area");
  const input = document.getElementById("image");
  const previewContainer = document.getElementById("preview-container");
  const previewImage = document.getElementById("preview-image");
  const loading = document.getElementById("loading");
  const responseData = document.getElementById("response-data");
  const explicitToggle = document.getElementById("explicitToggle");
  const defaultDiagnosisImage = document.getElementById("default-diagnosis-image");
  const diagnosisImage = document.getElementById("diagnosis-image");

  let chartInstance = null; // Track the current chart instance

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
    responseData.style.display = "none"
    responseData.innerHTML = "";

    // Destroy previous chart if it exists
    if (chartInstance) {
      chartInstance.destroy();
    }

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
        responseData.style.display = "block"
        if (data.prediction && data.confidence) {
          responseData.innerHTML = `
            <strong>Result:</strong> ${data.confidence.toFixed(2)}% confidence it is ${data.prediction}.
          `;
          updateAnalysisBox(data.prediction, data.confidence, new Date().toLocaleDateString());
          updateDiagnosisInfo(data.prediction);
        } else {
          responseData.innerHTML = `<span>No valid result returned. Please try again.</span>`;
        }
      })
      .catch((error) => {
        loading.style.display = "none";
        responseData.innerHTML = `<span style="color:red;">An error occurred: ${error.message}</span>`;
      });
  }

  // Event listener for the toggle switch
  explicitToggle.addEventListener("change", (event) => {
    if (event.target.checked) {
      // Show the default image if toggle is checked
      defaultDiagnosisImage.style.display = "block";
      diagnosisImage.style.display = "none";
    } else {
      // Hide the diagnosis image if toggle is unchecked
      defaultDiagnosisImage.style.display = "none";
      diagnosisImage.style.display = "block";
    }
  });

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
      const analysisDate = document.getElementById("analysis-date");
      const analysisResult = document.getElementById("analysis-result");
      const analysisActions = document.getElementById("analysis-actions");

      // Update date
      analysisDate.textContent = date || "N/A";

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

    // Function to fetch diagnoses and populate the accordion
  async function populateDiagnosisAccordion() {
    const accordion = document.getElementById("accordion");

    try {
      const response = await fetch("labels");
      if (!response.ok) throw new Error("Failed to fetch diagnoses.");

      const diagnoses = await response.json();

      // Clear existing accordion items
      accordion.innerHTML = '';

      // Iterate through diagnoses and create accordion items
      for (const diagnosis of diagnoses) {
        const diagnosisName = diagnosis;
        const diagnosisId = diagnosisName.replace(/\s+/g, '-').toLowerCase();

        // Create a new accordion item
        const accordionItem = document.createElement("div");
        accordionItem.classList.add("card");

        accordionItem.innerHTML = `
          <div class="card-header" id="heading-${diagnosisId}">
            <h5 class="mb-0">
              <button class="btn btn-link collapsed" data-toggle="collapse" data-target="#collapse-${diagnosisId}" aria-expanded="false" aria-controls="collapse-${diagnosisId}">
                ${diagnosisName}
              </button>
            </h5>
          </div>
          <div id="collapse-${diagnosisId}" class="collapse" aria-labelledby="heading-${diagnosisId}" data-parent="#accordion">
            <div class="card-body" id="description-${diagnosisId}">
              Loading description...
            </div>
          </div>
        `;

        accordion.appendChild(accordionItem);

        // Fetch description for each diagnosis
        await updateDiagnosisDescription(diagnosisName, diagnosisId);
      }
    } catch (error) {
      console.error("Error populating accordion:", error);
    }
  }

  // Function to update the description for a specific diagnosis
  async function updateDiagnosisDescription(diagnosisName, diagnosisId) {
    try {
      const response = await fetch(`https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(diagnosisName)}`);
      if (!response.ok) throw new Error("Failed to fetch diagnosis description.");

      const data = await response.json();
      const descriptionElement = document.getElementById(`description-${diagnosisId}`);

      if (data.extract) {
        descriptionElement.innerHTML = data.extract;
      } else {
        descriptionElement.innerHTML = "Description not available.";
      }
    } catch (error) {
      console.error(`Error fetching description for ${diagnosisName}:`, error);
      document.getElementById(`description-${diagnosisId}`).innerHTML = "Failed to load description.";
    }
  }

  // Call the function to populate the accordion on page load
  populateDiagnosisAccordion();
});
