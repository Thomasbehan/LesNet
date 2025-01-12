document.addEventListener("DOMContentLoaded", () => {
  const cardsContainer = document.getElementById("diagnosis-cards");

  // Function to fetch diagnoses and populate the cards
  async function populateDiagnosisCards() {
    try {
      const response = await fetch("labels");
      if (!response.ok) throw new Error("Failed to fetch diagnoses.");

      const diagnoses = await response.json();

      // Clear existing cards
      cardsContainer.innerHTML = '';

      // Iterate through diagnoses and create cards
      for (const diagnosis of diagnoses) {
        const diagnosisName = diagnosis;
        const diagnosisId = diagnosisName.replace(/\s+/g, '-').toLowerCase();

        // Create a new card
        const card = document.createElement("div");
        card.classList.add("card");
        card.classList.add("m-2");
        card.classList.add("shadow");
        card.classList.add("border-0");
        card.style.width = "18rem";

        card.innerHTML = `
          <img class="card-img-top default-diagnosis-image" src="/static/doc.webp" alt="${diagnosisName}" />
          <img class="card-img-top diagnosis-image" style="display:none;" id="image-${diagnosisId}" src="" alt="${diagnosisName}" />
          <div class="card-body">
            <h5 class="card-title">${diagnosisName}</h5>
            <p class="card-text" id="description-${diagnosisId}">Loading description...</p>
          </div>
        `;

        cardsContainer.appendChild(card);

        // Fetch description and image for each diagnosis
        await updateDiagnosisDetails(diagnosisName, diagnosisId);
      }
    } catch (error) {
      console.error("Error populating cards:", error);
    }
  }

  // Function to update the description and image for a specific diagnosis
  async function updateDiagnosisDetails(diagnosisName, diagnosisId) {
    try {
      const response = await fetch(`https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(diagnosisName)}`);
      if (!response.ok) throw new Error("Failed to fetch diagnosis details.");

      const data = await response.json();
      const descriptionElement = document.getElementById(`description-${diagnosisId}`);
      const imageElement = document.getElementById(`image-${diagnosisId}`);

      if (data.extract) {
        descriptionElement.innerHTML = data.extract;
      } else {
        descriptionElement.innerHTML = "Description not available.";
      }

      if (data.thumbnail && data.thumbnail.source) {
        imageElement.src = data.thumbnail.source;
      } else {
        imageElement.src = "https://via.placeholder.com/150"; // Fallback image
      }
    } catch (error) {
      console.error(`Error fetching details for ${diagnosisName}:`, error);
      document.getElementById(`description-${diagnosisId}`).innerHTML = "Failed to load description.";
      document.getElementById(`image-${diagnosisId}`).src = "https://via.placeholder.com/150";
    }
  }

  // Call the function to populate the cards on page load
  populateDiagnosisCards();
});
