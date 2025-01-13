document.addEventListener("DOMContentLoaded", () => {
  const explicitToggle = document.getElementById("explicitToggle");

  // Event listener for the toggle switch
  explicitToggle.addEventListener("change", (event) => {
    const defaultDiagnosisImages = document.querySelectorAll(".default-diagnosis-image");
    const diagnosisImages = document.querySelectorAll(".diagnosis-image");
    if (event.target.checked) {
      // Show all default images if toggle is checked
      defaultDiagnosisImages.forEach(image => {
        image.style.display = "block";
      });
      // Hide all diagnosis images if toggle is checked
      diagnosisImages.forEach(image => {
        image.style.display = "none";
      });
    } else {
      // Hide all default images if toggle is unchecked
      defaultDiagnosisImages.forEach(image => {
        image.style.display = "none";
      });
      // Show all diagnosis images if toggle is unchecked
      diagnosisImages.forEach(image => {
        image.style.display = "block";
      });
    }
  });
});
