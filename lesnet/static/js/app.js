// Set constraints for the video stream
var constraints = { video: { facingMode: "user" }, audio: false };
var track = null;

// Define constants
const cameraView = document.querySelector("#camera--view"),
    cameraOutput = document.querySelector("#camera--output"),
    loadingMessage = document.querySelector("#loading"),
    resultsContainer = document.getElementById('resultsContainer'),
    cameraSensor = document.querySelector("#camera--sensor"),
    cameraTrigger = document.querySelector("#camera--trigger"),
    predictUrl = document.body.getAttribute('data-predict-url'),
    swUrl = document.body.getAttribute('sw-url');

// Access the device camera and stream to cameraView
function cameraStart() {
    navigator.mediaDevices
        .getUserMedia(constraints)
        .then(function(stream) {
            track = stream.getTracks()[0];
            cameraView.srcObject = stream;
        })
        .catch(function(error) {
            console.error("Oops. Something is broken.", error);
        });
}

// Take a picture when cameraTrigger is tapped
cameraTrigger.onclick = function() {
    cameraSensor.width = cameraView.videoWidth;
    cameraSensor.height = cameraView.videoHeight;
    const context = cameraSensor.getContext("2d");
    context.drawImage(cameraView, 0, 0);

    loadingMessage.style.display = "block";
    cameraOutput.style.display = "block";
    cameraSensor.toBlob(function(blob) {
        cameraOutput.src = URL.createObjectURL(blob);
        cameraOutput.classList.add("taken");

        makePrediction(blob);
    }, 'image/jpeg');
};

function handleFileChange(event) {
    const file = event.target.files[0];
    if (file) {
        makePrediction(file);
    }
}

function makePrediction(blob) {
    const formData = new FormData();
    formData.append('image', blob, 'capture.jpeg');

    const loadingElement = document.getElementById('loading');
    const resultsContainer = document.getElementById('response-data');

    loadingElement.style.display = "block";

    fetch(predictUrl, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        loadingElement.style.display = "none";
        if (data.prediction) {
            resultsContainer.innerHTML = `<span>${data.confidence.toFixed(2)}% it is ${data.prediction}</span>`;
        } else {
            var errorMessage = "An error occurred while processing the image.";
            if (data.error) {
                errorMessage = "<div class='alert alert-info'><span class='emoji'>&#129300;</span> " +
                               data.error + "</div>";
            } else {
                errorMessage = "<div class='alert alert-danger'>" + errorMessage + "</div>";
            }

            resultsContainer.innerHTML = errorMessage;
        }
    })
    .catch(() => {
        loadingElement.style.display = "none";
        resultsContainer.innerHTML = "<div class='alert alert-danger'>An error occurred while processing the image.</div>";
    });
}

// Start the video stream when the window loads
window.addEventListener("load", cameraStart, false);


// Install ServiceWorker
if ('serviceWorker' in navigator) {
  console.log('CLIENT: service worker registration in progress.');
  navigator.serviceWorker.register( swUrl , { scope : ' ' } ).then(function() {
    console.log('CLIENT: service worker registration complete.');
  }, function() {
    console.log('CLIENT: service worker registration failure.');
  });
} else {
  console.log('CLIENT: service worker is not supported.');
}

