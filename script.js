const video = document.getElementById('camera-view');
const titleContainer = document.querySelector('.title-container');
const homeContent = document.getElementById('home-content');
const topText = document.getElementById('top-text');
const loadingState = document.getElementById('loading-state');
const correctionState = document.getElementById('correction-state');
const correctionInput = document.getElementById('correction-input');
const backButton = document.getElementById('back-button');
const wrongButton = document.getElementById('wrong-button');

let type = "";
let classified = false;
let isThinking = false;
let classificationTimeout = null;
let resetTimeout = null;

// Stability timer variables
let stabilityTimer = null;
let stabilityStartTime = null;
let lastStablePrediction = null;

// API endpoint for your PyTorch model
const API_ENDPOINT = 'http://localhost:5000/predict';

let currentPrediction = null;
let isModelReady = false;
let items_counter = 0;

// Dashboard variables
let allPredictions = {};
let isDashboardOpen = false;

// Waste type categories for dashboard
const wasteTypes = [
    // PAPER
    "CARDBOARD", // 1
    "PAPER EGG TRAYS", // 2
    "TOILET PAPER AND PAPER TOWEL ROLLS", // 3
    "MIXED AND OTHER PAPER", // 4
    "PAPER TOWELS AND TISSUES", // 5
    "DISPOSABLE FOOD PACKAGING", // 6
    "GLITTER PAPER", // 7
    "CRAYON DRAWINGS", // 8

    // PLASTICS
    "PLASTIC BEVERAGE BOTTLES", // 9
    "SHAMPOO/SOAP/DETERGENT BOTTLES", // 10
    "PLASTIC BAGS", // 11
    "BUBBLE WRAP", // 12
    "PLASTIC PACKAGING", // 13
    "REUSABLE PLASTIC CONTAINERS", // 14
    "PACKAGING WITH FOIL", // 15
    "MELAMINE PRODUCTS", // 16
    "EXPIRED CREDIT CARDS", // 17
    "CONTAMINATED PLASTIC PACKAGING", // 18
    "TOYS", // 19

    // GLASS
    "GLASS BOTTLES AND JARS", // 20
    "DRINKING_WINE GLASSES", // 21
    "GLASSWARE CONTAINERS", // 22
    "TEMPERED GLASS", // 23
    "MIRRORS", // 24
    "GLASS WITH METAL WIRES", // 25
    "LIGHT BULBS", // 26

    // OTHER
    "PLANTS", // 27
    "WOOD", // 28
    "DIAPERS", // 29
    "STYROFOAM", // 30
    "PORCELAIN AND CERAMICS", // 31
    "OTHER" // 32
];

// Initialize predictions object
wasteTypes.forEach(type => {
    allPredictions[type] = 0;
});

// Check if backend is ready
async function checkBackend() {
    try {
        const response = await fetch('http://localhost:5000/health');
        const data = await response.json();
        isModelReady = data.model_loaded;
        console.log("Backend status:", data);
        
        if (isModelReady) {
            startCamera();
        } else {
            alert("Model not loaded on backend");
        }
    } catch (error) {
        alert("Backend not available: " + error);
    }
}

// Initialize when page loads
window.addEventListener('DOMContentLoaded', checkBackend);

// Start camera
function startCamera() {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => { 
            video.srcObject = stream;
            video.style.display = 'block';
            
            // Start prediction loop once camera is ready
            video.onloadeddata = () => {
                console.log("Video ready, starting prediction loop");
                predictionLoop();
            };
        })
        .catch(err => {
            alert("Camera access error: ", err);
        });
}

// Convert video frame to base64 image
function captureFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/png');
}

// Make prediction using PyTorch backend
async function makePrediction() {
    if (!isModelReady || video.readyState !== 4) return null;
    
    try {
        const imageData = captureFrame();
        
        const response = await fetch(API_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: imageData,
                return_all_predictions: true // Request all predictions for dashboard
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            // Update all predictions for dashboard
            if (result.all_predictions) {
                allPredictions = { ...result.all_predictions };
                console.log("Updated allPredictions:", allPredictions);
                if (isDashboardOpen) {
                    updateDashboard();
                }
            } else {
                console.log("No all_predictions in response:", result);
            }
            
            return {
                className: result.className,
                probability: result.probability
            };
        } else {
            console.error('Prediction failed:', result.error);
            return null;
        }
        
    } catch (error) {
        console.error('Prediction error:', error);
        return null;
    }
}

// Prediction loop
async function predictionLoop() {
    if (isModelReady) {
        try {
            const prediction = await makePrediction();
            
            if (prediction && prediction.probability > 0.25) {
                currentPrediction = prediction;
                
                // Clear any existing reset timeout
                if (resetTimeout) {
                    clearTimeout(resetTimeout);
                    resetTimeout = null;
                }
                
                // If not already classified and not thinking
                if (!classified && !isThinking) {
                    // Start or continue stability timer
                    if (!stabilityTimer) {
                        // Start new stability timer
                        stabilityStartTime = Date.now();
                        lastStablePrediction = prediction;
                        
                        stabilityTimer = setTimeout(() => {
                            // After 0.5 seconds of stable detection, start thinking
                            if (lastStablePrediction && !classified && !isThinking) {
                                showThinkingState();
                                classificationTimeout = setTimeout(() => {
                                    hideThinkingState();
                                    showResultState();
                                    classificationTimeout = null;
                                }, 1500); // 1.5 seconds to show thinking animation
                            }
                            stabilityTimer = null;
                            stabilityStartTime = null;
                            lastStablePrediction = null;
                        }, 500); // 0.5 second stability requirement
                    } else {
                        // Update the stable prediction (timer is already running)
                        lastStablePrediction = prediction;
                    }
                }
                
            } else {
                // Low confidence detection
                currentPrediction = null;
                
                // Clear stability timer if running
                if (stabilityTimer) {
                    clearTimeout(stabilityTimer);
                    stabilityTimer = null;
                    stabilityStartTime = null;
                    lastStablePrediction = null;
                }
                
                // Clear classification timeout if pending
                if (classificationTimeout) {
                    clearTimeout(classificationTimeout);
                    classificationTimeout = null;
                }
                
                // Hide thinking state if showing
                if (isThinking) {
                    hideThinkingState();
                }
            }
            
        } catch (error) {
            console.error("Prediction error:", error);
        }
    }
    
    // Continue the prediction loop with a small delay to avoid overwhelming the server
    setTimeout(() => {
        window.requestAnimationFrame(predictionLoop);
    }, 100); // 100ms delay between predictions
}

function showThinkingState() {
    if (isThinking) return;
    
    isThinking = true;
    homeContent.classList.add('hide');
    loadingState.classList.add('show');
}

function hideThinkingState() {
    if (!isThinking) return;
    
    isThinking = false;
    homeContent.classList.remove('hide');
    loadingState.classList.remove('show');
}

function showResultState() {
    const confidence = currentPrediction.probability;
    const confidencePercent = Math.round(confidence * 100);
    
    // Update the confidence percentage in the top text
    document.getElementById('confidence-percentage').textContent = `${confidencePercent}%`;
    
    topText.classList.add('show');
    type = currentPrediction.className.toUpperCase();
    classifyWaste(type, confidence);
}

function showCorrectionState() {
    if (autoReturnTimer) {
        clearTimeout(autoReturnTimer);
    }
    
    backButton.classList.remove('show');
    wrongButton.classList.remove('show');   

    // Hide captured image when showing correction state
    const imgElem = document.getElementById('captured-image');
    const videoElem = document.getElementById('camera-view');
    if (imgElem && videoElem) {
        imgElem.style.display = 'none';
        videoElem.style.display = 'none';
    }

    topText.classList.remove('show');
    homeContent.classList.add('hide');
    correctionState.classList.add('show');
    correctionInput.value = '';
    correctionInput.focus();
}

function hideCorrectionState() {
    correctionState.classList.remove('show');
    homeContent.classList.remove('hide');
    const imgElem = document.getElementById('captured-image');
    if (imgElem) {
        imgElem.style.display = 'block';
    }

    backButton.classList.add('show');
    wrongButton.classList.add('show');  
    topText.classList.add('show');
}

function submitCorrection() {
    const inputValue = correctionInput.value.trim();
    const wasteTypesForCorrection = [
        // PAPER
        "CARDBOARD", // 1
        "PAPER EGG TRAYS", // 2
        "TOILET PAPER AND PAPER TOWEL ROLLS", // 3
        "MIXED AND OTHER PAPER", // 4
        "PAPER TOWELS AND TISSUES", // 5
        "DISPOSABLE FOOD PACKAGING", // 6
        "GLITTER PAPER", // 7
        "CRAYON DRAWINGS", // 8

        // PLASTICS
        "PLASTIC BEVERAGE BOTTLES", // 9
        "SHAMPOO/SOAP/DETERGENT BOTTLES", // 10
        "PLASTIC BAGS", // 11
        "BUBBLE WRAP", // 12
        "PLASTIC PACKAGING", // 13
        "REUSABLE PLASTIC CONTAINERS", // 14
        "PACKAGING WITH FOIL", // 15
        "MELAMINE PRODUCTS", // 16
        "EXPIRED CREDIT CARDS", // 17
        "CONTAMINATED PLASTIC PACKAGING", // 18
        "TOYS", // 19

        // GLASS
        "GLASS BOTTLES AND JARS", // 20
        "DRINKING_WINE GLASSES", // 21
        "GLASSWARE CONTAINERS", // 22
        "TEMPERED GLASS", // 23
        "MIRRORS", // 24
        "GLASS WITH METAL WIRES", // 25
        "LIGHT BULBS", // 26

        // OTHER
        "PLANTS", // 27
        "WOOD", // 28
        "DIAPERS", // 29
        "STYROFOAM", // 30
        "PORCELAIN AND CERAMICS", // 31
        "OTHER" // 32
    ];
    
    // Check if input is a number between 1-32
    const numericInput = parseInt(inputValue);
    if (isNaN(numericInput) || numericInput < 1 || numericInput > 32) {
        correctionInput.classList.add('input-error');
        correctionInput.value = '';
        correctionInput.placeholder = 'Enter a number 1-32!';
        correctionInput.focus();
        return;
    }
    
    // Convert number to waste type (array is 0-indexed, input is 1-indexed)
    const correctedType = wasteTypesForCorrection[numericInput - 1];
    
    // Update confidence text to show 100% for corrected items
    document.getElementById('confidence-percentage').textContent = "100%";
    
    hideCorrectionState();
    classifyWaste(correctedType, 1.0);
}

// Add Enter key support for correction input
document.getElementById('correction-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        submitCorrection();
    }
});

let autoReturnTimer = null;

function classifyWaste(type, confidence) {
    document.getElementById('bottom-text').classList.add('result-state');
    topText.classList.add('show');
    backButton.classList.add('show');
    wrongButton.classList.add('show');

    if (autoReturnTimer) {
        clearTimeout(autoReturnTimer);
    }

    // Capture the current video frame and display as image
    const videoElem = document.getElementById('camera-view');
    const imgElem = document.getElementById('captured-image');
    if (videoElem && imgElem && videoElem.readyState === 4) {
        const canvas = document.createElement('canvas');
        canvas.width = videoElem.videoWidth;
        canvas.height = videoElem.videoHeight;
        const ctx = canvas.getContext('2d');
        // Mirror the image to match the video
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(videoElem, 0, 0, canvas.width, canvas.height);
        imgElem.src = canvas.toDataURL('image/png');
        imgElem.style.display = 'block';
        videoElem.style.display = 'none';
    }

    let instruction = "";
    switch (type) {
        // PAPERS
        case "CARDBOARD":
            instruction = "Cardboard is a paper product, please place it in the paper bin.";
            break;
        case "PAPER EGG TRAYS":
            instruction = "Paper egg trays are made from recycled paper pulp, please place them in the paper bin.";
            break;
        case "TOILET PAPER AND PAPER TOWEL ROLLS":
            instruction = "Toilet paper and paper towel rolls are typically made of cardboard, a paper product, so please place them in the paper bin.";
            break;
        case "MIXED AND OTHER PAPER":
            instruction = "Paper products can be placed in the paper bin.";
            break;
        case "PAPER TOWELS AND TISSUES":
            instruction = "Paper towels, tissues, and napkins are often used and unclean, and should be placed in the waste bin.";
            break;
        case "DISPOSABLE FOOD PACKAGING":
            instruction = "Disposable food and drink paper packaging is often used and unclean, and should be placed in the waste bin.";
            break;
        case "GLITTER PAPER":
            instruction = "Glitter paper is not practical to recycle as it is very difficult to separate the non-paper layer. Please place it in the waste bin.";
            break;
        case "CRAYON DRAWINGS":
            instruction = "During recycling, paper is broken down by water, but crayon wax prevents the process and thus makes the item unrecyclable. Please place it in the waste bin.";
            break;
        
        // PLASTICS
        case "PLASTIC BEVERAGE BOTTLES":
            instruction = "Plastic beverage bottles are recyclable, please place them in the plastic bin.";
            break;
        case "SHAMPOO/SOAP/DETERGENT BOTTLES":
            instruction = "Shampoo, soap, detergent, and other similar bottles are recyclable, please place them in the plastic bin.";
            break;
        case "PLASTIC BAGS":
            instruction = "Plastic bags are recyclable, please place them in the plastic bin.";
            break;
        case "BUBBLE WRAP":
            instruction = "Bubble wrap is recyclable, please place them in the plastic bin.";
            break;
        case "PLASTIC PACKAGING":
            instruction = "These items are recyclable, please place them in the plastic bin.";
            break;
        case "REUSABLE PLASTIC CONTAINERS":
            instruction = "Reusable food storage containers, such as Tupperware, are recyclable, please place them in the plastic bin.";
            break;
        case "PACKAGING WITH FOIL":
            instruction = "It is difficult to separate the foil from plastic, so it is not recyclable. Please place it in the waste bin.";
            break;
        case "MELAMINE PRODUCTS":
            instruction = "Melamine products, such as melamine tableware, are not recyclable and should be placed in the waste bin.";
            break;
        case "EXPIRED CREDIT CARDS":
            instruction = "Expired credit cards are not recyclable. Cancel the card, cut it in half, and place it in the waste bin.";
            break;
        case "CONTAMINATED PLASTIC PACKAGING":
            instruction = "Anything with food waste, even if just a bit of grease, is not recyclable and should be placed in the waste bin.";
            break;
        case "TOYS":
            instruction = "Toys should be donated if in good condition, else, because they are made of mixed materials, place in the waste bin.";
            break;

        // GLASS
        case "GLASS BOTTLES AND JARS":
            instruction = "Glass bottles and jars are recyclable, please place them carefully in the glass bin.";
            break;
        case "DRINKING_WINE GLASSES":
            instruction = "Drinking and wine glasses are recyclable, please place them carefully in the glass bin.";
            break;
        case "GLASSWARE CONTAINERS":
            instruction = "Glassware containers, such as Pyrex and oven-safe containers, are often used and unclean, and should be placed carefully in the waste bin.";
            break;
        case "TEMPERED GLASS":
            instruction = "Tempered glass has a different chemical composition compared to regular glass, making it unrecyclable. It should be placed carefully in the waste bin.";
            break;
        case "MIRRORS":
            instruction = "Mirrors are not recyclable because of the coatings on the glass that make it reflective. They should be placed carefully in the waste bin.";
            break;
        case "GLASS WITH METAL WIRES":
            instruction = "Glass with metal wires is difficult to process so it is not recyclable. It should be placed carefully in the waste bin.";
            break;
        case "LIGHT BULBS":
            instruction = "Light bulbs should be placed in the E-waste bin.";
            break;
            
        // OTHER
        case "PLANTS":
            instruction = "Plants are not recyclable and should be placed in the waste bin.";
            break;
        case "WOOD":
            instruction = "Wood is not recyclable and should be placed in the waste bin.";
            break;
        case "DIAPERS":
            instruction = "Diapers, even if clean, are not recyclable and should be placed in the waste bin.";
            break;
        case "STYROFOAM":
            instruction = "Styrofoam is not recyclable and should be placed in the waste bin.";
            break;
        case "PORCELAIN AND CERAMICS":
            instruction = "Porcelain and ceramics are not recyclable and should be placed in the waste bin.";
            break;

        default:
            instruction = "Please check with local waste management for proper disposal. Recyclopedia.sg is a good resource!";
    }

    // Change display to result state
    titleContainer.classList.add('result-state');
    document.getElementById('main-title').textContent = type;
    document.getElementById('byline').textContent = instruction;

    const bottomText = document.getElementById('bottom-text');
    bottomText.innerHTML = 'Is the classification correct?';

    classified = true;
}

function resetToHomeScreen() {
    if (!classified) return;

    // Clear all timers
    if (stabilityTimer) {
        clearTimeout(stabilityTimer);
        stabilityTimer = null;
        stabilityStartTime = null;
        lastStablePrediction = null;
    }

    if (classificationTimeout) {
        clearTimeout(classificationTimeout);
        classificationTimeout = null;
    }

    // Clear auto-return timer if it exists
    if (autoReturnTimer) {
        clearTimeout(autoReturnTimer);
        autoReturnTimer = null;
    }

    // Hide all states
    hideCorrectionState();

    // Hide buttons
    document.getElementById('bottom-text').classList.remove('result-state');
    backButton.classList.remove('show');
    wrongButton.classList.remove('show');
    topText.classList.remove('show');

    // Reset to initial state
    titleContainer.classList.remove('result-state');
    document.getElementById('main-title').textContent = "AI Waste Sorter";
    document.getElementById('byline').textContent = "by Vidacity";
    document.getElementById('bottom-text').innerHTML = '1. Hold your item up to the camera &nbsp;&nbsp; 2. Wait for classifiction &nbsp;&nbsp; 3. Follow the instructions!';
    
    // Reset confidence text to default
    document.getElementById('confidence-percentage').textContent = "";

    // Show video, hide captured image
    const videoElem = document.getElementById('camera-view');
    const imgElem = document.getElementById('captured-image');
    if (videoElem && imgElem) {
        videoElem.style.display = 'block';
        imgElem.style.display = 'none';
    }

    classified = false;
}

// Dashboard functions
function showDashboard() {
    isDashboardOpen = true;
    document.getElementById('dashboard-overlay').classList.add('show');
    initializeDashboard();
}

function hideDashboard(event) {
    if (event && event.target !== event.currentTarget) return;
    isDashboardOpen = false;
    document.getElementById('dashboard-overlay').classList.remove('show');
}

function initializeDashboard() {
    const grid = document.getElementById('predictions-grid');
    grid.innerHTML = '';
    
    wasteTypes.forEach((type, index) => {
        const item = document.createElement('div');
        item.className = 'prediction-item';
        item.id = `prediction-${index}`;
        
        const displayName = type.replace(/_/g, ' / ');
        
        item.innerHTML = `
            <div class="prediction-label">
                <span>${displayName}</span>
                <span class="prediction-value" id="value-${index}">0%</span>
            </div>
            <div class="prediction-bar-container">
                <div class="prediction-bar" id="bar-${index}" style="width: 0%"></div>
            </div>
        `;
        
        grid.appendChild(item);
    });
    
    updateDashboard();
}

function updateDashboard() {
    if (!isDashboardOpen) return;
    
    console.log("Updating dashboard with predictions:", allPredictions);
    
    // Sort predictions by value for top 5
    const sortedPredictions = Object.entries(allPredictions)
        .sort(([,a], [,b]) => b - a)
        .slice(0, 5);
    
    console.log("Top 5 predictions:", sortedPredictions);
    
    // Update top predictions
    const topList = document.getElementById('top-predictions-list');
    topList.innerHTML = sortedPredictions.map(([type, confidence]) => `
        <div class="top-prediction-item">
            <span class="top-prediction-name">${type.replace(/_/g, ' / ')}</span>
            <span class="top-prediction-confidence">${Math.round(confidence * 100)}%</span>
        </div>
    `).join('');
    
    // Update all prediction bars
    wasteTypes.forEach((type, index) => {
        const confidence = allPredictions[type] || 0;
        const percentage = Math.round(confidence * 100);
        
        const valueElement = document.getElementById(`value-${index}`);
        const barElement = document.getElementById(`bar-${index}`);
        const itemElement = document.getElementById(`prediction-${index}`);
        
        if (valueElement && barElement && itemElement) {
            valueElement.textContent = `${percentage}%`;
            barElement.style.width = `${percentage}%`;
            
            // Update bar color based on confidence
            barElement.classList.remove('high-confidence', 'medium-confidence');
            if (confidence > 0.7) {
                barElement.classList.add('high-confidence');
                itemElement.classList.add('active');
            } else if (confidence > 0.3) {
                barElement.classList.add('medium-confidence');
                itemElement.classList.remove('active');
            } else {
                itemElement.classList.remove('active');
            }
        }
    });
}

// Add keyboard shortcut for dashboard (press 'D')
document.addEventListener('keydown', function(e) {
    if (e.key.toLowerCase() === 'd' && !isDashboardOpen) {
        showDashboard();
    } else if (e.key === 'Escape' && isDashboardOpen) {
        hideDashboard();
    }
});