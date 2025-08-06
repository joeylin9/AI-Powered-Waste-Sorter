const video = document.getElementById('camera-view');
const titleContainer = document.querySelector('.title-container');
const homeContent = document.getElementById('home-content');
const topText = document.getElementById('top-text');
const loadingState = document.getElementById('loading-state');
const correctionState = document.getElementById('correction-state');
const correctionInput = document.getElementById('correction-input');
const yesButton = document.getElementById('yes-button');
const noButton = document.getElementById('no-button');
const backButton = document.getElementById('back-button');
const dottedOutline = document.querySelector('.dotted-outline');
let classified = false;
let isThinking = false;
let isCountingDown = false;

let correctionStep = 1; // 1 for category selection, 2 for item selection
let selectedCategoryIndex = null;

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
    "PAPER EGG TRAY", // 2
    "PAPER TOWEL ROLL", // 3
    "MIXED OR OTHER PAPER", // 4
    "PAPER TOWEL OR TISSUE", // 5
    "FOOD PACKAGING", // 6
    "RECEIPT", // 7
    "DRINK CARTON", // 8
    "GLITTER PAPER", // 9
    "CRAYON DRAWING", // 10

    // PLASTICS
    "PLASTIC BEVERAGE BOTTLE", // 11
    "SOAP BOTTLE", // 12
    "PLASTIC BAG", // 13
    "BUBBLE WRAP", // 14
    "PLASTIC PACKAGING", // 15
    "REUSABLE PLASTIC CONTAINER", // 16
    "PACKAGING WITH FOIL", // 17
    "MELAMINE PRODUCT", // 18
    "CREDIT CARD", // 19
    "DIRTY PLASTIC PACKAGING", // 20
    "TOY", // 21

    // GLASS
    "GLASS BOTTLE OR JAR", // 22
    "DRINKING OR WINE GLASS", // 23
    "GLASSWARE CONTAINER", // 24
    "TEMPERED GLASS", // 25
    "MIRROR", // 26
    "GLASS WITH METAL WIRES", // 27
    "LIGHT BULB", // 28

    // OTHER
    "PLANT", // 29
    "WOOD", // 30
    "DIAPER", // 31
    "STYROFOAM", // 32
    "PORCELAIN OR CERAMIC", // 33
    "OTHER" // 34
];

const wasteCategories = {
    "PAPER": {
        number: 1,
        items: [
            "CARDBOARD",
            "PAPER EGG TRAY",
            "PAPER TOWEL ROLL",
            "MIXED OR OTHER PAPER",
            "PAPER TOWEL OR TISSUE",
            "FOOD PACKAGING",
            "RECEIPT",
            "DRINK CARTON",
            "GLITTER PAPER",
            "CRAYON DRAWING"
        ]
    },
    "PLASTICS": {
        number: 2,
        items: [
            "PLASTIC BEVERAGE BOTTLE",
            "SOAP BOTTLE",
            "PLASTIC BAG",
            "BUBBLE WRAP",
            "PLASTIC PACKAGING",
            "REUSABLE PLASTIC CONTAINER",
            "PACKAGING WITH FOIL",
            "MELAMINE PRODUCT",
            "CREDIT CARD",
            "DIRTY PLASTIC PACKAGING",
            "TOY"
        ]
    },
    "GLASS": {
        number: 3,
        items: [
            "GLASS BOTTLE OR JAR",
            "DRINKING OR WINE GLASS",
            "GLASSWARE CONTAINER",
            "TEMPERED GLASS",
            "MIRROR",
            "GLASS WITH METAL WIRES",
            "LIGHT BULB"
        ]
    },
    "OTHER": {
        number: 4,
        items: [
            "PLANT",
            "WOOD",
            "DIAPER",
            "STYROFOAM",
            "PORCELAIN OR CERAMIC",
            "OTHER"
        ]
    }
};

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
    if (video.srcObject) return;

    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
            video.style.display = 'block';
        })
        .catch(err => {
            alert("Camera access error: ", err);
        });
}

function captureFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');

    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    return canvas.toDataURL('image/png');
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
    type = currentPrediction.className;
    classifyWaste(type, confidence);
}

function showCorrectionState() {
    backButton.classList.remove('show');
    yesButton.classList.remove('show');
    noButton.classList.remove('show');   

    // Hide captured image when showing correction state
    const imgElem = document.getElementById('captured-image');
    const videoElem = document.getElementById('camera-view');
    if (imgElem && videoElem) {
        imgElem.style.display = 'none';
        videoElem.style.display = 'none';
    }

    topText.classList.remove('show');
    homeContent.classList.add('hide');
    
    // Reset to step 1 and show category selection
    correctionStep = 1;
    selectedCategoryIndex = null;
    showCategorySelection();
    
    correctionState.classList.add('show');
    correctionInput.value = '';
    correctionInput.focus();
}

function showCategorySelection() {
    const correctionTitle = document.querySelector('.correction-title');
    const correctionSubtitle = document.querySelector('.correction-subtitle');
    const categoriesContainer = document.querySelector('.categories-container');
    
    correctionTitle.textContent = "Select the Material";
    correctionSubtitle.textContent = "Please enter the number corresponding to the main material category";
    correctionSubtitle.classList.remove('highlight');
    
    categoriesContainer.innerHTML = `
        <div class="category-selection-grid">
            <div class="main-category-item">
                <div class="main-category-number">1.</div>
                <div class="main-category-name">PAPER</div>
                <div class="main-category-description">Cardboard, receipts, paper packaging, etc.</div>
            </div>
            <div class="main-category-item">
                <div class="main-category-number">2.</div>
                <div class="main-category-name">PLASTICS</div>
                <div class="main-category-description">Beverage bottles, bubble wrap, chip bags, etc.</div>
            </div>
            <div class="main-category-item">
                <div class="main-category-number">3.</div>
                <div class="main-category-name">GLASS</div>
                <div class="main-category-description">Jars, drinking glasses, tempered glass etc.</div>
            </div>
            <div class="main-category-item">
                <div class="main-category-number">4.</div>
                <div class="main-category-name">OTHER</div>
                <div class="main-category-description">Wood, plants, ceramics, styrofoam, etc.</div>
            </div>
        </div>
    `;
    
    // Update input placeholder
    correctionInput.placeholder = "Enter number (1-4)";
    
    // Update button text
    document.getElementById('submit-button').textContent = "Next (+)";
}

let bubblePage = 0;

function showItemSelection(categoryName) {
    const correctionTitle = document.querySelector('.correction-title');
    const correctionSubtitle = document.querySelector('.correction-subtitle');
    const categoriesContainer = document.querySelector('.categories-container');

    correctionTitle.textContent = `Select the Specific Item`;
    correctionSubtitle.innerHTML = "Press <b>ENTER</b> to cycle through items";
    correctionSubtitle.classList.add('highlight');
    correctionInput.value = '';

    const category = wasteCategories[categoryName];
    const itemsPerPage = 4;
    const totalPages = Math.ceil(category.items.length / itemsPerPage);
    correctionInput.placeholder = `Enter a number (1-${category.items.length})`;

    // Build all pages of bubble items with numbering
    const allPages = generateAllPages(category.items, itemsPerPage);
    let itemNumber = 1;
    const allPagesHTML = allPages.map(pageItems => `
        <div class="circular-bubble-grid">
            ${pageItems.map(item => `
                <div class="bubble-item">
                    <img src="images/${item.replace(/ /g, "_").toLowerCase()}.png" alt="${item}" />
                    <div class="bubble-name">${itemNumber++}. ${item}</div>
                    <div class="bubble-example">e.g. example here</div>
                </div>
            `).join('')}
        </div>
    `).join('');

    categoriesContainer.innerHTML = `
        <div class="bubble-slider">
            <div class="bubble-wrapper" style="transform: translateX(0%);">
                ${allPagesHTML}
            </div>
        </div>
    `;

    // Handle ENTER key to shift to next page
    document.onkeydown = function(e) {
        if (correctionStep === 2 && e.key === 'Enter') {
            e.preventDefault();
            bubblePage = (bubblePage + 1) % totalPages;
            const wrapper = document.querySelector('.bubble-wrapper');
            if (wrapper) {
                wrapper.style.transform = `translateX(-${bubblePage * 100}%)`;
            }
        }
    };
}

function generateAllPages(items, itemsPerPage = 4) {
    const pages = [];
    for (let i = 0; i < items.length; i += itemsPerPage) {
        pages.push(items.slice(i, i + itemsPerPage));
    }
    return pages;
}

function generateAllPages(items, itemsPerPage = 4) {
    const pages = [];
    for (let i = 0; i < items.length; i += itemsPerPage) {
        pages.push(items.slice(i, i + itemsPerPage));
    }
    return pages;
}

function correctionBackButtonPress() {
    if (correctionStep === 1) {
        // If already at category selection, just hide correction state
        hideCorrectionState();
    } else if (correctionStep === 2) {
        // Go back to category selection
        correctionStep = 1;
        showCategorySelection();
        
        // Reset input
        correctionInput.value = '';
        correctionInput.placeholder = "Enter number (1-4)";
        correctionInput.focus();
    }
}

function hideCorrectionState() {
    correctionInput.classList.remove('input-error');
    correctionState.classList.remove('show');
    homeContent.classList.remove('hide');
    const imgElem = document.getElementById('captured-image');
    if (imgElem) {
        imgElem.style.display = 'block';
    }

    backButton.classList.add('show');
    yesButton.classList.add('show');
    noButton.classList.add('show');  
    topText.classList.add('show');

    selectedCategoryIndex = null;
}

function submitCorrection() {
    const inputValue = correctionInput.value.trim();
    const numericInput = parseInt(inputValue);
    
    if (correctionStep === 1) {
        // Category selection step
        if (isNaN(numericInput) || numericInput < 1 || numericInput > 4) {
            correctionInput.value = '';
            correctionInput.placeholder = 'Enter a number 1-4!';
            correctionInput.focus();
            correctionInput.classList.add('input-error');
            return;
        }
        
        // Find the selected category
        const categoryNames = Object.keys(wasteCategories);
        const selectedCategory = categoryNames.find(cat => wasteCategories[cat].number === numericInput);
        
        selectedCategoryIndex = numericInput;
        correctionStep = 2;
        
        // Show item selection for this category
        showItemSelection(selectedCategory);
        
    } else if (correctionStep === 2) {
        // Item selection step
        const categoryNames = Object.keys(wasteCategories);
        const selectedCategory = categoryNames.find(cat => wasteCategories[cat].number === selectedCategoryIndex);
        const category = wasteCategories[selectedCategory];
        
        if (isNaN(numericInput) || numericInput < 1 || numericInput > category.items.length) {
            correctionInput.value = '';
            correctionInput.placeholder = `Enter a number 1-${category.items.length}!`;
            correctionInput.focus();
            correctionInput.classList.add('input-error');
            return;
        }

        // Get the selected waste type
        const correctedType = category.items[numericInput - 1];
        
        // Update confidence text to show 100% for corrected items
        document.getElementById('confidence-percentage').textContent = "100%";
        
        hideCorrectionState();
        classifyWaste(correctedType, 1.0);
    }

    correctionInput.classList.remove('input-error');
}

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
                return_all_predictions: true
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        console.log("Full API response:", result); // Debug log
        
        if (result.success) {
            // Update all predictions for dashboard
            if (result.all_predictions) {
                allPredictions = { ...result.all_predictions };
                console.log("Updated allPredictions:", allPredictions);
                if (isDashboardOpen) {
                    updateDashboard();
                }
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

async function startManualClassification() {
    topText.classList.remove('show');
    backButton.classList.remove('show');
    yesButton.classList.remove('show');
    noButton.classList.remove('show');
    
    // Show only video with dotted outline
    dottedOutline.classList.add('show');
    video.style.display = 'block';
    const imgElem = document.getElementById('captured-image');
    imgElem.style.display = 'none';
    
    // Update header and byline for countdown
    isCountingDown = true;
    document.getElementById('main-title').textContent = "3";
    document.getElementById('main-title').style.marginTop = '4vh';
    document.getElementById('byline').textContent = "Hold the item up to the camera";
    document.getElementById('bottom-text').textContent = '';
    
    // Countdown: 3, 2, 1
    await new Promise(resolve => setTimeout(() => {
        document.getElementById('main-title').textContent = "2";
        setTimeout(() => {
            document.getElementById('main-title').textContent = "1";
            setTimeout(resolve, 1000);
        }, 1000);
    }, 1000));
    
    // Hide dotted outline and capture image
    dottedOutline.classList.remove('show');
    const imageData = captureFrame();
    imgElem.src = imageData;
    imgElem.style.display = 'block';
    video.style.display = 'none';

    // Show thinking state
    showThinkingState();
    
    // Make prediction with the captured image data
    try {
        const response = await fetch(API_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: imageData,
                return_all_predictions: true
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        console.log("Manual classification result:", result);
        
        if (result.success) {
            // Update all predictions for dashboard
            if (result.all_predictions) {
                allPredictions = { ...result.all_predictions };
                console.log("Updated allPredictions from manual:", allPredictions);
                if (isDashboardOpen) {
                    updateDashboard();
                }
            }
            
            currentPrediction = {
                className: result.className,
                probability: result.probability
            };
        } else {
            console.error('Manual prediction failed:', result.error);
            currentPrediction = { className: "UNKNOWN", probability: 0 };
        }
        
    } catch (error) {
        console.error('Manual prediction error:', error);
        currentPrediction = { className: "UNKNOWN", probability: 0 };
    }
    
    // Wait for thinking animation
    await new Promise(resolve => setTimeout(resolve, 2000));
    hideThinkingState();
    document.getElementById('main-title').style.marginTop = '10vh';

    // Show result
    showResultState();
}

function classifyWaste(type) {
    document.getElementById('bottom-text').classList.add('result-state');
    backButton.classList.add('show');
    topText.classList.add('show');
    yesButton.classList.add('show');
    noButton.classList.add('show');
    dottedOutline.classList.remove('show');

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
        case "PAPER EGG TRAY":
            instruction = "Paper egg trays are made from recycled paper pulp, please place them in the paper bin.";
            break;
        case "PAPER TOWEL ROLL":
            instruction = "Toilet paper and paper towel rolls are typically made of cardboard, a paper product, so please place them in the paper bin.";
            break;
        case "MIXED OR OTHER PAPER":
            instruction = "Paper products can be placed in the paper bin.";
            break;
        case "PAPER TOWEL OR TISSUE":
            instruction = "Paper towels, tissues, and napkins are often used and unclean, and should be placed in the waste bin.";
            break;
        case "FOOD PACKAGING":
            instruction = "Disposable food and drink paper packaging is often used and unclean, and should be placed in the waste bin.";
            break;
        case "RECEIPT":
            instruction = "There are different types of receipts (e.g. paper and thermal), but all are okay for the blue bins. Please place it in the paper bin.";
            break;
        case "DRINK CARTON":
            instruction = "Drink cartons are made of paper and has a waterproof lining, but can still be recycled. Please place it in the paper bin.";
            break;
        case "GLITTER PAPER":
            instruction = "Glitter paper is not practical to recycle as it is very difficult to separate the non-paper layer. Please place it in the waste bin.";
            break;
        case "CRAYON DRAWING":
            instruction = "Crayon wax prevents water from breaking down paper, so it can't be recycled. Please place it in the waste bin.";
            break;
        
        // PLASTICS
        case "PLASTIC BEVERAGE BOTTLE":
            instruction = "Plastic beverage bottles are recyclable, please place them in the plastic bin.";
            break;
        case "SOAP BOTTLE":
            instruction = "Shampoo, soap, detergent, and other similar bottles are recyclable, please place them in the plastic bin.";
            break;
        case "PLASTIC BAG":
            instruction = "Plastic bags are recyclable, please place them in the plastic bin.";
            break;
        case "BUBBLE WRAP":
            instruction = "Bubble wrap is recyclable, please place them in the plastic bin.";
            break;
        case "PLASTIC PACKAGING":
            instruction = "These items are recyclable, please place them in the plastic bin.";
            break;
        case "REUSABLE PLASTIC CONTAINER":
            instruction = "Reusable food storage containers, such as Tupperware, are recyclable, please place them in the plastic bin.";
            break;
        case "PACKAGING WITH FOIL":
            instruction = "It is difficult to separate the foil from plastic, so it is not recyclable. Please place it in the waste bin.";
            break;
        case "MELAMINE PRODUCT":
            instruction = "Melamine products, such as melamine tableware, are not recyclable and should be placed in the waste bin.";
            break;
        case "CREDIT CARD":
            instruction = "Credit cards are not recyclable. Cancel the card, cut it in half, and place it in the waste bin.";
            break;
        case "DIRTY PLASTIC PACKAGING":
            instruction = "Anything with food waste, even if just a bit of grease, is not recyclable and should be placed in the waste bin.";
            break;
        case "TOY":
            instruction = "Toys should be donated if in good condition, else, because they are made of mixed materials, place in the waste bin.";
            break;

        // GLASS
        case "GLASS BOTTLE OR JAR":
            instruction = "Glass bottles and jars are recyclable, please place them carefully in the glass bin.";
            break;
        case "DRINKING OR WINE GLASS":
            instruction = "Drinking and wine glasses are recyclable, please place them carefully in the glass bin.";
            break;
        case "GLASSWARE CONTAINER":
            instruction = "Glassware containers, such as Pyrex and oven-safe containers, are often used and unclean, and should be placed carefully in the waste bin.";
            break;
        case "TEMPERED GLASS":
            instruction = "Tempered glass has a different chemical composition compared to regular glass, making it unrecyclable. It should be placed carefully in the waste bin.";
            break;
        case "MIRROR":
            instruction = "Mirrors are not recyclable because of the coatings on the glass that make it reflective. They should be placed carefully in the waste bin.";
            break;
        case "GLASS WITH METAL WIRES":
            instruction = "Glass with metal wires is difficult to process so it is not recyclable. It should be placed carefully in the waste bin.";
            break;
        case "LIGHT BULB":
            instruction = "Light bulbs should be placed in the E-waste bin.";
            break;
            
        // OTHER
        case "PLANT":
            instruction = "Plants are not recyclable and should be placed in the waste bin.";
            break;
        case "WOOD":
            instruction = "Wood is not recyclable and should be placed in the waste bin.";
            break;
        case "DIAPER":
            instruction = "Diapers, even if clean, are not recyclable and should be placed in the waste bin.";
            break;
        case "STYROFOAM":
            instruction = "Styrofoam is not recyclable and should be placed in the waste bin.";
            break;
        case "PORCELAIN OR CERAMIC":
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
    document.getElementById('main-title').style.marginTop = '0';

    // Hide all states
    hideCorrectionState();

    // Hide buttons
    document.getElementById('bottom-text').classList.remove('result-state');
    backButton.classList.remove('show');
    yesButton.classList.remove('show');
    noButton.classList.remove('show');
    topText.classList.remove('show');

    // Reset to initial state
    titleContainer.classList.remove('result-state');
    document.getElementById('main-title').textContent = "AI Waste Sorter";
    document.getElementById('byline').textContent = "by Vidacity";
    document.getElementById('bottom-text').innerHTML = '1. Press <b>ENTER</b> to start classification &nbsp;&nbsp; 2. Hold the item up to the camera &nbsp;&nbsp; 3. Follow the instructions!';
    isCountingDown = false;
    
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
    startCamera();
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

document.addEventListener('keydown', function (e) {
    const correctionStateActive = correctionState.classList.contains('show');
    const inputFocused = document.activeElement === correctionInput;

    if (correctionStateActive && inputFocused) {
        if (e.key === '-' || e.key === '_') {
            e.preventDefault();
            document.getElementById('correction-back-button').click();
            return;
        }
        if (e.key === '+' || e.key === '=') {
            e.preventDefault();
            submitCorrection();
            return;
        }
    }

    if (e.key === 'Enter' && !classified && !homeContent.classList.contains('hide')) {
        e.preventDefault();
        if (isCountingDown) return;
        startManualClassification();
        return;
    }

    if ((e.key === '-' || e.key === '_') && yesButton.classList.contains('show')) {
        e.preventDefault();
        yesButton.click();
    }

    if ((e.key === '+' || e.key === '=') && noButton.classList.contains('show')) {
        e.preventDefault();
        noButton.click();
    }

    if (e.key === '9' && backButton.classList.contains('show')) {
        backButton.click();
    }
});
