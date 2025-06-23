const backendUrlInput = document.getElementById('backendUrl');
const loadTransactionBtn = document.getElementById('loadTransactionBtn');
const predictBtn = document.getElementById('predictBtn');
const transactionDisplay = document.getElementById('transaction-display');
const predictionResult = document.getElementById('prediction-result');
const totalPredictionsSpan = document.getElementById('totalPredictions');
const averageLatencySpan = document.getElementById('averageLatency');
const remainingTransactionsSpan = document.getElementById('remainingTransactions');
const predictionDistributionChartCtx = document.getElementById('predictionDistributionChart').getContext('2d');

let currentTransaction = null;
let predictionDistributionChart;

// Initialize Chart.js
function initializeChart() {
    predictionDistributionChart = new Chart(predictionDistributionChartCtx, {
        type: 'bar',
        data: {
            labels: ['Class 0 (Non-Fraud)', 'Class 1 (Fraud)'], // Adjust labels based on your classes
            datasets: [{
                label: 'Prediction Count',
                data: [0, 0],
                backgroundColor: ['rgba(54, 162, 235, 0.6)', 'rgba(255, 99, 132, 0.6)'],
                borderColor: ['rgba(54, 162, 235, 1)', 'rgba(255, 99, 132, 1)'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Count'
                    }
                }
            }
        }
    });
}

// Load backend URL from local storage or use default
function loadBackendUrl() {
    const savedUrl = localStorage.getItem('backendUrl');
    if (savedUrl) {
        backendUrlInput.value = savedUrl;
    }
}

// Save backend URL to local storage
function saveBackendUrl() {
    localStorage.setItem('backendUrl', backendUrlInput.value);
    alert('Backend URL saved!');
}


async function fetchNextTransaction() {
    predictionResult.textContent = ''; // Clear previous prediction
    try {
        const response = await fetch(`${backendUrlInput.value}/get-next-transaction`);
        if (!response.ok) {
            if (response.status === 404) {
                transactionDisplay.textContent = "No more transactions available in the production dataset.";
                loadTransactionBtn.disabled = true;
                predictBtn.disabled = true;
                return;
            }
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        currentTransaction = data.transaction;
        transactionDisplay.textContent = JSON.stringify(currentTransaction, null, 2);
        predictBtn.disabled = false;
        loadTransactionBtn.textContent = 'Load Next Transaction'; // Reset button text
    } catch (error) {
        console.error('Error loading transaction:', error);
        transactionDisplay.textContent = `<span class="error">Error loading transaction: ${error.message}</span>`;
        predictBtn.disabled = true;
    }
}

async function sendPredictionRequest() {
    if (!currentTransaction) {
        predictionResult.textContent = 'Please load a transaction first.';
        return;
    }

    predictBtn.disabled = true; // Disable button during prediction
    predictBtn.textContent = 'Predicting...';

    try {
        const response = await fetch(`${backendUrlInput.value}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(currentTransaction)
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
        }

        const data = await response.json();
        predictionResult.innerHTML = `
            <strong>Prediction:</strong> ${data.prediction_class === 1 ? '<span style="color: red;">Fraud</span>' : '<span style="color: green;">Not Fraud</span>'}<br>
            <strong>Probability:</strong> ${data.prediction_proba ? data.prediction_proba.map(p => p.toFixed(4)).join(', ') : 'N/A'}<br>
            <strong>Latency:</strong> ${data.latency_ms.toFixed(2)} ms
        `;
        fetchMonitoringData(); // Refresh monitoring data after each prediction
    } catch (error) {
        console.error('Error during prediction:', error);
        predictionResult.innerHTML = `<span class="error">Prediction failed: ${error.message}</span>`;
    } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = 'Predict Fraud';
    }
}

async function fetchMonitoringData() {
    try {
        const response = await fetch(`${backendUrlInput.value}/monitoring`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();

        totalPredictionsSpan.textContent = data.total_predictions;
        averageLatencySpan.textContent = data.average_latency_ms.toFixed(2);
        remainingTransactionsSpan.textContent = data.current_production_data_remaining;

        // Update chart
        const fraudCount = data.prediction_distribution['1'] || 0;
        const nonFraudCount = data.prediction_distribution['0'] || 0;
        predictionDistributionChart.data.datasets[0].data = [nonFraudCount, fraudCount];
        predictionDistributionChart.update();

    } catch (error) {
        console.error('Error fetching monitoring data:', error);
        // Display error on dashboard or log
        totalPredictionsSpan.textContent = 'Error';
        averageLatencySpan.textContent = 'Error';
    }
}

// Event Listeners
loadTransactionBtn.addEventListener('click', fetchNextTransaction);
predictBtn.addEventListener('click', sendPredictionRequest);

// On page load
document.addEventListener('DOMContentLoaded', () => {
    loadBackendUrl();
    initializeChart();
    fetchMonitoringData(); // Fetch initial monitoring data
});