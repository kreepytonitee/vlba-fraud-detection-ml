import React, { useState, useEffect, useRef, useCallback } from 'react';

// Define FastAPI backend URL
// const FASTAPI_BASE_URL = 'http://localhost:8000';
const FASTAPI_BASE_URL = 'https://fraud-detection-service-429587376884.europe-west10.run.app';

function App() {
    const [transaction, setTransaction] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [monitoringData, setMonitoringData] = useState(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [autoplaySpeed, setAutoplaySpeed] = useState(2000); // 2 seconds default

    const intervalRef = useRef(null);
    const metrics = useRef({
        truePositives: 0,
        falsePositives: 0,
        trueNegatives: 0,
        falseNegatives: 0,
        totalPredictions: 0,
        correctPredictions: 0,
    });

    // Function to calculate and update classification report metrics
    const updateClassificationMetrics = useCallback((predictedClass, trueLabel) => {
        metrics.current.totalPredictions++;

        if (predictedClass === trueLabel) {
            metrics.current.correctPredictions++;
        }

        if (predictedClass === 1 && trueLabel === 1) {
            metrics.current.truePositives++;
        } else if (predictedClass === 1 && trueLabel === 0) {
            metrics.current.falsePositives++;
        } else if (predictedClass === 0 && trueLabel === 0) {
            metrics.current.trueNegatives++;
        } else if (predictedClass === 0 && trueLabel === 1) {
            metrics.current.falseNegatives++;
        }
    }, []);

    // Function to fetch and process the next transaction
    const fetchNextTransaction = useCallback(async () => {
        setIsLoading(true);
        setError(null);
        try {
            // Fetch next transaction
            const transactionResponse = await fetch(`${FASTAPI_BASE_URL}/get-next-transaction`);
            if (!transactionResponse.ok) {
                if (transactionResponse.status === 404) {
                    throw new Error("No more transactions in the dataset.");
                }
                throw new Error(`HTTP error! status: ${transactionResponse.status}`);
            }
            const transactionData = await transactionResponse.json();
            const currentTransaction = transactionData.transaction;
            setTransaction(currentTransaction);

            // Send transaction for prediction
            const predictResponse = await fetch(`${FASTAPI_BASE_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(currentTransaction),
            });

            if (!predictResponse.ok) {
                throw new Error(`HTTP error! status: ${predictResponse.status}`);
            }
            const predictionData = await predictResponse.json();
            setPrediction(predictionData);

            // Update local classification metrics
            if (typeof predictionData.true_label !== 'undefined' && predictionData.true_label !== null) {
                updateClassificationMetrics(predictionData.prediction_class, predictionData.true_label);
            } else {
                console.warn("True label not found in prediction response. Metrics might be incomplete.");
                // Fallback to currentTransaction['Is Laundering'] if true_label is somehow missing from predictionData
                if (currentTransaction && typeof currentTransaction['Is Laundering'] !== 'undefined') {
                     updateClassificationMetrics(predictionData.prediction_class, currentTransaction['Is Laundering']);
                }
            }

        } catch (e) {
            console.error("Error fetching transaction or prediction:", e);
            setError(e.message);
            setIsPlaying(false); // Stop autoplay on error
        } finally {
            setIsLoading(false);
        }
    }, [updateClassificationMetrics]);

    // Function to fetch monitoring data
    const fetchMonitoringData = useCallback(async () => {
        try {
            const response = await fetch(`${FASTAPI_BASE_URL}/monitoring`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            setMonitoringData(data);
        } catch (e) {
            console.error("Error fetching monitoring data:", e);
            // Don't set error state for monitoring data, just log
        }
    }, []);

    // Auto-play control
    useEffect(() => {
        if (isPlaying) {
            intervalRef.current = setInterval(() => {
                fetchNextTransaction();
                fetchMonitoringData();
            }, autoplaySpeed);
        } else {
            clearInterval(intervalRef.current);
        }
        return () => clearInterval(intervalRef.current);
    }, [isPlaying, autoplaySpeed, fetchNextTransaction, fetchMonitoringData]);

    // Initial fetch of monitoring data on component mount
    useEffect(() => {
        fetchMonitoringData();
    }, [fetchMonitoringData]);

    const handleStartAutoplay = () => setIsPlaying(true);
    const handleStopAutoplay = () => setIsPlaying(false);

    const handleRestart = async () => {
        setIsPlaying(false);
        setIsLoading(true);
        setError(null);
        
        // Reset local metrics
        metrics.current = {
            truePositives: 0, falsePositives: 0, trueNegatives: 0, falseNegatives: 0,
            totalPredictions: 0, correctPredictions: 0,
        };
        setTransaction(null);
        setPrediction(null);
        setMonitoringData(null);

        try {
            // Reset dataset on backend
            const resetResponse = await fetch(`${FASTAPI_BASE_URL}/reset-dataset`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
            });

            if (!resetResponse.ok) {
                throw new Error(`Failed to reset dataset: ${resetResponse.status}`);
            }

            // Fetch fresh monitoring data
            await fetchMonitoringData();
        } catch (e) {
            setError(e.message);
        } finally {
            setIsLoading(false);
        }
    };

    // Calculate derived metrics for classification report
    const calculateMetrics = () => {
        const { truePositives, falsePositives, trueNegatives, falseNegatives, totalPredictions, correctPredictions } = metrics.current;

        const accuracy = totalPredictions > 0 ? (correctPredictions / totalPredictions) : 0;
        const precision = (truePositives + falsePositives) > 0 ? (truePositives / (truePositives + falsePositives)) : 0;
        const recall = (truePositives + falseNegatives) > 0 ? (truePositives / (truePositives + falseNegatives)) : 0;
        const f1Score = (precision + recall) > 0 ? (2 * precision * recall / (precision + recall)) : 0;

        return {
            accuracy: accuracy.toFixed(4),
            precision: precision.toFixed(4),
            recall: recall.toFixed(4),
            f1Score: f1Score.toFixed(4),
            confusionMatrix: {
                TP: truePositives,
                FP: falsePositives,
                TN: trueNegatives,
                FN: falseNegatives,
            },
        };
    };

    // Helper function to format time
    const formatTime = (seconds) => {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    };

    const classificationReport = calculateMetrics();

    return (
        <div className="min-h-screen bg-gray-100 p-4 sm:p-8 font-inter antialiased">
            <style>
                {`
                .font-inter {
                    font-family: 'Inter', sans-serif;
                }
                .card {
                    background-color: white;
                    border-radius: 0.75rem;
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                    padding: 1.5rem;
                }
                .fixed-height-panel {
                    min-height: 350px;
                    max-height: 450px;
                    overflow-y: auto;
                }
                .metric-card {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border-radius: 0.75rem;
                    padding: 1.5rem;
                    text-align: center;
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                }
                .status-indicator {
                    display: inline-block;
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    margin-right: 8px;
                }
                .status-active {
                    background-color: #10b981;
                    animation: pulse 2s infinite;
                }
                .status-inactive {
                    background-color: #6b7280;
                }
                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.5; }
                }
                .prediction-badge {
                    display: inline-block;
                    padding: 0.5rem 1rem;
                    border-radius: 0.5rem;
                    font-weight: bold;
                    font-size: 1.1rem;
                    text-align: center;
                    min-width: 120px;
                }
                .fraud-badge {
                    background-color: #fef2f2;
                    color: #dc2626;
                    border: 2px solid #fca5a5;
                }
                .not-fraud-badge {
                    background-color: #f0fdf4;
                    color: #16a34a;
                    border: 2px solid #86efac;
                }
                `}
            </style>

            <div className="max-w-7xl mx-auto">
                <h1 className="text-3xl sm:text-4xl font-bold text-center text-gray-800 mb-8">
                    Fraud Detection Dashboard
                </h1>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Left Column: Transaction Details & Prediction */}
                    <div className="lg:col-span-2 space-y-6">
                        <div className="card">
                            <h2 className="text-xl sm:text-2xl font-semibold text-gray-700 mb-4">
                                Transaction Details & Prediction
                            </h2>
                            <div className="fixed-height-panel bg-gray-50 p-4 rounded-lg border border-gray-200">
                                {isLoading && (
                                    <div className="flex items-center justify-center h-full">
                                        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
                                        <span className="ml-3 text-gray-600">Loading transaction and prediction...</span>
                                    </div>
                                )}
                                
                                {error && (
                                    <div className="bg-red-50 border border-red-200 rounded-md p-4">
                                        <p className="text-red-700 font-medium">Error: {error}</p>
                                    </div>
                                )}

                                {!isLoading && !error && !transaction && (
                                    <div className="flex items-center justify-center h-full text-gray-600">
                                        <div className="text-center">
                                            <p className="text-lg mb-2">Ready to start</p>
                                            <p className="text-sm">Click "Load Next Transaction" or "Start Autoplay" to begin.</p>
                                        </div>
                                    </div>
                                )}

                                {transaction && (
                                    <div className="space-y-4">
                                        <div className="bg-white p-4 rounded-lg border border-gray-200">
                                            <h3 className="text-lg font-medium text-gray-800 mb-3">
                                                Transaction ID: {transaction['Unnamed: 0']}
                                            </h3>
                                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm">
                                                <p><span className="font-semibold">Timestamp:</span> {transaction['Timestamp']}</p>
                                                <p><span className="font-semibold">From Bank:</span> {transaction['From Bank']} ({transaction['Account']})</p>
                                                <p><span className="font-semibold">To Bank:</span> {transaction['To Bank']} ({transaction['Account.1']})</p>
                                                <p><span className="font-semibold">Amount Received:</span> {transaction['Amount Received']} {transaction['Receiving Currency']}</p>
                                                <p><span className="font-semibold">Amount Paid:</span> {transaction['Amount Paid']} {transaction['Payment Currency']}</p>
                                                <p><span className="font-semibold">Payment Format:</span> {transaction['Payment Format']}</p>
                                            </div>
                                            <div className="mt-3 pt-3 border-t border-gray-200">
                                                <p className="text-sm">
                                                    <span className="font-semibold">True Label:</span>{' '}
                                                    <span className={`${transaction['Is Laundering'] === 1 ? 'text-red-600' : 'text-green-600'} font-bold`}>
                                                        {typeof transaction['Is Laundering'] === 'number' ? 
                                                            (transaction['Is Laundering'] === 1 ? 'Fraud' : 'Not Fraud') : 'N/A'}
                                                    </span>
                                                </p>
                                            </div>
                                        </div>

                                        {prediction && (
                                            <div className="bg-white p-4 rounded-lg border border-gray-200">
                                                <h3 className="text-xl font-semibold text-gray-800 mb-3">Prediction Result</h3>
                                                <div className="flex items-center justify-between mb-4">
                                                    <div className="flex items-center">
                                                        <span className="text-lg font-bold mr-3">Prediction:</span>
                                                        <span className={`prediction-badge ${prediction.prediction_class === 1 ? 'fraud-badge' : 'not-fraud-badge'}`}>
                                                            {prediction.prediction_class === 1 ? 'FRAUD' : 'NOT FRAUD'}
                                                        </span>
                                                    </div>
                                                    <div className="text-sm text-gray-600">
                                                        Confidence: {(Math.max(...prediction.prediction_probabilities) * 100).toFixed(2)}%
                                                    </div>
                                                </div>
                                                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm">
                                                    <p><span className="font-semibold">Prob. Not Fraud:</span> {prediction.prediction_probabilities[0].toFixed(4)}</p>
                                                    <p><span className="font-semibold">Prob. Fraud:</span> {prediction.prediction_probabilities[1].toFixed(4)}</p>
                                                    <p><span className="font-semibold">Prediction Time:</span> {prediction.latency_ms.toFixed(2)} ms</p>
                                                    <p><span className="font-semibold">Match:</span> 
                                                        <span className={`ml-1 font-bold ${
                                                            prediction.prediction_class === transaction['Is Laundering'] ? 'text-green-600' : 'text-red-600'
                                                        }`}>
                                                            {prediction.prediction_class === transaction['Is Laundering'] ? 'Correct' : 'Incorrect'}
                                                        </span>
                                                    </p>
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Controls */}
                        <div className="card">
                            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                                <button
                                    onClick={fetchNextTransaction}
                                    disabled={isLoading || isPlaying}
                                    className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-lg shadow-md transition duration-300 ease-in-out transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed text-sm"
                                >
                                    {isLoading ? 'Loading...' : 'Load Next'}
                                </button>
                                <button
                                    onClick={handleStartAutoplay}
                                    disabled={isPlaying || isLoading}
                                    className="bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-4 rounded-lg shadow-md transition duration-300 ease-in-out transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed text-sm"
                                >
                                    Start Auto
                                </button>
                                <button
                                    onClick={handleStopAutoplay}
                                    disabled={!isPlaying}
                                    className="bg-yellow-600 hover:bg-yellow-700 text-white font-bold py-3 px-4 rounded-lg shadow-md transition duration-300 ease-in-out transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed text-sm"
                                >
                                    Stop Auto
                                </button>
                                <button
                                    onClick={handleRestart}
                                    disabled={isLoading}
                                    className="bg-red-600 hover:bg-red-700 text-white font-bold py-3 px-4 rounded-lg shadow-md transition duration-300 ease-in-out transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed text-sm"
                                >
                                    Restart
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* Right Column: Monitoring Dashboards */}
                    <div className="lg:col-span-1 space-y-6">
                        {/* Key Metrics Overview */}
                        <div className="grid grid-cols-2 gap-3">
                            <div className="metric-card">
                                <div className="text-2xl font-bold">{metrics.current.totalPredictions}</div>
                                <div className="text-sm opacity-90">Total Predictions</div>
                            </div>
                            <div className="metric-card">
                                <div className="text-2xl font-bold">
                                    {metrics.current.totalPredictions > 0 ? 
                                        ((metrics.current.correctPredictions / metrics.current.totalPredictions) * 100).toFixed(1) : '0.0'}%
                                </div>
                                <div className="text-sm opacity-90">Accuracy</div>
                            </div>
                        </div>

                        {/* Model Monitoring */}
                        <div className="card">
                            <h2 className="text-xl font-semibold text-gray-700 mb-4">Model Performance</h2>
                            {monitoringData ? (
                                <div className="space-y-3">
                                    <div className="bg-gray-50 p-3 rounded-lg">
                                        <h4 className="font-semibold text-gray-800 mb-2">Prediction Distribution</h4>
                                        <div className="grid grid-cols-2 gap-2 text-sm">
                                            <p>Not Fraud: <span className="font-bold text-green-600">{monitoringData.prediction_distribution['0'] || 0}</span></p>
                                            <p>Fraud: <span className="font-bold text-red-600">{monitoringData.prediction_distribution['1'] || 0}</span></p>
                                        </div>
                                    </div>
                                    
                                    <div className="bg-gray-50 p-3 rounded-lg">
                                        <h4 className="font-semibold text-gray-800 mb-2">Live Classification Report</h4>
                                        <div className="grid grid-cols-2 gap-2 text-sm">
                                            <p>Accuracy: <span className="font-bold">{classificationReport.accuracy}</span></p>
                                            <p>Precision: <span className="font-bold">{classificationReport.precision}</span></p>
                                            <p>Recall: <span className="font-bold">{classificationReport.recall}</span></p>
                                            <p>F1-Score: <span className="font-bold">{classificationReport.f1Score}</span></p>
                                        </div>
                                    </div>

                                    <div className="bg-gray-50 p-3 rounded-lg">
                                        <h4 className="font-semibold text-gray-800 mb-2">Confusion Matrix</h4>
                                        <div className="grid grid-cols-2 gap-1 text-xs">
                                            <div className="bg-green-100 p-2 rounded text-center">
                                                <div className="font-bold text-green-800">TP: {classificationReport.confusionMatrix.TP}</div>
                                                <div className="text-green-600">True Positive</div>
                                            </div>
                                            <div className="bg-red-100 p-2 rounded text-center">
                                                <div className="font-bold text-red-800">FP: {classificationReport.confusionMatrix.FP}</div>
                                                <div className="text-red-600">False Positive</div>
                                            </div>
                                            <div className="bg-red-100 p-2 rounded text-center">
                                                <div className="font-bold text-red-800">FN: {classificationReport.confusionMatrix.FN}</div>
                                                <div className="text-red-600">False Negative</div>
                                            </div>
                                            <div className="bg-green-100 p-2 rounded text-center">
                                                <div className="font-bold text-green-800">TN: {classificationReport.confusionMatrix.TN}</div>
                                                <div className="text-green-600">True Negative</div>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="text-sm text-gray-600">
                                        <p>Avg. Latency: <span className="font-semibold">{monitoringData.average_latency_ms.toFixed(2)} ms</span></p>
                                        <p>Remaining: <span className="font-semibold">{monitoringData.current_production_data_remaining}</span></p>
                                    </div>
                                </div>
                            ) : (
                                <div className="flex items-center justify-center h-32">
                                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                                    <span className="ml-2 text-gray-600">Loading...</span>
                                </div>
                            )}
                        </div>

                        {/* Server Monitoring */}
                        <div className="card">
                            <h2 className="text-xl font-semibold text-gray-700 mb-4">Server Performance</h2>
                            {monitoringData ? (
                                <div className="space-y-3">
                                    <div className="bg-gray-50 p-3 rounded-lg">
                                        <h4 className="font-semibold text-gray-800 mb-2">Request Metrics</h4>
                                        <div className="space-y-2 text-sm">
                                            <p>Total Requests: <span className="font-bold text-blue-600">{monitoringData.total_requests}</span></p>
                                            <p>Avg. Response Time: <span className="font-bold">{monitoringData.avg_response_time_ms.toFixed(2)} ms</span></p>
                                            <p>Last Request: <span className="font-bold">{monitoringData.last_request_duration_ms.toFixed(2)} ms</span></p>
                                        </div>
                                    </div>
                                    
                                    <div className="bg-gray-50 p-3 rounded-lg">
                                        <h4 className="font-semibold text-gray-800 mb-2">System Health</h4>
                                        <div className="space-y-2 text-sm">
                                            <p>Error Count: <span className={`font-bold ${monitoringData.error_count > 0 ? 'text-red-600' : 'text-green-600'}`}>{monitoringData.error_count}</span></p>
                                            <p>Error Rate: <span className={`font-bold ${monitoringData.error_rate_percent > 5 ? 'text-red-600' : 'text-green-600'}`}>{monitoringData.error_rate_percent.toFixed(2)}%</span></p>
                                            <p>Uptime: <span className="font-bold text-green-600">{formatTime(monitoringData.uptime_seconds)}</span></p>
                                        </div>
                                    </div>

                                    <div className="text-xs text-gray-500 mt-3">
                                        <p>Last updated: {new Date().toLocaleTimeString()}</p>
                                    </div>
                                </div>
                            ) : (
                                <div className="flex items-center justify-center h-32">
                                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                                    <span className="ml-2 text-gray-600">Loading...</span>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default App;