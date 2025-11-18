document.addEventListener('DOMContentLoaded', function() {
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    const clearBtn = document.getElementById('clearBtn');
    const predictBtn = document.getElementById('predictBtn');
    const predictionsDiv = document.getElementById('predictions');
    
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;
    
    // Set canvas background to white
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    // Drawing functions
    function startDrawing(e) {
        isDrawing = true;
        [lastX, lastY] = [e.offsetX, e.offsetY];
    }
    
    function draw(e) {
        if (!isDrawing) return;
        
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
        
        [lastX, lastY] = [e.offsetX, e.offsetY];
    }
    
    function stopDrawing() {
        isDrawing = false;
    }
    
    // Event listeners for drawing
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    
    // Touch support for mobile devices
    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent('mousedown', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        canvas.dispatchEvent(mouseEvent);
    });
    
    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent('mousemove', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        canvas.dispatchEvent(mouseEvent);
    });
    
    canvas.addEventListener('touchend', (e) => {
        e.preventDefault();
        const mouseEvent = new MouseEvent('mouseup', {});
        canvas.dispatchEvent(mouseEvent);
    });
    
    // Clear canvas
    clearBtn.addEventListener('click', () => {
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        predictionsDiv.innerHTML = '<p class="text-muted">Draw a digit and click \'Predict\'</p>';
    });
    
    // Predict digit
    predictBtn.addEventListener('click', async () => {
        // Show loading state
        predictionsDiv.innerHTML = '<div class="text-center"><div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div><p class="mt-2">Predicting...</p></div>';
        
        // Convert canvas to base64
        const imageData = canvas.toDataURL('image/png');
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData })
            });
            
            const data = await response.json();
            
            if (data.success && data.predictions && data.predictions.length > 0) {
                // Clear previous predictions
                predictionsDiv.innerHTML = '';
                
                // Create a container for the top prediction
                const topPrediction = document.createElement('div');
                topPrediction.className = 'top-prediction mb-3';
                topPrediction.innerHTML = `
                    <div class="alert alert-success">
                        <h4>Predicted Digit: ${data.top_prediction.digit}</h4>
                        <div class="progress mt-2">
                            <div class="progress-bar bg-success" 
                                 role="progressbar" 
                                 style="width: ${(data.top_prediction.probability * 100).toFixed(1)}%"
                                 aria-valuenow="${(data.top_prediction.probability * 100).toFixed(1)}" 
                                 aria-valuemin="0" 
                                 aria-valuemax="100">
                                ${(data.top_prediction.probability * 100).toFixed(1)}%
                            </div>
                        </div>
                    </div>
                `;
                predictionsDiv.appendChild(topPrediction);
                
                // Add other predictions
                if (data.predictions.length > 1) {
                    const otherPredictions = document.createElement('div');
                    otherPredictions.className = 'other-predictions';
                    otherPredictions.innerHTML = '<h5>Other possibilities:</h5>';
                    
                    const listGroup = document.createElement('div');
                    listGroup.className = 'list-group';
                    
                    // Skip the first one (already shown as top prediction)
                    data.predictions.slice(1).forEach(pred => {
                        const predItem = document.createElement('div');
                        predItem.className = 'list-group-item d-flex justify-content-between align-items-center';
                        predItem.innerHTML = `
                            Digit ${pred.digit}
                            <span class="badge bg-primary rounded-pill">${(pred.probability * 100).toFixed(1)}%</span>
                        `;
                        listGroup.appendChild(predItem);
                    });
                    
                    otherPredictions.appendChild(listGroup);
                    predictionsDiv.appendChild(otherPredictions);
                }
                // Display predictions
                let html = '<div class="predictions-list">';
                data.predictions.forEach(pred => {
                    const percentage = (pred.probability * 100).toFixed(2);
                    const width = Math.min(100, percentage * 1.2); // Scale for better visualization
                    
                    html += `
                        <div class="prediction-item mb-2">
                            <div class="d-flex justify-content-between">
                                <span class="prediction-label">Digit: <strong>${pred.digit}</strong></span>
                                <span class="prediction-percentage">${percentage}%</span>
                            </div>
                            <div class="progress" style="height: 20px;">
                                <div class="progress-bar" role="progressbar" 
                                     style="width: ${width}%" 
                                     aria-valuenow="${percentage}" 
                                     aria-valuemin="0" 
                                     aria-valuemax="100">
                                </div>
                            </div>
                        </div>
                    `;
                });
                html += '</div>';
                
                predictionsDiv.innerHTML = html;
            } else {
                const errorMsg = data.error || 'Failed to get predictions. Please try again.';
                predictionsDiv.innerHTML = `
                    <div class="alert alert-danger">
                        <i class="bi bi-exclamation-triangle-fill me-2"></i>
                        ${errorMsg}
                    </div>
                `;
            }
        } catch (error) {
            console.error('Prediction error:', error);
            predictionsDiv.innerHTML = `
                <div class="alert alert-danger">
                    <i class="bi bi-exclamation-triangle-fill me-2"></i>
                    ${error.message || 'An error occurred while processing your request. Please try again.'}
                </div>
            `;
        }
        
        // Prevent scrolling when touching the canvas on mobile
        document.body.addEventListener('touchstart', function(e) {
            if (e.target === canvas) {
                e.preventDefault();
            }
        }, { passive: false });
        
        document.body.addEventListener('touchend', function(e) {
            if (e.target === canvas) {
                e.preventDefault();
            }
        }, { passive: false });
        
        document.body.addEventListener('touchmove', function(e) {
            if (e.target === canvas) {
                e.preventDefault();
            }
        }, { passive: false });
    });
    document.body.addEventListener('touchend', function(e) {
        if (e.target === canvas) {
            e.preventDefault();
        }
    }, { passive: false });
    
    document.body.addEventListener('touchmove', function(e) {
        if (e.target === canvas) {
            e.preventDefault();
        }
    }, { passive: false });
});
