document.getElementById('submit-btn').addEventListener('click', async (e) => {
    e.preventDefault();
    
    const API_ENDPOINTS = [
        'https://endo-detect.onrender.com/api/predict'
    ];
    // const API_ENDPOINTS = [
    //     'http://172.168.4.79:5000/api/predict'
    // ];


    const symptoms = {
        "Irregular / Missed periods": getRadioValue('irregular_missed_periods'),
        "Cramping": getRadioValue('cramping'),
        "Menstrual clots": getRadioValue('menstrual_clots'),
        "Infertility": getRadioValue('infertility'),
        "Pain / Chronic pain": getRadioValue('pain_chronic_pain'),
        "Diarrhea": getRadioValue('diarrhea'),
        "Long menstruation": getRadioValue('long_menstruation'),
        "Vomiting / constant vomiting": getRadioValue('vomiting_constant_vomiting'),
        "Migraines": getRadioValue('migraines'),
        "Extreme Bloating": getRadioValue('extreme_bloating'),
        "Leg pain": getRadioValue('leg_pain'),
        "Depression": getRadioValue('depression'),
        "Fertility Issues": getRadioValue('fertility_issues'),
        "Ovarian cysts": getRadioValue('ovarian_cysts'),
        "Painful urination": getRadioValue('painful_urination'),
        "Pain after Intercourse": getRadioValue('pain_after_intercourse'),
        "Digestive / GI problems": getRadioValue('digestive_gi_problems'),
        "Anaemia / Iron deficiency": getRadioValue('anaemia_iron_deficiency'),
        "Hip pain": getRadioValue('hip_pain'),
        "Vaginal Pain/Pressure": getRadioValue('vaginal_pain_pressure'),
        "Cysts (unspecified)": getRadioValue('cysts'),
        "Abnormal uterine bleeding": getRadioValue('abnormal_uterine_bleeding'),
        "Hormonal problems": getRadioValue('hormonal_problems'),
        "Feeling sick": getRadioValue('feeling_sick'),
        "Abdominal Cramps during Intercourse": getRadioValue('abdominal_cramps_during_intercourse'),
        "Insomnia / Sleeplessness": getRadioValue('insomnia_sleeplessness'),
        "Loss of appetite": getRadioValue('loss_of_appetite')
    };
    
    function getRadioValue(name) {
        const selected = document.querySelector(`input[name="${name}"]:checked`);
        return selected ? selected.value : '0';
    }

    try {
        const submitBtn = document.getElementById('submit-btn');
        submitBtn.disabled = true;
        submitBtn.textContent = 'Analyzing...';
        
        console.log("Sending symptoms:", symptoms);

        let lastError = null;
        for (const endpoint of API_ENDPOINTS) {
            try {
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(symptoms),
                    signal: AbortSignal.timeout(5000) // 5 second timeout
                });

                if (!response.ok) {
                    const error = await response.json().catch(() => ({}));
                    throw new Error(error.message || `HTTP error ${response.status}`);
                }

                const data = await response.json();
                displayResult(data);
                return; // Success - exit the function
                
            } catch (error) {
                console.error(`Failed with ${endpoint}:`, error);
                lastError = error;
                continue; // Try next endpoint
            }
        }

        throw lastError || new Error('All API connection attempts failed');

    } catch (error) {
        console.error('Final error:', error);
        showError(error.message || 'Failed to get diagnosis');
    } finally {
        const submitBtn = document.getElementById('submit-btn');
        submitBtn.disabled = false;
        submitBtn.textContent = 'Check Diagnosis';
    }
});

function displayResult(data) {
    const resultContainer = document.getElementById('result-container');
    const resultContent = document.getElementById('result-content');
    
    const diagnosis = data.prediction == 1 ? 
        'Endometriosis Detected' : 'No Endometriosis Detected';
    
    resultContent.innerHTML = `
        <p class="diagnosis-${data.prediction == 1 ? 'positive' : 'negative'}">
            Diagnosis: ${diagnosis}
        </p>
        <p class="confidence">Confidence: ${(data.confidence * 100).toFixed(1)}%</p>
        <div class="symptoms-list">
            <p>Based on:</p>
            <ul>
                ${Object.entries(data.features_used || {})
                  .filter(([_, val]) => val == 1)
                  .map(([key]) => `<li>${key}</li>`)
                  .join('')}
            </ul>
        </div>
        <p class="note">Note: This is a predictive tool, not a medical diagnosis.</p>
    `;
    
    resultContainer.classList.remove('hidden');
    resultContainer.scrollIntoView({ behavior: 'smooth' });
}

function showError(message) {
    const resultContent = document.getElementById('result-content');
    resultContent.innerHTML = `
        <p class="error-message">Error: ${message}</p>
        <div class="troubleshooting">
            <p>Please:</p>
            <ol>
                <li>Ensure backend is running at <strong>https://endo-detect.onrender.com</strong></li>
                <li>Check the browser console for details (F12 > Console)</li>
                <li>Try submitting different symptoms</li>
            </ol>
        </div>
    `;
    document.getElementById('result-container').classList.remove('hidden');
}
