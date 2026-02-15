let hotspots = [[40.0, 120.0], [60.0, 130.0], [30.0, 90.0]];
        let isRunning = false;
        let isPlaying = false;
        let playInterval = null;

        // Dark Mode Functions
        function toggleDarkMode() {
            const body = document.body;
            const isDark = body.classList.toggle('dark-mode');
            
            // Update icon and text
            const icon = document.getElementById('dark-mode-icon');
            const text = document.getElementById('dark-mode-text');
            
            if (isDark) {
                icon.textContent = '☀️';
                text.textContent = 'Light';
            } else {
                icon.textContent = '🌙';
                text.textContent = 'Dark';
            }
            
            // Save preference to localStorage
            localStorage.setItem('darkMode', isDark ? 'enabled' : 'disabled');
        }

        function loadDarkModePreference() {
            const darkMode = localStorage.getItem('darkMode');
            if (darkMode === 'enabled') {
                document.body.classList.add('dark-mode');
                document.getElementById('dark-mode-icon').textContent = '☀️';
                document.getElementById('dark-mode-text').textContent = 'Light';
            }
        }

        // Initialize
        window.onload = function() {
            loadDarkModePreference();
            loadState();
            renderHotspots();
            checkWindStatus();
        };

        async function loadState() {
            const response = await fetch('/api/state');
            const data = await response.json();
            hotspots = data.hotspots;
            
            document.getElementById('sigma-turb').value = data.sigma_turb;
            document.getElementById('sigma-val').textContent = data.sigma_turb;
            document.getElementById('npph').value = data.npph;
            document.getElementById('npph-val').textContent = data.npph;
            document.getElementById('dt').value = data.dt;
            document.getElementById('dt-val').textContent = data.dt;
            document.getElementById('wind-type').value = data.wind_type;
            document.getElementById('show-wind-vectors').checked = data.show_wind_vectors !== false;
            
            // Load emission controls
            const emissionMode = data.emission_mode || 'continuous';
            const emissionInterval = data.emission_interval || 30.0;
            document.getElementById('emission-mode').value = emissionMode;
            document.getElementById('emission-interval').value = emissionInterval;
            document.getElementById('emission-interval-val').textContent = emissionInterval.toFixed(1);
            
            // Show/hide interval control based on mode
            updateEmissionModeUI(emissionMode);
            
            isPlaying = data.is_playing || false;
            updatePlayButton();
            
            updateParticleCount();
            renderHotspots();
        }

        function updatePlayButton() {
            const btn = document.getElementById('play-pause-btn');
            if (isPlaying) {
                btn.innerHTML = '⏸️ Pause';
                btn.style.background = '#ffc107';
            } else {
                btn.innerHTML = '▶️ Play';
                btn.style.background = '#28a745';
            }
        }

        async function togglePlayPause() {
            try {
                const response = await fetch('/api/play', {method: 'POST'});
                const data = await response.json();
                
                if (data.success) {
                    isPlaying = data.is_playing;
                    updatePlayButton();
                    
                    if (isPlaying) {
                        startAutoPlay();
                    } else {
                        stopAutoPlay();
                    }
                }
            } catch (error) {
                console.error('Error toggling play/pause:', error);
            }
        }

        function startAutoPlay() {
            if (playInterval) return;
            
            // Use a slightly longer interval to avoid overwhelming the server
            playInterval = setInterval(async () => {
                if (!isPlaying) {
                    stopAutoPlay();
                    return;
                }
                
                // Only step if not currently running
                if (!isRunning) {
                    await stepSimulation();
                }
            }, 500); // 500ms interval for smoother playback
        }

        function stopAutoPlay() {
            if (playInterval) {
                clearInterval(playInterval);
                playInterval = null;
            }
        }

        function renderHotspots() {
            const list = document.getElementById('hotspot-list');
            list.innerHTML = '';
            hotspots.forEach((h, i) => {
                const item = document.createElement('div');
                item.className = 'hotspot-item';
                item.innerHTML = `
                    <span class="hotspot-coords">[${h[0].toFixed(1)}, ${h[1].toFixed(1)}]</span>
                    <button class="remove-btn" onclick="removeHotspot(${i})">Remove</button>
                `;
                list.appendChild(item);
            });
            updateParticleCount();
        }

        function updateParticleCount() {
            const npph = parseInt(document.getElementById('npph').value);
            const count = hotspots.length * npph;
            document.getElementById('particle-count').textContent = count;
        }

        async function addHotspot() {
            const x = parseFloat(document.getElementById('new-x').value);
            const y = parseFloat(document.getElementById('new-y').value);
            
            if (isNaN(x) || isNaN(y)) {
                alert('Please enter valid coordinates');
                return;
            }

            if (x < 0 || x > 50 || y < 0 || y > 50) {
                alert('Coordinates must be within domain [0-50, 0-50]');
                return;
            }

            hotspots.push([x, y]);
            await updateHotspots();
            renderHotspots();
        }

        async function removeHotspot(index) {
            hotspots.splice(index, 1);
            await updateHotspots();
            renderHotspots();
        }

        async function updateHotspots() {
            await fetch('/api/hotspots', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({hotspots: hotspots})
            });
        }

        async function updateParam(param, value) {
            const valSpan = document.getElementById(param.replace('_', '-') + '-val');
            valSpan.textContent = value;
            
            if (param === 'npph') {
                updateParticleCount();
            }

            await fetch('/api/params', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({[param]: value})
            });
        }

        async function updateBaseTimestep(value) {
            // Update display
            document.getElementById('dt-val').textContent = parseFloat(value).toFixed(1);
            
            // Calculate actual dt with multiplier
            const multiplier = parseFloat(document.getElementById('speed-multiplier').value);
            const actualDt = parseFloat(value) * multiplier;
            
            // Update actual dt display
            updateActualDt();
            
            // Send the actual dt to the server
            await fetch('/api/params', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({dt: actualDt})
            });
        }

        async function updateSpeedMultiplier(value) {
            const multiplier = parseFloat(value);
            document.getElementById('speed-multiplier-val').textContent = multiplier.toFixed(1) + 'x';
            
            // Get base dt value
            const baseDt = parseFloat(document.getElementById('dt').value);
            const actualDt = baseDt * multiplier;
            
            // Update display
            updateActualDt();
            
            // Send the actual dt to the server
            await fetch('/api/params', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({dt: actualDt})
            });
        }

        function setMultiplier(value) {
            const slider = document.getElementById('speed-multiplier');
            slider.value = value;
            updateSpeedMultiplier(value);
        }

        async function updateWindSpeedMultiplier(value) {
            const multiplier = parseFloat(value);
            document.getElementById('wind-speed-multiplier-val').textContent = multiplier.toFixed(1) + 'x';
            
            // Send to backend
            await fetch('/api/params', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({wind_speed_multiplier: multiplier})
            });
            
            console.log(`Wind speed multiplier set to: ${multiplier.toFixed(1)}x`);
        }

        function setWindSpeed(value) {
            const slider = document.getElementById('wind-speed-multiplier');
            slider.value = value;
            updateWindSpeedMultiplier(value);
        }

        async function updateEmissionMode() {
            const mode = document.getElementById('emission-mode').value;
            
            // Show/hide interval control
            updateEmissionModeUI(mode);
            
            // Send to backend
            await fetch('/api/emission', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({emission_mode: mode})
            });
            
            console.log(`Emission mode set to: ${mode}`);
        }

        function updateEmissionModeUI(mode) {
            const intervalControl = document.getElementById('emission-interval-control');
            if (mode === 'continuous') {
                intervalControl.style.display = 'block';
            } else {
                intervalControl.style.display = 'none';
            }
        }

        async function updateEmissionInterval(value) {
            const interval = parseFloat(value);
            document.getElementById('emission-interval-val').textContent = interval.toFixed(1);
            
            // Send to backend
            await fetch('/api/emission', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({emission_interval: interval})
            });
            
            console.log(`Emission interval set to: ${interval.toFixed(1)}s`);
        }

        async function updateDomainScale(scale) {
            // Domain scale configurations
            const scales = {
                'block': {
                    name: 'Block',
                    size: 5,
                    description: '5×5 km, ~42m resolution',
                    hotspots: [[1.25, 2.5], [1.875, 3.125], [0.9375, 2.1875]]
                },
                'neighborhood': {
                    name: 'Neighborhood',
                    size: 10,
                    description: '10×10 km, ~83m resolution',
                    hotspots: [[2.5, 5.0], [3.75, 6.25], [1.875, 4.375]]
                },
                'city': {
                    name: 'City',
                    size: 50,
                    description: '50×50 km, ~417m resolution',
                    hotspots: [[10.0, 30.0], [15.0, 32.5], [7.5, 22.5]]
                },
                'metro': {
                    name: 'Metro Region',
                    size: 100,
                    description: '100×100 km, ~833m resolution',
                    hotspots: [[20.0, 60.0], [30.0, 65.0], [15.0, 45.0]]
                },
                'regional': {
                    name: 'Regional',
                    size: 200,
                    description: '200×200 km, ~1.67km resolution',
                    hotspots: [[40.0, 120.0], [60.0, 130.0], [30.0, 90.0]]
                },
                'state': {
                    name: 'State',
                    size: 500,
                    description: '500×500 km, ~4.17km resolution',
                    hotspots: [[100.0, 300.0], [150.0, 325.0], [75.0, 225.0]]
                }
            };

            const config = scales[scale];
            
            // Update UI
            document.getElementById('domain-scale-name').textContent = config.name;
            document.getElementById('domain-info').textContent = config.description;
            
            try {
                // Send to backend
                const response = await fetch('/api/domain-scale', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        scale: scale,
                        domain_size: config.size,
                        hotspots: config.hotspots
                    })
                });
                
                if (response.ok) {
                    const result = await response.json();
                    console.log(`Domain scaled to: ${config.name} (${config.size}×${config.size} km)`);
                    
                    // Update sigma_turb UI if it was auto-scaled
                    if (result.sigma_turb !== undefined) {
                        document.getElementById('sigma-val').textContent = result.sigma_turb;
                        document.getElementById('sigma-turb').value = result.sigma_turb;
                        console.log(`Turbulent diffusion auto-scaled to: ${result.sigma_turb} m/s`);
                    }
                    
                    // Reset simulation to apply new domain
                    await resetSimulation();
                } else {
                    console.error('Failed to update domain scale');
                }
            } catch (error) {
                console.error('Error updating domain scale:', error);
            }
        }

        function updateActualDt() {
            const baseDt = parseFloat(document.getElementById('dt').value);
            const multiplier = parseFloat(document.getElementById('speed-multiplier').value);
            const actualDt = baseDt * multiplier;
            document.getElementById('actual-dt').textContent = actualDt.toFixed(1);
        }

        async function updateWindType() {
            const windType = document.getElementById('wind-type').value;
            await fetch('/api/params', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({wind_type: windType})
            });
            
            if (windType === 'real') {
                await checkWindStatus();
            }
        }

        async function toggleWindVectors() {
            const showVectors = document.getElementById('show-wind-vectors').checked;
            await fetch('/api/params', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({show_wind_vectors: showVectors})
            });
        }

        async function checkWindStatus() {
            const response = await fetch('/api/wind/status');
            const data = await response.json();
            
            const statusSpan = document.getElementById('wind-loaded-status');
            if (data.loaded && data.has_data) {
                statusSpan.textContent = `✅ Loaded (${data.time_steps} time steps)`;
                statusSpan.style.color = '#28a745';
            } else {
                statusSpan.textContent = '❌ No data loaded';
                statusSpan.style.color = '#dc3545';
            }
        }

        async function createSampleData() {
            const statusSpan = document.getElementById('wind-loaded-status');
            statusSpan.textContent = '⏳ Creating sample data...';
            statusSpan.style.color = '#ffc107';
            
            const response = await fetch('/api/wind/sample', {method: 'POST'});
            const data = await response.json();
            
            if (data.success) {
                alert('Sample wind data created successfully!\nFile: wind_data.npz');
                await checkWindStatus();
            } else {
                alert('Failed to create sample data');
                statusSpan.textContent = '❌ No data loaded';
                statusSpan.style.color = '#dc3545';
            }
        }

        async function uploadWindData() {
            const fileInput = document.getElementById('wind-file-input');
            const file = fileInput.files[0];
            
            if (!file) return;
            
            const statusSpan = document.getElementById('wind-loaded-status');
            statusSpan.textContent = '⏳ Uploading and loading...';
            statusSpan.style.color = '#ffc107';
            
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/api/wind/upload', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                alert('Wind data uploaded and loaded successfully!');
                await checkWindStatus();
            } else {
                alert('Failed to load wind data: ' + (data.error || 'Unknown error'));
                statusSpan.textContent = '❌ Load failed';
                statusSpan.style.color = '#dc3545';
            }
            
            // Reset file input
            fileInput.value = '';
        }

        function toggleERA5Panel() {
            const panel = document.getElementById('era5-panel');
            panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
            
            // Set default dates
            const today = new Date();
            const yesterday = new Date(today);
            yesterday.setDate(yesterday.getDate() - 2);
            const twoDaysAgo = new Date(today);
            twoDaysAgo.setDate(twoDaysAgo.getDate() - 3);
            
            document.getElementById('era5-start-date').value = twoDaysAgo.toISOString().split('T')[0];
            document.getElementById('era5-end-date').value = yesterday.toISOString().split('T')[0];
        }

        function toggleGFSPanel() {
            const panel = document.getElementById('gfs-panel');
            panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
        }

        async function fetchERA5Data() {
            const startDate = document.getElementById('era5-start-date').value;
            const endDate = document.getElementById('era5-end-date').value;
            const bbox = [
                parseFloat(document.getElementById('bbox-north').value),
                parseFloat(document.getElementById('bbox-west').value),
                parseFloat(document.getElementById('bbox-south').value),
                parseFloat(document.getElementById('bbox-east').value)
            ];

            if (!startDate || !endDate) {
                alert('Please select both start and end dates');
                return;
            }

            const statusSpan = document.getElementById('wind-loaded-status');
            statusSpan.textContent = '⏳ Fetching ERA5 data from CDS (this may take a few minutes)...';
            statusSpan.style.color = '#ffc107';

            // Note: API key should already be configured in .cdsapirc file
            // Fetch data
            const response = await fetch('/api/wind/fetch-era5', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    bbox: bbox,
                    start_date: startDate,
                    end_date: endDate
                })
            });

            const data = await response.json();

            if (data.success) {
                alert('ERA5 data fetched successfully from CDS!');
                await checkWindStatus();
                document.getElementById('era5-panel').style.display = 'none';
            } else {
                alert('Failed to fetch ERA5 data: ' + (data.error || 'Unknown error'));
                statusSpan.textContent = '❌ Fetch failed';
                statusSpan.style.color = '#dc3545';
            }
        }

        async function fetchGFSData() {
            const forecastHour = parseInt(document.getElementById('forecast-hour').value);
            const bbox = [
                parseFloat(document.getElementById('gfs-bbox-north').value),
                parseFloat(document.getElementById('gfs-bbox-west').value),
                parseFloat(document.getElementById('gfs-bbox-south').value),
                parseFloat(document.getElementById('gfs-bbox-east').value)
            ];

            const statusSpan = document.getElementById('wind-loaded-status');
            statusSpan.textContent = '⏳ Fetching GFS data...';
            statusSpan.style.color = '#ffc107';

            const response = await fetch('/api/wind/fetch-gfs', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    bbox: bbox,
                    forecast_hour: forecastHour
                })
            });

            const data = await response.json();

            if (data.success) {
                alert('GFS data fetched successfully!');
                await checkWindStatus();
                document.getElementById('gfs-panel').style.display = 'none';
            } else {
                alert('GFS fetch note: ' + (data.error || 'Using fallback data'));
                await checkWindStatus();
            }
        }

        async function stepSimulation() {
            if (isRunning) return;
            isRunning = true;
            
            try {
                const container = document.getElementById('canvas-container');
                container.innerHTML = '<p class="loading">Computing...</p>';
                
                const response = await fetch('/api/step', {method: 'POST'});
                const data = await response.json();
                
                if (data.success) {
                    container.innerHTML = `<img src="data:image/png;base64,${data.image}" alt="Simulation frame">`;
                    document.getElementById('frame-num').textContent = data.frame;
                    document.getElementById('time-val').textContent = data.time.toFixed(1);
                }
            } catch (error) {
                console.error('Error stepping simulation:', error);
            } finally {
                isRunning = false;
            }
        }

        async function runSimulation(steps) {
            if (isRunning) return;
            isRunning = true;
            
            const container = document.getElementById('canvas-container');
            container.innerHTML = `<p class="loading">Running ${steps} steps...</p>`;
            
            const response = await fetch('/api/run', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({n_steps: steps})
            });
            const data = await response.json();
            
            if (data.success && data.images.length > 0) {
                // Show last frame
                const lastImage = data.images[data.images.length - 1];
                container.innerHTML = `<img src="data:image/png;base64,${lastImage}" alt="Simulation frame">`;
                const finalTime = data.frame * parseFloat(document.getElementById('dt').value);
                document.getElementById('frame-num').textContent = data.frame;
                document.getElementById('time-val').textContent = finalTime.toFixed(1);
            }
            
            isRunning = false;
        }

        async function resetSimulation() {
            // Stop playing if active
            if (isPlaying) {
                await togglePlayPause();
            }
            
            const container = document.getElementById('canvas-container');
            container.innerHTML = '<p style="color: #999;">Click "Step Forward" or "Run" to start simulation</p>';
            
            await fetch('/api/reset', {method: 'POST'});
            
            document.getElementById('frame-num').textContent = '0';
            document.getElementById('time-val').textContent = '0.0';
        }

        // ============ Model Management Functions ============
        
        const MODEL_INFO = {
            'lagrangian': {
                description: 'Track individual particles through space - best for point sources',
                advantages: 'Excellent for point sources, mass conservative, handles complex terrain'
            },
            'eulerian': {
                description: 'Solve advection-diffusion PDE on fixed grid - best for large domains',
                advantages: 'Fast for regional scale, no statistical noise, easy to couple with weather models'
            },
            'gaussian_plume': {
                description: 'Analytical steady-state solution - fastest, regulatory standard',
                advantages: 'Extremely fast, no numerical errors, EPA/regulatory approved'
            },
            'puff': {
                description: 'Track expanding Gaussian puffs - good for intermittent sources',
                advantages: 'Time-varying winds, intermittent emissions, faster than full Lagrangian'
            },
            'semi_lagrangian': {
                description: 'Backward trajectory on grid - reduces numerical diffusion',
                advantages: 'Stable with large timesteps, reduced numerical diffusion, good accuracy'
            },
            'hybrid': {
                description: 'Particles near source, grid far away - best accuracy and efficiency',
                advantages: 'Best of both worlds: accuracy near source, efficiency at distance'
            }
        };
        
        async function loadModelState() {
            try {
                const response = await fetch('/api/models/current');
                const data = await response.json();
                
                if (data.success) {
                    const modelType = data.model_type;
                    const modelSelect = document.getElementById('model-type');
                    if (modelSelect) {
                        modelSelect.value = modelType;
                        updateModelInfo(modelType);
                    }
                }
            } catch (error) {
                console.error('Error loading model state:', error);
            }
        }
        
        async function changeModel() {
            const modelSelect = document.getElementById('model-type');
            const modelType = modelSelect.value;
            
            // Confirm if simulation is running
            if (isPlaying) {
                const confirm = window.confirm('Stop simulation and switch models?');
                if (!confirm) {
                    // Reload previous selection
                    await loadModelState();
                    return;
                }
                await togglePlayPause();
            }
            
            // Update UI immediately
            updateModelInfo(modelType);
            
            try {
                const response = await fetch('/api/models/select', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({model_type: modelType})
                });
                
                const data = await response.json();
                
                if (data.success) {
                    // Show success message
                    showNotification(`✓ Switched to ${data.model_info.name}`, 'success');
                    
                    // Reset visualization
                    const container = document.getElementById('canvas-container');
                    container.innerHTML = '<p style="color: #999;">Click "Step Forward" or "Run" to start simulation with new model</p>';
                    
                    // Reset frame counter
                    document.getElementById('frame-num').textContent = '0';
                    document.getElementById('time-val').textContent = '0.0';
                } else {
                    showNotification(`✗ Error: ${data.error}`, 'error');
                }
            } catch (error) {
                console.error('Error changing model:', error);
                showNotification('✗ Failed to switch models', 'error');
            }
        }
        
        function updateModelInfo(modelType) {
            const info = MODEL_INFO[modelType];
            if (info) {
                document.getElementById('model-desc').textContent = info.description;
                document.getElementById('model-advantages').textContent = `Advantages: ${info.advantages}`;
            }
        }
        
        function showNotification(message, type = 'info') {
            // Create notification element
            const notification = document.createElement('div');
            notification.className = `notification notification-${type}`;
            notification.textContent = message;
            notification.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 15px 20px;
                background: ${type === 'success' ? '#28a745' : type === 'error' ? '#dc3545' : '#007bff'};
                color: white;
                border-radius: 6px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.2);
                z-index: 10000;
                animation: slideIn 0.3s ease;
            `;
            
            document.body.appendChild(notification);
            
            // Remove after 3 seconds
            setTimeout(() => {
                notification.style.animation = 'slideOut 0.3s ease';
                setTimeout(() => notification.remove(), 300);
            }, 3000);
        }
        
        // Add CSS animations
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideIn {
                from { transform: translateX(400px); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes slideOut {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(400px); opacity: 0; }
            }
        `;
        document.head.appendChild(style);
        
        // Load model state on page load
        window.addEventListener('load', () => {
            loadModelState();
        });