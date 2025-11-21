const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const pipButton = document.getElementById('pipButton'); // Assuming pipButton is the ID for PiP
const fullscreenButton = document.getElementById('fullscreenButton'); // Get the new fullscreen button
const videoElement = document.getElementById('videoStream');
const webcamFeedElement = document.getElementById('webcamFeed'); // Webcam video element
const logsElement = document.getElementById('logs');
const serverIpInput = document.getElementById('serverIp');
const fpsCounterElement = document.getElementById('fpsCounter');
const detachVideoButton = document.getElementById('detachVideoButton');

const MAX_LOG_LINES = 50; // Maximum number of log lines to display
let logMessages = [];      // Array to store log messages

let ws = null;
let pc = null;
let userId = null;
let serverSettings = null; // To store settings like input_mode
let localWebcamStream = null; // To hold the webcam MediaStream
let frameCanvas = null; // Canvas for capturing frames
let webRTCConnected = false; // Flag to track WebRTC connection state

function log(message) {
    console.log(message); // Keep console logging

    logMessages.push(message);
    if (logMessages.length > MAX_LOG_LINES) {
        logMessages.shift(); // Remove the oldest message if the limit is exceeded
    }

    logsElement.textContent = logMessages.join('\n');
    logsElement.scrollTop = logsElement.scrollHeight; // Auto-scroll
}

// Generate a simple UUID for the client
function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

function generateControls(properties) {
    const container = document.getElementById('dynamicControlsContainer');
    if (!container) {
        log('Error: dynamicControlsContainer not found in HTML.');
        return;
    }
    container.innerHTML = ''; // Clear previous controls
    log('Generating dynamic controls...');

    for (const paramId in properties) {
        const param = properties[paramId];

        const controlDiv = document.createElement('div');
        controlDiv.classList.add('control-group');

        const label = document.createElement('label');
        label.htmlFor = `control-${paramId}`;
        label.textContent = param.title || paramId;
        controlDiv.appendChild(label);

        let inputElement;
        const fieldType = param.field || param.type; 

        if (paramId === 'prompt' && document.getElementById('promptInput')) {
            // Main prompt is handled by a dedicated textarea, ensure its default is set if specified
            const promptInput = document.getElementById('promptInput');
            if (param.default && promptInput.value !== param.default) { 
                promptInput.value = param.default;
            }
            log(`Main prompt input configured for '${paramId}'.`);
            continue; // Skip creating another control for the main 'prompt'
        }

        if (fieldType === 'range') {
            inputElement = document.createElement('input');
            inputElement.type = 'range';
            const min = param.min !== undefined ? parseFloat(param.min) : 0;
            const max = param.max !== undefined ? parseFloat(param.max) : 1;
            let step = param.step !== undefined ? parseFloat(param.step) : 0.01;
            // Ensure step allows reaching max if not perfectly divisible
            if ((max - min) > 0 && step > 0 && ((max - min) % step !== 0) && ((max - min)/step < 20) ) {
                 // Heuristic for small number of steps, make step smaller
                 step = (max-min) / 100; // Default to 100 steps if step is problematic
            }
            if (step === 0) step = 0.00001; // Avoid zero step for float ranges

            inputElement.min = min;
            inputElement.max = max;
            inputElement.step = step;
            inputElement.value = param.default !== undefined ? param.default : min;

            const valueSpan = document.createElement('span');
            valueSpan.id = `control-value-${paramId}`;
            valueSpan.textContent = ` ${inputElement.value}`;
            inputElement.oninput = () => { valueSpan.textContent = ` ${inputElement.value}`; };
            controlDiv.appendChild(inputElement);
            controlDiv.appendChild(valueSpan);
        } else if (fieldType === 'checkbox') {
            inputElement = document.createElement('input');
            inputElement.type = 'checkbox';
            inputElement.checked = param.default !== undefined ? Boolean(param.default) : false;
            controlDiv.appendChild(inputElement); 
        } else if (fieldType === 'textarea' || (param.type === 'string' && !param.options)) {
            inputElement = document.createElement('textarea');
            inputElement.rows = param.rows || 2; // Default to 2 rows for other textareas
            inputElement.value = param.default !== undefined ? param.default : '';
            controlDiv.appendChild(inputElement);
        } else if (fieldType === 'select' && param.options && Array.isArray(param.options)) {
            inputElement = document.createElement('select');
            param.options.forEach(optValue => {
                const option = document.createElement('option');
                option.value = optValue;
                option.textContent = optValue;
                if (param.default === optValue) {
                    option.selected = true;
                }
                inputElement.appendChild(option);
            });
            controlDiv.appendChild(inputElement);
        } else if (fieldType === 'multiselect') {
            // For multiselect, use a textarea for comma-separated values
            // A true multiselect UI is more complex; this is a simpler approach.
            log(`Using textarea for multiselect '${paramId}'. Enter comma-separated values.`);
            inputElement = document.createElement('textarea');
            inputElement.rows = param.rows || 2;
            // If default is an array, join it. Otherwise, use as is (might be string or empty).
            inputElement.value = param.default !== undefined ? (Array.isArray(param.default) ? param.default.join(', ') : param.default) : '';
            controlDiv.appendChild(inputElement);
        } else if (param.type === 'integer' || param.type === 'number') {
            inputElement = document.createElement('input');
            inputElement.type = 'number';
            if (param.min !== undefined) inputElement.min = param.min;
            if (param.max !== undefined) inputElement.max = param.max;
            if (param.step !== undefined) inputElement.step = param.step;
            inputElement.value = param.default !== undefined ? param.default : (param.min !== undefined ? param.min : 0);
            controlDiv.appendChild(inputElement);
        } else {
            log(`Unsupported control type '${fieldType}' for ${paramId}. Defaulting to text input.`);
            inputElement = document.createElement('input');
            inputElement.type = 'text';
            inputElement.value = param.default !== undefined ? param.default : '';
            controlDiv.appendChild(inputElement);
        }

        if (inputElement) {
             inputElement.id = `control-${paramId}`;
        }
        container.appendChild(controlDiv);
    }
    log('Dynamic controls generation complete.');
    if(startButton) startButton.disabled = false;
}

async function fetchSettings(baseURL, isInitialLoad = false) {
    log(`Fetching server settings from ${baseURL}...`);
    if(startButton && isInitialLoad) startButton.disabled = true; // Disable while fetching initial settings
    if(stopButton && isInitialLoad) stopButton.disabled = true;
    try {
        const response = await fetch(`${baseURL}/api/settings`);
        if (!response.ok) {
            serverSettings = null; // Clear old settings on error
            if (document.getElementById('dynamicControlsContainer')) document.getElementById('dynamicControlsContainer').innerHTML = '<p style="color:red;">Error fetching settings. Check server IP and if server is running.</p>';
            throw new Error(`Failed to fetch settings: ${response.status} ${response.statusText}`);
        }
        serverSettings = await response.json();
        log('Successfully fetched server settings.');
        if (serverSettings && serverSettings.input_params && serverSettings.input_params.properties) {
            generateControls(serverSettings.input_params.properties);
            if(startButton) startButton.disabled = false; // Enable start button after successful fetch
        } else {
            log('No input_params.properties found in server settings to generate controls.');
            if (document.getElementById('dynamicControlsContainer')) document.getElementById('dynamicControlsContainer').innerHTML = '<p>Server settings loaded, but no controllable parameters defined.</p>';
            if(startButton) startButton.disabled = false; // Still enable, might be a text-only pipeline
        }
    } catch (error) {
        log(`Error fetching server settings: ${error.toString()}`);
        serverSettings = null; 
        if (document.getElementById('dynamicControlsContainer')) {
             document.getElementById('dynamicControlsContainer').innerHTML = `<p style="color:red;">Error fetching settings: ${error.message}. Check server IP and ensure the server is running correctly.</p>`;
        }
        if(startButton) startButton.disabled = true; // Keep start disabled if settings fail
        // stopStream(); // Don't call stopStream here as it might not be running
    }
}

// Function to be called on page load or when server IP changes
async function loadInitialSettings() {
    const serverIp = serverIpInput.value.trim();
    if (!serverIp) {
        log('Server IP is empty. Please enter a server IP to load settings.');
        if (document.getElementById('dynamicControlsContainer')) {
             document.getElementById('dynamicControlsContainer').innerHTML = '<p style="color:orange;">Please enter a Server IP to load settings.</p>';
        }
        if(startButton) startButton.disabled = true;
        return;
    }
    const HTTP_URL = `https://${serverIp}:7860`;
    await fetchSettings(HTTP_URL, true);
}

// Function to capture a frame from the webcam video element
async function captureWebcamFrame() {
    if (!localWebcamStream || !webcamFeedElement.readyState >= 3) { // Check if webcam stream is active and video has data
        log('Webcam not ready or stream not available.');
        return null;
    }

    if (!frameCanvas) {
        frameCanvas = document.createElement('canvas');
    }

    // Set canvas dimensions to match video element (or desired output)
    // Use videoWidth/videoHeight which gives the intrinsic size of the video
    const width = webcamFeedElement.videoWidth;
    const height = webcamFeedElement.videoHeight;
    frameCanvas.width = width;
    frameCanvas.height = height;

    const ctx = frameCanvas.getContext('2d');
    ctx.drawImage(webcamFeedElement, 0, 0, width, height);

    return new Promise((resolve) => {
        // Using JPEG for potentially smaller size, adjust quality as needed
        frameCanvas.toBlob(blob => {
            if (blob) {
                resolve(blob.arrayBuffer()); // Resolve with ArrayBuffer
            } else {
                log('Error converting canvas to Blob.');
                resolve(null);
            }
        }, 'image/jpeg', 0.9); // 0.9 is JPEG quality
    });
}

// Function to start the webcam
async function startWebcam() {
    if (localWebcamStream) {
        log('Webcam already started.');
        return true;
    }
    try {
        log('Requesting webcam access...');
        // Request a resolution close to what the server might expect, if known
        // Otherwise, default constraints are fine.
        const constraints = { 
            video: { 
                width: { ideal: 540 }, 
                height: { ideal: 960 } 
            } 
        };
        localWebcamStream = await navigator.mediaDevices.getUserMedia(constraints);
        webcamFeedElement.srcObject = localWebcamStream;
        await webcamFeedElement.play(); // Ensure video starts playing
        log('Webcam access granted and stream started.');
        return true;
    } catch (error) {
        log(`Error accessing webcam: ${error.toString()}`);
        localWebcamStream = null;
        return false;
    }
}

// Function to stop the webcam
function stopWebcam() {
    if (localWebcamStream) {
        log('Stopping webcam stream...');
        localWebcamStream.getTracks().forEach(track => track.stop());
        webcamFeedElement.srcObject = null;
        localWebcamStream = null;
        log('Webcam stream stopped.');
    }
}

async function startStream() {
    if (ws || pc) {
        log('Stream already started or in progress. Please stop first.');
        return;
    }

    if (!serverSettings) {
        log('Server settings not loaded. Please ensure Server IP is correct and settings are loaded.');
        // Attempt to load them now if user forgot, or guide them.
        await loadInitialSettings(); 
        if (!serverSettings) { // If still not loaded after attempt
            log('Failed to load server settings. Aborting stream start.');
            return;
        }
    }
    
    // Webcam can be started here or earlier if desired, but ensure settings are loaded first
    // For now, keeping webcam start here to ensure it's tied to a stream attempt.
    const webcamStarted = await startWebcam();
    if (!webcamStarted) {
        log('Failed to start webcam. Aborting stream.');
        // stopWebcam(); // startWebcam handles its own cleanup on failure
        return;
    }

    userId = generateUUID();
    const serverIp = serverIpInput.value.trim(); // Use the current IP value
    // HTTP_URL is already used by loadInitialSettings, WS_URL is needed here.
    const WS_URL = `wss://${serverIp}:7860`;

    log(`Generated User ID: ${userId}`);
    log(`Attempting to connect to server at: ${WS_URL}`);
    log('Starting stream...');
    if(startButton) startButton.disabled = true;
    if(stopButton) stopButton.disabled = false;
    if(serverIpInput) serverIpInput.disabled = true; // Disable IP input during stream

    // DO NOT fetch settings here again, they should be pre-loaded
    // await fetchSettings(HTTP_URL); // REMOVED

    ws = new WebSocket(`${WS_URL}/api/ws/${userId}`);

    ws.onopen = async () => {
        log('WebSocket connection established.');
        // FPS counter will be initialized in pc.ontrack -> videoElement.onplaying
        if (fpsCounterElement) fpsCounterElement.textContent = 'FPS: --';
        
        // 1. Create RTCPeerConnection
        const configuration = {
            iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] // Example STUN server
        };
        pc = new RTCPeerConnection(configuration);
        log('RTCPeerConnection created.');

        // Add a video transceiver for receiving video from the server
        // This is crucial for the offer/answer to correctly negotiate video reception.
        if (pc.addTransceiver) { // Check if addTransceiver is supported (it should be)
            pc.addTransceiver('video', { direction: 'recvonly' });
            log('Added video transceiver (recvonly).');
        } else {
            log('Warning: pc.addTransceiver is not supported by this browser/WebRTC version.');
            // Older syntax might be pc.addTrack(track, stream) for sending, 
            // but for receiving, offers are typically made by the side that wants to send.
            // However, for modern WebRTC, addTransceiver is standard for offer/answer.
        }

        // 2. Handle ICE candidates
        pc.onicecandidate = event => {
            if (event.candidate) {
                log('Sending ICE candidate to server...');
                ws.send(JSON.stringify({ 
                    type: 'icecandidate',
                    candidate: {
                        candidate: event.candidate.candidate,
                        sdpMid: event.candidate.sdpMid,
                        sdpMLineIndex: event.candidate.sdpMLineIndex
                    }
                }));
            } else {
                log('All ICE candidates have been sent.');
            }
        };

        // 3. Handle incoming tracks
        pc.ontrack = event => {
            log('Received remote track!');
            if (videoElement.srcObject !== event.streams[0]) {
                videoElement.srcObject = event.streams[0];
                log('Attached remote stream to video element.');

                videoElement.onloadedmetadata = () => {
                    log('Video metadata loaded.');
                };

                videoElement.onplaying = () => {
                    log('Video playback started. Initializing FPS counter.');
                    if (fpsCounterElement) fpsCounterElement.textContent = 'FPS: --';
                    
                    let videoFramesRenderedThisSecond = 0;
                    let lastFpsUpdateTime = performance.now();

                    function videoFrameRenderedCallback(now, metadata) {
                        videoFramesRenderedThisSecond++;
                        
                        if (now - lastFpsUpdateTime >= 1000) { // Update display every second
                            if (fpsCounterElement) {
                                fpsCounterElement.textContent = `FPS: ${videoFramesRenderedThisSecond}`;
                            }
                            videoFramesRenderedThisSecond = 0;
                            lastFpsUpdateTime = now;
                        }
                        
                        if (videoElement.srcObject && !videoElement.paused && videoElement.HAVE_CURRENT_DATA) { 
                            try {
                                videoElement.requestVideoFrameCallback(videoFrameRenderedCallback);
                            } catch (e) {
                                // log('Error re-registering videoFrameCallback, stream might have ended.');
                            }
                        } else {
                            if (fpsCounterElement) fpsCounterElement.textContent = 'FPS: --';
                        }
                    }

                    if ('requestVideoFrameCallback' in videoElement) {
                        log('Using requestVideoFrameCallback for FPS counting.');
                        videoElement.requestVideoFrameCallback(videoFrameRenderedCallback);
                    } else {
                        log('requestVideoFrameCallback not supported. FPS counter will be less accurate (event-based).');
                        let eventFramesThisSecond = 0;
                        let lastEventFpsTime = performance.now();
                        videoElement.ontimeupdate = () => { 
                            eventFramesThisSecond++;
                            if (performance.now() - lastEventFpsTime >= 1000) {
                                if (fpsCounterElement) fpsCounterElement.textContent = `FPS: ${eventFramesThisSecond} (event)`;
                                eventFramesThisSecond = 0;
                                lastEventFpsTime = performance.now();
                            }
                        };
                    }
                };
            }
            // A simple way to count frames for FPS for this specific context:
            // Count each time ontrack fires with a potentially new stream or if we had a mechanism for discrete frames
            // However, for a continuous stream, this ontrack only fires once usually.
            // We will use a generic interval counter for now and tie FPS update to frame display if possible.
            // The current `updateFPS` called by `setInterval` is a general UI update rate.
            // To make it more about video frames:
            // We can increment `frameCount` when actual frame data is processed if we had such a hook.
            // For WebRTC, the browser handles rendering, so we monitor the video element itself.
            // Let's refine frameCount increment inside a requestVideoFrameCallback if available or ontimeupdate.

            // Use requestVideoFrameCallback if available for more accurate render FPS
            if ('requestVideoFrameCallback' in videoElement) {
                let lastVideoFrameTime = performance.now();
            } else {
                // Fallback: Use ontimeupdate (less frequent but better than nothing)
                videoElement.ontimeupdate = () => {
                    // frameCount++; // frameCount is not defined or used.
                };
            }
        };
        
        pc.onconnectionstatechange = event => {
            log(`PeerConnection state changed: ${pc.connectionState}`);
            if (pc.connectionState === 'connected') {
                log('WebRTC connection established!');
                webRTCConnected = true; // Set flag when connected
                // KICKSTART the process by telling server we are ready for it to ask for a frame
                if (ws && ws.readyState === WebSocket.OPEN) {
                    log('Sending initial next_frame to kickstart server stream loop...');
                    // This first "next_frame" tells the server to start its _video_stream_loop.
                    // That loop will then send back a "send_frame" to request actual data.
                    ws.send(JSON.stringify({ status: 'next_frame' }));
                }
            } else if (['failed', 'disconnected', 'closed'].includes(pc.connectionState)) {
                log('WebRTC connection failed or closed.');
                webRTCConnected = false; // Reset flag
                stopStream();
            }
        };

        try {
            // 4. Create offer
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);
            log('Offer created and local description set. Sending to server...');
            ws.send(JSON.stringify({ sdp: { type: offer.type, sdp: offer.sdp } }));
        } catch (error) {
            log(`Error creating offer: ${error.toString()}`);
            stopStream();
        }
    };

    ws.onmessage = async event => {
        const message = JSON.parse(event.data);
        // log(`Received WebSocket message: ${JSON.stringify(message, null, 2)}`);

        if (message.type === 'answer') {
            log('Received answer from server.');
            try {
                await pc.setRemoteDescription(new RTCSessionDescription(message.sdp));
                log('Remote description set.');
            } catch (error) {
                log(`Error setting remote description: ${error.toString()}`);
                stopStream();
            }
        } else if (message.type === 'icecandidate') {
            // log('Received ICE candidate from server.');
            try {
                if (message.candidate && message.candidate.candidate) {
                    await pc.addIceCandidate(new RTCIceCandidate(message.candidate));
                    log('Added ICE candidate.');
                } else {
                    log('Received null or invalid ICE candidate from server.');
                }
            } catch (error) {
                log(`Error adding ICE candidate: ${error.toString()}`);
                // Not critical enough to stop stream for a single bad candidate usually
            }
        } else if (message.status === 'send_frame') {
            if (!webRTCConnected) {
                log('Received send_frame, but WebRTC not connected yet. Waiting...');
                return; 
            }

            const params = {}; 

            if (serverSettings && serverSettings.input_params && serverSettings.input_params.properties) {
                for (const paramId in serverSettings.input_params.properties) {
                    const paramConfig = serverSettings.input_params.properties[paramId];

                    if (paramId === 'prompt' && document.getElementById('promptInput')) {
                         params.prompt = document.getElementById('promptInput').value;
                         continue;
                    }
                    
                    const controlElement = document.getElementById(`control-${paramId}`);
                    if (controlElement) {
                        const fieldType = paramConfig.field || paramConfig.type;
                        if (fieldType === 'checkbox') {
                            params[paramId] = controlElement.checked;
                        } else if (fieldType === 'range' || paramConfig.type === 'number' || paramConfig.type === 'integer') {
                            const val = controlElement.value;
                            params[paramId] = paramConfig.type === 'integer' ? parseInt(val) : parseFloat(val);
                            if (isNaN(params[paramId])) { // Handle parsing errors
                               params[paramId] = paramConfig.default !== undefined ? paramConfig.default : 0; 
                            }
                        } else if (fieldType === 'multiselect') {
                            // Parse comma-separated string into an array of strings
                            const rawValue = controlElement.value.trim();
                            if (rawValue) {
                                params[paramId] = rawValue.split(',').map(item => item.trim()).filter(item => item);
                            } else {
                                // Send empty list if input is empty, as Pydantic expects a list for lora_models
                                params[paramId] = []; 
                            }
                        } else { // textarea, select, text
                            params[paramId] = controlElement.value;
                        }
                    } else {
                        // If control wasn't generated (e.g. due to type or if main prompt was skipped)
                        // and it's not the main prompt, use default from settings.
                        if (paramId !== 'prompt') { 
                           params[paramId] = paramConfig.default !== undefined ? paramConfig.default : null;
                            // log(`Control for ${paramId} not found, using default: ${params[paramId]}`);
                        }
                    }
                }
            } else {
                log("Server settings or input_params not available for param construction. Sending minimal params.");
                params.prompt = document.getElementById('promptInput')?.value || 'default prompt (fallback)';
            }

            // Ensure critical defaults if not set by controls or missing in settings
            // Many of these might come from server defaults now via the loop above.
            // params.seed = params.seed !== undefined ? params.seed : Math.floor(Math.random() * 1000000);
            // params.width = params.width !== undefined ? params.width : 512; 
            // params.height = params.height !== undefined ? params.height : 512;

            const currentInputMode = serverSettings?.info?.properties?.input_mode?.default || 'text';
            if (currentInputMode === 'image') {
                params.pipeline_type = 'img2img';
            } else {
                params.pipeline_type = 'txt2img'; 
                if (localWebcamStream) { // If webcam is on but mode is text
                    log("Warning: Server input mode is not 'image', but webcam is active. Pipeline set to txt2img.");
                }
            }
            
            // Make sure all expected params by the server are present, even if null from default
            if (serverSettings && serverSettings.input_params && serverSettings.input_params.properties) {
                for (const paramId in serverSettings.input_params.properties) {
                    if (!(paramId in params)) {
                        if (paramId === 'prompt' && document.getElementById('promptInput')) {
                            params.prompt = document.getElementById('promptInput').value;
                        } else {
                            params[paramId] = serverSettings.input_params.properties[paramId].default !== undefined ? 
                                              serverSettings.input_params.properties[paramId].default : null;
                        }
                    }
                }
            }

            log(`Sending params: ${JSON.stringify(Object.keys(params))}`); // Log keys to check
            // log(`Sending params: ${JSON.stringify(params, null, 1)}`); // Verbose log

            // Send params first
            ws.send(JSON.stringify(params));
            // log(`Sent params JSON.`);

            // Then, if server expects an image, capture and send webcam frame
            if (currentInputMode === 'image') {
                const frameBuffer = await captureWebcamFrame();
                if (frameBuffer) {
                    ws.send(frameBuffer);
                    // log(`Sent webcam frame (${(frameBuffer.byteLength / 1024).toFixed(2)} KB)`);
                } else {
                    log('Failed to capture webcam frame. Will send params and next_frame only.');
                }
            }

            // Finally, send the 'next_frame' status to signal server all data is sent
            ws.send(JSON.stringify({ status: 'next_frame' }));
            // log('Sent next_frame status after data.');

        } else if (message.status === 'timeout') {
            log(`Session timed out by server: ${message.message}`);
            stopStream();
        } else if (message.status === 'connected' || message.status === 'wait') {
            log(`Received server status message: ${message.status}. Ignoring.`);
            // These are initial messages from the server, client can ignore them.
        } else {
            log(`Received unknown WebSocket message: ${JSON.stringify(message)}`);
        }
    };

    ws.onerror = error => {
        log(`WebSocket error: ${error.message || 'Unknown error'}`);
        console.error('WebSocket error:', error);
        if (fpsCounterElement) fpsCounterElement.textContent = 'FPS: --';
        if(startButton) startButton.disabled = false;
        if(stopButton) stopButton.disabled = true;
        if(serverIpInput) serverIpInput.disabled = false;
        // Consider if stopStream() should be called here for full cleanup
        // stopStream(); // If ws fails, might need to cleanup pc etc.
    };

    ws.onclose = event => {
        log(`WebSocket connection closed. Code: ${event.code}, Reason: ${event.reason || 'No reason given'}`);
        if (fpsCounterElement) fpsCounterElement.textContent = 'FPS: --';
        // Re-enable IP input and start button, disable stop button
        if(startButton) startButton.disabled = false;
        if(stopButton) stopButton.disabled = true;
        if(serverIpInput) serverIpInput.disabled = false;
        // Do not call stopStream() from here if it's also called by stopStream itself, to avoid loops
        // stopStream(); // Let the main stopStream handle full cleanup if called explicitly or by error
    };
}

function stopStream() {
    log('Stopping stream...');
    stopWebcam(); // Stop webcam first
    if (ws) {
        // Send a disconnect message if the WebSocket is still open
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ status: 'disconnect' })); 
        }
        ws.close();
        ws = null;
    }
    if (pc) {
        pc.close();
        pc = null;
    }
    webRTCConnected = false; // Reset flag on stop
    videoElement.srcObject = null;
    if (videoElement) { // Clear event handlers to prevent errors after element might be reused/gc'd
        videoElement.onloadedmetadata = null;
        videoElement.onplaying = null;
        videoElement.ontimeupdate = null;
        if ('requestVideoFrameCallback' in videoElement && videoElement.cancelVideoFrameCallback) {
            // There isn't a direct way to cancel all rVFCs other than letting them not re-register
        }
    }
    if(startButton) startButton.disabled = false;
    if(stopButton) stopButton.disabled = true;
    if(serverIpInput) serverIpInput.disabled = false; // Re-enable IP input on stop
    serverSettings = null; // Reset settings
    log('Stream stopped and resources cleaned up.');
    // logsElement.textContent = ''; // Optionally clear logs on stop
}

// Event listener for the detach video button
if (detachVideoButton) {
    detachVideoButton.addEventListener('click', async () => {
        if (!videoElement) {
            log('Video element not found.');
            return;
        }
        if (document.pictureInPictureEnabled && videoElement.readyState >= 3) { // readyState 3 (HAVE_FUTURE_DATA) or 4 (HAVE_ENOUGH_DATA)
            try {
                if (document.pictureInPictureElement === videoElement) {
                    await document.exitPictureInPicture();
                    log('Exited Picture-in-Picture mode.');
                } else {
                    await videoElement.requestPictureInPicture();
                    log('Requested Picture-in-Picture mode.');
                }
            } catch (error) {
                log(`Error with Picture-in-Picture: ${error.toString()}`);
            }
        } else if (!document.pictureInPictureEnabled) {
            log('Picture-in-Picture is not enabled in this browser.');
        } else if (videoElement.readyState < 3) {
            log('Video is not ready for Picture-in-Picture yet.');
        }
    });
}

// Add event listener for server IP changes to reload settings
if (serverIpInput) {
    serverIpInput.addEventListener('change', loadInitialSettings);
}

// Event listener for Picture-in-Picture button
if (pipButton && videoElement) {
    pipButton.addEventListener('click', async () => {
        try {
            if (videoElement !== document.pictureInPictureElement) {
                await videoElement.requestPictureInPicture();
            } else {
                await document.exitPictureInPicture();
            }
        } catch (error) {
            console.error('Error toggling Picture-in-Picture:', error);
            log('Error toggling PiP: ' + error.toString(), 'error');
        }
    });
}

// Event listener for Fullscreen button
if (fullscreenButton && videoElement) {
    fullscreenButton.addEventListener('click', () => {
        if (!document.fullscreenElement) {
            videoElement.requestFullscreen().catch(err => {
                log(`Error attempting to enable full-screen mode: ${err.message} (${err.name})`, 'error');
                console.error(`Error attempting to enable full-screen mode: ${err.message} (${err.name})`);
            });
        } else {
            if (document.exitFullscreen) {
                document.exitFullscreen();
            }
        }
    });
}

// Optional: Update button text based on fullscreen state changes (e.g., Esc key)
document.addEventListener('fullscreenchange', () => {
    if (fullscreenButton) {
        if (document.fullscreenElement) {
            fullscreenButton.textContent = 'Exit Fullscreen';
        } else {
            fullscreenButton.textContent = 'Toggle Fullscreen';
        }
    }
});

// Initial load of settings when the script runs
document.addEventListener('DOMContentLoaded', () => {
    log('DOM fully loaded. Attempting to load initial server settings...');
    loadInitialSettings();
    if(startButton) startButton.disabled = true; // Keep disabled until settings load successfully
    if(stopButton) stopButton.disabled = true;

    const urlParams = new URLSearchParams(window.location.search);
    const isKioskMode = urlParams.get('kiosk') === 'true';

    if (isKioskMode) {
        log('Kiosk Mode: Activated. Will attempt to start stream and go fullscreen.');

        // Wait 5 seconds from DOMContentLoaded before attempting to click Start Stream
        setTimeout(() => {
            const attemptClickStartButton = () => {
                if (startButton && !startButton.disabled) {
                    log('Kiosk Mode: Clicking Start Stream button.');
                    startButton.click();

                    // After clicking start, wait another 5 seconds for fullscreen
                    setTimeout(() => {
                        if (fullscreenButton) {
                            log('Kiosk Mode: Clicking Toggle Fullscreen button.');
                            fullscreenButton.click();
                        } else {
                            log('Kiosk Mode: Fullscreen button not found.');
                        }
                    }, 5000); // 5s delay for fullscreen after start is clicked
                } else if (startButton) {
                    log('Kiosk Mode: Start Stream button is not enabled yet. Retrying in 1 second...');
                    setTimeout(attemptClickStartButton, 1000); // Retry every second
                } else {
                    log('Kiosk Mode: Start Stream button not found.');
                    // No retry here, if button fundamentally doesn't exist, retrying won't help.
                }
            };
            attemptClickStartButton(); // Start the process of trying to click the start button
        }, 5000); // Initial 5s delay from page load
    }
});

startButton.addEventListener('click', startStream);
stopButton.addEventListener('click', stopStream);

log('WebRTC client initialized. Enter Server IP and click "Start Stream". Webcam will be requested.');