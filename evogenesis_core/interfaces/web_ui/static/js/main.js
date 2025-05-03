/**
 * EvoGenesis Control Panel Main Script
 * 
 * This script provides the core functionality for the EvoGenesis Control Panel UI.
 */

// Global websocket connection
let socket = null;

// Connect to the WebSocket server
function connectWebSocket() {
    const wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${location.host}/ws`;
    
    socket = new WebSocket(wsUrl);
    
    socket.onopen = function(e) {
        console.log("WebSocket connection established");
        // Subscribe to system events by default
        socket.send(JSON.stringify({
            subscribe: ["system", "system.status"]
        }));
        // Show connection notification
        showNotification("Connected to EvoGenesis server", "success");
    };
    
    socket.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            // Dispatch the event to handlers
            const customEvent = new CustomEvent('ws-message', { detail: data });
            document.dispatchEvent(customEvent);
        } catch (error) {
            console.error("Error parsing WebSocket message:", error);
        }
    };
    
    socket.onclose = function(event) {
        if (event.wasClean) {
            console.log(`WebSocket connection closed cleanly, code=${event.code} reason=${event.reason}`);
            showNotification(`Connection closed: ${event.reason}`, "info");
        } else {
            console.log('WebSocket connection died');
            showNotification("Connection lost. Attempting to reconnect...", "warning");
            // Try to reconnect after a delay
            setTimeout(connectWebSocket, 5000);
        }
    };
    
    socket.onerror = function(error) {
        console.error(`WebSocket error: ${error.message}`);
        showNotification("Connection error occurred", "error");
    };
    
    return socket;
}

// Subscribe to additional topics
function subscribeToTopics(topics) {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
            subscribe: topics
        }));
    }
}

// API helper functions
async function fetchApi(endpoint, options = {}) {
    try {
        const response = await fetch(endpoint, options);
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error(`Error fetching ${endpoint}:`, error);
        return null;
    }
}

// System control functions
async function startSystem() {
    try {
        const result = await fetchApi('/api/system/start', { method: 'POST' });
        showNotification('System started successfully', 'success');
        return result;
    } catch (error) {
        showNotification('Failed to start system', 'error');
        console.error('Error starting system:', error);
    }
}

async function pauseSystem() {
    try {
        const result = await fetchApi('/api/system/pause', { method: 'POST' });
        showNotification('System paused', 'warning');
        return result;
    } catch (error) {
        showNotification('Failed to pause system', 'error');
        console.error('Error pausing system:', error);
    }
}

async function stopSystem() {
    try {
        const result = await fetchApi('/api/system/stop', { method: 'POST' });
        showNotification('System stopped', 'info');
        return result;
    } catch (error) {
        showNotification('Failed to stop system', 'error');
        console.error('Error stopping system:', error);
    }
}

// UI helper functions
function showNotification(message, type = 'info') {
    // Log to console for debugging
    console.log(`[${type.toUpperCase()}] ${message}`);
    
    // Create toast container if it doesn't exist
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
        document.body.appendChild(toastContainer);
    }
    
    // Create toast element
    const toastId = `toast-${Date.now()}`;
    const toast = document.createElement('div');
    toast.className = `toast show`;
    toast.id = toastId;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');
    
    // Get the appropriate color based on notification type
    const bgColor = type === 'error' ? 'var(--danger-color)' : 
                    type === 'success' ? 'var(--success-color)' : 
                    type === 'warning' ? 'var(--warning-color)' : 
                    'var(--primary-color)';
    
    // Get the appropriate icon
    const icon = type === 'error' ? 'fa-exclamation-circle' : 
                type === 'success' ? 'fa-check-circle' : 
                type === 'warning' ? 'fa-exclamation-triangle' : 
                'fa-info-circle';
    
    // Set toast content
    toast.innerHTML = `
        <div class="toast-header" style="background-color: ${bgColor}; color: white;">
            <i class="fas ${icon} me-2"></i>
            <strong class="me-auto">${type.charAt(0).toUpperCase() + type.slice(1)}</strong>
            <small>just now</small>
            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
        <div class="toast-body">
            ${message}
        </div>
    `;
    
    // Add toast to container
    toastContainer.appendChild(toast);
    
    // Add event listener for close button
    const closeButton = toast.querySelector('.btn-close');
    closeButton.addEventListener('click', function() {
        toast.remove();
    });
    
    // Auto-remove toast after 5 seconds
    setTimeout(() => {
        if (document.getElementById(toastId)) {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }
    }, 5000);
}

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', function() {
    // Connect to WebSocket
    connectWebSocket();
    
    // Setup sidebar toggle
    const sidebarToggle = document.getElementById('sidebarCollapse');
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', function() {
            document.getElementById('sidebar').classList.toggle('active');
        });
    }
    
    // Setup system control buttons
    const startButton = document.getElementById('start-system');
    const pauseButton = document.getElementById('pause-system');
    const stopButton = document.getElementById('stop-system');
    
    if (startButton) {
        startButton.addEventListener('click', startSystem);
    }
    
    if (pauseButton) {
        pauseButton.addEventListener('click', pauseSystem);
    }
    
    if (stopButton) {
        stopButton.addEventListener('click', stopSystem);
    }
});

// Export helper functions for use in other scripts
window.EvoGenesis = {
    api: {
        fetch: fetchApi
    },
    ws: {
        connect: connectWebSocket,
        subscribe: subscribeToTopics
    },
    system: {
        start: startSystem,
        pause: pauseSystem,
        stop: stopSystem
    },
    ui: {
        showNotification: showNotification
    }
};
