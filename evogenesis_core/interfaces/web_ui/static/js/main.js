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
        } else {
            console.log('WebSocket connection died');
            // Try to reconnect after a delay
            setTimeout(connectWebSocket, 5000);
        }
    };
    
    socket.onerror = function(error) {
        console.error(`WebSocket error: ${error.message}`);
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
    // This function would show a toast notification
    // For now, just log to console
    console.log(`[${type.toUpperCase()}] ${message}`);
    
    // If we had a notification library (like Toastify) we would use it here
    // For example:
    // Toastify({
    //   text: message,
    //   duration: 3000,
    //   close: true,
    //   gravity: "top",
    //   position: "right",
    //   backgroundColor: type === 'error' ? "#ef4444" : type === 'success' ? "#22c55e" : type === 'warning' ? "#f59e0b" : "#6366f1",
    // }).showToast();
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
