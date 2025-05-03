/**
 * EvoGenesis API Integration Testing Utilities
 * 
 * This module provides helper functions for testing API endpoints
 * and ensuring proper integration between the frontend and backend.
 */

// Namespace for API testing utilities
window.EvoGenesis = window.EvoGenesis || {};
window.EvoGenesis.apiTesting = {
    /**
     * Test an API endpoint and display the results
     * 
     * @param {string} endpoint - The API endpoint to test (e.g., '/api/agents')
     * @param {string} method - HTTP method (GET, POST, PUT, DELETE)
     * @param {object} payload - Request payload for POST/PUT requests
     * @param {function} onSuccess - Callback function on successful response
     * @param {function} onError - Callback function on error
     */
    testEndpoint: async function(endpoint, method = 'GET', payload = null, onSuccess = null, onError = null) {
        try {
            const startTime = performance.now();
            const options = {
                method: method,
                headers: {
                    'Content-Type': 'application/json',
                    'X-Testing': 'true'
                }
            };
            
            if (payload && (method === 'POST' || method === 'PUT')) {
                options.body = JSON.stringify(payload);
            }
            
            console.log(`Testing endpoint: ${endpoint} (${method})`);
            if (payload) {
                console.log('Payload:', payload);
            }
            
            const response = await fetch(endpoint, options);
            const endTime = performance.now();
            const responseTime = (endTime - startTime).toFixed(2);
            
            let data;
            let contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                data = await response.json();
            } else {
                data = await response.text();
            }
            
            const result = {
                success: response.ok,
                status: response.status,
                statusText: response.statusText,
                responseTime: `${responseTime}ms`,
                data: data,
                headers: Object.fromEntries([...response.headers])
            };
            
            this._displayTestResult(endpoint, method, result);
            
            if (response.ok && onSuccess) {
                onSuccess(result);
            } else if (!response.ok && onError) {
                onError(result);
            }
            
            return result;
        } catch (error) {
            const errorResult = {
                success: false,
                error: error.message,
                endpoint: endpoint,
                method: method
            };
            
            this._displayTestResult(endpoint, method, errorResult);
            
            if (onError) {
                onError(errorResult);
            }
            
            return errorResult;
        }
    },
    
    /**
     * Test multiple API endpoints in sequence
     * 
     * @param {Array} tests - Array of test objects with endpoint, method, and payload properties
     * @param {function} onComplete - Callback function when all tests are complete
     */
    testEndpointSequence: async function(tests, onComplete = null) {
        const results = [];
        const startTime = performance.now();
        
        for (const test of tests) {
            const result = await this.testEndpoint(
                test.endpoint, 
                test.method || 'GET', 
                test.payload || null,
                test.onSuccess,
                test.onError
            );
            
            results.push({
                test: test,
                result: result
            });
        }
        
        const endTime = performance.now();
        const totalTime = (endTime - startTime).toFixed(2);
        
        console.log(`Completed ${tests.length} API tests in ${totalTime}ms`);
        
        if (onComplete) {
            onComplete(results, totalTime);
        }
        
        return {
            results: results,
            totalTime: totalTime
        };
    },
    
    /**
     * Create a test report container on the page
     * 
     * @param {string} containerId - ID for the container element
     * @returns {HTMLElement} The created container
     */
    createTestReportContainer: function(containerId = 'api-test-results') {
        let container = document.getElementById(containerId);
        
        if (!container) {
            container = document.createElement('div');
            container.id = containerId;
            container.className = 'api-test-container card mt-3';
            
            const header = document.createElement('div');
            header.className = 'card-header d-flex justify-content-between align-items-center';
            header.innerHTML = `
                <h5 class="card-title mb-0">API Integration Test Results</h5>
                <button class="btn btn-sm btn-outline-secondary" id="clear-test-results">
                    <i class='bx bx-trash'></i> Clear
                </button>
            `;
            
            const body = document.createElement('div');
            body.className = 'card-body';
            body.id = `${containerId}-body`;
            
            container.appendChild(header);
            container.appendChild(body);
            
            document.body.appendChild(container);
            
            document.getElementById('clear-test-results').addEventListener('click', () => {
                document.getElementById(`${containerId}-body`).innerHTML = '';
            });
        }
        
        return container;
    },
    
    /**
     * Display a test result in the test report container
     * 
     * @private
     * @param {string} endpoint - The API endpoint tested
     * @param {string} method - HTTP method used
     * @param {object} result - Test result object
     */
    _displayTestResult: function(endpoint, method, result) {
        const container = this.createTestReportContainer();
        const resultsBody = document.getElementById(`${container.id}-body`);
        
        const resultCard = document.createElement('div');
        resultCard.className = `test-result mb-3 ${result.success ? 'test-success' : 'test-failure'}`;
        
        const statusClass = result.success ? 'success' : 'danger';
        const statusText = result.success ? 'Success' : 'Failed';
        
        resultCard.innerHTML = `
            <div class="test-header d-flex justify-content-between align-items-center">
                <div>
                    <span class="badge bg-${statusClass}">${statusText}</span>
                    <span class="ms-2 fw-bold">${method}</span>
                    <span class="ms-2">${endpoint}</span>
                </div>
                <div>
                    ${result.responseTime ? `<span class="badge bg-secondary">${result.responseTime}</span>` : ''}
                    ${result.status ? `<span class="badge bg-info ms-2">${result.status}</span>` : ''}
                </div>
            </div>
            <div class="test-body mt-2">
                <div class="test-result-data">
                    <pre class="mb-0"><code>${this._syntaxHighlight(JSON.stringify(result.data || result.error, null, 2))}</code></pre>
                </div>
            </div>
        `;
        
        resultsBody.appendChild(resultCard);
        resultsBody.scrollTop = resultsBody.scrollHeight;
    },
    
    /**
     * Add syntax highlighting to JSON string
     * 
     * @private
     * @param {string} json - JSON string to highlight
     * @returns {string} HTML string with syntax highlighting
     */
    _syntaxHighlight: function(json) {
        // Simple syntax highlighting for JSON
        if (!json) return '';
        
        json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
            let cls = 'json-number';
            if (/^"/.test(match)) {
                if (/:$/.test(match)) {
                    cls = 'json-key';
                } else {
                    cls = 'json-string';
                }
            } else if (/true|false/.test(match)) {
                cls = 'json-boolean';
            } else if (/null/.test(match)) {
                cls = 'json-null';
            }
            return '<span class="' + cls + '">' + match + '</span>';
        });
    },
    
    /**
     * Generate a test data payload based on content type
     * 
     * @param {string} type - Type of content to generate (agent, team, task, etc.)
     * @returns {object} Generated test data
     */
    generateTestData: function(type) {
        const timestamp = new Date().toISOString();
        
        switch (type) {
            case 'agent':
                return {
                    name: `Test Agent ${Math.floor(Math.random() * 1000)}`,
                    type: 'assistant',
                    description: 'This is a test agent created for API testing',
                    capabilities: ['test', 'api_testing', 'integration_testing'],
                    model: 'gpt-4',
                    created_at: timestamp
                };
                
            case 'team':
                return {
                    name: `Test Team ${Math.floor(Math.random() * 1000)}`,
                    goal: 'Test API endpoints and validate responses',
                    type: 'testing',
                    description: 'This is a test team created for API testing',
                    agents: [
                        { id: 'test-agent-1', role: 'tester' },
                        { id: 'test-agent-2', role: 'validator' }
                    ],
                    created_at: timestamp
                };
                
            case 'task':
                return {
                    name: `Test Task ${Math.floor(Math.random() * 1000)}`,
                    description: 'This is a test task created for API testing',
                    priority: 'medium',
                    status: 'pending',
                    created_at: timestamp
                };
                
            case 'tool':
                return {
                    name: `Test Tool ${Math.floor(Math.random() * 1000)}`,
                    description: 'This is a test tool created for API testing',
                    category: 'testing',
                    type: 'internal',
                    config: {
                        test_mode: true,
                        timeout: 5000
                    }
                };
                
            case 'project':
                return {
                    name: `Test Project ${Math.floor(Math.random() * 1000)}`,
                    description: 'This is a test project created for API testing',
                    type: 'testing',
                    requirements: ['api_testing', 'integration_testing'],
                    agents: [
                        { type: 'coordinator', name: 'Test Coordinator' },
                        { type: 'tester', name: 'Test Agent' }
                    ]
                };
                
            default:
                return {
                    test: true,
                    timestamp: timestamp,
                    message: 'Test data'
                };
        }
    }
};

// CSS styles for test results
(function() {
    const style = document.createElement('style');
    style.textContent = `
        .api-test-container {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 600px;
            max-width: 90vw;
            max-height: 80vh;
            z-index: 9999;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
            display: flex;
            flex-direction: column;
        }
        
        .api-test-container .card-body {
            padding: 10px;
            overflow-y: auto;
            max-height: 500px;
        }
        
        .test-result {
            padding: 8px;
            border-radius: 5px;
            background-color: #f8f9fa;
            border-left: 3px solid #ddd;
        }
        
        .test-success {
            border-left-color: var(--success-color, #107c10);
        }
        
        .test-failure {
            border-left-color: var(--danger-color, #d13438);
        }
        
        .test-result-data {
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 12px;
            background-color: #f0f0f0;
            border-radius: 3px;
            padding: 5px;
            overflow-x: auto;
        }
        
        .test-result-data pre {
            background-color: transparent;
            padding: 0;
            margin: 0;
        }
        
        .json-key {
            color: #0451a5;
        }
        
        .json-string {
            color: #a31515;
        }
        
        .json-number {
            color: #098658;
        }
        
        .json-boolean {
            color: #0000ff;
        }
        
        .json-null {
            color: #808080;
        }
    `;
    document.head.appendChild(style);
})();

// Expose API testing endpoint for quick console access
window.testAPI = function(endpoint, method, payload) {
    return window.EvoGenesis.apiTesting.testEndpoint(endpoint, method, payload);
};

// Add batch testing function to global scope
window.testAPIBatch = function(tests) {
    return window.EvoGenesis.apiTesting.testEndpointSequence(tests);
};

// Sample usage:
// testAPI('/api/agents', 'GET');
// testAPI('/api/teams', 'POST', EvoGenesis.apiTesting.generateTestData('team'));
// testAPIBatch([
//     { endpoint: '/api/agents', method: 'GET' },
//     { endpoint: '/api/teams', method: 'GET' },
//     { endpoint: '/api/tasks', method: 'GET' }
// ]);
