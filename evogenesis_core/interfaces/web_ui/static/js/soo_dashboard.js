/**
 * Strategic Opportunity Observatory Dashboard
 * This script handles the dashboard functionality for the Strategic Opportunity Observatory
 */

// Global variables
let currentFilter = 'all';
let currentPage = 1;
let opportunitiesPerPage = 10;
let currentOpportunityId = null;

// Initialize the dashboard when the document is ready
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
    
    // Set up event listeners
    setupFilters();
    setupPagination();
    setupModals();
    
    // Refresh data every 30 seconds
    setInterval(refreshDashboardData, 30000);
});

/**
 * Initialize the dashboard with data
 */
function initializeDashboard() {
    fetchDashboardStats();
    fetchOpportunities();
    fetchHeuristics();
    fetchSignalSources();
}

/**
 * Fetch and display the dashboard statistics
 */
function fetchDashboardStats() {
    fetch('/api/soo/stats')
        .then(response => response.json())
        .then(data => {
            updateStatsCards(data);
        })
        .catch(error => {
            console.error('Error fetching dashboard stats:', error);
        });
}

/**
 * Update the statistics cards with the data
 */
function updateStatsCards(data) {
    // Update opportunities card
    document.getElementById('total-opportunities').textContent = data.opportunities.total;
    
    // Calculate progress percentages
    const opportunityProgress = Math.min(100, (data.opportunities.total / 100) * 100);
    document.getElementById('opportunity-progress').style.width = `${opportunityProgress}%`;
    
    // Update validated opportunities
    const validatedCount = data.opportunities.by_status.validated || 0;
    document.getElementById('validated-opportunities').textContent = validatedCount;
    
    const validationRate = data.opportunities.total > 0 
        ? Math.round((validatedCount / data.opportunities.total) * 100) 
        : 0;
    document.getElementById('validated-ratio').textContent = `${validationRate}% validation rate`;
    document.getElementById('validated-progress').style.width = `${validationRate}%`;
    
    // Update signal sources
    document.getElementById('signal-sources').textContent = data.signals.sources;
    const lastUpdate = data.signals.last_update > 0 
        ? new Date(data.signals.last_update * 1000).toLocaleString() 
        : 'Never';
    document.getElementById('last-signal-update').textContent = `Last update: ${lastUpdate}`;
    
    // Update mining heuristics
    document.getElementById('mining-heuristics').textContent = data.miners.heuristics;
    document.getElementById('active-miners').textContent = `${data.miners.active_miners} active miners`;
}

/**
 * Fetch and display opportunities
 */
function fetchOpportunities() {
    const url = `/api/soo/opportunities?filter=${currentFilter}&page=${currentPage}&limit=${opportunitiesPerPage}`;
    
    fetch(url)
        .then(response => response.json())
        .then(data => {
            displayOpportunities(data.opportunities);
            updatePagination(data.pagination);
        })
        .catch(error => {
            console.error('Error fetching opportunities:', error);
            document.getElementById('opportunities-table').innerHTML = 
                `<tr><td colspan="6" class="text-center">Error loading opportunities: ${error.message}</td></tr>`;
        });
}

/**
 * Display opportunities in the table
 */
function displayOpportunities(opportunities) {
    const tableBody = document.getElementById('opportunities-table');
    
    if (!opportunities || opportunities.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="6" class="text-center">No opportunities found</td></tr>';
        return;
    }
    
    let html = '';
    opportunities.forEach(opportunity => {
        const confidenceClass = getConfidenceClass(opportunity.confidence);
        const statusClass = getStatusClass(opportunity.status);
        
        html += `
        <tr data-id="${opportunity.id}" class="opportunity-row">
            <td>${opportunity.title}</td>
            <td><span class="badge bg-secondary">${opportunity.opportunity_type}</span></td>
            <td><span class="badge ${confidenceClass}">${opportunity.confidence}</span></td>
            <td><span class="badge ${statusClass}">${opportunity.status}</span></td>
            <td>
                ${opportunity.combined_score !== null ? 
                    `<div class="progress" style="height: 6px;">
                        <div class="progress-bar" role="progressbar" 
                             style="width: ${Math.round(opportunity.combined_score * 100)}%"></div>
                    </div>
                    <small>${Math.round(opportunity.combined_score * 100)}%</small>` : 
                    '<small class="text-muted">Not calculated</small>'}
            </td>
            <td>
                <button class="btn btn-sm btn-outline-primary view-opportunity" data-id="${opportunity.id}">
                    <i class="fas fa-eye"></i>
                </button>
            </td>
        </tr>`;
    });
    
    tableBody.innerHTML = html;
    
    // Add event listeners to view buttons
    document.querySelectorAll('.view-opportunity').forEach(button => {
        button.addEventListener('click', function() {
            const opportunityId = this.getAttribute('data-id');
            viewOpportunityDetails(opportunityId);
        });
    });
    
    // Make rows clickable
    document.querySelectorAll('.opportunity-row').forEach(row => {
        row.addEventListener('click', function() {
            const opportunityId = this.getAttribute('data-id');
            viewOpportunityDetails(opportunityId);
        });
    });
}

/**
 * Get the CSS class for confidence levels
 */
function getConfidenceClass(confidence) {
    switch(confidence) {
        case 'speculative': return 'bg-secondary';
        case 'plausible': return 'bg-info';
        case 'promising': return 'bg-primary';
        case 'probable': return 'bg-success';
        case 'certain': return 'bg-success';
        default: return 'bg-secondary';
    }
}

/**
 * Get the CSS class for status
 */
function getStatusClass(status) {
    switch(status) {
        case 'candidate': return 'bg-secondary';
        case 'evaluating': return 'bg-info';
        case 'validated': return 'bg-primary';
        case 'simulated': return 'bg-warning';
        case 'valued': return 'bg-success';
        case 'approved': return 'bg-success';
        case 'rejected': return 'bg-danger';
        case 'archived': return 'bg-secondary';
        default: return 'bg-secondary';
    }
}

/**
 * Update pagination controls
 */
function updatePagination(pagination) {
    const paginationElement = document.querySelector('.pagination');
    if (!pagination) return;
    
    let html = '';
    
    // Previous button
    html += `
    <li class="page-item ${pagination.current_page === 1 ? 'disabled' : ''}">
        <a class="page-link" href="#" data-page="${pagination.current_page - 1}">Previous</a>
    </li>`;
    
    // Page numbers
    for (let i = 1; i <= pagination.total_pages; i++) {
        html += `
        <li class="page-item ${pagination.current_page === i ? 'active' : ''}">
            <a class="page-link" href="#" data-page="${i}">${i}</a>
        </li>`;
    }
    
    // Next button
    html += `
    <li class="page-item ${pagination.current_page === pagination.total_pages ? 'disabled' : ''}">
        <a class="page-link" href="#" data-page="${pagination.current_page + 1}">Next</a>
    </li>`;
    
    paginationElement.innerHTML = html;
    
    // Add event listeners to pagination links
    document.querySelectorAll('.pagination .page-link').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const page = parseInt(this.getAttribute('data-page'));
            if (page && !isNaN(page)) {
                currentPage = page;
                fetchOpportunities();
            }
        });
    });
}

/**
 * Fetch and display heuristics
 */
function fetchHeuristics() {
    fetch('/api/soo/heuristics')
        .then(response => response.json())
        .then(data => {
            displayHeuristics(data.heuristics);
        })
        .catch(error => {
            console.error('Error fetching heuristics:', error);
            document.getElementById('heuristics-list').innerHTML = 
                `<li class="list-group-item text-center">Error loading heuristics: ${error.message}</li>`;
        });
}

/**
 * Display heuristics in the list
 */
function displayHeuristics(heuristics) {
    const listElement = document.getElementById('heuristics-list');
    
    if (!heuristics || heuristics.length === 0) {
        listElement.innerHTML = '<li class="list-group-item text-center">No heuristics found</li>';
        return;
    }
    
    let html = '';
    heuristics.forEach(heuristic => {
        const successRate = heuristic.success_count + heuristic.failure_count > 0 ?
            Math.round((heuristic.success_count / (heuristic.success_count + heuristic.failure_count)) * 100) : 0;
        
        html += `
        <li class="list-group-item">
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <h6 class="mb-0">${heuristic.name}</h6>
                    <small class="text-muted">Generation ${heuristic.generation}</small>
                </div>
                <span class="badge bg-${successRate > 50 ? 'success' : 'warning'}">${successRate}%</span>
            </div>
            <small class="text-muted d-block mt-1">${heuristic.description}</small>
            <div class="progress mt-2" style="height: 4px;">
                <div class="progress-bar ${successRate > 50 ? 'bg-success' : 'bg-warning'}" 
                     role="progressbar" style="width: ${successRate}%"></div>
            </div>
        </li>`;
    });
    
    listElement.innerHTML = html;
}

/**
 * Fetch and display signal sources
 */
function fetchSignalSources() {
    fetch('/api/soo/sources')
        .then(response => response.json())
        .then(data => {
            displaySignalSources(data.sources);
        })
        .catch(error => {
            console.error('Error fetching signal sources:', error);
            document.getElementById('sources-list').innerHTML = 
                `<li class="list-group-item text-center">Error loading signal sources: ${error.message}</li>`;
        });
}

/**
 * Display signal sources in the list
 */
function displaySignalSources(sources) {
    const listElement = document.getElementById('sources-list');
    
    if (!sources || sources.length === 0) {
        listElement.innerHTML = '<li class="list-group-item text-center">No signal sources found</li>';
        return;
    }
    
    let html = '';
    sources.forEach(source => {
        const lastUpdate = source.last_update ? 
            new Date(source.last_update * 1000).toLocaleString() : 'Never';
        
        html += `
        <li class="list-group-item">
            <div class="d-flex justify-content-between align-items-center">
                <h6 class="mb-0">${source.name}</h6>
                <span class="badge ${source.enabled ? 'bg-success' : 'bg-secondary'}">
                    ${source.enabled ? 'Active' : 'Inactive'}
                </span>
            </div>
            <small class="text-muted">Type: ${source.source_type}</small>
            <small class="text-muted d-block">Last update: ${lastUpdate}</small>
        </li>`;
    });
    
    listElement.innerHTML = html;
}

/**
 * View details of a specific opportunity
 */
function viewOpportunityDetails(opportunityId) {
    currentOpportunityId = opportunityId;
    
    // Show the modal
    const modal = new bootstrap.Modal(document.getElementById('viewOpportunityModal'));
    modal.show();
    
    // Clear and show loading state
    document.getElementById('opportunity-details').innerHTML = `
        <div class="text-center py-5">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
        </div>`;
    
    // Fetch opportunity details
    fetch(`/api/soo/opportunity/${opportunityId}`)
        .then(response => response.json())
        .then(opportunity => {
            displayOpportunityDetails(opportunity);
        })
        .catch(error => {
            console.error('Error fetching opportunity details:', error);
            document.getElementById('opportunity-details').innerHTML = `
                <div class="alert alert-danger">
                    Error loading opportunity details: ${error.message}
                </div>`;
        });
}

/**
 * Display detailed information about an opportunity
 */
function displayOpportunityDetails(opportunity) {
    const detailsElement = document.getElementById('opportunity-details');
    
    const confidenceClass = getConfidenceClass(opportunity.confidence);
    const statusClass = getStatusClass(opportunity.status);
    
    let html = `
    <div class="mb-4">
        <div class="d-flex justify-content-between align-items-center mb-3">
            <h3>${opportunity.title}</h3>
            <div>
                <span class="badge ${statusClass} me-2">${opportunity.status}</span>
                <span class="badge ${confidenceClass}">${opportunity.confidence}</span>
            </div>
        </div>
        <p class="lead">${opportunity.description}</p>
        <div class="mb-3">
            ${opportunity.tags.map(tag => `<span class="badge bg-secondary me-1">${tag}</span>`).join('')}
        </div>
        <small class="text-muted">Discovered ${new Date(opportunity.discovered_at * 1000).toLocaleString()}</small>
    </div>
    
    <div class="row mb-4">
        <div class="col-md-6">
            <div class="card">
                <div class="card-header bg-light">
                    <h5 class="mb-0">Business Metrics</h5>
                </div>
                <div class="card-body">
                    <div class="row g-3">
                        <div class="col-sm-6">
                            <div class="mb-3">
                                <label class="form-label text-muted">Estimated TAM</label>
                                <h5>${opportunity.estimated_tam ? '$' + opportunity.estimated_tam.toLocaleString() : 'Not estimated'}</h5>
                            </div>
                        </div>
                        <div class="col-sm-6">
                            <div class="mb-3">
                                <label class="form-label text-muted">Growth Rate</label>
                                <h5>${opportunity.estimated_growth_rate ? (opportunity.estimated_growth_rate * 100).toFixed(2) + '%' : 'Not estimated'}</h5>
                            </div>
                        </div>
                        <div class="col-sm-6">
                            <div class="mb-3">
                                <label class="form-label text-muted">Required Investment</label>
                                <h5>${opportunity.required_investment ? '$' + opportunity.required_investment.toLocaleString() : 'Not estimated'}</h5>
                            </div>
                        </div>
                        <div class="col-sm-6">
                            <div class="mb-3">
                                <label class="form-label text-muted">Net Present Value</label>
                                <h5>${opportunity.npv ? '$' + opportunity.npv.toLocaleString() : 'Not calculated'}</h5>
                            </div>
                        </div>
                        <div class="col-sm-6">
                            <div class="mb-3">
                                <label class="form-label text-muted">Time to Market</label>
                                <h5>${opportunity.time_to_market ? opportunity.time_to_market + ' months' : 'Not estimated'}</h5>
                            </div>
                        </div>
                        <div class="col-sm-6">
                            <div class="mb-3">
                                <label class="form-label text-muted">Time to Breakeven</label>
                                <h5>${opportunity.time_to_breakeven ? opportunity.time_to_breakeven + ' months' : 'Not estimated'}</h5>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="col-md-6">
            <div class="card">
                <div class="card-header bg-light">
                    <h5 class="mb-0">Evaluation Scores</h5>
                </div>
                <div class="card-body">
                    <div class="mb-3">
                        <label class="form-label d-flex justify-content-between">
                            <span>Impact Score</span>
                            <span>${opportunity.impact_score !== null ? (opportunity.impact_score * 100).toFixed(0) + '%' : 'N/A'}</span>
                        </label>
                        <div class="progress" style="height: 10px;">
                            <div class="progress-bar bg-primary" role="progressbar" 
                                 style="width: ${opportunity.impact_score !== null ? opportunity.impact_score * 100 : 0}%"></div>
                        </div>
                    </div>
                    
                    <div class="mb-3">
                        <label class="form-label d-flex justify-content-between">
                            <span>Feasibility Score</span>
                            <span>${opportunity.feasibility_score !== null ? (opportunity.feasibility_score * 100).toFixed(0) + '%' : 'N/A'}</span>
                        </label>
                        <div class="progress" style="height: 10px;">
                            <div class="progress-bar bg-success" role="progressbar" 
                                 style="width: ${opportunity.feasibility_score !== null ? opportunity.feasibility_score * 100 : 0}%"></div>
                        </div>
                    </div>
                    
                    <div class="mb-3">
                        <label class="form-label d-flex justify-content-between">
                            <span>Risk Score</span>
                            <span>${opportunity.risk_score !== null ? (opportunity.risk_score * 100).toFixed(0) + '%' : 'N/A'}</span>
                        </label>
                        <div class="progress" style="height: 10px;">
                            <div class="progress-bar bg-danger" role="progressbar" 
                                 style="width: ${opportunity.risk_score !== null ? opportunity.risk_score * 100 : 0}%"></div>
                        </div>
                    </div>
                    
                    <div class="mb-3">
                        <label class="form-label d-flex justify-content-between">
                            <span>Combined Score</span>
                            <span>${opportunity.combined_score !== null ? (opportunity.combined_score * 100).toFixed(0) + '%' : 'N/A'}</span>
                        </label>
                        <div class="progress" style="height: 10px;">
                            <div class="progress-bar bg-warning" role="progressbar" 
                                 style="width: ${opportunity.combined_score !== null ? opportunity.combined_score * 100 : 0}%"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="row">
        <div class="col-md-6">
            <div class="card mb-4">
                <div class="card-header bg-light">
                    <h5 class="mb-0">Supporting Evidence</h5>
                </div>
                <div class="card-body p-0">
                    <ul class="list-group list-group-flush">
                        ${opportunity.evidence.length > 0 ? 
                            opportunity.evidence.map(evidence => `
                                <li class="list-group-item">
                                    <div>
                                        <strong>${evidence.source}</strong>
                                        ${evidence.url ? `<a href="${evidence.url}" target="_blank" class="ms-2"><i class="fas fa-external-link-alt"></i></a>` : ''}
                                    </div>
                                    <p class="mb-0">${evidence.content}</p>
                                    <small class="text-muted">Added ${new Date(evidence.added_at * 1000).toLocaleString()}</small>
                                </li>
                            `).join('') : 
                            '<li class="list-group-item text-center">No evidence available</li>'}
                    </ul>
                </div>
            </div>
        </div>
        
        <div class="col-md-6">
            <div class="card mb-4">
                <div class="card-header bg-light">
                    <h5 class="mb-0">Governance Feedback</h5>
                </div>
                <div class="card-body p-0">
                    <ul class="list-group list-group-flush">
                        ${opportunity.governance_comments.length > 0 ? 
                            opportunity.governance_comments.map(comment => `
                                <li class="list-group-item">
                                    <div class="d-flex justify-content-between align-items-center mb-1">
                                        <strong>${comment.user_id}</strong>
                                        <small class="text-muted">${new Date(comment.timestamp * 1000).toLocaleString()}</small>
                                    </div>
                                    <p class="mb-0">${comment.comment}</p>
                                    ${comment.rating ? 
                                        `<div class="mt-1">
                                            ${'★'.repeat(comment.rating)}${'☆'.repeat(5 - comment.rating)}
                                        </div>` : ''}
                                </li>
                            `).join('') : 
                            '<li class="list-group-item text-center">No feedback available</li>'}
                    </ul>
                </div>
                <div class="card-footer">
                    <div class="mb-3">
                        <label for="feedbackComment" class="form-label">Your Feedback</label>
                        <textarea class="form-control" id="feedbackComment" rows="3"></textarea>
                    </div>
                </div>
            </div>
        </div>
    </div>`;
    
    detailsElement.innerHTML = html;
    
    // Update modal buttons based on status
    updateModalButtons(opportunity);
}

/**
 * Update the modal buttons based on opportunity status
 */
function updateModalButtons(opportunity) {
    const approveBtn = document.getElementById('approveOpportunityBtn');
    const rejectBtn = document.getElementById('rejectOpportunityBtn');
    
    // Disable buttons for already approved/rejected opportunities
    if (opportunity.status === 'approved' || opportunity.status === 'rejected') {
        approveBtn.disabled = true;
        rejectBtn.disabled = true;
    } else {
        approveBtn.disabled = false;
        rejectBtn.disabled = false;
        
        // Add event listeners
        approveBtn.onclick = () => submitFeedback(true);
        rejectBtn.onclick = () => submitFeedback(false);
    }
}

/**
 * Submit feedback for an opportunity
 */
function submitFeedback(approved) {
    const comment = document.getElementById('feedbackComment').value;
    
    if (!comment) {
        alert('Please provide feedback before approving or rejecting.');
        return;
    }
    
    const feedbackData = {
        opportunity_id: currentOpportunityId,
        approved: approved,
        user_id: 'current_user', // This would need to be dynamically set based on the logged-in user
        feedback: comment
    };
    
    fetch('/api/soo/feedback', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(feedbackData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Close the modal
            bootstrap.Modal.getInstance(document.getElementById('viewOpportunityModal')).hide();
            
            // Show success message
            alert(`Opportunity successfully ${approved ? 'approved' : 'rejected'}.`);
            
            // Refresh opportunities
            fetchOpportunities();
            fetchDashboardStats();
        } else {
            alert(`Error: ${data.message}`);
        }
    })
    .catch(error => {
        console.error('Error submitting feedback:', error);
        alert(`Error submitting feedback: ${error.message}`);
    });
}

/**
 * Set up filter dropdown events
 */
function setupFilters() {
    document.querySelectorAll('[data-filter]').forEach(element => {
        element.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Update active class
            document.querySelectorAll('[data-filter]').forEach(el => {
                el.classList.remove('active');
            });
            this.classList.add('active');
            
            // Update filter dropdown button text
            document.getElementById('filterDropdown').textContent = `Filter: ${this.textContent}`;
            
            // Update current filter and fetch opportunities
            currentFilter = this.getAttribute('data-filter');
            currentPage = 1; // Reset to first page
            fetchOpportunities();
        });
    });
}

/**
 * Set up pagination events
 */
function setupPagination() {
    // Event delegation is used in updatePagination()
}

/**
 * Set up modal events
 */
function setupModals() {
    // Evolution Modal
    document.querySelector('[data-bs-target="#evolutionModal"]').addEventListener('click', function() {
        // Populate evolution modal if needed
    });
    
    // Add Heuristic Modal submit
    const addHeuristicForm = document.getElementById('addHeuristicForm');
    if (addHeuristicForm) {
        addHeuristicForm.addEventListener('submit', function(e) {
            e.preventDefault();
            submitNewHeuristic();
        });
    }
    
    // Add Source Modal submit
    const addSourceForm = document.getElementById('addSourceForm');
    if (addSourceForm) {
        addSourceForm.addEventListener('submit', function(e) {
            e.preventDefault();
            submitNewSource();
        });
    }
}

/**
 * Submit a new heuristic
 */
function submitNewHeuristic() {
    // Implementation would depend on form structure
    alert('This functionality would be implemented based on the form structure');
}

/**
 * Submit a new signal source
 */
function submitNewSource() {
    // Implementation would depend on form structure
    alert('This functionality would be implemented based on the form structure');
}

/**
 * Refresh all dashboard data
 */
function refreshDashboardData() {
    fetchDashboardStats();
    fetchOpportunities();
    fetchHeuristics();
    fetchSignalSources();
}
