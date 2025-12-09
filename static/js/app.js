/**
 * SDG Movies - Frontend JavaScript
 */

// State
let currentSDG = null;
let currentGenre = '';

// DOM Elements
const tabButtons = document.querySelectorAll('.tab');
const tabContents = document.querySelectorAll('.tab-content');
const sdgButtonsContainer = document.getElementById('sdg-buttons');
const genreFilter = document.getElementById('genre-filter');
const genreSelect = document.getElementById('genre-select');
const moviesSection = document.getElementById('movies-section');
const moviesGrid = document.getElementById('movies-grid');
const resultsTitle = document.getElementById('results-title');
const searchInput = document.getElementById('search-input');
const searchBtn = document.getElementById('search-btn');
const suggestionsContainer = document.getElementById('suggestions');
const similarSection = document.getElementById('similar-section');
const similarGrid = document.getElementById('similar-grid');
const similarTitle = document.getElementById('similar-title');
const loading = document.getElementById('loading');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    loadSDGs();
    loadGenres();
    initSearch();
});

// Tab Navigation
function initTabs() {
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.dataset.tab;

            // Update buttons
            tabButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Update content
            tabContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === `${tabId}-tab`) {
                    content.classList.add('active');
                }
            });
        });
    });
}

// Load SDG Categories
async function loadSDGs() {
    try {
        const response = await fetch('/api/sdgs');
        const sdgs = await response.json();

        sdgButtonsContainer.innerHTML = sdgs.map(sdg => `
            <button class="sdg-btn" data-sdg="${sdg.name}" style="--sdg-color: ${sdg.color}">
                <span class="sdg-icon">${sdg.icon}</span>
                <span class="sdg-name">${sdg.name}</span>
            </button>
        `).join('');

        // Add click handlers
        document.querySelectorAll('.sdg-btn').forEach(btn => {
            btn.addEventListener('click', () => selectSDG(btn));
        });
    } catch (error) {
        console.error('Error loading SDGs:', error);
    }
}

// Select SDG
async function selectSDG(btn) {
    // Update button states
    document.querySelectorAll('.sdg-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    currentSDG = btn.dataset.sdg;

    // Show filters and load movies
    genreFilter.style.display = 'flex';
    await loadMovies();
}

// Load Genres
async function loadGenres() {
    try {
        const response = await fetch('/api/genres');
        const genres = await response.json();

        genreSelect.innerHTML = '<option value="">All Genres</option>' +
            genres.map(g => `<option value="${g}">${g}</option>`).join('');

        genreSelect.addEventListener('change', () => {
            currentGenre = genreSelect.value;
            if (currentSDG) loadMovies();
        });
    } catch (error) {
        console.error('Error loading genres:', error);
    }
}

// Load Movies
async function loadMovies() {
    if (!currentSDG) return;

    showLoading(true);

    try {
        const params = new URLSearchParams({
            sdg: currentSDG,
            genre: currentGenre,
            limit: 12
        });

        const response = await fetch(`/api/movies?${params}`);
        const movies = await response.json();

        resultsTitle.textContent = `Movies for ${currentSDG}`;
        moviesSection.style.display = 'block';

        renderMovies(moviesGrid, movies, 'confidence');
    } catch (error) {
        console.error('Error loading movies:', error);
        moviesGrid.innerHTML = '<p style="color: var(--text-muted);">Error loading movies. Please try again.</p>';
    }

    showLoading(false);
}

// Search Functionality
function initSearch() {
    let searchTimeout;

    searchInput.addEventListener('input', () => {
        clearTimeout(searchTimeout);
        const query = searchInput.value.trim();

        if (query.length > 2) {
            searchTimeout = setTimeout(() => showSuggestions(query), 300);
        } else {
            suggestionsContainer.innerHTML = '';
        }
    });

    searchBtn.addEventListener('click', search);
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') search();
    });
}

async function showSuggestions(query) {
    try {
        const response = await fetch(`/api/search?q=${encodeURIComponent(query)}&limit=5`);
        const movies = await response.json();

        if (movies.length > 0) {
            suggestionsContainer.innerHTML = movies.map(m => `
                <div class="suggestion-item" data-title="${m.title}">
                    <strong>${m.title}</strong>
                    <span style="color: var(--text-muted); margin-left: 8px;">⭐ ${m.rating.toFixed(1)}</span>
                </div>
            `).join('');

            document.querySelectorAll('.suggestion-item').forEach(item => {
                item.addEventListener('click', () => {
                    searchInput.value = item.dataset.title;
                    suggestionsContainer.innerHTML = '';
                    search();
                });
            });
        } else {
            suggestionsContainer.innerHTML = '';
        }
    } catch (error) {
        console.error('Error fetching suggestions:', error);
    }
}

async function search() {
    const query = searchInput.value.trim();
    if (!query) return;

    suggestionsContainer.innerHTML = '';
    showLoading(true);

    try {
        const response = await fetch(`/api/similar/${encodeURIComponent(query)}`);
        const data = await response.json();

        if (data.error) {
            similarSection.style.display = 'block';
            similarGrid.innerHTML = `<p style="color: var(--text-muted);">No movie found matching "${query}". Try a different title!</p>`;
        } else {
            similarTitle.textContent = `Movies similar to "${data.query}"`;
            similarSection.style.display = 'block';
            renderMovies(similarGrid, data.movies, 'similarity');
        }
    } catch (error) {
        console.error('Error searching:', error);
    }

    showLoading(false);
}

// Render Movies
function renderMovies(container, movies, scoreType) {
    if (movies.length === 0) {
        container.innerHTML = '<p style="color: var(--text-muted);">No movies found.</p>';
        return;
    }

    container.innerHTML = movies.map(movie => `
        <div class="movie-card">
            <div class="movie-poster">
                ${movie.poster
            ? `<img src="${movie.poster}" alt="${movie.title}" loading="lazy">`
            : `<div class="movie-poster-placeholder">🎬</div>`
        }
            </div>
            <div class="movie-info">
                <h4 class="movie-title" title="${movie.title}">${movie.title}</h4>
                <div class="movie-rating">
                    <span>⭐</span>
                    <span>${movie.rating.toFixed(1)}</span>
                </div>
                <div class="movie-genres">
                    ${movie.genres.slice(0, 2).map(g => `<span class="genre-tag">${g}</span>`).join('')}
                </div>
                <div class="score-bar">
                    <div class="score-label">
                        <span>${scoreType === 'confidence' ? 'ML Confidence' : 'Similarity'}</span>
                        <span>${movie[scoreType]}%</span>
                    </div>
                    <div class="score-track">
                        <div class="score-fill" style="width: ${movie[scoreType]}%"></div>
                    </div>
                </div>
            </div>
        </div>
    `).join('');
}

// Loading State
function showLoading(show) {
    loading.style.display = show ? 'flex' : 'none';
}
