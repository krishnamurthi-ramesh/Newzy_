// Document ready function
document.addEventListener('DOMContentLoaded', function() {
    // Immediately hide the loading overlay
    const overlay = document.querySelector('.loading-overlay');
    if (overlay) {
        overlay.classList.add('hidden');
    }

    // Initialize UI components
    initializeNavbar();
    initializeScrollFeatures();
    initializeLanguageSelector();
    initializeFormValidation();
    initializeTooltips();
    initializeAnimations();
    
    // Handle all error messages
    displayFlashMessages();
    
    // Event listener for publish button
    const publishButton = document.getElementById('publish-button');
    if (publishButton) {
        publishButton.addEventListener('click', async function() {
            // Show loading state
            publishButton.disabled = true;
            publishButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Publishing...';
            
            try {
                // Get selected articles
                const selectedArticles = [];
                document.querySelectorAll('.article-checkbox:checked').forEach(checkbox => {
                    const articleCard = checkbox.closest('.article-card');
                    if (articleCard) {
                        // Get article data from the hidden input
                        const articleDataInput = articleCard.querySelector('input[name="article_data"]');
                        let articleData;
                        try {
                            articleData = JSON.parse(articleDataInput.value);
                        } catch (e) {
                            console.error('Error parsing article data:', e);
                            return;
                        }
                        
                        // Add article to list
                        selectedArticles.push({
                            title: articleData.title || '',
                            url: articleData.url || '',
                            summary: articleData.summary || '',
                            source: articleData.source || 'Unknown Source',
                            image_url: articleData.image_url || ''
                        });
                    }
                });
                
                if (selectedArticles.length === 0) {
                    showNotification('Error', 'Please select at least one article to publish', 'warning');
                    return;
                }
                
                // Get topic and location from search form
                const searchForm = document.getElementById('search-form');
                const topic = searchForm.querySelector('#search-query').value.trim();
                const location = searchForm.querySelector('#search-location').value.trim();
                
                if (!topic) {
                    showNotification('Error', 'Please enter a search topic', 'warning');
                    return;
                }
                
                // Publish roundup
                await publishRoundup(selectedArticles, topic, location);
            } catch (error) {
                console.error('Error preparing articles:', error);
                showNotification('Error', 'Failed to prepare articles: ' + error.message, 'error');
            } finally {
                // Reset button state
                publishButton.disabled = false;
                publishButton.innerHTML = 'Publish to Dev.to';
            }
        });
    }
});

// Navbar functionality
function initializeNavbar() {
    const navbar = document.querySelector('.navbar');
    if (!navbar) return;
    
    window.addEventListener('scroll', function() {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });
    
    // Mobile menu toggle
    const navbarToggler = document.querySelector('.navbar-toggler');
    if (navbarToggler) {
        navbarToggler.addEventListener('click', function() {
            document.body.classList.toggle('mobile-menu-open');
        });
    }
}

// Scroll features (progress bar, back to top)
function initializeScrollFeatures() {
    const backToTop = document.querySelector('.back-to-top');
    const scrollProgressBar = document.querySelector('.scroll-progress-bar');
    
    if (backToTop) {
        // Show/hide back to top button
        window.addEventListener('scroll', () => {
            if (window.scrollY > 200) {
                backToTop.classList.add('show');
            } else {
                backToTop.classList.remove('show');
            }
        });
        
        // Back to top button click event with smooth animation
        backToTop.addEventListener('click', function(e) {
            e.preventDefault();
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });
    }
    
    if (scrollProgressBar) {
        // Update scroll progress bar
        window.addEventListener('scroll', () => {
            const scrollPosition = window.scrollY;
            const windowHeight = window.innerHeight;
            const documentHeight = document.documentElement.scrollHeight;
            const progress = (scrollPosition / (documentHeight - windowHeight)) * 100;
            scrollProgressBar.style.width = `${progress}%`;
        });
    }
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const targetId = this.getAttribute('href');
            if (targetId !== '#') {
                e.preventDefault();
                document.querySelector(targetId).scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Language selector functionality
function initializeLanguageSelector() {
    const languageFlags = document.querySelectorAll('.language-flag');
    const languageInput = document.querySelector('input[name="language"]');
    
    if (languageFlags.length && languageInput) {
        languageFlags.forEach(flag => {
            flag.addEventListener('click', () => {
                // Visual selection
                languageFlags.forEach(f => f.classList.remove('selected'));
                flag.classList.add('selected');
                
                // Update hidden input
                const langCode = flag.getAttribute('data-lang');
                languageInput.value = langCode;
                
                // Add subtle animation
                flag.classList.add('pulse');
                setTimeout(() => {
                    flag.classList.remove('pulse');
                }, 500);
            });
        });
        
        // Set default selection
        const defaultFlag = document.querySelector('.language-flag[data-lang="en"]');
        if (defaultFlag) {
            defaultFlag.classList.add('selected');
            languageInput.value = 'en';
        }
    }
}

// Form validation with better UX
function initializeFormValidation() {
    const searchForm = document.querySelector('.search-form');
    if (!searchForm) return;
    
    searchForm.addEventListener('submit', (e) => {
        let isValid = true;
        
        // Validate query input
        const query = document.getElementById('query');
        if (query && query.value.trim() === '') {
            e.preventDefault();
            isValid = false;
            query.classList.add('is-invalid');
            
            // Show validation message
            let errorMsg = query.nextElementSibling;
            if (!errorMsg || !errorMsg.classList.contains('invalid-feedback')) {
                errorMsg = document.createElement('div');
                errorMsg.classList.add('invalid-feedback');
                query.parentNode.insertBefore(errorMsg, query.nextSibling);
            }
            errorMsg.textContent = 'Please enter a search query';
            
            // Focus on the input
            query.focus();
            
            // Remove error state after typing or after 3 seconds
            query.addEventListener('input', function() {
                if (this.value.trim() !== '') {
                    this.classList.remove('is-invalid');
                }
            });
            
            setTimeout(() => {
                query.classList.remove('is-invalid');
            }, 3000);
        }
        
        // Show loading overlay if form is valid
        if (isValid) {
            const loadingOverlay = document.getElementById('loadingOverlay');
            if (loadingOverlay) {
                loadingOverlay.classList.remove('hidden');
                loadingOverlay.classList.add('fade-in');
            }
        }
    });
}

// Initialize tooltips
function initializeTooltips() {
    const tooltipElements = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    if (tooltipElements.length) {
        tooltipElements.forEach(el => {
            new bootstrap.Tooltip(el);
        });
    }
}

// Animate elements when they come into view
function initializeAnimations() {
    const animatedElements = document.querySelectorAll('.animate-on-scroll');
    
    if (animatedElements.length) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animated');
                    observer.unobserve(entry.target);
                }
            });
        }, {
            threshold: 0.1
        });
        
        animatedElements.forEach(el => {
            observer.observe(el);
        });
    }
}

// Display flash messages with auto-dismiss
function displayFlashMessages() {
    const flashMessages = document.querySelectorAll('.alert');
    
    flashMessages.forEach(message => {
        // Add close button if not present
        if (!message.querySelector('.btn-close')) {
            const closeBtn = document.createElement('button');
            closeBtn.classList.add('btn-close');
            closeBtn.setAttribute('data-bs-dismiss', 'alert');
            closeBtn.setAttribute('aria-label', 'Close');
            message.appendChild(closeBtn);
        }
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            message.classList.add('fade-out');
            setTimeout(() => {
                message.remove();
            }, 500);
        }, 5000);
    });
}

// Function to handle publishing news roundup
async function publishRoundup(articles, topic, location) {
    try {
        console.log('Publishing articles:', articles);
        console.log('Topic:', topic);
        console.log('Location:', location);

        const response = await fetch('/news/publish/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')
            },
            body: JSON.stringify({
                articles: articles,
                topic: topic,
                location: location
            })
        });

        let data;
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            data = await response.json();
        } else {
            const text = await response.text();
            throw new Error('Invalid response: ' + text);
        }
        
        console.log('Response:', data);
        
        if (response.ok) {
            // Show success message
            showNotification('Success!', data.message, 'success');
            
            // Open Dev.to post in new tab
            if (data.url) {
                window.open(data.url, '_blank');
            }
        } else {
            // Show error message
            showNotification('Error', data.error || 'Failed to publish roundup', 'error');
        }
    } catch (error) {
        console.error('Error publishing roundup:', error);
        showNotification('Error', 'Failed to publish roundup: ' + error.message, 'error');
    }
}

// Function to get CSRF token from cookies
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

// Function to show notifications
function showNotification(title, message, type = 'info') {
    // You can customize this based on your UI framework
    // For example, using Bootstrap alerts:
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.role = 'alert';
    
    alertDiv.innerHTML = `
        <strong>${title}</strong> ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Add to notification area
    const notificationArea = document.getElementById('notification-area');
    if (notificationArea) {
        notificationArea.appendChild(alertDiv);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            alertDiv.remove();
        }, 5000);
    }
}
