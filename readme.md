# SmartScrapAI - Autonomous News Aggregation System

## Overview
SmartScrapAI is an intelligent news aggregation system that automatically collects, processes, and publishes news articles. It uses AI to search, summarize, and create comprehensive news roundups on specific topics.

## Features
- 🔍 Smart News Search: Intelligent search across multiple news sources
- 📝 AI-Powered Summarization: Automatic article summarization using BART
- 🌐 Multi-language Support: Translates content into multiple languages
- 🖼️ AI Image Generation: Creates relevant images for articles
- 📊 Batch Processing: Efficient handling of multiple articles
- 🔄 Real-time Processing: Immediate article fetching and processing
- 📱 Responsive UI: Modern and user-friendly interface
- 🚀 One-Click Publishing: Direct integration with publishing platforms
- cache to handle Previous searched results for faster results/response.

## Contributors
- **Krishnamurthi** - Core Developer

## Tech Stack
- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript, jQuery
- **AI/ML**: 
  - BART for summarization
  - Google Translator for multi-language support
  - Pollinations.ai for image generation
- **External Services**:
  - DuckDuckGo for news search
  - Dev.to for publishing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SmartScrapAI.git
cd SmartScrapAI
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
export HASHNODE_API_TOKEN=your_token_here
export PUBLICATION_ID=your_publication_id
```

## Usage

1. Start the Flask server:
```bash
python manage.py runserver
```

2. Open your browser and navigate to:
```
http://localhost:8000
```

3. Enter your search query, select language preferences, and click search.

4. Select articles to include in your roundup and click "Publish" to create a new blog post.

## Configuration

The application can be configured through environment variables:

- `Dev.to API Key`: Get this from https://dev.to/settings/account
- `SECRET_KEY`: Flask application secret key
- `PORT`: Server port (default: 8000)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [Newspaper3k](https://newspaper.readthedocs.io/) for article extraction
- [Transformers](https://huggingface.co/transformers/) for AI-powered summarization
- [DuckDuckGo Search](https://github.com/deedy5/duckduckgo_search) for news search
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [Dev.to](https://dev.to/) for publishing integration

## Contact

For support or inquiries:
- **Krishnamurthi**
  - Email: kiccha1703@gmail.com
  - LinkedIn: [Krishna Murthi](https://www.linkedin.com/in/krishna9003762619murthi)

For general support, please open an issue in the GitHub repository.
