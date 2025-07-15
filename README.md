# AI Research Platform - Backend 🚀

FastAPI backend server with AI integrations for research enhancement and image generation.

## 🔧 Environment Setup

1. **Clone and switch to backend branch:**
```bash
git clone https://github.com/priyansh1913/ai-research-platform.git
cd ai-research-platform
git checkout backend
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure API keys:**
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys:
# TOGETHER_API_KEY=your_together_api_key_here
# REPLICATE_API_KEY=your_replicate_api_key_here  
# HUGGINGFACE_API_KEY=your_huggingface_api_key_here
```

4. **Start the server:**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 🎯 Features

- **Together AI Integration**: Research enhancement with Mistral-7B
- **Hugging Face**: FREE image generation with Stable Diffusion XL
- **Replicate**: Backup image generation service
- **Multi-Subject Optimization**: Advanced prompt engineering
- **FastAPI**: High-performance REST API with automatic docs

## 📋 API Endpoints

- `POST /research` - Enhanced research analysis
- `POST /generate-image` - AI image generation  
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation

## 🔑 API Key Setup

### Get Your API Keys:
1. **Together AI**: https://api.together.xyz/
2. **Replicate**: https://replicate.com/account/api-tokens
3. **Hugging Face**: https://huggingface.co/settings/tokens

### Configuration:
The application reads API keys from environment variables. Configure them in your `.env` file:

```bash
TOGETHER_API_KEY=your_together_api_key_here
REPLICATE_API_KEY=your_replicate_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
```

## 🌐 Tech Stack

- **FastAPI**: Modern, fast web framework
- **Together AI API**: Research enhancement
- **Hugging Face API**: Image generation
- **Replicate API**: Backup image generation
- **PyYAML**: Configuration management
- **Uvicorn**: ASGI server

## 🚀 Quick Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables  
cp .env.example .env
# Edit .env with your API keys

# Run development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# View API docs at: http://localhost:8000/docs
```

## 📂 Project Structure

```
backend/
├── main.py                              # FastAPI application
├── research_orchestrator.py             # Research enhancement logic
├── huggingface_image_generator.py       # Image generation
├── replicate_image_generator.py         # Backup image generation
├── research_service.py                  # Research services
├── together_open_deep_research.py       # Together AI integration
├── configs/
│   └── open_deep_researcher_config.yaml # Configuration template
├── requirements.txt                     # Python dependencies
├── .env.example                         # Environment variables template
└── README.md                           # This file
```


