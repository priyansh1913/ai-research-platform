# 🔬 AI Research Platform

A powerful AI-powered research application with modern frontend and backend architecture, featuring advanced research enhancement and AI image generation capabilities.

## 🏗️ Repository Architecture

This repository uses a **branch-based architecture** for clean separation of concerns:

```
ai-research-platform/
├── main branch      # 📋 Project overview & documentation (you are here)
├── frontend branch  # ⚛️ React application with modern UI
└── backend branch   # 🚀 FastAPI server with AI integrations
```

## 🌟 Features Overview

### 🎨 Frontend Features
- **Modern React UI** with Tailwind CSS styling
- **Interactive Data Visualization** using Chart.js
- **Dynamic Flowcharts** with Mermaid.js integration
- **Responsive Design** optimized for all devices
- **Real-time AI Integration** for research and image generation

### 🤖 Backend Features
- **FastAPI REST API** with automatic documentation
- **Together AI Integration** using Mistral-7B for research enhancement
- **Hugging Face Stable Diffusion XL** for high-quality image generation
- **Replicate API** as backup image generation service
- **Advanced Prompt Engineering** with multi-subject optimization
- **Secure Environment Configuration** with no hardcoded API keys

## 🚀 How to Run the Program

```bash
# Terminal 1 - Backend
git clone https://github.com/priyansh1913/ai-research-platform.git
cd ai-research-platform
git checkout backend
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 - Frontend  
cd ai-research-platform
git checkout frontend
npm install
npm start
```

## 🔑 API Keys Required

To run the backend, you'll need API keys from:

1. **Together AI** - https://api.together.xyz/
   - Used for research enhancement with Mistral-7B model
   
2. **Hugging Face** - https://huggingface.co/settings/tokens  
   - Used for FREE image generation with Stable Diffusion XL
   
3. **Replicate** - https://replicate.com/account/api-tokens
   - Used as backup for image generation

## 📋 API Endpoints

The backend provides these REST API endpoints:

- `POST /research` - Enhanced research analysis with AI
- `POST /generate-image` - AI-powered image generation
- `GET /health` - Application health check
- `GET /docs` - Interactive API documentation (Swagger UI)

## 🌐 Tech Stack

### Frontend Stack
- **React 18** - Modern component-based UI framework
- **Tailwind CSS** - Utility-first styling framework  
- **Chart.js** - Dynamic data visualization library
- **Mermaid.js** - Flowchart and diagram generation
- **Axios** - HTTP client for API communication

### Backend Stack
- **FastAPI** - High-performance Python web framework
- **Together AI** - Advanced language model integration
- **Hugging Face Transformers** - State-of-the-art ML models
- **Replicate** - Cloud-based AI model hosting
- **PyYAML** - Configuration management
- **Uvicorn** - Lightning-fast ASGI server

## 📂 Branch Navigation

```bash
# View all available branches
git branch -a

# Switch to frontend development
git checkout frontend

# Switch to backend development  
git checkout backend

# Return to project overview
git checkout main
```

## 📖 Documentation

Each branch contains comprehensive documentation:

- **Frontend Branch**: Component architecture, styling guide, API integration
- **Backend Branch**: API documentation, AI service setup, deployment guide
- **Main Branch**: Project overview, setup instructions, architecture decisions




