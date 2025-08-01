# 🔬 AI Research Platform

A powerful AI-powered research application with modern frontend and backend architecture, featuring advanced research enhancement and AI image generation capabilities.

## 🏗️ Repository Structure

```
ai-research-platform/
├── README.md           # 📋 Project overview & setup guide
├── backend/            # 🚀 FastAPI server with AI integrations
│   ├── main.py         #     FastAPI application entry point
│   ├── requirements.txt#     Python dependencies
│   ├── .env.example    #     Environment variables template
│   └── configs/        #     Configuration files
└── frontend/           # ⚛️ React application
    ├── src/            #     React components and logic
    ├── public/         #     Static assets
    └── package.json    #     Node.js dependencies
```

**🎯 Simple Structure**: Everything you need is in the main branch - no branch switching required!

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

## ⚡ Quick Start (Everything in Main Branch!)

### Backend Setup (Terminal 1):
```bash
git clone https://github.com/priyansh1913/ai-research-platform.git
cd ai-research-platform
cd backend
python -m venv venv
.\venv\Scripts\activate          # Windows
# source venv/bin/activate       # Mac/Linux
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
uvicorn main:app --reload --port 8000
```

### Frontend Setup (Terminal 2):
```bash
cd ai-research-platform/frontend
npm install
npm start
```

### 🔑 Get Your API Keys (Required):
- **Hugging Face**: https://huggingface.co/settings/tokens (Free)
- **Together AI**: https://api.together.xyz/ (Free tier available)
- **Replicate**: https://replicate.com/account/api-tokens (Pay per use)

### 🌐 Access Your App:
- **Web Interface**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

That's it! Full AI research platform running in 5 minutes! 🎉

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

## 📂 Project Navigation

### Directory Structure:
- **`backend/`** - All Python/FastAPI server code
- **`frontend/`** - All React application code  
- **`README.md`** - This setup guide

### No Branch Switching Required!
Everything you need is in the main branch. Simply:
1. `cd backend` - Work on API and AI features
2. `cd frontend` - Work on UI and React components
3. Both can run simultaneously for full-stack development

## 🛠️ Development Workflow

### 1. Clone the Repository
```bash
git clone https://github.com/priyansh1913/ai-research-platform.git
cd ai-research-platform
```

### 2. Setup Backend
```bash
cd backend
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
uvicorn main:app --reload --port 8000
```

### 3. Setup Frontend (New Terminal)
```bash
cd frontend
npm install
npm start
```

## 🚀 Deployment

### Frontend Deployment
The React frontend can be deployed to:
- **Vercel** (recommended for React)
- **Netlify** 
- **GitHub Pages**
- **AWS S3 + CloudFront**

### Backend Deployment
The FastAPI backend can be deployed to:
- **Railway** (recommended for Python)
- **Render**
- **Heroku**
- **AWS EC2 + Docker**
- **Google Cloud Run**

## 📖 Documentation

Each branch contains comprehensive documentation:

- **Frontend Branch**: Component architecture, styling guide, API integration
- **Backend Branch**: API documentation, AI service setup, deployment guide
- **Main Branch**: Project overview, setup instructions, architecture decisions



