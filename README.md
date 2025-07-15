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

## 🚀 Quick Start

### Option 1: Frontend Development
```bash
git clone https://github.com/priyansh1913/ai-research-platform.git
cd ai-research-platform
git checkout frontend
npm install
npm start
# Frontend runs on http://localhost:3000
```

### Option 2: Backend Development
```bash
git clone https://github.com/priyansh1913/ai-research-platform.git
cd ai-research-platform
git checkout backend
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
uvicorn main:app --reload --host 0.0.0.0 --port 8000
# Backend runs on http://localhost:8000
```

### Option 3: Full Stack Development
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

## 🛠️ Development Workflow

### 1. Clone the Repository
```bash
git clone https://github.com/priyansh1913/ai-research-platform.git
cd ai-research-platform
```

### 2. Choose Your Development Focus
- **Frontend**: `git checkout frontend` - Work on React UI components
- **Backend**: `git checkout backend` - Work on API and AI integrations
- **Full Stack**: Use both branches in separate terminal sessions

### 3. Environment Setup
Each branch contains detailed setup instructions in its respective README:
- Frontend README: React development setup and component architecture
- Backend README: Python environment, API keys, and service configuration

## 🔒 Security & Best Practices

- ✅ **No hardcoded API keys** - All secrets use environment variables
- ✅ **GitHub security compliance** - Passes all security scans
- ✅ **Comprehensive .gitignore** - Protects sensitive files and dependencies
- ✅ **Environment templates** - Easy setup with `.env.example` files
- ✅ **Branch isolation** - Clean separation of frontend and backend code

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

## 🤝 Contributing

1. Fork the repository
2. Choose the appropriate branch (`frontend` or `backend`)
3. Create a feature branch: `git checkout -b feature/amazing-feature`
4. Make your changes and commit: `git commit -m "Add amazing feature"`
5. Push to your fork: `git push origin feature/amazing-feature`
6. Create a Pull Request to the appropriate branch

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

- **Issues**: Report bugs or request features via GitHub Issues
- **Documentation**: Detailed setup guides in each branch's README
- **API Docs**: Interactive documentation at `http://localhost:8000/docs` when backend is running

---

**🎯 Ready to start?** Choose your development path above and dive into the respective branch for detailed instructions!

**⭐ Enjoying the project?** Give it a star on GitHub and share it with other developers!
