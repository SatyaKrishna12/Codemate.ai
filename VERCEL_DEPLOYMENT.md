# 🚀 Vercel Deployment Guide for Deep Researcher Agent

## 📋 Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **Vercel CLI**: Install globally
   ```bash
   npm i -g vercel
   ```
3. **GROQ API Key**: Get from [groq.com](https://groq.com)

## 🛠️ Deployment Steps

### 1. **Prepare for Deployment**
```bash
# Login to Vercel
vercel login

# Initialize project
vercel
```

### 2. **Configure Environment Variables**
In Vercel Dashboard or CLI:
```bash
vercel env add GROQ_API_KEY
# Enter your GROQ API key when prompted
```

### 3. **Deploy**
```bash
# Deploy to preview
vercel

# Deploy to production
vercel --prod
```

## 📁 **Files Created for Vercel:**

- `vercel.json` - Vercel configuration
- `main_vercel.py` - Simplified app for serverless
- `requirements-vercel.txt` - Lightweight dependencies
- `static/` - Frontend files
- `api/index.py` - Alternative API structure

## ⚡ **Serverless Optimizations:**

### **Removed Heavy Dependencies:**
- ❌ `torch` (>500MB)
- ❌ `sentence-transformers` (>1GB)
- ❌ `transformers` (>500MB)
- ✅ Kept: `groq`, `fastapi`, `spacy` (lightweight)

### **Simplified Features:**
- ✅ File upload (10MB limit)
- ✅ Basic document processing
- ✅ Chat interface
- ✅ Session management
- ⚠️ Limited AI processing (due to size constraints)

## 🌐 **Production Considerations:**

### **Option A: Hybrid Deployment**
- **Frontend**: Vercel (static files)
- **Backend**: Railway/Render/AWS (full AI features)

### **Option B: Full Vercel (Current)**
- **Pros**: Simple deployment, auto-scaling
- **Cons**: Limited AI capabilities, file size restrictions

### **Option C: Vercel + External AI**
- Use Vercel for API routing
- External AI service for heavy processing

## 🔧 **Environment Variables Needed:**

```env
GROQ_API_KEY=your_groq_api_key_here
```

## 📊 **Vercel Limits to Consider:**

- **Function Size**: 50MB compressed
- **Function Duration**: 30s (hobby), 900s (pro)
- **Memory**: 1024MB max
- **File Upload**: Recommend <10MB per file

## 🚀 **Quick Deploy Commands:**

```bash
# Clone and deploy
git clone <your-repo-url>
cd Researcher-Agent
vercel
```

## 🔍 **Testing Deployment:**

1. **Health Check**: `https://your-app.vercel.app/api/v1/health`
2. **Main Page**: `https://your-app.vercel.app/`
3. **Chat Interface**: `https://your-app.vercel.app/chat`

## 🐛 **Troubleshooting:**

### **Common Issues:**
1. **Import Errors**: Check `requirements-vercel.txt`
2. **Timeout**: Reduce function complexity
3. **Memory**: Use lighter models/libraries
4. **File Upload**: Check size limits

### **Debugging:**
```bash
# Check logs
vercel logs <deployment-url>

# Local testing
vercel dev
```

## 📈 **Scaling Options:**

### **For High Traffic:**
- Use Vercel Pro plan
- Implement caching
- Use CDN for static assets
- Consider database for sessions

### **For Full AI Features:**
- Deploy backend to Railway/Render
- Use Vercel only for frontend
- Implement microservices architecture

## 🎯 **Next Steps:**

1. Deploy to Vercel
2. Test all endpoints
3. Configure custom domain
4. Set up monitoring
5. Implement caching if needed

Your Deep Researcher Agent is now ready for Vercel deployment! 🚀