# Deep Researcher Agent - Windows PowerShell Startup Script

Write-Host "🔧 Deep Researcher Agent - Startup Script" -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor Cyan

# Check if virtual environment exists
$venvPath = ".\researcher_env\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    Write-Host "✅ Virtual environment found" -ForegroundColor Green
    
    # Activate virtual environment
    Write-Host "🔄 Activating virtual environment..." -ForegroundColor Yellow
    & $venvPath
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Virtual environment activated" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to activate virtual environment" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "❌ Virtual environment not found at: $venvPath" -ForegroundColor Red
    Write-Host "Please create the virtual environment first:" -ForegroundColor Yellow
    Write-Host "  python -m venv researcher_env" -ForegroundColor Cyan
    exit 1
}

# Install dependencies if requirements.txt exists
if (Test-Path "requirements.txt") {
    Write-Host "📦 Installing/updating dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "⚠️  requirements.txt not found" -ForegroundColor Yellow
}

# Create directories
$directories = @("data", "data\uploads", "data\vectors", "logs")
foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "✅ Created directory: $dir" -ForegroundColor Green
    } else {
        Write-Host "✅ Directory exists: $dir" -ForegroundColor Green
    }
}

# Check for .env file
if (!(Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Write-Host "⚠️  .env file not found. Copying from .env.example..." -ForegroundColor Yellow
        Copy-Item ".env.example" ".env"
        Write-Host "✅ Created .env file from .env.example" -ForegroundColor Green
        Write-Host "💡 Please review and modify .env file as needed" -ForegroundColor Cyan
    } else {
        Write-Host "⚠️  No .env file found. Using default configuration." -ForegroundColor Yellow
    }
} else {
    Write-Host "✅ Configuration file (.env) found" -ForegroundColor Green
}

Write-Host "`n🚀 Starting Deep Researcher Agent..." -ForegroundColor Cyan
Write-Host "📖 API Documentation: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "🔍 Health check: http://localhost:8000/api/v1/health" -ForegroundColor Cyan
Write-Host "`nPress Ctrl+C to stop the application`n" -ForegroundColor Yellow

# Start the application
try {
    python -m uvicorn main_simplified:app --host 127.0.0.1 --port 8000 --reload
} catch {
    Write-Host "❌ Error starting application: $_" -ForegroundColor Red
    exit 1
}
