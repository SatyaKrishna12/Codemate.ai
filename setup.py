"""
Simple setup script for the Researcher Agent application.
Handles environment setup and application startup.
"""

import os
import sys
import subprocess
from pathlib import Path


def create_env_file():
    """Create .env file from example if it doesn't exist."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        with open(env_example) as src, open(env_file, 'w') as dst:
            dst.write(src.read())
        print("✅ Created .env file from .env.example")
    elif env_file.exists():
        print("✅ .env file already exists")
    else:
        print("⚠️  No .env.example file found")


def create_directories():
    """Create necessary directories."""
    directories = [
        "data",
        "data/uploads", 
        "data/vectors",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Directory: {directory}")


def check_dependencies():
    """Check if basic dependencies are available."""
    required = ["fastapi", "uvicorn", "pydantic"]
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package}")
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install them with: pip install " + " ".join(missing))
        return False
    
    return True


def main():
    """Main setup function."""
    print("🔧 Researcher Agent - Simple Setup")
    print("=" * 40)
    
    # Create environment file
    print("\n📄 Setting up environment...")
    create_env_file()
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        print("\n❌ Setup incomplete due to missing dependencies")
        return False
    
    print("\n✅ Setup complete!")
    print("\n🚀 To start the application, run:")
    print("   python -m uvicorn main_simplified:app --host 0.0.0.0 --port 8000 --reload")
    print("\n📖 API docs will be available at: http://localhost:8000/docs")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
