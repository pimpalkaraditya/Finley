#!/bin/bash

echo "========================================="
echo "  Finley Streamlit App Setup"
echo "========================================="
echo ""

# Check Python version
echo "✓ Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python $python_version detected"

# Create virtual environment
echo ""
echo "✓ Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo ""
echo "✓ Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install dependencies
echo ""
echo "✓ Installing dependencies..."
pip install -r requirements.txt --quiet

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "✓ Creating .env file..."
    cp .env.example .env
    echo "  ⚠️  Please edit .env and add your OpenAI API key"
else
    echo ""
    echo "✓ .env file already exists"
fi

# Create conversations directory
echo ""
echo "✓ Creating conversations directory..."
mkdir -p finley_conversations

echo ""
echo "========================================="
echo "  Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Edit .env and add your OPENAI_API_KEY"
echo "  2. Run: streamlit run app.py"
echo "  3. Open http://localhost:8501"
echo ""
echo "For help, see README.md"
echo ""
