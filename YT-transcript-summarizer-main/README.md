# YouTube Transcript Summarizer

A full-stack application that generates AI-powered summaries from YouTube video transcripts using OpenAI's GPT models.

## Features

- 🎥 Extract transcripts from YouTube videos
- 🤖 Generate intelligent summaries using OpenAI GPT
- 🌐 Modern web interface with React
- 🚀 One-command startup script
- ⚡ Fast and responsive design
- 🛡️ Robust error handling

## Quick Start

1. **Clone or create the project:**
   ```bash
   mkdir youtube-summarizer
   cd youtube-summarizer
   ```

2. **Set up the project structure:**
   ```bash
   mkdir api frontend static
   touch api/__init__.py
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python main.py
   ```

5. **Open your browser:**
   - The app will automatically open at `http://localhost:8080`
   - API runs on `http://localhost:5000`

## Project Structure

```
youtube-summarizer/
├── main.py                 # Startup script
├── requirements.txt        # Dependencies
├── README.md              # This file
├── api/
│   ├── __init__.py        # Package init
│   └── app.py             # Flask API
├── frontend/
│   └── index.html         # React frontend (auto-generated)
└── static/
    └── favicon.ico        # Optional favicon
```

## Usage

1. **Get a YouTube video ID:**
   - From URL: `https://www.youtube.com/watch?v=VIDEO_ID`
   - Or just paste the full URL

2. **Get an OpenAI API key:**
   - Sign up at [OpenAI](https://openai.com)
   - Create an API key in your dashboard

3. **Generate summary:**
   - Enter the video ID/URL
   - Enter your API key
   - Click "Generate Summary"

## API Endpoints

- `GET /` - Health check
- `GET /api/health` - Detailed health check
- `POST /api/summarize` - Generate summary

### Example API Request:
```bash
curl -X POST http://localhost:5000/api/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "dQw4w9WgXcQ",
    "api_key": "sk-your-openai-key"
  }'
```

## Configuration

Set environment variables to customize:

- `PORT` - API server port (default: 5000)
- `FRONTEND_PORT` - Frontend port (default: 8080)

## Error Handling

The application handles various error scenarios:
- Invalid video IDs
- Videos without transcripts
- Invalid OpenAI API keys
- Network connectivity issues
- API quota limits

## Development

To modify the frontend:
1. Edit the React component in `main.py` (line ~80)
2. Restart the application

To modify the API:
1. Edit `api/app.py`
2. Restart the application

## Dependencies

- **Flask** - Web framework
- **Flask-CORS** - Cross-origin support
- **youtube-transcript-api** - YouTube transcript extraction
- **openai** - OpenAI API client

## Troubleshooting

**"No module named 'flask'"**
```bash
pip install -r requirements.txt
```

**"Port already in use"**
```bash
export PORT=5001
export FRONTEND_PORT=8081
python main.py
```

**"No transcript found"**
- Video may not have transcripts enabled
- Try a different video
- Check if video is public

**"Invalid API key"**
- Verify your OpenAI API key
- Check if you have available credits

## License

MIT License - Feel free to use and modify!

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

If you encounter issues:
1. Check the error messages in the browser console
2. Verify your API key is valid
3. Ensure the video has transcripts available
4. Check your internet connection