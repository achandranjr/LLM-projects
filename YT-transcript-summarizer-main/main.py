#!/usr/bin/env python3
"""
YouTube Transcript Summarizer - Main Startup Script
Starts both the Flask API server and the frontend server.
"""

import os
import sys
import time
import signal
import threading
import subprocess
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

# Configuration
API_PORT = int(os.environ.get('PORT', 5000))
FRONTEND_PORT = int(os.environ.get('FRONTEND_PORT', 8080))
FRONTEND_DIR = Path(__file__).parent / 'frontend'

class FrontendHandler(SimpleHTTPRequestHandler):
    """Custom handler to serve the frontend files"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)
    
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def log_message(self, format, *args):
        """Override to customize logging"""
        print(f"[Frontend] {format % args}")

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'youtube_transcript_api',
        'openai'
    ]
    
    # Check for either Flask or FastAPI
    has_flask = False
    has_fastapi = False
    
    try:
        import flask
        has_flask = True
    except ImportError:
        pass
    
    try:
        import fastapi
        import uvicorn
        has_fastapi = True
    except ImportError:
        pass
    
    if not has_flask and not has_fastapi:
        print("❌ Missing web framework:")
        print("   Install either: pip install flask flask-cors")
        print("   Or: pip install fastapi uvicorn")
        return False
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def create_frontend_html():
    """Create the frontend HTML file if it doesn't exist"""
    html_file = FRONTEND_DIR / 'index.html'
    
    if not FRONTEND_DIR.exists():
        FRONTEND_DIR.mkdir(parents=True)
    
    if not html_file.exists():
        html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YouTube Transcript Summarizer</title>
    <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/lucide/0.263.1/lucide.min.css">
    <script src="https://unpkg.com/lucide@latest/dist/umd/lucide.js"></script>
</head>
<body>
    <div id="root"></div>
    
    <script type="text/babel">
        const { useState } = React;
        const { Play, FileText, Key, Loader2, AlertCircle, CheckCircle } = lucide;

        function YouTubeTranscriptSummarizer() {
          const [videoId, setVideoId] = useState('');
          const [apiKey, setApiKey] = useState('');
          const [summary, setSummary] = useState('');
          const [loading, setLoading] = useState(false);
          const [error, setError] = useState('');
          const [success, setSuccess] = useState(false);

          const extractVideoId = (input) => {
            const patterns = [
              /(?:youtube\\.com\\/watch\\?v=|youtu\\.be\\/|youtube\\.com\\/embed\\/)([^&\\n?#]+)/,
              /^([a-zA-Z0-9_-]{11})$/
            ];
            
            for (const pattern of patterns) {
              const match = input.match(pattern);
              if (match) return match[1];
            }
            return input;
          };

          const handleSubmit = async () => {
            setLoading(true);
            setError('');
            setSummary('');
            setSuccess(false);

            try {
              const cleanVideoId = extractVideoId(videoId.trim());
              
              if (!cleanVideoId || !apiKey.trim()) {
                throw new Error('Please provide both video ID/URL and OpenAI API key');
              }

              const response = await fetch('http://localhost:5000/api/summarize', {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                  video_id: cleanVideoId,
                  api_key: apiKey.trim()
                })
              });

              if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to generate summary');
              }

              const data = await response.json();
              setSummary(data.summary);
              setSuccess(true);
            } catch (err) {
              setError(err.message);
            } finally {
              setLoading(false);
            }
          };

          const clearForm = () => {
            setVideoId('');
            setApiKey('');
            setSummary('');
            setError('');
            setSuccess(false);
          };

          return React.createElement('div', {
            className: "min-h-screen bg-gradient-to-br from-red-50 via-white to-red-50"
          }, 
            React.createElement('div', {
              className: "container mx-auto px-4 py-8 max-w-4xl"
            },
              // Header
              React.createElement('div', {
                className: "text-center mb-8"
              },
                React.createElement('div', {
                  className: "flex items-center justify-center mb-4"
                },
                  React.createElement('div', {
                    className: "bg-red-500 p-3 rounded-full mr-3"
                  },
                    React.createElement(Play, {
                      className: "w-8 h-8 text-white"
                    })
                  ),
                  React.createElement('h1', {
                    className: "text-4xl font-bold text-gray-800"
                  }, "YouTube Summarizer")
                ),
                React.createElement('p', {
                  className: "text-gray-600 text-lg"
                }, "Generate AI-powered summaries from YouTube video transcripts")
              ),
              
              // Form
              React.createElement('div', {
                className: "bg-white rounded-xl shadow-lg p-8 mb-8"
              },
                React.createElement('div', {
                  className: "space-y-6"
                },
                  // Video ID Input
                  React.createElement('div', null,
                    React.createElement('label', {
                      className: "flex items-center text-sm font-medium text-gray-700 mb-2"
                    },
                      React.createElement(Play, {
                        className: "w-4 h-4 mr-2 text-red-500"
                      }),
                      "YouTube Video ID or URL"
                    ),
                    React.createElement('input', {
                      type: "text",
                      value: videoId,
                      onChange: (e) => setVideoId(e.target.value),
                      placeholder: "e.g., dQw4w9WgXcQ or https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                      className: "w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent transition-all duration-200"
                    })
                  ),
                  
                  // API Key Input
                  React.createElement('div', null,
                    React.createElement('label', {
                      className: "flex items-center text-sm font-medium text-gray-700 mb-2"
                    },
                      React.createElement(Key, {
                        className: "w-4 h-4 mr-2 text-blue-500"
                      }),
                      "OpenAI API Key"
                    ),
                    React.createElement('input', {
                      type: "password",
                      value: apiKey,
                      onChange: (e) => setApiKey(e.target.value),
                      placeholder: "sk-...",
                      className: "w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                    })
                  ),
                  
                  // Buttons
                  React.createElement('div', {
                    className: "flex gap-4"
                  },
                    React.createElement('button', {
                      onClick: handleSubmit,
                      disabled: loading,
                      className: "flex-1 bg-gradient-to-r from-red-500 to-red-600 text-white py-3 px-6 rounded-lg font-medium hover:from-red-600 hover:to-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center"
                    },
                      loading ? 
                        React.createElement(React.Fragment, null,
                          React.createElement(Loader2, {
                            className: "w-5 h-5 mr-2 animate-spin"
                          }),
                          "Generating Summary..."
                        ) :
                        React.createElement(React.Fragment, null,
                          React.createElement(FileText, {
                            className: "w-5 h-5 mr-2"
                          }),
                          "Generate Summary"
                        )
                    ),
                    React.createElement('button', {
                      onClick: clearForm,
                      className: "px-6 py-3 border border-gray-300 text-gray-700 rounded-lg font-medium hover:bg-gray-50 transition-all duration-200"
                    }, "Clear")
                  )
                )
              ),
              
              // Error
              error && React.createElement('div', {
                className: "bg-red-50 border border-red-200 rounded-lg p-4 mb-6 flex items-start"
              },
                React.createElement(AlertCircle, {
                  className: "w-5 h-5 text-red-500 mr-3 mt-0.5 flex-shrink-0"
                }),
                React.createElement('div', null,
                  React.createElement('h3', {
                    className: "font-medium text-red-800"
                  }, "Error"),
                  React.createElement('p', {
                    className: "text-red-700 mt-1"
                  }, error)
                )
              ),
              
              // Success
              success && summary && React.createElement('div', {
                className: "bg-green-50 border border-green-200 rounded-lg p-4 mb-6 flex items-start"
              },
                React.createElement(CheckCircle, {
                  className: "w-5 h-5 text-green-500 mr-3 mt-0.5 flex-shrink-0"
                }),
                React.createElement('div', null,
                  React.createElement('h3', {
                    className: "font-medium text-green-800"
                  }, "Summary Generated Successfully!"),
                  React.createElement('p', {
                    className: "text-green-700 mt-1"
                  }, "Your video transcript has been summarized below.")
                )
              ),
              
              // Summary
              summary && React.createElement('div', {
                className: "bg-white rounded-xl shadow-lg p-8"
              },
                React.createElement('h2', {
                  className: "text-2xl font-bold text-gray-800 mb-4 flex items-center"
                },
                  React.createElement(FileText, {
                    className: "w-6 h-6 mr-2 text-blue-500"
                  }),
                  "Summary"
                ),
                React.createElement('div', {
                  className: "bg-gray-50 rounded-lg p-6 whitespace-pre-wrap text-gray-700 leading-relaxed"
                }, summary)
              )
            )
          );
        }

        ReactDOM.render(React.createElement(YouTubeTranscriptSummarizer), document.getElementById('root'));
    </script>
</body>
</html>'''
        
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        print(f"✅ Created frontend HTML file: {html_file}")

def start_api_server():
    """Start the API server (supports both Flask and FastAPI)"""
    try:
        print(f"🚀 Starting API server on port {API_PORT}...")
        
        # Import and check what type of app we have
        sys.path.insert(0, str(Path(__file__).parent / 'api'))
        from app import app
        
        # Check if it's FastAPI or Flask
        app_type = type(app).__name__
        
        if app_type == 'FastAPI':
            # Run FastAPI with uvicorn
            import uvicorn
            uvicorn.run(
                "app:app",
                host='0.0.0.0',
                port=API_PORT,
                reload=False,
                access_log=False,
                log_level="error"  # Reduce uvicorn logging
            )
        else:
            # Run Flask normally
            app.run(
                host='0.0.0.0',
                port=API_PORT,
                debug=False,
                use_reloader=False
            )
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        sys.exit(1)

def start_frontend_server():
    """Start the frontend HTTP server"""
    try:
        print(f"🌐 Starting frontend server on port {FRONTEND_PORT}...")
        
        server = HTTPServer(('localhost', FRONTEND_PORT), FrontendHandler)
        server.serve_forever()
    except Exception as e:
        print(f"❌ Failed to start frontend server: {e}")
        sys.exit(1)

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n🛑 Shutting down servers...")
    sys.exit(0)

def main():
    """Main function to start both servers"""
    print("=" * 60)
    print("🎥 YouTube Transcript Summarizer")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create frontend HTML if needed
    create_frontend_html()
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start API server in a separate thread
    api_thread = threading.Thread(target=start_api_server, daemon=True)
    api_thread.start()
    
    # Give API server time to start
    time.sleep(2)
    
    # Start frontend server in a separate thread
    frontend_thread = threading.Thread(target=start_frontend_server, daemon=True)
    frontend_thread.start()
    
    # Give frontend server time to start
    time.sleep(1)
    
    # Print startup information
    print("\n✅ Both servers are running!")
    print(f"📡 API Server: http://localhost:{API_PORT}")
    print(f"🌐 Frontend: http://localhost:{FRONTEND_PORT}")
    print("\n💡 Usage:")
    print("   1. Open your browser to the frontend URL")
    print("   2. Enter a YouTube video ID or URL")
    print("   3. Enter your OpenAI API key")
    print("   4. Click 'Generate Summary'")
    print("\n⚠️  Press Ctrl+C to stop both servers")
    
    # Optionally open browser
    try:
        webbrowser.open(f'http://localhost:{FRONTEND_PORT}')
        print("🔗 Opening browser...")
    except:
        pass
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down servers...")
        sys.exit(0)

if __name__ == '__main__':
    main()