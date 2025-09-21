"""
Vercel serverless function for Deep Researcher Agent
"""
from http.server import BaseHTTPRequestHandler
import json
from datetime import datetime
from urllib.parse import urlparse, parse_qs

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.handle_request('GET')
    
    def do_POST(self):
        self.handle_request('POST')
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def handle_request(self, method):
        try:
            # Parse the URL
            parsed_url = urlparse(self.path)
            path = parsed_url.path
            query_params = parse_qs(parsed_url.query)
            
            # Get request body for POST requests
            request_data = {}
            if method == 'POST':
                content_length = int(self.headers.get('Content-Length', 0))
                if content_length > 0:
                    body = self.rfile.read(content_length).decode('utf-8')
                    try:
                        request_data = json.loads(body)
                    except:
                        request_data = {}
            
            # Route the request
            response_data = self.route_request(path, method, request_data, query_params)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            
            self.wfile.write(json.dumps(response_data, indent=2).encode())
            
        except Exception as e:
            self.send_error(500, f"Internal Server Error: {str(e)}")
    
    def route_request(self, path, method, data, query):
        """Route requests to appropriate handlers"""
        
        # Normalize path
        if path.endswith('/'):
            path = path[:-1]
        
        if path == '' or path == '/api':
            return {
                'message': 'Deep Researcher Agent API',
                'status': 'active',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0',
                'endpoints': ['/api', '/api/health', '/api/test', '/api/chat', '/api/research']
            }
        
        elif path == '/api/health' or path == '/health':
            return {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'service': 'Deep Researcher Agent'
            }
        
        elif path == '/api/test' or path == '/test':
            return {
                'test': 'success',
                'vercel': 'working',
                'timestamp': datetime.now().isoformat(),
                'method': method,
                'path': path
            }
        
        elif path == '/api/chat':
            if method == 'POST':
                return {
                    'response': f"Chat received: {data.get('message', 'No message')}",
                    'chat_id': data.get('chat_id', 'default'),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success'
                }
            else:
                return {'error': 'Use POST method', 'allowed_methods': ['POST']}
        
        elif path == '/api/research':
            if method == 'POST':
                return {
                    'query': data.get('query', 'No query'),
                    'depth': data.get('depth', 'deep'),
                    'result': f"Research query: {data.get('query', 'No query')}",
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success'
                }
            else:
                return {'error': 'Use POST method', 'allowed_methods': ['POST']}
        
        else:
            return {
                'error': 'Not Found',
                'message': f'Path {path} not found',
                'available_endpoints': ['/api', '/api/health', '/api/test', '/api/chat', '/api/research'],
                'timestamp': datetime.now().isoformat()
            }