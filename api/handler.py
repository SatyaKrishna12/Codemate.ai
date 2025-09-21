"""
Simple Vercel-compatible handler for Deep Researcher Agent
"""
import json
from datetime import datetime
from urllib.parse import parse_qs

def handler(event, context):
    """
    Vercel serverless function handler
    Compatible with Vercel's Python runtime
    """
    try:
        # Handle different event structures
        if hasattr(event, 'get'):
            # Vercel event object
            path = event.get('path', '/')
            method = event.get('httpMethod', 'GET')
            headers = event.get('headers', {})
            query = event.get('queryStringParameters') or {}
            body = event.get('body', '')
        else:
            # Direct call or different structure
            path = getattr(event, 'path', '/')
            method = getattr(event, 'method', 'GET')
            headers = {}
            query = {}
            body = ''

        # Parse body if it's JSON
        request_data = {}
        if body and method in ['POST', 'PUT', 'PATCH']:
            try:
                request_data = json.loads(body) if isinstance(body, str) else body
            except:
                request_data = {}

        # Simple routing
        if path == '/' or path == '/api' or path == '/api/':
            response_data = {
                'message': 'Deep Researcher Agent API',
                'status': 'active',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0',
                'endpoints': [
                    '/api',
                    '/api/health',
                    '/api/test',
                    '/api/chat',
                    '/api/research'
                ]
            }
        elif path == '/api/health' or path == '/health':
            response_data = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'service': 'Deep Researcher Agent',
                'version': '1.0.0'
            }
        elif path == '/api/test' or path == '/test':
            response_data = {
                'test': 'success',
                'vercel': 'working',
                'timestamp': datetime.now().isoformat(),
                'method': method,
                'path': path
            }
        elif path == '/api/chat':
            if method == 'POST':
                response_data = {
                    'response': f"Chat endpoint received: {request_data.get('message', 'No message')}",
                    'chat_id': request_data.get('chat_id', 'default'),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success'
                }
            else:
                response_data = {
                    'error': 'Method not allowed',
                    'message': 'Use POST method for chat endpoint',
                    'allowed_methods': ['POST']
                }
        elif path == '/api/research':
            if method == 'POST':
                response_data = {
                    'query': request_data.get('query', 'No query provided'),
                    'depth': request_data.get('depth', 'deep'),
                    'result': f"Research query received: {request_data.get('query', 'No query')}",
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success'
                }
            else:
                response_data = {
                    'error': 'Method not allowed',
                    'message': 'Use POST method for research endpoint',
                    'allowed_methods': ['POST']
                }
        else:
            response_data = {
                'error': 'Not Found',
                'message': f'Path {path} not found',
                'available_endpoints': ['/api', '/api/health', '/api/test', '/api/chat', '/api/research'],
                'timestamp': datetime.now().isoformat()
            }

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization'
            },
            'body': json.dumps(response_data, indent=2)
        }

    except Exception as e:
        error_response = {
            'error': 'Internal Server Error',
            'message': str(e),
            'timestamp': datetime.now().isoformat(),
            'status': 'error'
        }
        
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(error_response, indent=2)
        }

# For backwards compatibility and different calling patterns
app = handler  # Some frameworks expect 'app'
main = handler  # Some expect 'main'