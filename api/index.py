import json
from datetime import datetime

def handler(event, context):
    """
    Simple Vercel serverless function handler
    This should work with any Vercel Python runtime
    """
    try:
        # Get the path from the event
        path = event.get('path', '/') if event else '/'
        method = event.get('httpMethod', 'GET') if event else 'GET'
        
        # Simple routing
        if path == '/' or path == '/api':
            response_data = {
                'message': 'Hello from Python on Vercel!',
                'timestamp': datetime.now().isoformat(),
                'status': 'working',
                'path': path,
                'method': method
            }
        elif path == '/test':
            response_data = {
                'test': 'success',
                'vercel': 'working',
                'timestamp': datetime.now().isoformat()
            }
        else:
            response_data = {
                'error': 'Path not found',
                'path': path,
                'available_paths': ['/', '/test', '/api']
            }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps(response_data)
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': str(e),
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            })
        }