from app import app
import os

if __name__ == '__main__':
    # Development server
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
else:
    # Production Gunicorn server
    app.config['PROPAGATE_EXCEPTIONS'] = True
