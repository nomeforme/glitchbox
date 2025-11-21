import http.server
import ssl
import os

PORT = 8080
CERTFILE = './cert.pem' # Path to your certificate
KEYFILE = './key.pem'   # Path to your private key

# Navigate to the directory where this script is located
# This ensures files are served from frontend_efficient
os.chdir(os.path.dirname(os.path.abspath(__file__)))

httpd = http.server.HTTPServer(('0.0.0.0', PORT), http.server.SimpleHTTPRequestHandler)
httpd.socket = ssl.wrap_socket(httpd.socket,
                                server_side=True,
                                certfile=CERTFILE,
                                keyfile=KEYFILE,
                                ssl_version=ssl.PROTOCOL_TLS_SERVER) # Updated for modern SSL

print(f"Serving HTTPS on https://0.0.0.0:{PORT} from directory {os.getcwd()}...")
httpd.serve_forever()