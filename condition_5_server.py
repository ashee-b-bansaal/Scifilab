import argparse
import socket
import json
import threading
import datetime

# TCP server configuration
HOST = '192.168.0.38'  # Listen on all available interfaces
PORT = 9000       # Default port (same as in the Android app)
BUFFER_SIZE = 1024

# File to save chat logs
LOG_FILE = 'chat_logs.txt'

def handle_client(client_socket, client_address):
    """Handle incoming client connections"""
    print(f"Connection from {client_address}")
    
    try:
        # Receive data from client
        data = client_socket.recv(BUFFER_SIZE).decode('utf-8').strip()
        
        if data:
            # Parse JSON message
            try:
                message_json = json.loads(data)
                timestamp = message_json.get('timestamp', str(datetime.datetime.now()))
                sender = message_json.get('sender', 'Unknown')
                message_text = message_json.get('message', '')
                
                # Format message for display and logging
                formatted_message = f"[{timestamp}] {sender}: {message_text}"
                print(formatted_message)
                
                # Save to log file
                with open(LOG_FILE, 'a') as log_file:
                    log_file.write(formatted_message + '\n')
                
            except json.JSONDecodeError:
                print(f"Received non-JSON data: {data}")
    except Exception as e:
        print(f"Error handling client: {e}")
    finally:
        # Close the connection
        client_socket.close()

def main():
    """Main server function"""
    # Create a TCP/IP socket
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Allow reuse of address
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Bind the socket to the address
    server.bind((HOST, PORT))
    
    # Listen for incoming connections (max 5 queued connections)
    server.listen(5)
    
    print(f"TCP Server started on {HOST}:{PORT}")
    print(f"Waiting for connections...")
    
    try:
        while True:
            # Wait for a connection
            client_socket, client_address = server.accept()
            
            # Start a new thread to handle the client
            client_thread = threading.Thread(
                target=handle_client,
                args=(client_socket, client_address)
            )
            client_thread.daemon = True
            client_thread.start()
            
    except KeyboardInterrupt:
        print("Server shutting down...")
    finally:
        server.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        required = True
    )
    args = parser.parse_args()
    LOG_FILE = args.path
    main()
