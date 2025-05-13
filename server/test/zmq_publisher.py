import zmq
import time
import random

def main():
    # Create a ZMQ Context
    context = zmq.Context()
    
    # Create a publisher socket
    publisher = context.socket(zmq.PUB)
    
    # Bind the socket to a port
    publisher.bind("tcp://*:5555")
    
    print("Publisher started, sending messages...")
    
    try:
        while True:
            # Generate a random message
            message = f"Message {random.randint(1, 100)}"
            
            # Send the message
            publisher.send_string(message)
            print(f"Sent: {message}")
            
            # Wait for a short time
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nPublisher stopped by user")
    finally:
        # Clean up
        publisher.close()
        context.term()

if __name__ == "__main__":
    main()