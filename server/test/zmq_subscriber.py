import zmq

ip = "100.115.148.103"
port = 5555

def main():
    # Create a ZMQ Context
    context = zmq.Context()
    
    # Create a subscriber socket
    subscriber = context.socket(zmq.SUB)
    
    # Connect to the publisher
    subscriber.connect(f"tcp://{ip}:{port}")
    
    # Subscribe to all messages
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
    
    print("Subscriber started, waiting for messages...")
    
    try:
        while True:
            # Receive the message
            message = subscriber.recv_string()
            print(f"Received: {message}")
            
    except KeyboardInterrupt:
        print("\nSubscriber stopped by user")
    finally:
        # Clean up
        subscriber.close()
        context.term()

if __name__ == "__main__":
    main()