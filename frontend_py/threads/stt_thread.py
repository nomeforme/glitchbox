from RealtimeSTT import AudioToTextRecorder
from PySide6.QtCore import QThread, Signal
from config import STT_DEVICE

class SpeechToTextThread(QThread):
    """Thread for handling real-time speech-to-text processing"""
    
    # Signal emitted when new text is transcribed
    transcription_updated = Signal(str)
    
    def __init__(self, input_device_index=None):
        super().__init__()
        self.input_device_index = input_device_index
        self.recorder = None
        self.running = False
        
    def on_transcription_update(self, text):
        """Callback function for real-time transcription updates"""
        # Emit the signal with the new transcription
        self.transcription_updated.emit(text)
        print(f"[STT] Transcription: {text}")
        
    def run(self):
        """Main thread execution"""
        self.running = True
        
        try:
            # Initialize the AudioToTextRecorder
            print(f"[STT] Initializing AudioToTextRecorder with input device index: {self.input_device_index}")
            self.recorder = AudioToTextRecorder(
                input_device_index=self.input_device_index,
                enable_realtime_transcription=True,
                use_main_model_for_realtime=True,
                print_transcription_time=True,
                on_realtime_transcription_update=self.on_transcription_update,
                no_log_file=True,  # Disable log file generation
                device=STT_DEVICE,
            )
            
            print("[STT] Speech-to-text system initialized, wait until it says 'speak now'")
            
            # Keep processing text while the thread is running
            while self.running:
                # Use the full model for final transcription
                self.recorder.text(self.on_transcription_update)
                
        except Exception as e:
            print(f"[STT] Error in speech-to-text processing: {e}")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up resources"""
        print("[STT] Cleaning up recorder resources")
        self.running = False
        self.recorder = None

    def stop(self):
        """Stop the speech-to-text processing"""
        if not self.isRunning():
            print("[STT] Thread is not running, nothing to stop")
            return

        print("[STT] Stopping speech-to-text thread")

        # Set running flag to False first to exit the loop
        self.running = False

        # Then shutdown the recorder to unblock the text() call
        # This makes the blocking text() call return immediately
        if self.recorder:
            try:
                print("[STT] Shutting down AudioToTextRecorder")
                self.recorder.shutdown()
                print("[STT] AudioToTextRecorder shutdown complete")
            except Exception as e:
                print(f"[STT] Error shutting down recorder: {e}")

        # Wait for the thread to finish naturally
        print("[STT] Waiting for thread to finish...")
        if not self.wait(5000):  # 5 second timeout
            print("[STT] WARNING: Thread did not finish in time, forcefully terminating")
            self.terminate()
            self.wait()  # Wait for termination to complete
            print("[STT] Thread forcefully terminated")
        else:
            print("[STT] Thread finished cleanly")