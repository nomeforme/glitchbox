from RealtimeSTT import AudioToTextRecorder
from PySide6.QtCore import QThread, Signal

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
        self.running = False
        
        # Release the AudioToTextRecorder resources if available
        if self.recorder:
            try:
                self.recorder.close()
                print("[STT] Recorder resources released")
            except Exception as e:
                print(f"[STT] Error releasing recorder resources: {e}")
            finally:
                self.recorder = None
                
    def stop(self):
        """Stop the speech-to-text processing"""
        print("[STT] Stopping speech-to-text thread")
        self.running = False
        
        # Wait for the thread to finish with timeout
        if not self.wait(2000):  # 2 second timeout
            print("[STT] Thread did not finish in time, terminating")
            self.terminate()