"""Simple Voice-Activated Light Controller
Turns on light when voice is detected and fades it out when silence is detected.
"""

import asyncio
import numpy as np
import pyaudio
from scipy import signal
import time
from tapo import ApiClient

# --- Configuration ---
TAPO_USERNAME = "joaopaulo.passos@gmail.com"
TAPO_PASSWORD = "123456!#"
BULB_IP = "10.203.0.251"

# Audio settings
CHUNK_SIZE = 1024
SAMPLE_RATE = 44100
INPUT_DEVICE_INDEX = None
MIC_CHANNELS = 2  # Use stereo input to access right channel

# Voice detection settings
VOICE_THRESHOLD = 0.05  # Adjust this based on your microphone
SILENCE_DURATION = 2.5  # Seconds of silence before considering speech ended

# Light settings
MIN_BRIGHTNESS = 1  # Minimum allowed by the bulb
INITIAL_BRIGHTNESS = 1  # Quick initial response
MAX_BRIGHTNESS = 75
FADE_IN_STEP = 5    # Brightness increase per step
FADE_OUT_STEP = 5  # Brightness decrease per step
FADE_INTERVAL = 0.05  # Time between fade steps in seconds
MIN_API_INTERVAL = 0.05  # Minimum time between API calls to prevent overwhelming the bulb

class VoiceDetector:
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.silence_start_time = None  # Track when silence began
        self.is_speaking = False  # Binary state: speaking or not
        
        # Bandpass filter for voice frequencies (80Hz - 3000Hz)
        nyquist = sample_rate / 2.0
        low = 80 / nyquist
        high = 3000 / nyquist
        self.b, self.a = signal.butter(4, [low, high], btype='band')

    def process_chunk(self, audio_chunk_bytes):
        audio_data_float32 = np.frombuffer(audio_chunk_bytes, dtype=np.float32)
        current_time = time.monotonic()
        
        if len(audio_data_float32) == 0:
            return False
            
        try:
            filtered_audio = signal.filtfilt(self.b, self.a, audio_data_float32)
        except ValueError:
            filtered_audio = audio_data_float32

        rms = np.sqrt(np.mean(filtered_audio**2))
        is_voice = rms > VOICE_THRESHOLD
        
        # Handle silence duration
        if is_voice:
            self.silence_start_time = None
            self.is_speaking = True
        else:
            if self.silence_start_time is None:
                self.silence_start_time = current_time
            elif (current_time - self.silence_start_time) >= SILENCE_DURATION:
                self.is_speaking = False
        
        return self.is_speaking

class VoiceReactiveBulb:
    def __init__(self):
        self.device = None
        self.running = False
        self.detector = VoiceDetector()
        self.current_brightness = MIN_BRIGHTNESS  # Start at minimum brightness
        self.target_brightness = MIN_BRIGHTNESS   # Start at minimum brightness
        self.update_lock = asyncio.Lock()
        self.last_api_call_time = 0
        
    async def connect(self):
        client = ApiClient(TAPO_USERNAME, TAPO_PASSWORD)
        try:
            self.device = await client.l530(BULB_IP)
            
            # First turn off the bulb to ensure clean state
            await self.device.off()
            await asyncio.sleep(0.1)  # Small delay to ensure off command is processed
            
            # Then turn on and set initial state
            await self.device.on()
            await asyncio.sleep(0.1)  # Small delay to ensure on command is processed
            
            # Set color and brightness
            await self.device.set_hue_saturation(0, 100)
            await asyncio.sleep(0.1)  # Small delay to ensure color is set
            
            # Set initial brightness
            await self.device.set_brightness(MIN_BRIGHTNESS)
            await asyncio.sleep(0.1)  # Small delay to ensure brightness is set
            
            print(f"Connected to bulb ({BULB_IP}) and set to initial state")
            return True
        except Exception as e:
            print(f"Error connecting to bulb: {e}")
            self.device = None
            return False

    async def set_brightness(self, brightness):
        if not self.device:
            return
            
        current_time = time.monotonic()
        if current_time - self.last_api_call_time < MIN_API_INTERVAL:
            return
            
        async with self.update_lock:
            try:
                # Ensure brightness is always between MIN_BRIGHTNESS and MAX_BRIGHTNESS
                brightness = max(MIN_BRIGHTNESS, min(brightness, MAX_BRIGHTNESS))
                await self.device.set_brightness(brightness)
                self.current_brightness = brightness
                self.last_api_call_time = current_time
            except Exception as e:
                print(f"Error setting brightness: {e}")
                self.device = None

    async def run(self):
        if not await self.connect():
            print("Failed to connect to bulb. Exiting.")
            return

        p = pyaudio.PyAudio()
        self.running = True
        
        async def fade_task():
            while self.running:
                if self.current_brightness < self.target_brightness:
                    # Fade in - calculate next brightness based on time
                    time_since_last = time.monotonic() - self.last_api_call_time
                    if time_since_last >= MIN_API_INTERVAL:
                        steps = int(time_since_last / FADE_INTERVAL)
                        if steps > 0:
                            new_brightness = min(
                                self.current_brightness + (FADE_IN_STEP * steps),
                                self.target_brightness
                            )
                            await self.set_brightness(new_brightness)
                elif self.current_brightness > self.target_brightness:
                    # Fade out - calculate next brightness based on time
                    time_since_last = time.monotonic() - self.last_api_call_time
                    if time_since_last >= MIN_API_INTERVAL:
                        steps = int(time_since_last / FADE_INTERVAL)
                        if steps > 0:
                            new_brightness = max(
                                self.current_brightness - (FADE_OUT_STEP * steps),
                                self.target_brightness
                            )
                            await self.set_brightness(new_brightness)
                await asyncio.sleep(FADE_INTERVAL)
        
        fade_task_handle = asyncio.create_task(fade_task())
        
        def audio_stream_callback(in_data, frame_count, time_info, status_flags):
            if not self.running:
                return (None, pyaudio.paComplete)
            
            audio_array_int16 = np.frombuffer(in_data, dtype=np.int16)
            if MIC_CHANNELS == 2:
                audio_array_int16 = audio_array_int16[1::2]  # Use right channel (index 1)
            audio_array_float32 = audio_array_int16.astype(np.float32) / 32768.0
            
            is_speaking = self.detector.process_chunk(audio_array_float32.tobytes())
            
            # Simple binary state: either fade up or fade down
            self.target_brightness = MAX_BRIGHTNESS if is_speaking else MIN_BRIGHTNESS
            
            return (None, pyaudio.paContinue)

        stream = None
        try:
            stream = p.open(format=pyaudio.paInt16,
                          channels=MIC_CHANNELS,
                          rate=SAMPLE_RATE,
                          input=True,
                          frames_per_buffer=CHUNK_SIZE,
                          input_device_index=INPUT_DEVICE_INDEX,
                          stream_callback=audio_stream_callback)
            print("Microphone stream opened. Listening...")
            stream.start_stream()
        except Exception as e:
            print(f"Error opening microphone stream: {e}")
            p.terminate()
            self.running = False
            if fade_task_handle:
                await asyncio.wait_for(fade_task_handle, timeout=1.0)
            return

        try:
            while self.running:
                if not self.device:
                    print("Bulb connection lost, attempting to reconnect...")
                    if not await self.connect():
                        await asyncio.sleep(5)
                        continue
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.running = False
            if stream:
                stream.stop_stream()
                stream.close()
            p.terminate()
            
            if fade_task_handle:
                fade_task_handle.cancel()
                try:
                    await fade_task_handle
                except asyncio.CancelledError:
                    pass

            if self.device:
                try:
                    await self.device.off()
                except Exception as e:
                    print(f"Error turning off bulb: {e}")
            print("Cleanup complete.")

async def main():
    controller = VoiceReactiveBulb()
    await controller.run()

if __name__ == "__main__":
    print("Simple Voice-Activated Light Controller")
    print("======================================")
    print("Press Ctrl+C to stop.")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Application terminated by user.")
    except Exception as e:
        print(f"An error occurred: {e}")