from RealtimeSTT import AudioToTextRecorder

def process_text(text):
    print(text)

if __name__ == '__main__':
    print("Wait until it says 'speak now'")
    recorder = AudioToTextRecorder(
        input_device_index=7,
        enable_realtime_transcription=True,
        use_main_model_for_realtime=True,
        print_transcription_time=True,
        on_realtime_transcription_update=process_text,
    )

    while True:
        recorder.text(process_text)