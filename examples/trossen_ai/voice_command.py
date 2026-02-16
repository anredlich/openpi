"""
Voice command listener for robot control.
Uses Silero VAD to detect speech automatically and faster-whisper for transcription.
Runs in a background thread — commands are placed in a queue for your robot code to consume.

Usage standalone:
    python voice_command.py

Usage in your robot code:
    from voice_command import VoiceCommandListener

    listener = VoiceCommandListener()
    listener.start()

    # In your robot loop:
    command = listener.get_command()  # Non-blocking, returns None if no command
    if command:
        print(f"Got command: {command}")

    # When done:
    listener.stop()
"""

import sounddevice as sd
import numpy as np
import tempfile
import os
import threading
import queue
import torch
from scipy.io.wavfile import write as wav_write
from faster_whisper import WhisperModel

# Configuration
DEVICE_INDEX = 10           # ReSpeaker device index
SAMPLE_RATE = 16000         # 16kHz for both VAD and Whisper
CHANNELS = 1                # Mono
VAD_THRESHOLD = 0.2         # Speech detection threshold (0-1, lower = more sensitive)
SILENCE_DURATION = 2.0      # Seconds of silence to end a command
CHUNK_SAMPLES = 512         # Silero VAD requires exactly 512 samples at 16kHz
MIN_SPEECH_DURATION = 0.5   # Minimum speech duration in seconds to transcribe


class VoiceCommandListener:
    # Prompt to guide Whisper toward your domain vocabulary
    INITIAL_PROMPT = "pick up place cube bucket red blue green yellow pink brown"

    def __init__(self, device_index=DEVICE_INDEX, whisper_model="medium",
                 vad_threshold=VAD_THRESHOLD, silence_duration=SILENCE_DURATION):
        self.device_index = device_index
        self.vad_threshold = vad_threshold
        self.silence_duration = silence_duration
        self.command_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = None

        # Load models
        print("Loading Whisper model...")
        self.whisper_model = WhisperModel(whisper_model, device="cuda", compute_type="float16")
        print("Whisper model loaded.")

        print("Loading Silero VAD model...")
        self.vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                           model='silero_vad', force_reload=False)
        print("VAD model loaded.")

    def start(self):
        """Start listening in a background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        print("\n🎙️  Listening for voice commands... (speak naturally)\n")

    def stop(self):
        """Stop listening."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)
        print("\n🔇 Voice listener stopped.")

    def get_command(self):
        """Non-blocking: get the next command or None."""
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None

    def get_command_blocking(self, timeout=None):
        """Blocking: wait for the next command."""
        try:
            return self.command_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _listen_loop(self):
        """Main listening loop — runs in background thread."""
        chunk_samples = CHUNK_SAMPLES  # 512 samples
        silence_chunks_needed = int(self.silence_duration * SAMPLE_RATE / chunk_samples)

        while not self._stop_event.is_set():
            try:
                self._wait_for_speech_and_transcribe(chunk_samples, silence_chunks_needed)
            except Exception as e:
                if not self._stop_event.is_set():
                    print(f"  [Error: {e}]")

    def _wait_for_speech_and_transcribe(self, chunk_samples, silence_chunks_needed):
        """Wait for speech, record until silence, then transcribe."""
        speech_chunks = []
        silence_count = 0
        is_speaking = False

        # Rolling buffer to capture audio BEFORE speech is detected
        # This prevents clipping the start of the utterance
        from collections import deque
        PRE_BUFFER_CHUNKS = 15  # ~0.5s of pre-speech audio at 512 samples/chunk
        pre_buffer = deque(maxlen=PRE_BUFFER_CHUNKS)

        # Open audio stream
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                                dtype='float32', device=self.device_index,
                                blocksize=chunk_samples)

        with stream:
            while not self._stop_event.is_set():
                audio_chunk, _ = stream.read(chunk_samples)
                audio_mono = audio_chunk[:, 0]

                # Run VAD
                audio_tensor = torch.from_numpy(audio_mono)
                speech_prob = self.vad_model(audio_tensor, SAMPLE_RATE).item()

                if speech_prob >= self.vad_threshold:
                    # Speech detected
                    if not is_speaking:
                        is_speaking = True
                        print("  🗣️  Speech detected...")
                        # Prepend the pre-buffer so we don't lose the start
                        speech_chunks.extend(list(pre_buffer))
                    speech_chunks.append(audio_mono.copy())
                    silence_count = 0
                elif is_speaking:
                    # Still recording but silence detected
                    speech_chunks.append(audio_mono.copy())
                    silence_count += 1

                    if silence_count >= silence_chunks_needed:
                        # End of utterance
                        break
                else:
                    # Not speaking yet — keep rolling buffer
                    pre_buffer.append(audio_mono.copy())

        if not speech_chunks:
            return

        # Check minimum duration
        audio_data = np.concatenate(speech_chunks)
        duration = len(audio_data) / SAMPLE_RATE

        if duration < MIN_SPEECH_DURATION:
            return

        # Convert to int16 for wav file
        audio_int16 = (audio_data * 32767).astype(np.int16)

        # Save to temp file and transcribe
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_write(tmp.name, SAMPLE_RATE, audio_int16)
        tmp.close()

        segments, _ = self.whisper_model.transcribe(tmp.name, beam_size=5, language="en",
                                                     initial_prompt=self.INITIAL_PROMPT)
        text = " ".join([seg.text.strip() for seg in segments]).strip()

        os.unlink(tmp.name)

        if text and not text.startswith("[") and len(text) > 1:
            print(f"  ✅ Command: \"{text}\"")
            self.command_queue.put(text)
        else:
            print("  (no speech recognized)")


def main():
    """Standalone mode — print commands as they are detected."""
    listener = VoiceCommandListener()
    listener.start()

    print("=" * 50)
    print("  ROBOT VOICE COMMAND LISTENER")
    print("=" * 50)
    print("  Speak naturally — commands are auto-detected")
    print("  Press Ctrl+C to exit")
    print("=" * 50)

    try:
        while True:
            command = listener.get_command_blocking(timeout=1.0)
            if command:
                print(f"\n  >> COMMAND READY: \"{command}\"\n")
                if 'exit' in command.lower() or 'stop' in command.lower():
                    break
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()


if __name__ == "__main__":
    main()
