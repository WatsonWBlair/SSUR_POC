import asyncio
import websockets
import wave
import json
import numpy as np
import os


class AudioStreamClient:
    def __init__(self, server_uri="ws://localhost:8765"):
        self.server_uri = server_uri
        self.received_audio_buffer = bytearray()
        self.response_audio_params = None
        
    async def create_test_audio(self, filename="test_input.wav"):
        """Create a test audio file to send to the server"""
        try:
            # Generate a simple sine wave (880 Hz for 2 seconds)
            sample_rate = 44100
            duration = 2.0
            frequency = 880.0
            
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            wave_data = np.sin(2 * np.pi * frequency * t)
            
            # Convert to 16-bit PCM
            wave_data = (wave_data * 32767).astype(np.int16)
            
            # Save as WAV file
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(1)  # mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(wave_data.tobytes())
            
            print(f"Created test audio file: {filename}")
            return filename
            
        except Exception as e:
            print(f"Error creating test audio: {e}")
            return None
    
    async def send_audio_file(self, websocket, filename):
        """Send audio file to the server"""
        try:
            with wave.open(filename, 'rb') as wav_file:
                # Send audio parameters first
                params = {
                    "action": "audio_params",
                    "params": {
                        "channels": wav_file.getnchannels(),
                        "sample_width": wav_file.getsampwidth(),
                        "sample_rate": wav_file.getframerate(),
                        "total_frames": wav_file.getnframes()
                    }
                }
                await websocket.send(json.dumps(params))
                
                # Signal start of streaming
                await websocket.send(json.dumps({"action": "start_streaming"}))
                
                # Stream audio data in chunks
                chunk_size = 1024  # Smaller chunks for better streaming
                total_sent = 0
                
                while True:
                    audio_chunk = wav_file.readframes(chunk_size)
                    if not audio_chunk:
                        break
                    
                    await websocket.send(audio_chunk)
                    total_sent += len(audio_chunk)
                    print(f"Sent {len(audio_chunk)} bytes (total: {total_sent})")
                    
                    # Small delay to simulate real-time streaming
                    await asyncio.sleep(0.02)
                
                # Signal end of streaming
                await websocket.send(json.dumps({
                    "action": "end_streaming",
                    "audio_params": params["params"]
                }))
                
                print(f"Finished sending audio file ({total_sent} bytes total)")
                
        except Exception as e:
            print(f"Error sending audio file: {e}")
    
    async def receive_response_audio(self, websocket):
        """Receive and save the response audio from server"""
        print("Waiting for response audio...")
        
        async for message in websocket:
            if isinstance(message, str):
                try:
                    data = json.loads(message)
                    
                    if data.get("action") == "response_audio_start":
                        self.response_audio_params = data.get("params", {})
                        print(f"Starting to receive response audio: {self.response_audio_params}")
                        
                    elif data.get("action") == "response_audio_end":
                        print("Response audio streaming complete")
                        await self.save_response_audio()
                        break
                        
                    elif data.get("status") == "ready":
                        print(f"Server status: {data.get('message')}")
                        
                    elif data.get("status") == "error":
                        print(f"Server error: {data.get('message')}")
                        break
                        
                except json.JSONDecodeError:
                    print("Received invalid JSON from server")
                    
            elif isinstance(message, bytes):
                # Received audio data
                self.received_audio_buffer.extend(message)
                print(f"Received {len(message)} bytes of response audio")
    
    async def save_response_audio(self):
        """Save the received response audio to a file"""
        if not self.received_audio_buffer or not self.response_audio_params:
            print("No response audio data to save")
            return
        
        try:
            filename = "received_response_audio.wav"
            
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(self.response_audio_params.get("channels", 1))
                wav_file.setsampwidth(self.response_audio_params.get("sample_width", 2))
                wav_file.setframerate(self.response_audio_params.get("sample_rate", 44100))
                wav_file.writeframes(self.received_audio_buffer)
            
            print(f"Saved response audio to {filename} ({len(self.received_audio_buffer)} bytes)")
            
        except Exception as e:
            print(f"Error saving response audio: {e}")
    
    async def connect_and_stream(self):
        """Main method to connect to server and handle audio streaming"""
        try:
            # Create test audio file
            test_file = await self.create_test_audio()
            if not test_file:
                return
            
            # Connect to server
            print(f"Connecting to server at {self.server_uri}")
            async with websockets.connect(self.server_uri) as websocket:
                print("Connected to server!")
                
                # Send audio file
                await self.send_audio_file(websocket, test_file)
                
                # Receive response audio
                await self.receive_response_audio(websocket)
                
                print("Audio streaming session complete!")
                
        except websockets.ConnectionClosed:
            print("Could not connect to server. Make sure the server is running.")
        except ConnectionRefusedError:
            print("Could not connect to server. Make sure the server is running.")
        except Exception as e:
            print(f"Error during streaming session: {e}")


async def main():
    client = AudioStreamClient()
    await client.connect_and_stream()


if __name__ == "__main__":
    asyncio.run(main())
