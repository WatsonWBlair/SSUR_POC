import asyncio
import websockets
import wave
import json
import os
import io
from pathlib import Path


class AudioStreamServer:
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.response_audio_path = "response_audio.wav"
        
    async def handle_client(self, websocket):
        """Handle incoming WebSocket connections"""
        print(f"Client connected from {websocket.remote_address}")
        
        # Buffer to store incoming audio data
        audio_buffer = bytearray()
        current_audio_params = {}
        
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # Received audio data
                    audio_buffer.extend(message)
                    print(f"Received {len(message)} bytes of audio data (total: {len(audio_buffer)})")
                    
                elif isinstance(message, str):
                    # Received JSON control message
                    try:
                        data = json.loads(message)
                        
                        if data.get("action") == "start_streaming":
                            print("Client started streaming")
                            await websocket.send(json.dumps({
                                "status": "ready", 
                                "message": "Server ready to receive audio"
                            }))
                            
                        elif data.get("action") == "end_streaming":
                            print("Client ended streaming, processing audio...")
                            # Save received audio
                            if audio_buffer:
                                audio_params = data.get("audio_params", current_audio_params)
                                await self.save_received_audio(audio_buffer, audio_params)
                            
                            # Stream response audio back
                            await self.stream_response_audio(websocket)
                            break  # End the connection after streaming response
                            
                        elif data.get("action") == "audio_params":
                            # Store audio parameters for later use
                            current_audio_params = data.get("params", {})
                            print(f"Received audio parameters: {current_audio_params}")
                            
                    except json.JSONDecodeError:
                        print("Received invalid JSON message")
                        
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected")
        except Exception as e:
            print(f"Error handling client: {e}")
            import traceback
            traceback.print_exc()
            try:
                await websocket.send(json.dumps({
                    "status": "error", 
                    "message": str(e)
                }))
            except:
                # Connection might be closed, ignore send error
                pass
    
    async def save_received_audio(self, audio_data, audio_params):
        """Save received audio data to a WAV file"""
        try:
            print(f"Saving audio data: {len(audio_data)} bytes with params: {audio_params}")
            
            # Default audio parameters
            sample_rate = audio_params.get("sample_rate", 44100)
            channels = audio_params.get("channels", 1)
            sample_width = audio_params.get("sample_width", 2)
            
            # Validate parameters
            if sample_rate <= 0 or channels <= 0 or sample_width <= 0:
                print("Using default audio parameters due to invalid values")
                sample_rate, channels, sample_width = 44100, 1, 2
            
            # Save as received_audio.wav
            filename = "received_audio.wav"
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data)
            
            print(f"Successfully saved received audio to {filename}")
            
        except Exception as e:
            print(f"Error saving received audio: {e}")
            import traceback
            traceback.print_exc()
    
    async def stream_response_audio(self, websocket):
        """Stream a response audio file back to the client"""
        try:
            if not os.path.exists(self.response_audio_path):
                # If response audio doesn't exist, create a simple tone
                await self.create_sample_response_audio()
            
            # Read and stream the response audio file
            with wave.open(self.response_audio_path, 'rb') as wav_file:
                # Send audio parameters first
                params = {
                    "action": "response_audio_start",
                    "params": {
                        "channels": wav_file.getnchannels(),
                        "sample_width": wav_file.getsampwidth(),
                        "sample_rate": wav_file.getframerate(),
                        "total_frames": wav_file.getnframes()
                    }
                }
                await websocket.send(json.dumps(params))
                
                # Stream audio data in chunks
                chunk_size = 1024  # Smaller chunks for better streaming
                while True:
                    audio_chunk = wav_file.readframes(chunk_size)
                    if not audio_chunk:
                        break
                    
                    await websocket.send(audio_chunk)
                    # Small delay to control streaming rate
                    await asyncio.sleep(0.02)
                
                # Signal end of audio stream
                await websocket.send(json.dumps({
                    "action": "response_audio_end",
                    "message": "Response audio streaming complete"
                }))
                
                print("Response audio streaming complete")
                
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected during audio streaming")
        except Exception as e:
            print(f"Error streaming response audio: {e}")
            try:
                await websocket.send(json.dumps({
                    "status": "error",
                    "message": f"Error streaming response audio: {str(e)}"
                }))
            except:
                # Connection might be closed, ignore send error
                pass
    
    async def create_sample_response_audio(self):
        """Create a sample response audio file if none exists"""
        try:
            import numpy as np
            
            # Generate a simple tone (440 Hz for 3 seconds)
            sample_rate = 44100
            duration = 3.0
            frequency = 440.0
            
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            wave_data = np.sin(2 * np.pi * frequency * t)
            
            # Convert to 16-bit PCM
            wave_data = (wave_data * 32767).astype(np.int16)
            
            # Save as WAV file
            with wave.open(self.response_audio_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(wave_data.tobytes())
            
            print(f"Created sample response audio: {self.response_audio_path}")
            
        except Exception as e:
            print(f"Error creating sample response audio: {e}")
    
    async def start_server(self):
        """Start the WebSocket server"""
        print(f"Starting audio streaming server on {self.host}:{self.port}")
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            print(f"Audio server running on ws://{self.host}:{self.port}")
            print("Waiting for client connections...")
            await asyncio.Future()  # Run forever


if __name__ == "__main__":
    server = AudioStreamServer()
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        print("\nServer stopped by user")
