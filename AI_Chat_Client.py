import gradio as gr
import subprocess
import threading
import asyncio
import websockets
import wave
import json
import numpy as np
import os

AI_AUDIO_OUTPUT_PATH = "ai_output.wav"
WS_URL = "ws://localhost:8000/ws"

received_audio_buffer = bytearray()
response_audio_params = {}


def play_audio_file(file_path):
    subprocess.run(['ffplay', '-nodisp', '-autoexit', file_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


async def send_audio_stream(websocket, audio_path):
    with wave.open(audio_path, 'rb') as wav_file:
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
        await websocket.send(json.dumps({"action": "start_streaming"}))

        chunk_size = 1024
        while True:
            chunk = wav_file.readframes(chunk_size)
            if not chunk:
                break
            await websocket.send(chunk)
            await asyncio.sleep(0.02)

        await websocket.send(json.dumps({"action": "end_streaming", "audio_params": params["params"]}))


async def receive_audio_stream(websocket):
    global received_audio_buffer, response_audio_params
    async for message in websocket:
        if isinstance(message, str):
            try:
                data = json.loads(message)
                if data.get("action") == "response_audio_start":
                    response_audio_params = data.get("params", {})
                elif data.get("action") == "response_audio_end":
                    await save_audio()
                    break
            except json.JSONDecodeError:
                print("Invalid JSON")
        elif isinstance(message, bytes):
            received_audio_buffer.extend(message)


async def save_audio():
    if not received_audio_buffer or not response_audio_params:
        return
    with wave.open(AI_AUDIO_OUTPUT_PATH, 'wb') as wav_file:
        wav_file.setnchannels(response_audio_params.get("channels", 1))
        wav_file.setsampwidth(response_audio_params.get("sample_width", 2))
        wav_file.setframerate(response_audio_params.get("sample_rate", 44100))
        wav_file.writeframes(received_audio_buffer)


def handle_voice_input(audio_path):
    global received_audio_buffer, response_audio_params
    received_audio_buffer = bytearray()
    response_audio_params = {}

    async def run():
        try:
            async with websockets.connect(WS_URL) as websocket:
                await send_audio_stream(websocket, audio_path)
                await receive_audio_stream(websocket)
        except Exception as e:
            print(f"WebSocket error: {e}")
            return None, "WebSocket connection failed", None

    asyncio.run(run())

    if os.path.exists(AI_AUDIO_OUTPUT_PATH):
        threading.Thread(target=play_audio_file, args=(AI_AUDIO_OUTPUT_PATH,)).start()
        return AI_AUDIO_OUTPUT_PATH, "Streaming complete", None
    else:
        return None, "Failed to receive audio", None


with gr.Blocks() as demo:
    gr.Markdown("🎙️ **Voice AI Streaming Client**")

    with gr.Row():
        mic = gr.Audio(label="🎤 Record Your Voice")
        response_audio = gr.Audio(label="🗣️ AI Response", interactive=False)

    latency_text = gr.Textbox(label="⏱️ Status", interactive=False)
    submit_btn = gr.Button("Send to AI")

    submit_btn.click(
        fn=handle_voice_input,
        inputs=[mic],
        outputs=[response_audio, latency_text, gr.State()]
    )

    demo.launch(server_name="0.0.0.0", server_port=7000)
