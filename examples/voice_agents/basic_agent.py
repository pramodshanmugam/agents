import logging
from pathlib import Path
import asyncio

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import deepgram, openai, silero
# from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents.tts.tts import TTSCapabilities, ChunkedStream, SynthesizedAudio, TTS, APIConnectOptions, AudioEmitter
import numpy as np
import socketio
import json
import websockets
from livekit import rtc
# uncomment to enable Krisp background voice/noise cancellation
# from livekit.plugins import noise_cancellation

logger = logging.getLogger("basic-agent")

load_dotenv()

# =============================================================================
# CHANGE THIS LINE TO SWITCH BETWEEN ROLES:
# Available options: "hotel_receptionist", "ai_recruiter", "f1_visa_interviewer"
# =============================================================================
SELECTED_ROLE = "hotel_receptionist"
# =============================================================================


class AvaTalkTTS(TTS):
    """Custom TTS that sends text to AvaTalk and returns empty audio frames for LiveKit compatibility"""
    
    def __init__(self, avatalk_client):
        super().__init__(
            capabilities=TTSCapabilities(streaming=False),
            sample_rate=24000,
            num_channels=1
        )
        self.avatalk_client = avatalk_client
        self._label = "AvaTalkTTS"
        self._text_buffer = ""  # Buffer to collect text chunks
        self._buffer_lock = asyncio.Lock()  # Lock for thread safety
    
    @property
    def label(self) -> str:
        return self._label

    def synthesize(self, text: str, **kwargs) -> ChunkedStream:
        """Create a ChunkedStream that sends text to AvaTalk and returns empty audio"""
        print(f"[AvaTalkTTS] synthesize called with text: {text}")
        
        # Add text to buffer
        asyncio.create_task(self._add_to_buffer(text))
        
        return AvaTalkChunkedStream(self, text, self.avatalk_client)
    
    async def _add_to_buffer(self, text: str):
        """Add text to buffer and send if buffer is getting long"""
        async with self._buffer_lock:
            self._text_buffer += text
            
            # If buffer is getting long or contains sentence endings, send it
            if len(self._text_buffer) > 200 or any(char in self._text_buffer for char in '.!?'):
                await self._send_buffered_text()
    
    async def _send_buffered_text(self):
        """Send the buffered text to AvaTalk"""
        if self._text_buffer:
            try:
                print(f"[AvaTalkTTS] Sending buffered text to AvaTalk: {self._text_buffer}")
                self.avatalk_client.send_tts(self._text_buffer)
                print(f"[AvaTalkTTS] Buffered text sent successfully")
            except Exception as e:
                print(f"[AvaTalkTTS] Error sending buffered text: {e}")
                logger.error(f"AvaTalkTTS error: {e}")
            finally:
                self._text_buffer = ""  # Clear buffer after sending

class AvaTalkChunkedStream(ChunkedStream):
    """ChunkedStream implementation for AvaTalk that sends text to AvaTalk and returns empty audio frames"""
    
    def __init__(self, tts: AvaTalkTTS, input_text: str, avatalk_client):
        super().__init__(tts=tts, input_text=input_text, conn_options=APIConnectOptions())
        self.avatalk_client = avatalk_client
        self._full_text = input_text  # Store the complete text
    
    async def _run(self, output_emitter: AudioEmitter) -> None:
        """Send complete text to AvaTalk and emit empty audio frame"""
        # Text is now handled by the TTS class buffering system
        # Just emit empty audio frame to satisfy the framework
        
        # Initialize the output emitter
        output_emitter.initialize(
            request_id="avatalk_request",
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/pcm",
            frame_size_ms=200,
            stream=False
        )
        
        # Create an empty audio frame to satisfy the framework
        empty_audio_data = np.zeros(1024, dtype=np.int16)  # 1024 samples of silence
        empty_frame_bytes = empty_audio_data.tobytes()
        
        # Push the empty audio data through the emitter
        output_emitter.push(empty_frame_bytes)
        output_emitter.flush()

class AvaTalkClient:
    def __init__(self, server_url="https://0j0t0qdf-8088.use.devtunnels.ms/", session_id="visa_interview", backend_id="visa_agent"):
        self.sio = socketio.Client()
        self.server_url = server_url
        self.session_id = session_id
        self.backend_id = backend_id

        self.sio.on('connect', self._on_socket_up)
        self.sio.on('backend_connected', self.on_connected)
        self.sio.on('backend_error', self.on_error)
        self.sio.on('backend_tts_sent', self.on_tts_sent)

    def connect(self):
        self.sio.connect(self.server_url)
        
    def _on_socket_up(self):
        self.sio.emit('backend_connect', {
            'session_id': self.session_id,
            'backend_id': self.backend_id
        })

    def send_tts(self, text):
        self.sio.emit('backend_direct_tts', {
            'session_id': self.session_id,
            'text': text,
            'backend_id': self.backend_id
        })

    def on_connected(self, data):
        print(f"[AvaTalk] Connected to session: {data['session_id']}")

    def on_error(self, data):
        print(f"[AvaTalk] Error: {data['message']}")

    def on_tts_sent(self, data):
        print(f"[AvaTalk] TTS sent: {data['text'][:50]}...")

async def listen_websocket(room_name: str, agent: AgentSession, ctx: JobContext, my_shutdown_hook,chat_context):
    """Listen for WebSocket messages from the backend"""
    
    websocket = None  # Initialize websocket variable
    try:
        uri = f"{WEBSOCKET_URL}/{room_name}"
        async with websockets.connect(uri) as websocket:
            logger.info(f"Connected to WebSocket for room {room_name}")
            
            # Send connection message
            await websocket.send(json.dumps({
                "type": "connection",
                "room": room_name,
                "content": {
                    "agent_type": F1_INTERVIEW_AGENT
                }
            }))

            while True:
                message = await websocket.recv()
                data = json.loads(message)
                logger.info(f"Received WebSocket message: {data}")

                message_type = data.get("type")
                if message_type == "time_warning":
                    remaining_seconds = data.get("remaining_seconds", 60)
                    logger.info(f"Websocket: Remaining seconds: {remaining_seconds}")
                    wrap_up_message = "We have one minute remaining."
                    await agent.say(wrap_up_message)
                    wrap_up_instruction = "Wrap up and end the interview. Do not ask any more questions."
            
                    chat_context.messages.append(
                        llm.ChatMessage(
                            role="system",
                            content=wrap_up_instruction
                        )
                    )
                    logger.info("WebSocket: Time warning message sent")

                elif message_type == "end_interview":
                    end_message = "Our time is up. Thank you for participating in this mock F1 visa interview."
                    await agent.say(end_message)
                    logger.info("WebSocket: End interview message sent")
                    # Trigger shutdown
                    ctx.add_shutdown_callback(my_shutdown_hook)
                    # Clean up the room
                    await room_manager.cleanup_room(ctx.room.name)
                    # Shutdown the agent
                    ctx.shutdown(reason="Interview time completed")
                    break


                # Send acknowledgment
                await websocket.send(json.dumps({
                    "type": "acknowledgment",
                    "room": room_name,
                    "content": {
                        "original_type": message_type,
                        "status": "received"
                    }
                }))

    except websockets.ConnectionClosed as e:
        logger.error(f"WebSocket connection closed: {e}")
    except Exception as e:
        logger.error(f"WebSocket Error: {e}")
    finally:
        # Ensure WebSocket is closed
        if websocket:
            try:
                await websocket.close()
                logger.info("WebSocket connection closed")
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")

class MyAgent(Agent):
    def __init__(self, role: str = "hotel_receptionist") -> None:
        # Load the appropriate prompt based on the role
        prompt = self._load_prompt(role)
        
        super().__init__(instructions=prompt)
        self.role = role

    def _load_prompt(self, role: str) -> str:
        """Load prompt from the prompts folder based on the role."""
        prompts_dir = Path(__file__).parent / "prompts"
        prompt_file = prompts_dir / f"{role}.txt"
        
        if not prompt_file.exists():
            logger.warning(f"Prompt file {prompt_file} not found, using default prompt")
            return "Your name is Kelly. You would interact with users via voice. with that in mind keep your responses concise and to the point. You are curious and friendly, and have a sense of humor."
        
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Error loading prompt from {prompt_file}: {e}")
            return "Your name is Kelly. You would interact with users via voice. with that in mind keep your responses concise and to the point. You are curious and friendly, and have a sense of humor."

    async def on_enter(self):
        # Generate a reply according to the agent's instructions
        # The agent will use the loaded prompt to generate an appropriate greeting
        self.session.generate_reply()

    # all functions annotated with @function_tool will be passed to the LLM when this
    # agent is active
    @function_tool
    async def lookup_weather(
        self, context: RunContext, location: str, latitude: str, longitude: str
    ):
        """Called when the user asks for weather related information.
        Ensure the user's location (city or region) is provided.
        When given a location, please estimate the latitude and longitude of the location and
        do not ask the user for them.

        Args:
            location: The location they are asking for
            latitude: The latitude of the location, do not ask user for it
            longitude: The longitude of the location, do not ask user for it
        """

        logger.info(f"Looking up weather for {location}")

        return "sunny with a temperature of 70 degrees."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):

    # Try to connect to AvaTalk, but don't fail if it's unavailable
    avatalk_client = None
   
    session_id = "visa_interview"  # <-- Hardcoded session ID
    avatalk_client = AvaTalkClient(server_url="https://0j0t0qdf-8088.use.devtunnels.ms/", session_id=session_id)
    avatalk_client.connect()
    logger.info("Successfully connected to AvaTalk TTS service")

    # Use the hardcoded role
    role = SELECTED_ROLE
    
    # Validate role
    valid_roles = ["hotel_receptionist", "ai_recruiter", "f1_visa_interviewer"]
    if role not in valid_roles:
        logger.warning(f"Invalid role '{role}', using default 'hotel_receptionist'")
        role = "hotel_receptionist"
    
    logger.info(f"Starting agent with role: {role}")
    
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
        "role": role,
    }

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        # any combination of STT, LLM, TTS, or realtime API can be used
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=openai.STT(model="whisper-1"),
        tts=AvaTalkTTS(avatalk_client) if avatalk_client else openai.TTS(voice="ash"),
        # use LiveKit's turn detection model
    )

    # log metrics as they are emitted, and total usage after session is over
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    # shutdown callbacks are triggered when the session is over
    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=MyAgent(role=role),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # uncomment to enable Krisp BVC noise cancellation
            # noise_cancellation=noise_cancellation.BVC(),
        ),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )

    # join the room when agent is ready
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
