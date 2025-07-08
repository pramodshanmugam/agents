import logging
from pathlib import Path
import numpy as np

# Try to import socketio with proper error handling
try:
    import socketio
except ImportError:
    try:
        import python_socketio as socketio
    except ImportError:
        print("ERROR: Please install python-socketio package:")
        print("pip install python-socketio")
        raise ImportError("python-socketio package is required for AvaTalk integration")

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
from livekit.agents.tts.tts import TTSCapabilities, ChunkedStream, SynthesizedAudio, TTS
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

# =============================================================================
# AVA TALK CONFIGURATION:
# Set to False to use standard OpenAI TTS instead of AvaTalk
# =============================================================================
USE_AVATALK = False  # Set to True to enable AvaTalk integration
# =============================================================================

# AvaTalk Integration Classes
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
    
    @property
    def label(self) -> str:
        return self._label

    def synthesize(self, text: str, **kwargs) -> ChunkedStream:
        """Create a ChunkedStream that sends text to AvaTalk and returns empty audio"""
        return AvaTalkChunkedStream(self, text, self.avatalk_client)

class AvaTalkChunkedStream(ChunkedStream):
    """ChunkedStream implementation for AvaTalk that sends text to AvaTalk and returns empty audio frames"""
    
    def __init__(self, tts: AvaTalkTTS, input_text: str, avatalk_client):
        super().__init__(tts=tts, input_text=input_text, conn_options={})
        self.avatalk_client = avatalk_client
    
    async def _run(self) -> None:
        """Send text to AvaTalk and emit empty audio frame"""
        try:
            print(f"[AvaTalkTTS] Sending to AvaTalk: {self.input_text[:50]}...")
            self.avatalk_client.send_tts(self.input_text)
            print(f"[AvaTalkTTS] Text sent to AvaTalk successfully")
        except Exception as e:
            print(f"[AvaTalkTTS] Error sending to AvaTalk: {e}")
            logger.error(f"AvaTalkTTS error: {e}")
        
        # Create an empty audio frame to satisfy the framework
        empty_audio_data = np.zeros(1024, dtype=np.int16)  # 1024 samples of silence
        empty_frame = rtc.AudioFrame(
            data=empty_audio_data.tobytes(),
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            samples_per_channel=len(empty_audio_data)
        )
        
        # Emit the empty audio frame
        self._event_ch.send_nowait(SynthesizedAudio(
            frame=empty_frame,
            request_id="avatalk_request",
            is_final=True
        ))

class AvaTalkClient:
    def __init__(self, server_url="https://0j0t0qdf-8088.use.devtunnels.ms/", session_id="basic_agent", backend_id="basic_agent"):
        try:
            self.sio = socketio.Client()
        except AttributeError:
            print("ERROR: Invalid socketio package. Please install python-socketio:")
            print("pip install python-socketio")
            raise ImportError("python-socketio package is required")
            
        self.server_url = server_url
        self.session_id = session_id
        self.backend_id = backend_id

        self.sio.on('connect', self._on_socket_up)
        self.sio.on('backend_connected', self.on_connected)
        self.sio.on('backend_error', self.on_error)
        self.sio.on('backend_tts_sent', self.on_tts_sent)

    def connect(self):
        try:
            self.sio.connect(self.server_url)
        except Exception as e:
            print(f"[AvaTalk] Connection error: {e}")
            logger.error(f"AvaTalk connection error: {e}")

    def _on_socket_up(self):
        self.sio.emit('backend_connect', {
            'session_id': self.session_id,
            'backend_id': self.backend_id
        })

    def send_tts(self, text):
        try:
            self.sio.emit('backend_direct_tts', {
                'session_id': self.session_id,
                'text': text,
                'backend_id': self.backend_id
            })
        except Exception as e:
            print(f"[AvaTalk] Error sending TTS: {e}")
            logger.error(f"AvaTalk TTS error: {e}")

    def on_connected(self, data):
        print(f"[AvaTalk] Connected to session: {data['session_id']}")

    def on_error(self, data):
        print(f"[AvaTalk] Error: {data['message']}")

    def on_tts_sent(self, data):
        print(f"[AvaTalk] TTS sent: {data['text'][:50]}...")


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
        # The role-specific greeting is already included in the prompt
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
    # Use the hardcoded role
    role = SELECTED_ROLE
    
    # Validate role
    valid_roles = ["hotel_receptionist", "ai_recruiter", "f1_visa_interviewer"]
    if role not in valid_roles:
        logger.warning(f"Invalid role '{role}', using default 'hotel_receptionist'")
        role = "hotel_receptionist"
    
    logger.info(f"Starting agent with role: {role}")
    
    # AvaTalk integration: initialize client with error handling
    avatalk_client = None
    use_avatalk = USE_AVATALK
    
    if use_avatalk:
        try:
            session_id = "visa_interview"  # Create session ID based on role
            avatalk_client = AvaTalkClient(
                server_url="https://0j0t0qdf-8088.use.devtunnels.ms/", 
                session_id=session_id,
                backend_id=f"{role}_agent"
            )
            avatalk_client.connect()
            logger.info("AvaTalk client initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize AvaTalk client: {e}")
            logger.info("Falling back to standard TTS")
            avatalk_client = None
            use_avatalk = False
    
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
        "role": role,
    }

    # Choose TTS based on AvaTalk availability
    if use_avatalk and avatalk_client:
        tts = AvaTalkTTS(avatalk_client)
        logger.info("Using AvaTalk TTS")
    else:
        tts = openai.TTS(voice="ash")
        logger.info("Using standard OpenAI TTS")

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        # any combination of STT, LLM, TTS, or realtime API can be used
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=openai.STT(model="whisper-1"),
        tts=tts
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
