import logging
from pathlib import Path

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
from livekit.plugins.turn_detector.multilingual import MultilingualModel

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
        # Generate an appropriate greeting based on the role
        if self.role == "hotel_receptionist":
            self.session.generate_reply("Welcome to the Grand Plaza Hotel! How may I assist you today?")
        elif self.role == "ai_recruiter":
            self.session.generate_reply("Hello! I'm Alex from TechCorp Innovations. Thank you for your interest in our company. I'll be conducting your initial screening interview today.")
        elif self.role == "f1_visa_interviewer":
            self.session.generate_reply("Good day. I'm Officer Chen, and I'll be conducting your F1 student visa interview today. Please confirm your full name and the purpose of your visit.")
        else:
            # when the agent is added to the session, it'll generate a reply
            # according to its instructions
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
        tts=openai.TTS(voice="ash")
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
