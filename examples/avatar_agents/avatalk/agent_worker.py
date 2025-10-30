import logging
import os

from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, WorkerType, cli
from livekit.agents.voice.room_io import RoomOutputOptions
from livekit.plugins.avatalk.avatar import AvatarSession


logger = logging.getLogger("avatalk-avatar-example")
logger.setLevel(logging.INFO)

load_dotenv()


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt="deepgram/nova-2",
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm="openai/gpt-4.1-mini",
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts="elevenlabs/eleven_multilingual_v2:2EiwWnXFnvU5JabPnv8n",
        
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        # turn_detection=MultilingualModel(),
        # vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
        # sometimes background noise could interrupt the agent session, these are considered false positive interruptions
        # when it's detected, you may resume the agent's speech
        resume_false_interruption=True,
        false_interruption_timeout=1.0,
    )

    avatalk_ws_url = os.getenv("AVATALK_WS_URL")
    if not avatalk_ws_url:
        raise ValueError("AVATALK_WS_URL is not set")

    await session.start(
        agent=Agent(instructions="Talk to me!"),
        room=ctx.room,
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )
    # attach avatalk after session started so we can chain the existing audio sink
    avatar = AvatarSession()
    await avatar.start(session, room=ctx.room)
    session.generate_reply(instructions="say hello to the user")
    



if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM))


