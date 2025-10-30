import logging
import os

from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, WorkerType, cli
from livekit.plugins.avatalk.avatar import AvatarSession


logger = logging.getLogger("avatalk-avatar-example")
logger.setLevel(logging.INFO)

load_dotenv()


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    session = AgentSession(
        llm="openai/gpt-4.1-mini",
        tts="elevenlabs/eleven_multilingual_v2:21m00Tcm4TlvDq8ikWAM",
        preemptive_generation=True,
        resume_false_interruption=False,
    )

    avatalk_ws_url = os.getenv("AVATALK_WS_URL")
    if not avatalk_ws_url:
        raise ValueError("AVATALK_WS_URL is not set")

    avatar = AvatarSession()
    await avatar.start(session, room=ctx.room)

    await session.start(
        agent=Agent(instructions="Talk to me!"),
        room=ctx.room,
    )
    session.generate_reply(instructions="say hello to the user")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM))


