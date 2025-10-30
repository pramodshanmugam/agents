import logging
import asyncio
import os

from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, WorkerType, cli
from livekit import rtc
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
        tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        # turn_detection=MultilingualModel(),
        # vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=False,
        # sometimes background noise could interrupt the agent session, these are considered false positive interruptions
        # when it's detected, you may resume the agent's speech
        resume_false_interruption=False,
        false_interruption_timeout=1.0,
    )

    avatalk_ws_url = os.getenv("AVATALK_WS_URL")
    if not avatalk_ws_url:
        raise ValueError("AVATALK_WS_URL is not set")

    # Start avatar first to have WS ready
    avatar = AvatarSession()
    await avatar.start(session, room=ctx.room)

    # Custom Agent that mirrors TTS frames out-of-band without touching room output
    class AvatalkTeeAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="Talk to me!")

        async def tts_node(self, text_stream, model_settings):  # type: ignore[override]
            # Buffer entire utterance (accept latency), send to Avatalk, then play audio aligned
            frames: list[rtc.AudioFrame] = []
            async for frame in super().tts_node(text_stream, model_settings):
                frames.append(frame)
            if not frames:
                return
            combined = rtc.combine_audio_frames(frames)
            if combined.sample_rate != 16000 or combined.num_channels != 1:
                resampler = rtc.AudioResampler(
                    input_rate=combined.sample_rate,
                    output_rate=16000,
                    num_channels=1,
                    quality=rtc.AudioResamplerQuality.HIGH,
                )
                combined_16k = resampler.resample(combined)
            else:
                combined_16k = combined
            wav_bytes = combined_16k.to_wav_bytes()
            # Reset events and send to Avatalk
            if hasattr(avatar, "first_frame_event"):
                avatar.first_frame_event = asyncio.Event()
            try:
                from livekit.plugins.avatalk.ws_client import AvatalkWSClient  # type: ignore
                ws = getattr(avatar, "_ws", None)
                if ws is not None and isinstance(ws, AvatalkWSClient):
                    await ws.send_wav_segment(wav_bytes, 16000, 1)
            except Exception:
                pass
            # Allow video to start now and flush buffered frames paced to FPS
            if hasattr(avatar, "start_video_now"):
                await avatar.start_video_now()  # type: ignore
            # Play frames to room with natural pacing
            for f in frames:
                await asyncio.sleep(max(0.0, f.duration))
                yield f

    await session.start(
        agent=AvatalkTeeAgent(),
        room=ctx.room,
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )
    session.generate_reply(instructions="say hello to the user")
    



if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM))


