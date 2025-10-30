import logging
import asyncio
import os

from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, WorkerType, cli
from livekit import rtc
from livekit.agents.voice.room_io import RoomOutputOptions
from livekit.plugins.avatalk.avatar import AvatarSession
from livekit.plugins import elevenlabs


logger = logging.getLogger("avatalk-avatar-example")
logger.setLevel(logging.INFO)

load_dotenv()


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    
    # Create ElevenLabs TTS with custom voice
    elevenlabs_api_key = os.getenv("ELEVEN_API_KEY")
    if not elevenlabs_api_key:
        raise ValueError("ELEVEN_API_KEY is not set")
    
    custom_tts = elevenlabs.TTS(
        model="eleven_multilingual_v2",
        voice_id="nH3dKNst9otlofjj0qlP",
        api_key=elevenlabs_api_key,
    )
    
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt="deepgram/nova-2",
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm="openai/gpt-4.1-mini",
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # Pass the custom TTS instance directly
        tts=custom_tts,

        
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

    # Load MAP Bengaluru knowledge base
    kb_path = os.path.join(os.path.dirname(__file__), "map_bengaluru_knowledgebase_v1.json")
    with open(kb_path, "r") as f:
        import json
        kb_data = json.load(f)
    
    # Custom Agent that mirrors TTS frames out-of-band without touching room output
    class AvatalkTeeAgent(Agent):
        def __init__(self) -> None:
            instructions = """You are a friendly and knowledgeable museum assistant at the Museum of Art & Photography (MAP) in Bengaluru, India. Your role is to help visitors with information about the museum, its collection, exhibitions, events, facilities, and services.

Key information about MAP:
- Located on Kasturba Road Cross, near Cubbon Park and MG Road
- Open Tuesday-Sunday (Closed Mondays)
- Free entry on Tuesdays from 14:00-18:30
- Collection of 60,000+ artworks spanning 10th century to present
- Features digital experiences, rooftop restaurant (Cumulus by SMOOR), and MAP Store
- Fully accessible with wheelchair access, elevators, ISL support, and more

Use the knowledge base provided to answer visitor questions accurately. Be warm, helpful, and enthusiastic about the museum's offerings. If you don't know something specific, guide visitors to contact hello@map-india.org or visit map-india.org.

Always provide concise, visitor-friendly responses. Mention practical details like timings, locations, and contact information when relevant."""
            super().__init__(instructions=instructions)

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
                resampled_frames = resampler.push(combined)
                resampled_frames.extend(resampler.flush())
                if resampled_frames:
                    combined_16k = rtc.combine_audio_frames(resampled_frames)
                else:
                    combined_16k = combined
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
    session.generate_reply(instructions="Greet the visitor warmly and introduce yourself as a MAP museum assistant. Ask how you can help them today.")
    



if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM))


