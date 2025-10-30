from __future__ import annotations

import asyncio
import io
import os
from dataclasses import dataclass
from typing import Optional

from PIL import Image

from livekit import rtc, api
from livekit.agents import (
    AgentSession,
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    get_job_context,
    utils,
)
from livekit.agents.voice.io import AudioOutput, AudioOutputCapabilities

from .ws_client import AvatalkWSClient


DEFAULT_SAMPLE_RATE = 16000
DEFAULT_IMG_SIZE = 96


@dataclass
class _BufferedSegment:
    sample_rate: int
    num_channels: int
    pcm_bytes: bytearray


class _AvatalkAudioOutput(AudioOutput):
    def __init__(self, *, on_segment_ready, sample_rate: Optional[int] = None) -> None:
        super().__init__(
            label="AvatalkAudioOutput",
            next_in_chain=None,
            sample_rate=sample_rate,
            capabilities=AudioOutputCapabilities(pause=True),
        )
        self._current: Optional[_BufferedSegment] = None
        self._on_segment_ready = on_segment_ready

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        if self._current is None:
            self._current = _BufferedSegment(
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
                pcm_bytes=bytearray(),
            )
        # frame.data is PCM16 little-endian
        self._current.pcm_bytes.extend(bytes(frame.data))

    def flush(self) -> None:
        super().flush()
        if not self._current:
            return
        seg = self._current
        self._current = None
        wav_bytes = _pcm16_to_wav(seg.pcm_bytes, seg.sample_rate, seg.num_channels)
        # hand off synchronously
        self._on_segment_ready(wav_bytes, seg.sample_rate, seg.num_channels)

    def clear_buffer(self) -> None:
        # Drop any buffered audio for the current segment
        self._current = None


def _pcm16_to_wav(pcm: bytearray, sample_rate: int, num_channels: int) -> bytes:
    # Minimal RIFF/WAVE header for PCM16
    byte_rate = sample_rate * num_channels * 2
    block_align = num_channels * 2
    data_size = len(pcm)
    riff_size = 36 + data_size
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write((riff_size).to_bytes(4, "little"))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write((16).to_bytes(4, "little"))  # PCM fmt chunk size
    buf.write((1).to_bytes(2, "little"))   # audio format PCM
    buf.write((num_channels).to_bytes(2, "little"))
    buf.write((sample_rate).to_bytes(4, "little"))
    buf.write((byte_rate).to_bytes(4, "little"))
    buf.write((block_align).to_bytes(2, "little"))
    buf.write((16).to_bytes(2, "little"))  # bits per sample
    buf.write(b"data")
    buf.write((data_size).to_bytes(4, "little"))
    buf.write(pcm)
    return buf.getvalue()


class AvatarSession:
    def __init__(
        self,
        *,
        ws_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        img_size: Optional[int] = None,
        fps: Optional[int] = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        self._ws_url = ws_url or os.getenv("AVATALK_WS_URL")
        if not self._ws_url:
            raise ValueError("AVATALK_WS_URL must be set")
        self._auth_token = auth_token or os.getenv("AVATALK_AUTH")
        self._img_size = img_size or int(os.getenv("AVATALK_IMG_SIZE", str(DEFAULT_IMG_SIZE)))
        self._fps = fps or int(os.getenv("AVATALK_FPS", "15"))
        self._conn_options = conn_options

        self._video_source: Optional[rtc.VideoSource] = None
        self._video_pub: Optional[rtc.LocalTrackPublication] = None
        self._ws: Optional[AvatalkWSClient] = None
        self._first_frame_received: bool = False
        self._placeholder_task: Optional[asyncio.Task] = None

    async def start(self, agent_session: AgentSession, room: rtc.Room) -> None:
        # Ensure room is connected before publishing
        if not room.isconnected():
            connected_fut: asyncio.Future[None] = asyncio.get_event_loop().create_future()

            def _on_state_changed(state: rtc.ConnectionState) -> None:
                if room.isconnected() and not connected_fut.done():
                    connected_fut.set_result(None)

            room.on("connection_state_changed", _on_state_changed)
            await connected_fut

        # Publish a video track from the worker participant
        self._video_source = rtc.VideoSource(width=self._img_size, height=self._img_size)
        video_track = rtc.LocalVideoTrack.create_video_track("avatalk_avatar", self._video_source)
        video_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
        self._video_pub = await room.local_participant.publish_track(video_track, video_options)

        # Start placeholder frames so track is immediately visible
        async def _placeholder_loop() -> None:
            if not self._video_source:
                return
            color = bytes([0, 0, 0, 255])
            buf = bytearray(self._img_size * self._img_size * 4)
            while not self._first_frame_received:
                buf[:] = color * self._img_size * self._img_size
                frame = rtc.VideoFrame(self._img_size, self._img_size, rtc.VideoBufferType.RGBA, buf)
                self._video_source.capture_frame(frame)
                await asyncio.sleep(0.2)

        self._placeholder_task = asyncio.create_task(_placeholder_loop())

        # Prepare WS connection
        # Try to include room/identity for observability on the gateway side
        try:
            job_ctx = get_job_context()
            local_identity = job_ctx.token_claims().identity
            room_name = room.name
        except Exception:
            local_identity = room.local_participant.identity
            room_name = room.name

        self._ws = AvatalkWSClient(
            url=self._ws_url,
            auth_token=self._auth_token,
            query={"room": room_name, "identity": local_identity},
            fps=self._fps,
        )

        # Frame callback: decode JPEG → RGBA → push to video source
        def _on_frame(jpeg_bytes: bytes) -> None:
            if not self._video_source:
                return
            with Image.open(io.BytesIO(jpeg_bytes)) as im:
                if im.mode != "RGBA":
                    im = im.convert("RGBA")
                if im.size != (self._img_size, self._img_size):
                    im = im.resize((self._img_size, self._img_size))
                rgba = im.tobytes()
            frame = rtc.VideoFrame(self._img_size, self._img_size, rtc.VideoBufferType.RGBA, rgba)
            self._video_source.capture_frame(frame)
            if not self._first_frame_received:
                self._first_frame_received = True

        await self._ws.connect(on_frame=_on_frame)

        # Wire audio output to forward WAV segments
        def _on_segment_ready(wav_bytes: bytes, sample_rate: int, num_channels: int) -> None:
            # send full WAV as one or more chunks
            asyncio.create_task(self._ws.send_wav_segment(wav_bytes, sample_rate, num_channels))

        agent_session.output.audio = _AvatalkAudioOutput(on_segment_ready=_on_segment_ready)

    async def aclose(self) -> None:
        if self._ws:
            await self._ws.close()
        if self._video_pub:
            await self._video_pub.unpublish()
        if self._placeholder_task:
            self._placeholder_task.cancel()


