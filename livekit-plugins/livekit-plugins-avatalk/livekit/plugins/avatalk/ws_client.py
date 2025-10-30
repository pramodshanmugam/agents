from __future__ import annotations

import asyncio
import base64
import json
import urllib.parse
from typing import Any, Callable, Optional
import contextlib

import aiohttp


class AvatalkWSClient:
    def __init__(
        self,
        *,
        url: str,
        auth_token: Optional[str] = None,
        query: Optional[dict[str, str]] = None,
        fps: int = 15,
    ) -> None:
        self._url = url
        self._auth = auth_token
        self._query = query or {}
        self._fps = fps
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._recv_task: Optional[asyncio.Task[Any]] = None
        self._on_frame: Optional[Callable[[bytes], None]] = None

    async def connect(self, *, on_frame: Callable[[bytes], None]) -> None:
        self._on_frame = on_frame
        self._session = aiohttp.ClientSession()
        url = self._url
        if self._query:
            qs = urllib.parse.urlencode(self._query)
            sep = "&" if ("?" in url) else "?"
            url = f"{url}{sep}{qs}"
        headers = {}
        if self._auth:
            headers["Authorization"] = f"Bearer {self._auth}"
        self._ws = await self._session.ws_connect(url, headers=headers)
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def _recv_loop(self) -> None:
        assert self._ws is not None
        async for msg in self._ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            try:
                data = json.loads(msg.data)
            except Exception:
                # malformed json
                continue
            mtype = data.get("type")
            if mtype == "Frame":
                b64 = data.get("jpeg_b64")
                if isinstance(b64, str) and self._on_frame:
                    try:
                        jpeg = base64.b64decode(b64)
                        self._on_frame(jpeg)
                    except Exception:
                        # decoding/rendering issue
                        pass
            # other messages (AudioData, Complete, Error) can be handled/logged later

    async def send_wav_segment(self, wav_bytes: bytes, sample_rate: int, num_channels: int) -> None:
        assert self._ws is not None
        # Start
        start = {
            "type": "UploadWavStart",
            "sample_rate": int(sample_rate),
            "channels": int(num_channels),
            "total_bytes": len(wav_bytes),
        }
        await self._ws.send_str(json.dumps(start))
        # Chunk in ~128KB pieces
        chunk_size = 128 * 1024
        for i in range(0, len(wav_bytes), chunk_size):
            chunk = wav_bytes[i : i + chunk_size]
            msg = {"type": "UploadWavChunk", "chunk_b64": base64.b64encode(chunk).decode("ascii")}
            await self._ws.send_str(json.dumps(msg))
        # End
        await self._ws.send_str(json.dumps({"type": "UploadWavEnd"}))

    async def cancel(self) -> None:
        if self._ws:
            await self._ws.send_str(json.dumps({"type": "Cancel"}))

    async def close(self) -> None:
        if self._recv_task:
            self._recv_task.cancel()
            with contextlib.suppress(Exception):
                await self._recv_task
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()


