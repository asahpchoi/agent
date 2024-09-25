from __future__ import annotations

import asyncio
import os
import io
from io import BytesIO
from dataclasses import dataclass
from typing import Any, List, Literal

import aiohttp
from livekit.agents import tts, utils, tokenize
from livekit import rtc

from .log import logger
from .models import TTSEncoding
from enum import Enum

class TTSEngines(str, Enum):
    MINIMAX = "minimax"

_Encoding = Literal["mp3", "wav"]

@dataclass
class Voice:
    id: str
    name: str
    voice_engine: str

DEFAULT_VOICE = Voice(
    id="default",
    name="Default",
    voice_engine=TTSEngines.MINIMAX.value
)

API_BASE_URL = "https://api.minimax.chat/v1/text_to_speech"
@dataclass
class _TTSOptions:
    api_key: str
    group_id: str
    voice: Voice
    base_url: str
    encoding: _Encoding

class TTS(tts.TTS):
    MINIMAX_TTS_SAMPLE_RATE = 24000
    MINIMAX_TTS_CHANNELS = 1

    def __init__(
            self,
            *,
            voice: Voice | str = DEFAULT_VOICE,
            api_key: str | None = None,
            group_id: str | None = None,
            base_url: str | None = None,
            encoding: _Encoding = "wav",
            http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=self.MINIMAX_TTS_SAMPLE_RATE,
            num_channels=self.MINIMAX_TTS_CHANNELS,
        )
        logger.info(f"Initializing TTS with sample_rate: {self.MINIMAX_TTS_SAMPLE_RATE}, channels: {self.MINIMAX_TTS_CHANNELS}")
        api_key = api_key or os.environ.get("MINIMAX_API_KEY")
        if not api_key:
            raise ValueError("MINIMAX_API_KEY must be set")

        group_id = group_id or os.environ.get("MINIMAX_GROUP_ID")
        if not group_id:
            raise ValueError("MINIMAX_GROUP_ID must be set")

        if isinstance(voice, str):
            voice = Voice(id=voice, name=voice, voice_engine=TTSEngines.MINIMAX.value)

        self._opts = _TTSOptions(
            voice=voice,
            group_id=group_id,
            api_key=api_key,
            base_url=base_url or API_BASE_URL,
            encoding=encoding,
        )
        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def list_voices(self) -> List[Voice]:
        # MiniMax might not provide a list of voices, so we'll return a default list
        return [DEFAULT_VOICE]

    def synthesize(self, text: str) -> "ChunkedStream":
        return ChunkedStream(text, self._opts, self._ensure_session())

class ChunkedStream(tts.ChunkedStream):
    def __init__(
            self, text: str, opts: _TTSOptions, session: aiohttp.ClientSession
    ) -> None:
        super().__init__()
        self._text, self._opts, self._session = text, opts, session
        self._sample_rate = TTS.MINIMAX_TTS_SAMPLE_RATE
        self._num_channels = TTS.MINIMAX_TTS_CHANNELS

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        try:
            logger.info(f"Starting _main_task with sample_rate: {self._sample_rate}, channels: {self._num_channels}")
            stream = utils.audio.AudioByteStream(
                sample_rate=self._sample_rate, num_channels=self._num_channels
            )
            logger.debug(f"AudioByteStream created with sample_rate: {self._sample_rate}, channels: {self._num_channels}")
            request_id = utils.shortuuid()
            segment_id = utils.shortuuid()

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._opts.api_key}"
            }
            json_data = {
                "text": self._text,
                "model": "speech-01",  # Adjust this based on MiniMax's model names
                "voice_id": self._opts.voice.id,
                "group_id": self._opts.group_id,
                "audio_format": self._opts.encoding.upper(),
                "sample_rate": self._sample_rate
            }

            logger.debug(f"Sending request to MiniMax API: {json_data}")

            async with self._session.post(url=self._opts.base_url, headers=headers, json=json_data) as resp:
                if resp.status != 200:
                    content = await resp.text()
                    logger.error(f"MiniMax API returned error: {resp.status} - {content}")
                    return

                audio_content = await resp.read()
                content_type = resp.headers.get('Content-Type', '')
                logger.debug(f"Received audio content of type: {content_type}, length: {len(audio_content)} bytes")

                audio_buffer = BytesIO(audio_content)

                if 'mp3' in content_type.lower():
                    mp3_decoder = utils.codecs.Mp3StreamDecoder()
                    while True:
                        chunk = audio_buffer.read(4096)  # Read in chunks
                        if not chunk:
                            break
                        for frame in mp3_decoder.decode_chunk(chunk):
                            await self._send_audio_frame(request_id, segment_id, frame)
                elif 'wav' in content_type.lower() or self._opts.encoding == "wav":
                    for frame in stream.write(audio_content):
                        await self._send_audio_frame(request_id, segment_id, frame)
                    for frame in stream.flush():
                        await self._send_audio_frame(request_id, segment_id, frame)
                else:
                    logger.error(f"Unsupported audio format: {content_type}")
                    return

            logger.info(f"Audio synthesis completed for text: {self._text[:30]}...")
        except Exception as e:
            logger.error(f"Error in _main_task: {str(e)}", exc_info=True)
            raise

    async def _send_audio_frame(self, request_id: str, segment_id: str, frame: bytes) -> None:
        try:
            self._event_ch.send_nowait(
                tts.SynthesizedAudio(
                    request_id=request_id,
                    segment_id=segment_id,
                    frame=frame,
                )
            )
        except Exception as e:
            logger.error(f"Error sending audio frame: {str(e)}", exc_info=True)

