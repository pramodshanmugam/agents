import logging
from pathlib import Path
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os

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
from livekit.agents.llm import llm
# uncomment to enable Krisp background voice/noise cancellation
# from livekit.plugins import noise_cancellation

logger = logging.getLogger("basic-agent")

load_dotenv()

# =============================================================================
# CHANGE THIS LINE TO SWITCH BETWEEN ROLES:
# Available options: "hotel_receptionist", "ai_recruiter", "f1_visa_interviewer", "art_museum"
# =============================================================================
SELECTED_ROLE = "art_museum"
# =============================================================================

# Define missing variables
WEBSOCKET_URL = "ws://localhost:8080"  # Default WebSocket URL
F1_INTERVIEW_AGENT = "art_museum"  # Agent type for F1 interview

# Simple room manager for cleanup
class RoomManager:
    async def cleanup_room(self, room_name: str):
        logger.info(f"Cleaning up room: {room_name}")
        # Add any cleanup logic here

room_manager = RoomManager()

async def my_shutdown_hook():
    """Shutdown hook for cleanup operations"""
    logger.info("Executing shutdown hook")
    # Add any cleanup operations here

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
    def __init__(self, server_url="https://5fhh2h7n-8088.use.devtunnels.ms/", session_id="visa_interview", backend_id="visa_agent"):
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
        
        # Load knowledge base if it's the art museum role
        self.knowledge_base = None
        if role == "art_museum":
            self.knowledge_base = self._load_knowledge_base()
        
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

    def _load_knowledge_base(self) -> dict:
        """Load the MAP Bengaluru knowledge base JSON file."""
        try:
            knowledge_base_path = Path(__file__).parent / "prompts" / "map_bengaluru_knowledgebase_v1.json"
            with open(knowledge_base_path, 'r', encoding='utf-8') as f:
                knowledge_base = json.load(f)
            logger.info("Successfully loaded MAP Bengaluru knowledge base")
            return knowledge_base
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            return {}

    async def on_enter(self):
        # Provide a simple greeting based on the role
        if self.role == "hotel_receptionist":
            greeting = "Hello! Welcome to the hotel Bengaluru. I'm here to help you with information about our hotel. How can I assist you today?"
        else:
            # For other roles, use the default behavior
            self.session.generate_reply()
            return
        
        # Send the greeting directly
        await self.session.say(greeting)

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

    @function_tool
    async def get_museum_info(
        self, context: RunContext, info_type: str = "general"
    ):
        """Get information about MAP Bengaluru museum.
        
        Args:
            info_type: Type of information requested (general, hours, exhibitions, events, accessibility, contact, etc.)
        """
        if not self.knowledge_base:
            return "I don't have access to the museum knowledge base."
        
        try:
            if info_type == "general":
                museum_info = self.knowledge_base.get("core", {}).get("museum", {})
                return f"MAP Bengaluru is located at {museum_info.get('address', 'Kasturba Road Cross, Bengaluru')}. It's a private art museum that opened in February 2023."
            
            elif info_type == "hours":
                hours = self.knowledge_base.get("core", {}).get("museum", {}).get("hours", {})
                return f"Museum hours: Tuesday-Sunday 10:00-18:30 (Tuesday 14:00-18:30 is free entry), Friday-Saturday until 19:30. Closed Mondays."
            
            elif info_type == "exhibitions":
                exhibitions = self.knowledge_base.get("programming", {}).get("exhibitions", {}).get("current", [])
                if exhibitions:
                    exhibition_list = []
                    for exhibition in exhibitions:
                        title = exhibition.get("title", "")
                        dates = exhibition.get("dates", {})
                        summary = exhibition.get("summary", "")
                        exhibition_list.append(f"• {title} ({dates.get('start', '')} to {dates.get('end', '')}): {summary}")
                    return "Current exhibitions:\n" + "\n".join(exhibition_list)
                return "No current exhibitions information available."
            
            elif info_type == "events":
                events = self.knowledge_base.get("programming", {}).get("events", {}).get("upcoming_sample_oct_2025", [])
                if events:
                    event_list = []
                    for event in events[:5]:  # Show first 5 events
                        title = event.get("title", "")
                        date = event.get("date", "")
                        event_type = event.get("type", "")
                        event_list.append(f"• {title} ({date}) - {event_type}")
                    return "Upcoming events:\n" + "\n".join(event_list)
                return "No upcoming events information available."
            
            elif info_type == "accessibility":
                accessibility = self.knowledge_base.get("core", {}).get("museum", {}).get("accessibility", {})
                features = []
                if accessibility.get("wheelchair_access"): features.append("wheelchair accessible")
                if accessibility.get("elevator"): features.append("elevator access")
                if accessibility.get("tactile_and_braille"): features.append("tactile and braille resources")
                if accessibility.get("isl_and_transcripts"): features.append("ISL and transcripts")
                if accessibility.get("service_animals_allowed"): features.append("service animals allowed")
                return f"Accessibility features: {', '.join(features)}. Contact {accessibility.get('contact_for_support', 'hello@map-india.org')} for support."
            
            elif info_type == "contact":
                contacts = self.knowledge_base.get("core", {}).get("museum", {}).get("contacts", {})
                return f"Contact information: General inquiries - {contacts.get('general_email', 'hello@map-india.org')}, Education - {contacts.get('education_email', 'education@map-india.org')}, Venue hire - {contacts.get('venue_hire', 'bookings@map-india.org')}"
            
            else:
                return "I can provide information about: general museum info, hours, exhibitions, events, accessibility, and contact details. Please specify what you'd like to know."
                
        except Exception as e:
            logger.error(f"Error accessing museum information: {e}")
            return "I'm sorry, I encountered an error accessing the museum information."

    @function_tool
    async def search_museum_knowledge(
        self, context: RunContext, query: str
    ):
        """Search the MAP Bengaluru knowledge base for specific information.
        
        Args:
            query: The search query or question about the museum
        """
        if not self.knowledge_base:
            return "I don't have access to the museum knowledge base."
        
        try:
            # Simple keyword-based search in the knowledge base
            query_lower = query.lower()
            
            # Check FAQs first
            faqs = self.knowledge_base.get("faqs", [])
            for faq in faqs:
                if any(keyword in faq.get("q", "").lower() for keyword in query_lower.split()):
                    return f"Q: {faq.get('q', '')}\nA: {faq.get('a', '')}"
            
            # Check core museum information
            museum_info = self.knowledge_base.get("core", {}).get("museum", {})
            
            if "hours" in query_lower or "time" in query_lower or "open" in query_lower:
                hours = museum_info.get("hours", {})
                return f"Museum hours: Tuesday-Sunday 10:00-18:30 (Tuesday 14:00-18:30 is free entry), Friday-Saturday until 19:30. Closed Mondays."
            
            if "address" in query_lower or "location" in query_lower or "where" in query_lower:
                address = museum_info.get("address", "")
                getting_there = museum_info.get("getting_there", {})
                metro_stations = getting_there.get("nearest_metro", [])
                return f"MAP is located at {address}. Nearest metro stations: {', '.join(metro_stations)}."
            
            if "founder" in query_lower or "director" in query_lower or "leadership" in query_lower:
                leadership = museum_info.get("leadership", {})
                return f"Founder: {leadership.get('founder', 'Abhishek Poddar')}. Acting Director: {leadership.get('acting_director', 'Harish Vasudevan')}."
            
            if "exhibition" in query_lower or "display" in query_lower or "show" in query_lower:
                exhibitions = self.knowledge_base.get("programming", {}).get("exhibitions", {}).get("current", [])
                if exhibitions:
                    exhibition_list = []
                    for exhibition in exhibitions:
                        title = exhibition.get("title", "")
                        dates = exhibition.get("dates", {})
                        summary = exhibition.get("summary", "")
                        exhibition_list.append(f"• {title} ({dates.get('start', '')} to {dates.get('end', '')}): {summary}")
                    return "Current exhibitions:\n" + "\n".join(exhibition_list)
                return "No current exhibitions information available."
            
            # Default response
            return "I found some information but couldn't match your specific query. You can ask about museum hours, location, exhibitions, events, accessibility, or contact information."
            
        except Exception as e:
            logger.error(f"Error searching museum knowledge: {e}")
            return "I'm sorry, I encountered an error searching the museum information."

    @function_tool
    async def send_reservation_email(
        self, 
        context: RunContext, 
        recipient_email: str, 
        customer_name: str,
        reservation_date: str,
        reservation_time: str,
        number_of_guests: str = "2",
        special_requests: str = "None"
    ):
        """Sends a reservation confirmation email to the customer.
        
        Args:
            recipient_email: The customer's email address
            customer_name: The customer's name
            reservation_date: The date of the reservation
            reservation_time: The time of the reservation
            number_of_guests: Number of guests (default: 2)
            special_requests: Any special requests (default: None)
        """
        try:
            # Create email content
            subject = f"Reservation Confirmation - {customer_name}"
            
            body = f"""
Dear {customer_name},

Thank you for your reservation with us!

Reservation Details:
- Date: {reservation_date}
- Time: {reservation_time}
- Number of Guests: {number_of_guests}
- Special Requests: {special_requests}

Your reservation has been confirmed. We look forward to serving you!

If you need to make any changes to your reservation, please contact us at least 24 hours in advance.

Best regards,
The Restaurant Team
            """
            
            msg = MIMEMultipart()
            msg['From'] = os.getenv("EMAIL_SENDER_ADDRESS")
            msg['To'] = recipient_email
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))

            smtp_server = os.getenv("SMTP_SERVER")
            smtp_port = int(os.getenv("SMTP_PORT", 587))
            smtp_username = os.getenv("SMTP_USERNAME")
            smtp_password = os.getenv("SMTP_PASSWORD")

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_username, smtp_password)
                server.send_message(msg)
                
            logger.info(f"Reservation confirmation email sent to {recipient_email}")
            return f"Reservation confirmation email sent successfully to {recipient_email}"
            
        except Exception as e:
            logger.error(f"Error sending reservation email: {e}")
            return f"Failed to send reservation email: {str(e)}"


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):

    # Try to connect to AvaTalk, but don't fail if it's unavailable
    avatalk_client = None
   
    session_id = "visa_interview"  # <-- Hardcoded session ID
    avatalk_client = AvaTalkClient(server_url="https://5fhh2h7n-8088.use.devtunnels.ms/", session_id=session_id)
    avatalk_client.connect()
    logger.info("Successfully connected to AvaTalk TTS service")

    # Use the hardcoded role
    role = SELECTED_ROLE
    
    # Validate role
    valid_roles = ["hotel_receptionist", "ai_recruiter", "f1_visa_interviewer", "art_museum"]
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
