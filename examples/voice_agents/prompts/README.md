# Multi-Role Voice Agent Prompts

This folder contains specialized prompts for different voice agent roles that can be used with the LiveKit Agents framework.

## Available Roles

### 1. Hotel Receptionist (`hotel_receptionist.txt`)
- **Character**: Sarah, a professional hotel receptionist at Grand Plaza Hotel
- **Capabilities**: Check-ins, check-outs, reservations, guest services, problem resolution
- **Personality**: Warm, friendly, professional, patient with travelers
- **Best for**: Hospitality training, customer service scenarios, hotel management simulations

### 2. AI Recruiter (`ai_recruiter.txt`)
- **Character**: Alex, a senior AI recruiter at TechCorp Innovations
- **Capabilities**: Technical screening, candidate assessment, cultural fit evaluation
- **Personality**: Professional, confident, thorough, constructive
- **Best for**: Interview practice, recruitment training, technical assessment scenarios

### 3. F1 Visa Interviewer (`f1_visa_interviewer.txt`)
- **Character**: Officer Michael Chen, a senior consular officer
- **Capabilities**: Visa application verification, academic intent assessment, financial evaluation
- **Personality**: Professional, authoritative, impartial, thorough
- **Best for**: Visa interview practice, immigration training, academic counseling

## How to Use

### Option 1: Use the Multi-Role Agent (Recommended)

1. Set the environment variable to choose your role:
   ```bash
   # For Hotel Receptionist
   export AGENT_ROLE=hotel_receptionist
   
   # For AI Recruiter
   export AGENT_ROLE=ai_recruiter
   
   # For F1 Visa Interviewer
   export AGENT_ROLE=f1_visa_interviewer
   ```

2. Run the multi-role agent:
   ```bash
   python multi_role_agent.py
   ```

### Option 2: Modify the Basic Agent

1. Copy the content from your chosen prompt file
2. Replace the instructions in `basic_agent.py`:
   ```python
   super().__init__(
       instructions="[PASTE YOUR PROMPT HERE]"
   )
   ```

3. Run the basic agent:
   ```bash
   python basic_agent.py
   ```

## Customization

You can customize any prompt by editing the corresponding `.txt` file. Each prompt includes:

- **Character definition** with background and personality
- **Communication style** guidelines
- **Core responsibilities** and capabilities
- **Specific guidelines** for interactions
- **Conversation flow** or interview structure

## Tips for Best Results

1. **Environment Setup**: Ensure your `.env` file has the necessary API keys
2. **Voice Selection**: Consider changing the TTS voice to match the character's personality
3. **Testing**: Test each role with appropriate scenarios to ensure realistic responses
4. **Customization**: Modify prompts to match your specific use case or requirements

## Adding New Roles

To add a new role:

1. Create a new `.txt` file in this folder (e.g., `customer_service.txt`)
2. Follow the same structure as existing prompts
3. Add the role name to the `valid_roles` list in `multi_role_agent.py`
4. Add appropriate greeting logic in the `on_enter` method

## Example Usage Scenarios

### Hotel Receptionist
- "I'd like to check in for my reservation"
- "What amenities do you have available?"
- "I need to extend my stay by one night"
- "There's an issue with my room"

### AI Recruiter
- "Tell me about your experience with Python"
- "How do you handle difficult team situations?"
- "What are your career goals?"
- "Describe a challenging project you worked on"

### F1 Visa Interviewer
- "What is your purpose of visit to the United States?"
- "How will you fund your studies?"
- "What are your plans after graduation?"
- "Tell me about your family and ties to your home country" 