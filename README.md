# Command-line client for OpenAI API
## Installation
```bash
git clone https://github.com/akademi4eg/aka-assistant.git
cd aka-assistant
pip3 install -r requirements.txt
```

## Usage
```bash
OPENAI_API_KEY="YOUR_API_KEY_HERE" python3 main.py
```
You would see a `[gpt-3.5-turbo T0]>>` prompt. The `T0` part would be updated after every response to indicate number of tokens used in this session.

### Multiline input
Pressing enter sends a command to OpenAI. If you want to enter multiple lines, use `ESC-Enter` to add a new line.

### Commands

* /save NAME -- saves conversation
* /load NAME -- loads conversation
* /clear -- clears conversation state
* /drop -- removes last request-response from conversation state
* /image -- generates an image from a prompts and displays it (tested only on Mac). Images are saved in `./images`
* /asr -- records your voice until you press Enter, afterwards sends results to chatgpt. Note: there may be problems if you disconnect your headset while CLI tool was already running. Audios and Whisper transcripts are stored in `./audios`
* /doc URL -- downloads and summarizes a pdf document
