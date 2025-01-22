### Ignore command recognition during TTS execution ###

import serial
import openai
import pyaudio
import wave
import os
from datetime import datetime
from gtts import gTTS
from playsound import playsound
import speech_recognition as sr
import threading  # Parallel execution
import requests   # Fetching data

# OpenAI API key setup
openai.api_key = ''   # Substitute a private API key
api_key = 'a654363fc0bc630e21c2f31424130db7'
location = "Siheung-si"  # Location for weather information
weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric&lang=kr"

# System prompt setup
system_prompt = """
You are a friendly and helpful assistant. When responding to users, make sure to sound polite, kind, and encouraging. If you're unable to provide certain information, politely explain it and suggest alternatives where possible. Your answers should be clear, helpful, and should make users feel comfortable.
"""

# Initialization
history = []
arduino = serial.Serial('COM5', 9600, timeout=1)  # Arduino serial port

# Audio recording function
def record_audio(file_path, record_seconds=5):
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    rate = 44100
    p = pyaudio.PyAudio()
    print("녹음 중...")  # Recording...
    stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
    frames = []
    for _ in range(0, int(rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)
    print("녹음 완료.")  # Recording complete.
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

# Get GPT response
def get_gpt_response(transcript, history):
    system_message = {"role": "system", "content": system_prompt.strip()}
    messages = [system_message] + history + [{"role": "user", "content": transcript}]
    response = openai.ChatCompletion.create(model="gpt-4o", messages=messages)
    gpt_response = response.choices[0].message.content
    print("GPT Response:", gpt_response)
    return gpt_response

tts_lock = threading.Lock()   # Lock object for synchronization
is_recognizing = True         # Flag for voice recognition status
def play_gpt_response_with_tts(gpt_response):
    global is_recognizing
    with tts_lock:  # Synchronize file saving and deletion
        print("TTS Response:", gpt_response)
        tts = gTTS(text=gpt_response, lang='ko')
        # Generate unique filename based on current time
        speech_file_path = f"speech_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.mp3"
        try:
            is_recognizing = False  # Disable voice recognition
            tts.save(speech_file_path)
            playsound(speech_file_path)
        finally:
            # Delete the file after playback
            if os.path.exists(speech_file_path):
                os.remove(speech_file_path)
            is_recognizing = True  # Re-enable voice recognition

# Fetch weather information (using OpenWeather API)
def get_weather():
    response = requests.get(weather_url)
    data = response.json()
    
    if response.status_code == 200:
        main_data = data['main']
        weather_data = data['weather'][0]
        temperature = main_data['temp']
        description = weather_data['description']
        humidity = main_data['humidity']
        wind_speed = data['wind']['speed']
        
        weather_info = f"{location}의 현재 기온은 {temperature}°C이고, 날씨는 {description}입니다. 습도는 {humidity}%, 바람 속도는 {wind_speed}m/s입니다."
        return weather_info
    else:
        return "날씨 정보를 가져오는 데 실패했습니다."

# Get current time
def get_current_time():
    now = datetime.now()
    return now.strftime("%Y년 %m월 %d일 %H시 %M분입니다.")

# Process commands
def process_command(command):
    if "불" in command:
        if "켜" in command:
            arduino.write(b'1')  # Turn on LED
            print("LED 켜기 명령 실행")  # Executed: Turn on LED
            play_gpt_response_with_tts("불을 켰습니다.")
        elif "꺼" in command:
            arduino.write(b'0')  # Turn off LED
            print("LED 끄기 명령 실행")  # Executed: Turn off LED
            play_gpt_response_with_tts("불을 껐습니다.")
    elif "몇 시" in command:
        current_time = get_current_time()
        print(f"현재 시간: {current_time}")  # Current Time
        play_gpt_response_with_tts(f"현재 시간은 {current_time}")
    elif "날씨" in command:
        weather_response = get_weather()  # Fetch weather information
        play_gpt_response_with_tts(weather_response)
    else:
        gpt_response = get_gpt_response(command, history[-10:])
        play_gpt_response_with_tts(gpt_response)

# Monitor Arduino switch state
is_weather_shown = False  # Track if weather information is displayed
def listen_for_switch_in_thread():
    global is_weather_shown, is_recognizing
    while True:
        if arduino.in_waiting > 0:
            arduino_data = arduino.readline().decode('utf-8').strip()
            print(f"Arduino Data: {arduino_data}")
            if arduino_data == "2" and not is_weather_shown:  # Switch ON and weather not yet displayed
                print("스위치 ON 감지. 실시간 날씨 정보를 가져옵니다.")  # Switch ON detected. Fetching real-time weather.
                weather_response = get_weather()
                is_recognizing = False
                play_gpt_response_with_tts(weather_response)
                is_weather_shown = True  # Set weather displayed
                is_recognizing = True
            elif arduino_data != "2":
                is_weather_shown = False  # Reset weather display state if switch is not "2"

# Detect trigger word and process commands
def listen_and_execute():
    global is_recognizing
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    while True:
        if is_recognizing:  # Execute only when voice recognition is enabled
            print("대기 중... '지니야'와 명령을 말씀하세요.")  # Waiting... Please say "Hey Genie" followed by your command.
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)

            try:
                transcript = recognizer.recognize_google(audio, language='ko-KR')
                print(f"사용자: {transcript}")  # User

                # Check for trigger word "지니야"
                if "지니야" in transcript:
                    # Extract command by removing the trigger word
                    command = transcript.replace("지니야", "").strip()
                    print(f"추출된 명령: {command}")  # Extracted Command

                    # Process the command
                    if command:
                        # Check for termination command
                        if "종료" in command or "끝내" in command:
                            play_gpt_response_with_tts("프로그램을 종료합니다. 좋은 하루 보내세요!")  # Program termination message
                            os._exit(0)
                        else:
                            # Execute other commands
                            process_command(command)
                    else:
                        play_gpt_response_with_tts("네, 무엇을 도와드릴까요?")  # Yes, how can I assist you?
            except Exception as e:
                print("")

# Main execution
if __name__ == "__main__":
    print("프로그램 시작!")  # Program started!

    # Create threads
    switch_thread = threading.Thread(target=listen_for_switch_in_thread, daemon=True)
    voice_thread = threading.Thread(target=listen_and_execute, daemon=True)

    # Start threads
    switch_thread.start()
    voice_thread.start()

    # Keep main thread running
    switch_thread.join()
    voice_thread.join()
