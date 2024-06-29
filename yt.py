import re
from youtube_transcript_api import YouTubeTranscriptApi

def extract_video_id(youtube_url):
    """
    Extracts the video ID from a YouTube URL.
    
    Args:
    youtube_url (str): The URL of the YouTube video.
    
    Returns:
    str: The extracted video ID or an error message if the URL is invalid.
    """
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, youtube_url)
    if match:
        return match.group(1)
    else:
        return "Invalid YouTube URL"

def get_youtube_transcript(video_id):
    """
    Fetches the transcript of the given YouTube video ID and returns it as a formatted string.
    """
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = "\n".join([entry['text'] for entry in transcript])
        return transcript_text
    except Exception as e:
        return str(e)

def save_transcript_to_file(transcript_text, filename):
    """
    Saves the transcript text to a file.
    
    Args:
    transcript_text (str): The transcript text.
    filename (str): The name of the file to save the transcript.
    """
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(transcript_text)



import tiktoken

import tiktoken

def count_characters_and_convert_to_tokens(file_path: str) -> int:
    # Count characters in the file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        char_count = len(text)

    # Initialize the tokenizer
    enc = tiktoken.get_encoding("cl100k_base")  # Using a known encoding

    # Convert the text into tokens
    tokens = enc.encode(text)
    token_count = len(tokens)

    return token_count






# Example usage
def youtube_api(extract_video_id, get_youtube_transcript, save_transcript_to_file):
    youtube_url = 'https://www.youtube.com/watch?v=W7cimx9zhsM'  # Replace with your YouTube video URL
    video_id = extract_video_id(youtube_url)
    if video_id != "Invalid YouTube URL":
        transcript_text = get_youtube_transcript(video_id)
        
        if "Could not retrieve a transcript" not in transcript_text:
            save_transcript_to_file(transcript_text, 'transcript.txt')
            print("Transcript saved to transcript.txt")
        else:
            print(transcript_text)
    else:
        print("Invalid YouTube URL")

if __name__ == "__main__":
    # Example usage
    file_path = 'transcript.txt'
    tokens = count_characters_and_convert_to_tokens(file_path)
    print(f'Token count: {tokens}')
   # youtube_api(extract_video_id, get_youtube_transcript, save_transcript_to_file)
