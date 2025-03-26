import sys
import re
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter
from transformers import pipeline


def main(video_id):
    print(f"Fetching transcript for video: {video_id}")
    transcript = fetch_transcript(video_id)

    if not transcript:
        print("Failed to retrieve transcript.")
        return

    print("Preprocessing transcript data...")
    processed_transcript = preprocess_transcript(transcript)

    if not processed_transcript:
        print("Failed to preprocess transcript.")
        return

    print("Generating abstractive summary...")
    summary = generate_summary(processed_transcript)

    if not summary:
        print("Failed to generate summary.")
        return

    print("\nAbstractive Summary:")
    print(summary)


def fetch_transcript(video_id):
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id)

        formatter = JSONFormatter()
        return formatter.format_transcript(transcript)

    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None


def clean_text(text):
    # Remove timestamps and unnecessary characters
    text = re.sub(r"\[\d+:\d+\]|\(\d+:\d+\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_transcript(transcript_data):
    try:
        print(f"Type of transcript_data: {type(transcript_data)}")
        print(f"Transcript data: {transcript_data}")

        if isinstance(transcript_data, str):
            import json

            try:
                transcript_data = json.loads(transcript_data)
            except json.JSONDecodeError:
                print("Error: transcript_data is not valid JSON.")
                return None

        if not isinstance(transcript_data, list):
            print(
                "Error: Expected a list of transcript entries, but got:",
                type(transcript_data),
            )
            return None

        cleaned_transcript = [clean_text(entry["text"]) for entry in transcript_data]

        return " ".join(cleaned_transcript)

    except Exception as e:
        print(f"Error processing transcript_data: {e}")
        return None


def generate_summary(text: str) -> str:
    # initialize the summarization pipeline from Hugging Face
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    max_chunk_size = 1000
    chunks = [text[i : i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]

    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
        summaries.append(summary[0]["summary_text"])

    return " ".join(summaries)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <youtube_video_id>")
    else:
        video_id = sys.argv[1]
        main(video_id)
