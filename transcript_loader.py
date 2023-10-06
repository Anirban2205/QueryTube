from pytube import extract
from youtube_transcript_api import YouTubeTranscriptApi

def aquire_transcript(code):
  code = extract.video_id(code)
  str = YouTubeTranscriptApi.get_transcript(code)
  s = " "
  for d in str:
    s = s + d['text'] + " "
  return s

if __name__ == "__main__":
    link = input("Enter the link: ")
    text = aquire_transcript(link)
    print(text)