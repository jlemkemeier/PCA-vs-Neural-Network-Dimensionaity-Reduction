from pydub import AudioSegment

for x in range(1, 101):
	try:
		sound = AudioSegment.from_mp3("/Users/jlemkemeier/Desktop/Originals/"+str(x) + ".mp3")
	except:
		sound = AudioSegment.from_mp3("/Users/jlemkemeier/Desktop/Originals/"+str(x) + ".m4a")

	print(x)
	halfway_point = len(sound)/2
	five_seconds = sound[halfway_point: halfway_point+250]
	five_seconds = five_seconds.set_frame_rate(8000)
	five_seconds.export("/Users/jlemkemeier/Desktop/Cuts/"+str(x) + "c.wav", format="wav", parameters = ["-ac", "1"])
