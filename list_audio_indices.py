import sounddevice
import pyaudio
import subprocess
p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')


print("_________INPUT_INDICES_________")
for i in range(0, numdevices):
    if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
print("________OUTPUT_INDICES_________")
output = subprocess.run(["python", '-m', 'sounddevice'], capture_output=True, text=True)

print(output.stdout)
