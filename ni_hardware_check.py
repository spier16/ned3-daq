import nidaqmx
system = nidaqmx.system.System.local()
for device in system.devices:
    print(device.name, [ch.name for ch in device.ai_physical_chans])