from sister.sister import Camera
import pprint

camera = Camera('../../data/cameras/usb_camera.xml')
pprint.pprint(camera.__dict__)