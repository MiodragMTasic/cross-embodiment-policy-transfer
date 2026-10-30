"""Fixed visual-only preprocessing selected on saved frames; no object coordinates."""
import io
from PIL import Image, ImageDraw

EXTERIOR_CROP = (160, 120, 560, 345)
OUTPUT_SIZE = (640, 360)

def encode_views(exterior_rgb, wrist_rgb):
    encoded=[]
    for index, rgb in enumerate((exterior_rgb, wrist_rgb)):
        if getattr(rgb, 'dtype', None) is None or str(rgb.dtype) != 'uint8' or len(rgb.shape)!=3 or rgb.shape[2]!=3:
            raise ValueError('expected actual uint8 RGB image')
        view=Image.fromarray(rgb).resize(OUTPUT_SIZE, Image.Resampling.LANCZOS)
        # Match development video preprocessing; mask labels even though live images have none.
        ImageDraw.Draw(view).rectangle((0,0,639,30),fill='black')
        if index==0:
            view=view.crop(EXTERIOR_CROP).resize(OUTPUT_SIZE,Image.Resampling.LANCZOS)
        out=io.BytesIO();view.save(out,format='PNG');encoded.append(out.getvalue())
    return tuple(encoded)
