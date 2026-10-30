import base64,io,json
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
from vision_images import encode_views

HERE=Path(__file__).resolve().parents[1]

class VisionImagesTests(unittest.TestCase):
    @unittest.skipUnless((HERE/'saved_frame_check_v3/l4003_t1200/1200_exterior_640.png').is_file(), 'licensed saved-frame fixture is outside the code-only export')
    def test_encoded_images_match_selected_saved_frame_recipe(self):
        case=HERE/'saved_frame_check_v3/l4003_t1200'
        ext=np.asarray(Image.open(case/'1200_exterior_640.png').convert('RGB'))
        wrist=np.asarray(Image.open(case/'1200_wrist_640.png').convert('RGB'))
        got=encode_views(ext,wrist)
        body=json.loads((case/'640x360/crop_request.json').read_text())
        expected=[x for x in body['messages'][1]['content'] if x['type']=='image_url'][-2:]
        for actual,item in zip(got,expected):
            reference=base64.b64decode(item['image_url']['url'].split(',')[1])
            a=Image.open(io.BytesIO(actual)); b=Image.open(io.BytesIO(reference))
            self.assertEqual(a.size,(640,360))
            np.testing.assert_array_equal(np.asarray(a),np.asarray(b))

    def test_invalid_arrays_rejected(self):
        for a in [np.zeros((360,640,3),dtype=float),np.zeros((360,640),dtype='uint8')]:
            with self.assertRaises(ValueError):encode_views(a,a)

if __name__=='__main__':unittest.main()
