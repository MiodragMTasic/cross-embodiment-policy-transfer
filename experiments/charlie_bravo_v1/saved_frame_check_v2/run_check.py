from pathlib import Path
import importlib.util
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('screen',HERE.parent/'saved_frame_check/check_saved_frames.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.HERE=HERE
m.CAP=1.0
m.run()
