import unittest
from pathlib import Path
from unittest.mock import Mock
from models.svc.vevo2.cli import build_parser, run_task

class RoutingTests(unittest.TestCase):
    def test_all_tasks(self):
        parser = build_parser()
        for task in ("convert", "tts", "edit", "style", "melody"):
            argv = [task, "--input", "a.wav", "--output", "out.wav", "--steps", "12"]
            if task in ("convert", "style", "melody"):
                argv += ["--reference", "b.wav"]
            if task != "convert":
                argv += ["--text", "hello"]
            args = parser.parse_args(argv)
            pipe = Mock()
            run_task(pipe, args)
            if task == "convert":
                pipe.inference_fm.assert_called_once_with(
                    src_wav_path="a.wav", timbre_ref_wav_path="b.wav",
                    use_pitch_shift=True, flow_matching_steps=12)
                pipe.inference_ar_and_fm.assert_not_called()
            else:
                pipe.inference_ar_and_fm.assert_called_once()
                kw = pipe.inference_ar_and_fm.call_args.kwargs
                self.assertEqual(kw["use_prosody_code"], task != "tts")
                self.assertEqual(kw["flow_matching_steps"], 12)
                self.assertEqual(kw["style_ref_wav_path"], "b.wav" if task in ("style", "melody") else "a.wav")
                self.assertEqual(kw["timbre_ref_wav_path"], "b.wav" if task == "melody" else "a.wav")

if __name__ == "__main__":
    unittest.main()
