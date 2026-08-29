import tempfile
import unittest
from pathlib import Path

from PIL import Image

from process_people_dataset import Detection, preserve_original, save_crops, upper_body_square


class PeopleDatasetTests(unittest.TestCase):
    def detection(self):
        points = [[0.0, 0.0] for _ in range(17)]
        scores = [0.0 for _ in range(17)]
        points[0], points[5], points[6] = [50.0, 30.0], [35.0, 60.0], [65.0, 60.0]
        points[11], points[12] = [40.0, 115.0], [60.0, 115.0]
        for index in (0, 5, 6, 11, 12):
            scores[index] = 0.9
        return Detection(0.91, [20.0, 10.0, 80.0, 180.0], points, scores)

    def test_upper_body_box_is_square_and_in_bounds(self):
        box = upper_body_square(self.detection(), 100, 200)
        self.assertEqual(box[2] - box[0], box[3] - box[1])
        self.assertGreaterEqual(min(box), 0)
        self.assertLessEqual(box[2], 100)
        self.assertLessEqual(box[3], 200)

    def test_crop_naming_and_dimensions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, crop_dir = root / "historic.photo.jpg", root / "crops"
            crop_dir.mkdir()
            Image.new("RGB", (100, 200), "gray").save(source)
            outputs = save_crops(source, [self.detection()], crop_dir, 256, False)
            self.assertEqual(outputs[0].name, "historic.photo_person_01.jpg")
            with Image.open(outputs[0]) as image:
                self.assertEqual(image.size, (256, 256))

    def test_original_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination = root / "source.jpg", root / "original" / "source.jpg"
            source.write_bytes(b"first")
            preserve_original(source, destination)
            source.write_bytes(b"different")
            with self.assertRaises(RuntimeError):
                preserve_original(source, destination)
            self.assertEqual(destination.read_bytes(), b"first")


if __name__ == "__main__":
    unittest.main()
