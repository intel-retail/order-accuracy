import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException, UploadFile
from api import endpoints


class SingleUploadGuardTests(unittest.TestCase):
    def setUp(self):
        self.previous_active_id = endpoints._active_single_upload_id
        endpoints._active_single_upload_id = None

    def tearDown(self):
        endpoints._active_single_upload_id = self.previous_active_id

    @patch.object(endpoints, "get_video_history")
    def test_processing_history_blocks_upload_after_slot_reset(self, history):
        history.return_value = [{"video_id": "first", "status": "processing"}]

        self.assertFalse(endpoints._claim_single_upload("second"))
        self.assertIsNone(endpoints._active_single_upload_id)

    @patch.object(endpoints, "get_video_history")
    def test_active_slot_blocks_upload(self, history):
        history.return_value = []

        self.assertTrue(endpoints._claim_single_upload("first"))
        self.assertFalse(endpoints._claim_single_upload("second"))
        endpoints._release_single_upload("first")
        self.assertTrue(endpoints._claim_single_upload("second"))

    @patch.object(endpoints, "get_video_history")
    @patch.object(endpoints, "run_pipeline_async")
    def test_upload_endpoint_rejects_processing_video(self, pipeline, history):
        history.return_value = [{"video_id": "first", "status": "processing"}]
        upload = next(route.endpoint for route in endpoints.create_app().routes
                      if route.path == "/upload-video")

        with tempfile.TemporaryDirectory() as uploads_dir, patch.object(
            endpoints, "UPLOADS_DIR", Path(uploads_dir)
        ):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(upload(
                    file=UploadFile(filename="test.mp4", file=io.BytesIO(b"test")),
                    video_id="second", status_managed_by_ui=False,
                ))
            self.assertEqual(raised.exception.status_code, 409)
            self.assertEqual(list(Path(uploads_dir).iterdir()), [])
        pipeline.assert_not_called()


if __name__ == "__main__":
    unittest.main()