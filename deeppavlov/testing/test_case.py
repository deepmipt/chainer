from unittest import TestCase

import os
import shutil
import logging


class DPTestCase(TestCase):

    def setUp(self):
        logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                            level=logging.DEBUG)
        self.TEST_DIR = "/tmp/deeppavlov"
        os.makedirs(self.TEST_DIR, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.TEST_DIR)
