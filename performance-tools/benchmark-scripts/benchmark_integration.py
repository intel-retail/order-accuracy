'''
* Copyright (C) 2024 Intel Corporation.
*
* SPDX-License-Identifier: Apache-2.0
'''

import unittest
import benchmark
import os


class Integration(unittest.TestCase):

    def test_convert_csv_results_to_json(self):
        csv_dir = './test_src'
        found_json = False
        benchmark.convert_csv_results_to_json(csv_dir, 'test')
        for entry in os.scandir(csv_dir):
            if entry.is_file() and entry.name == 'test.json':
                found_json = True
                os.remove(os.path.join(csv_dir, 'test.json'))
        self.assertEqual(found_json, True, "json output successful")


if __name__ == '__main__':
    unittest.main()
