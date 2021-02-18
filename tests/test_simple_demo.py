import os
import sys
import unittest

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.split(os.path.dirname(__file__))[0])

import test_utility  # noqa: E402

simple_demo_path = os.path.join(os.getcwd(), "demo")
sys.path.append(simple_demo_path)

from simple_demo import main  # noqa: E402


class test_simple_demo(test_utility.TestCaseTimer):
    def test_simple_demo(self):
        main()


if __name__ == "__main__":
    t = test_simple_demo()
    t.test_simple_demo()
    unittest.main()
