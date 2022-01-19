import os
import unittest
import sys

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.split(os.path.dirname(__file__))[0])

import test_utility  # noqa: E402

theoph_path = os.path.join(os.getcwd(), "demo", "theoph")
sys.path.append(theoph_path)

from theoph_demo import main  # noqa: E402


class test_theoph_demo(test_utility.TestCase):
    def test_theoph_no_exceptions(self):
        main(plot=False)


if __name__ == "__main__":
    t = test_theoph_demo()
    t.test_theoph_no_exceptions()
    unittest.main()
