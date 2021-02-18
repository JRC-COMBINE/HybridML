import os
import sys
import time
import unittest

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.split(os.path.dirname(__file__))[0])

import test_utility  # noqa: E402

loader = unittest.TestLoader()
tests = loader.discover(".")
if tests.countTestCases() == 0:
    tests = loader.discover(os.path.dirname(__file__))
testRunner = unittest.runner.TextTestRunner()
start = time.time()
testRunner.run(tests)
end = time.time()
print(f"Running the tests in {end-start:.4f}s.")

sorted_tests = test_utility.profiling_list.copy()
sorted_tests.sort(key=lambda t: t[1], reverse=True)

for test, duration in sorted_tests:
    try:
        case = str(test).split(".")[-1].split(" ")[0]
        method = str(test).split("=")[1][:-2]
        print(f"{duration:1.3f}s\t{case}.{method}")
    except:  # noqa E722
        print(f"{duration:1.3f}s\t{test}")
