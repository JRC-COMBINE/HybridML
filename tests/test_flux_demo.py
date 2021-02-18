import os
import sys
import unittest

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.split(os.path.dirname(__file__))[0])

import test_utility  # noqa: E402

flux_path = os.path.join(os.getcwd(), "demo", "flux")
sys.path.append(flux_path)

from flux_demo import main  # noqa: E402
from flux_demo import make_config as make_flux_config  # noqa: E402
from structured_flux_demo import make_config as make_structured_flux_config  # noqa: E402


class test_flux_demo(test_utility.TestCaseTimer):
    def test_flux_no_exception(self):
        kwargs = make_flux_config()
        self.execute_main(kwargs, "Flux Model")

    def test_structured_flux_no_exception(self):
        kwargs = make_structured_flux_config(flux_path)
        self.execute_main(kwargs, "Structured Flux Model")

    def execute_main(self, kwargs, model_name):
        print("Executing: ", model_name)
        kwargs["train_epochs"] = 10
        kwargs["plot"] = False
        main(**kwargs)


if __name__ == "__main__":
    # t = test_flux_demo()
    # t.test_flux()
    # t.test_structured_flux()
    unittest.main()
