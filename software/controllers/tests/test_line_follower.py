import pytest

from controllers import line_follower


@pytest.fixture
def defaultArgs():
    class args():
        def __init__(self):
            self.image_channel = "IMAGE"
            self.table_channel = "TABLE"
            self.command_rate = 1.0
    return args()


class TestLineFollower():
    def test_construction(self, defaultArgs):
        LF = line_follower.LineFollower(defaultArgs)
