# test_environment.py
import unittest
from unittest.mock import MagicMock, patch
from pokegym.environment import Environment  # Adjust the import path based on your actual project structure

class TestMilestoneCompletionLogic(unittest.TestCase):
    def setUp(self):
        patcher = patch('pokegym.environment.Environment.__init__', return_value=None)
        self.addCleanup(patcher.stop)
        self.mock_env_init = patcher.start()

        # Create an instance of Environment without calling its constructor
        self.env = Environment()
        self.env.shared_data = {
            1: {'badge_1': 1, 'mt_moon_completion': 0},
            2: {'badge_1': 1, 'mt_moon_completion': 1},
            3: {'badge_1': 0, 'mt_moon_completion': 1}
        }
        self.env.milestones = {'badge_1': 0, 'mt_moon_completion': 0}
        self.env.milestone_threshold_values = [0.5, 0.5]  # 50% threshold for testing

    # @patch('pokegym.environment.Environment.trigger_state_loading')
    # def test_assess_and_trigger_state_loads(self, mock_trigger):
    #     # Assuming 'assess_and_trigger_state_loads' is a method to be tested
    #     self.env.assess_and_trigger_state_loads()
    #     mock_trigger.assert_called_with('mt_moon_completion', 1)
        
    @patch('pokegym.environment.Environment.trigger_state_loading')
    def test_assess_and_trigger_state_loads(self, mock_trigger):
        print("Starting test...")
        self.env.assess_and_trigger_state_loads()
        print(f"Shared data: {self.env.shared_data}")
        print("Expected to trigger loading for 'mt_moon_completion' with at least one environment.")
        mock_trigger.assert_called_with('mt_moon_completion', 1)
        print("Test completed successfully.")
            

if __name__ == '__main__':
    unittest.main()
