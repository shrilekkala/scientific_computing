import unittest
from src.tools import is_a_number

class TestIsANumber(unittest.TestCase):

    def test_integer(self):
        result = is_a_number(100)
        self.assertTrue(result)

    def test_float(self):
        result = is_a_number(0.01)
        self.assertTrue(result)

    def test_string(self):
        result = is_a_number("abc")
        self.assertFalse(result)

    def test_boolean(self):
        result = is_a_number(True)
        self.assertFalse(result)

    def test_list(self):
        result = is_a_number([0, 1, 5])
        self.assertFalse(result)

    def test_none(self):
        result = is_a_number(None)
        self.assertFalse(result)

if __name__ == "__main__":
    unittest.main()