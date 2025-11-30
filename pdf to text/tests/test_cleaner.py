import unittest
from text import clean_text

class TestCleaner(unittest.TestCase):

    def test_remove_cid_and_bullets_and_urls(self):
        old = '(cid:240)linkedin.com/in/peterprabhuj • §github.com/peterprabhuj'
        out = clean_text(old)
        self.assertIn('linkedin.com/in/peterprabhuj', out)
        self.assertIn('github.com/peterprabhuj', out)
        self.assertNotIn('(cid:240)', out)

    def test_conservative_case_split(self):
        old = 'InnovativeSoftwareEngineeringstudentspecializinginIoT'
        out = clean_text(old)
        self.assertIn('Innovative Software Engineering', out)
        self.assertIn('IoT', out)

    def test_hyphenation_join(self):
        old = 'detectwrong-waydriving,andpreventsignalviolations,\nby85%+'
        out = clean_text(old)
        # we at least expect the percentage separated from text
        self.assertIn('by 85%+', out.replace('\n', ' '))

if __name__ == '__main__':
    unittest.main()
