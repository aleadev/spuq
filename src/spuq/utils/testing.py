import sys


from numpy.testing import *

class _TestCase(TestCase):
    def __init__(self, method=None):
        if method is None:
            method = "dummy"
        super(_TestCase, self).__init__(method)

    def setUp(self):
        """Just call setup in case somebody didn't use the braindead
        capitalisation"""
        if hasattr(self, "setup"):
            self.setup()
    
    def myAssertIsInstance(self, obj, cls, msg=None):
        """Same as self.assertTrue(isinstance(obj, cls)), with a nicer
        default message."""
        if not isinstance(obj, cls):
            #standardMsg = '%r is not an instance of %r' % (obj, cls)
            #self.fail(self._formatMessage(msg, standardMsg))
            # TODO: remove this ugly hack
            assert_true(type(obj)==cls)
            
    def dummy(self):
        pass

del TestCase.assertEquals
del TestCase.failIf
del TestCase.failIfAlmostEqual
del TestCase.failIfEqual
del TestCase.failUnless
del TestCase.failUnlessAlmostEqual
del TestCase.failUnlessEqual

TestCase = _TestCase

_tc = TestCase()

#assert_equal = _tc.assertEqual
assert_not_equal = _tc.assertNotEqual
assert_true = _tc.assertTrue
assert_false = _tc.assertFalse

if sys.hexversion >= 0x02070000:
    assert_is = _tc.assertIs
    assert_is_not = _tc.assertIsNot
    assert_is_none = _tc.assertIsNone
    assert_is_not_none = _tc.assertIsNotNone
    assert_in = _tc.assertIn
    assert_not_in = _tc.assertNotIn
    assert_is_instance = _tc.assertIsInstance
    assert_not_is_instance = _tc.assertNotIsInstance
else:
    assert_is_instance = _tc.myAssertIsInstance


assert_raises = _tc.assertRaises
if sys.hexversion >= 0x02070000:
    assert_raises_regexp = _tc.assertRaisesRegexp

assert_almost_equal = _tc.assertAlmostEqual
assert_not_almost_equal= _tc.assertNotAlmostEqual
if sys.hexversion >= 0x02070000:
    assert_greater = _tc.assertGreater
    assert_greater_equal = _tc.assertGreaterEqual
    assert_less = _tc.assertLess
    assert_less_equal = _tc.assertLessEqual
    assert_regexp_matches = _tc.assertRegexpMatches
    assert_not_regexp_matches = _tc.assertNotRegexpMatches
    assert_items_equal = _tc.assertItemsEqual
