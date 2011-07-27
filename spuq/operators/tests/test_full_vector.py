from numpy.testing import *
import numpy as np
from spuq.operators.full_vector import FullVector

class TestFullVector(TestCase):

    def test_equals(self):
        """Make sure we can compare vectors for equality
        """
        fv1=FullVector( np.array([1.0,2,3]) )
        fv2=FullVector( np.array([1.0,2,3]) )
        fv3=FullVector( np.array([1.0,2]) )
        fv4=FullVector( np.array([1.0,2,4]) )
        self.assertEqual( fv1, fv2 )
        self.assertNotEqual( fv1, fv3 )
        self.assertNotEqual( fv1, fv4 )

    def test_mul(self):
        """Make sure we can multiply vectors with scalars
        """
        fv1=FullVector( np.array([1.0,2,3]) )
        fv2=FullVector( np.array([2.5,5,7.5]) )
        self.assertEqual( 2.5*fv1, fv2 )
        self.assertEqual( fv1*2.5, fv2 )


if __name__ == "__main__":
    run_module_suite()
