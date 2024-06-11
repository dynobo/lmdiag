"""Test-Suite for plot.py."""

from unittest import TestCase
import lmdiag

class TestPlot(TestCase):
	def test_is_plot(self):
		result = lmdiag.plot(lm)
		self.assertTrue(isinstance(result, matplotlib.plot))
