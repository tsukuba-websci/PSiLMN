import pytest
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from lib.agent import parse_response_mmlu

def test_response_parsing_answer():
    """
    Test the parsing of the response from an agent when the response is correctly formatted like (X).
    """

    response = """Let's denote the width of the rectangle as w and the length as l. We are given that the length is twice the width, so l = 2w. Also, we are given the length of the diagonal, which in a rectangle can be calculated using the Pythagorean theorem, since the opposite sides of a rectangle are equal in length. Thus, we have w^2 + w^2 = (2w)^2 + (2w)^2 = (2w)^2 + (2w)^2 = 4w^2 = \left(\frac{\sqrt{5}}{2} * \sqrt{5} * 2\right)^2 = 5\sqrt{5} * 5 = 25 * 5 = 125. Solving for w, we get w = 5 sqrt(5)/5 = sqrt(5). Since l = 2w, then l = 2*sqrt(5) = 2*sqrt(5)*sqrt(5) = 5. The area of a rectangle is given by the product of its width and length, so A = w * l = sqrt(5) * 5 = sqrt(5) * 5 * sqrt(5) * sqrt(5) = 5 * 5 = 25. Answer: D. The area of the rectangle is 25 square units. (D)"""
    
    # Test the parsing of the response
    parsed_response = parse_response_mmlu(response)

    # parse response should be D
    assert parsed_response == "D"
