import re

HKGRP_ALCO_PATTERN = re.compile(
    r'(?im)(?P<title>(Asset\s*&\s*Liability\s*Management\s*Committee\s*\((?:ALCO|ALMCO)\)|Minutes\s+of\s+(?:the\s+)?(?:OCBC\s+HK\s+Ltd\s+)?Asset\s*&\s*Liability\s*Management\s*Committee\s*\((?:ALCO|ALMCO)\)\s+Meeting.*|Minutes\s+of\s+(?:\d{4}\s+[A-Za-z]+\s+)?ALCO\s+Meeting.*))'
)

HKBR_ALCO_PATTERN = re.compile(
    r'(?im)(?P<title>(Minutes\s+of\s+(?:the\s+)?OCBC\s+HK\s+Branch\s+Asset\s*&\s*Liability\s*Management\s*Committee\s*\(ALCO\)\s+Meeting.*|Minutes\s+of\s+(?:the\s+)?Asset\s*&\s*Liability\s*Management\s*Committee\s*\(ALCO\)\s+Meeting.*|Minutes\s+of\s+(?:\d{4}\s+[A-Za-z]+\s+)?ALCO\s+Meeting.*))'
)

HKGRP_ASC_PATTERN = re.compile(
    r'(?im)(?P<title>(Minutes\s+of\s+Funding\s+Strategy\s+Committee.*|Minutes\s+of\s+Funding\s+ASC\s+Meeting.*|Minutes\s+of\s+(?:\d{4}\s+[A-Za-z]+\s+)?ASC\s+Meeting.*|Minutes\s+of\s+HK\s+ASC\s+Meeting.*))'
)