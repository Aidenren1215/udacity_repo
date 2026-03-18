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

import re

HKGRP_ALCO_PATTERN = re.compile(
    r'(?im)(?P<title>('
    r'Asset\s*(?:&\s*)?Liability\s*Management\s*Committee\s*\((?:ALCO|ALMCO)\)'
    r'|Minutes\s+of\s+(?:the\s+)?(?:OCBC\s+HK\s+Ltd\s+)?Asset\s*(?:&\s*)?Liability\s*Management\s*Committee\s*\((?:ALCO|ALMCO)\)\s+Meeting.*'
    r'|Minutes\s+of\s+(?:\d{4}\s+[A-Za-z]+\s+)?ALCO\s+Meeting.*'
    r'|agenda\s+items\s+to\s+the\s+Asset\s*(?:&\s*)?Liability\s*Management\s*Committee\s*\(ALCO\)'
    r'))'
)

HKBR_ALCO_PATTERN = re.compile(
    r'(?im)(?P<title>('
    r'Minutes\s+of\s+(?:the\s+)?OCBC\s+HK\s+Branch\s+Asset\s*(?:&\s*)?Liability\s*Management\s*Committee\s*\(ALCO\)\s+Meeting.*'
    r'|Minutes\s+of\s+(?:the\s+)?Asset\s*(?:&\s*)?Liability\s*Management\s*Committee\s*\(ALCO\)\s+Meeting.*'
    r'|Minutes\s+of\s+(?:\d{4}\s+[A-Za-z]+\s+)?ALCO\s+Meeting.*'
    r'))'
)

HKGRP_ASC_PATTERN = re.compile(
    r'(?im)(?P<title>('
    r'Minutes\s+of\s+Funding\s+Strategy\s+Committee.*'
    r'|Minutes\s+of\s+Funding\s+ASC\s+Meeting.*'
    r'|Minutes\s+of\s+(?:\d{4}\s+[A-Za-z]+\s+)?ASC\s+Meeting.*'
    r'|Minutes\s+of\s+HK\s+ASC\s+Meeting.*'
    r'|Minutes\s+of\s+the\s+ALCO\s+Sub-Committee\s*\(ASC\)\s+Meeting.*'
    r'))'
)

import os

folder = "files/alco minutes hk/ASC minutes"

for filename in os.listdir(folder):
    if "ASC Minutes" in filename:
        new_name = filename.replace("ASC Minutes", "ALCO Sub-committee minutes")

        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_name)

        print(f"{filename}  ->  {new_name}")  # 先确认
        os.rename(old_path, new_path)