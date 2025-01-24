"""
Filename: exceptions.py
Authors:
    Piotr Patek, email: piotr.patek.stud@pw.edu.pl
    Jan Potaszyński, email: jan.potaszynski.stud@pw.edu.pl
Date: January 2025
Version: 1.0
Description:
    This file contains custom exceptions.
Dependencies: None
"""

class ModelAlreadyTrainedError(Exception):
    pass


class ModelNotTrainedError(Exception):
    pass


class MeasuresOfQualityNotCompiledError(Exception):
    pass

class ClassNotExistingError(Exception):
    pass
