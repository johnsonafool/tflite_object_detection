"""
This module provides the support for Tracker Object. 
This creates object for the specific tracker based on the name of the tracker provided 
in the command line of the demo.

To add more trackers here, simply replicate the SortTracker() code and replace it with 
the new tracker as required.

Developer simply needs to instantiate the object of ObjectTracker(trackerObjectName) with a valid 
trackerObjectName.

"""

import os
import sys


class ObjectTracker(object):
    def __init__(self, trackerObjectName):
        if trackerObjectName == "sort":  # Add more trackers in elif whenever needed
            self.trackerObject = SortTracker()
        else:
            print("Invalid Tracker Name")
            self.trackerObject = None


class SortTracker(ObjectTracker):
    def __init__(self):
        sys.path.append(
            os.path.join(os.path.dirname(__file__), "../third_party", "sort-master")
        )
        from sort import Sort

        self.mot_tracker = Sort()
