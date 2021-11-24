import re
import sys
import luigi
from multidim.tasks import SentimentToPyIrt

if __name__ == "__main__":
    luigi.build([SentimentToPyIrt()], local_scheduler=True)
