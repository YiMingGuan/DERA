import logging
from ..load_strategy_interface import KG, Entity, Relation, KGs
from bidict import bidict
from isodate.isoerror import ISO8601Error
import re, os
import hashlib
import pickle
logger = logging.getLogger(__name__)

class SRPRSEntity(Entity):
    def __init__(self, url):
        super().__init__(url)
        self.translated_url = None
     
    def parse_url(self):
        name = self.url.split('/')[-1].replace("_", " ")
        return name


