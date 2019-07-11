import requests
from bs4 import BeautifulSoup
import ssl, urllib
import traceback

base_url = 'https://www.google.co.kr/search'

def spider(max_pages):
    page=1
    while page < max_pages:
