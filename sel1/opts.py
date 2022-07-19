from selenium.webdriver import Firefox
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options

profile_path = r'/Users/ekaeo/Library/Application Support/Firefox/Profiles/8z3fsuns.default-release'
options=Options()
options.set_preference('profile', profile_path)
service = Service('/usr/local/Cellar/geckodriver/0.31.0/bin/geckodriver')
driver = Firefox(service=service, options=options)
driver.get("https://www.skinport.com")
driver.quit()
