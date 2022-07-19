from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
import time

profile_path = "Users/ekaeo/Library/Application Support/Firefox/Profiles/8z3fsuns.default-release"
options = Options()
options.set_preference('profile', profile_path)

driver = webdriver.Firefox(options = options)
driver.get("https://skinport.com/market?sort=percent&order=desc&pricegt=2948&pricelt=7699")
print(driver.title)
assert "Buy" in driver.title
elem = driver.find_element(By.CLASS_NAME, 'ItemPreview-mainAction')
elem2 = driver.find_element(By.CLASS_NAME, 'ItemPreview-itemName')
print(elem2.text)
#elem2 = driver.find.element(by.ID, 'title')

#elem.clear()

time.sleep(5)
ele2.blah
#elem.send_keys("pycon")
elem.send_keys(Keys.RETURN)
time.sleep(5)
assert "No results found." not in driver.page_source
driver.close()
