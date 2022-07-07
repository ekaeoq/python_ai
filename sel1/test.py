from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time

driver = webdriver.Firefox()
driver.get("https://skinport.com/market?sort=percent&order=desc&pricegt=2948&pricelt=7699")
print(driver.title)
assert "Buy" in driver.title
elem = driver.find_element(By.CLASS_NAME, 'ItemPreview-mainAction')
elem2 = driver.find_element(By.CLASS_NAME, 'ItemPreview-itemName')
print(elem2.text)
#elem2 = driver.find.element(by.ID, 'title')

#elem.clear()

time.sleep(5)
#elem.send_keys("pycon")
elem.send_keys(Keys.RETURN)
time.sleep(5)
assert "No results found." not in driver.page_source
driver.close()
