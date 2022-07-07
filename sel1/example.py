from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time

driver = webdriver.Firefox()
driver.get("http://www.python.org")

print(driver.title)
#assert "Python" in driver.title
#elem = driver.find_element(By.NAME, "q")
#time.sleep(3)
#elem.clear()
#elem.send_keys("pycon")
#elem.send_keys(Keys.RETURN)
#time.sleep(3)
#assert "No results found." not in driver.page_source
#time.sleep(3)
#driver.close()
