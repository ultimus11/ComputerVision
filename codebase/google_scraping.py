from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time

# Initialize the WebDriver (use the path to your WebDriver if it's not in PATH)
driver = webdriver.Chrome()

def get_company_details(company_name):
    try:
        # Open Google
        driver.get("https://www.google.com")

        # Find the search box, enter the company name, and perform the search
        search_box = driver.find_element(By.NAME, "q")
        search_box.send_keys(company_name)
        search_box.send_keys(Keys.RETURN)

        time.sleep(3)  # Wait for the results to load

        # Extract company details
        details = {}

        # Example: Extracting the company website
        try:
            website_element = driver.find_element(By.CSS_SELECTOR, 'div.yuRUbf a')
            details['Website'] = website_element.get_attribute('href')
        except:
            details['Website'] = None

        # Example: Extracting the company address from Google Business profile
        try:
            address_element = driver.find_element(By.CSS_SELECTOR, 'div.LrzXr')
            details['Address'] = address_element.text
        except:
            details['Address'] = None

        # Example: Extracting phone number from Google Business profile
        try:
            phone_element = driver.find_element(By.CSS_SELECTOR, 'span.LrzXr.zdqRlf.kno-fv')
            details['Phone'] = phone_element.text
        except:
            details['Phone'] = None

        # Example: Extracting rating from Google Business profile
        try:
            rating_element = driver.find_element(By.CSS_SELECTOR, 'div.BHMmbe')
            details['Rating'] = rating_element.text
        except:
            details['Rating'] = None

        return details

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.quit()

# Example usage
company_name = "OpenAI"
details = get_company_details(company_name)
print(details)
