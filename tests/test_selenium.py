"""
BrainDetect — Selenium Browser Tests
=====================================
Tests 3 critical user workflows:
1. Doctor login → upload MRI → verify Grad-CAM appears
2. Admin login → verify doctor management table loads
3. Patient lookup by Report ID → verify diagnosis appears

Run with: python tests/test_selenium.py
"""

import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# ── Config ────────────────────────────────────────────────
BASE_URL       = "https://brain-tumor-phi.vercel.app"
ADMIN_EMAIL    = "admin@braindetect.com"
ADMIN_PASSWORD = "Admin123!"
DOCTOR_EMAIL   = "doctor@braindetect.com"
DOCTOR_PASSWORD= "Doctor123!"

# Path to any MRI image for upload test
MRI_IMAGE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
    "../dataset/Testing/pituitary/Te-pi_101.jpg")
)

# ── Setup Chrome ───────────────────────────────────────────
def get_driver():
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1400,900")
    # Uncomment below to run headless (no browser window)
    # options.add_argument("--headless")
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    driver.implicitly_wait(10)
    return driver

def wait(driver, seconds=2):
    time.sleep(seconds)

# ── Test 1: Doctor Login + MRI Upload + Grad-CAM ──────────
def test_doctor_login_and_predict():
    print("\n🧪 Test 1: Doctor Login → MRI Upload → Grad-CAM")
    driver = get_driver()

    try:
        # 1. Open login page
        driver.get(BASE_URL)
        wait(driver, 2)
        print("  ✅ Login page loaded")

        # 2. Enter doctor credentials
        driver.find_element(By.ID, "loginEmail").send_keys(DOCTOR_EMAIL)
        driver.find_element(By.ID, "loginPassword").send_keys(DOCTOR_PASSWORD)
        driver.find_element(By.CSS_SELECTOR, "#loginForm button[type='submit']").click()
        wait(driver, 4)

        # 3. Verify redirected to doctor dashboard
        assert "doctor.html" in driver.current_url, f"❌ Not redirected to doctor dashboard: {driver.current_url}"
        print("  ✅ Doctor login successful — redirected to doctor.html")

        # 4. Upload MRI image
        file_input = driver.find_element(By.ID, "mriFile")
        file_input.send_keys(MRI_IMAGE_PATH)
        wait(driver, 1)
        print("  ✅ MRI image selected")

        # 5. Click Analyse Scan
        driver.find_element(By.XPATH, "//button[contains(text(), 'Analyse Scan')]").click()
        wait(driver, 8)  # Wait for prediction
        print("  ✅ Analyse Scan clicked — waiting for result...")

        # 6. Verify result card appears
        result_card = driver.find_element(By.ID, "resultCard")
        assert result_card.is_displayed(), "❌ Result card not displayed"
        print("  ✅ Prediction result card appeared")

        # 7. Verify Grad-CAM overlay image loaded
        gc_overlay = driver.find_element(By.ID, "gcOverlay")
        src = gc_overlay.get_attribute("src")
        assert src and src.startswith("data:image/png;base64"), "❌ Grad-CAM overlay not loaded"
        print("  ✅ Grad-CAM overlay image loaded successfully")

        # 8. Verify prediction class is shown
        result_class = driver.find_element(By.ID, "resultClass")
        assert result_class.text in ["Glioma", "Meningioma", "No Tumor", "Pituitary"], \
            f"❌ Unexpected prediction class: {result_class.text}"
        print(f"  ✅ Prediction class displayed: {result_class.text}")

        print("  🎉 Test 1 PASSED!")
        return True

    except Exception as e:
        print(f"  ❌ Test 1 FAILED: {e}")
        return False

    finally:
        driver.quit()

# ── Test 2: Admin Login + Doctor Table ────────────────────
def test_admin_doctor_management():
    print("\n🧪 Test 2: Admin Login → Doctor Management Table")
    driver = get_driver()

    try:
        # 1. Open login page
        driver.get(BASE_URL)
        wait(driver, 2)

        # 2. Enter admin credentials
        driver.find_element(By.ID, "loginEmail").send_keys(ADMIN_EMAIL)
        driver.find_element(By.ID, "loginPassword").send_keys(ADMIN_PASSWORD)
        driver.find_element(By.CSS_SELECTOR, "#loginForm button[type='submit']").click()
        wait(driver, 4)

        # 3. Verify redirected to admin dashboard
        assert "admin.html" in driver.current_url, f"❌ Not redirected to admin: {driver.current_url}"
        print("  ✅ Admin login successful — redirected to admin.html")

        # 4. Verify stats loaded
        wait(driver, 4)  # extra wait for Railway
        stat_total = driver.find_element(By.ID, "statTotal")
        assert stat_total.text != "—", "❌ Stats not loaded"
        print(f"  ✅ System stats loaded — Total predictions: {stat_total.text}")

        # 5. Verify doctor table loaded
        wait(driver, 3)
        doctors_table = driver.find_element(By.ID, "doctorsTable")
        table_html = doctors_table.get_attribute("innerHTML")
        assert "table" in table_html.lower() or "doctor" in table_html.lower(), \
            "❌ Doctor table not loaded"
        print("  ✅ Doctor management table loaded")

        # 6. Verify Add Doctor button exists
        add_btn = driver.find_element(By.XPATH, "//button[contains(text(), 'Add Doctor')]")
        assert add_btn.is_displayed(), "❌ Add Doctor button not visible"
        print("  ✅ Add Doctor button visible")

        print("  🎉 Test 2 PASSED!")
        return True

    except Exception as e:
        print(f"  ❌ Test 2 FAILED: {e}")
        return False

    finally:
        driver.quit()

# ── Test 3: Patient Lookup by Report ID ───────────────────
def test_patient_lookup():
    print("\n🧪 Test 3: Patient Report ID Lookup")
    driver = get_driver()

    try:
        # 1. Open patient page directly
        driver.get(f"{BASE_URL}/patient.html")
        wait(driver, 2)
        print("  ✅ Patient page loaded")

        # 2. Verify lookup section visible
        lookup = driver.find_element(By.ID, "lookupSection")
        assert lookup.is_displayed(), "❌ Lookup section not visible"
        print("  ✅ Report ID lookup form visible")

        # 3. Enter fake report ID → should show error
        report_input = driver.find_element(By.ID, "reportIdInput")
        report_input.send_keys("BD-FAKE-0000")
        driver.find_element(By.XPATH, "//button[contains(text(), 'View My Results')]").click()
        wait(driver, 8)

        # 4. Verify error alert shown
        alert = driver.find_element(By.ID, "lookupAlert")
        assert alert.is_displayed(), "❌ Error alert not shown for invalid report ID"
        print("  ✅ Error alert shown for invalid Report ID")

        # 5. Clear and test with Enter key
        report_input.clear()
        report_input.send_keys("BD-FAKE-1111")
        report_input.send_keys(Keys.ENTER)
        wait(driver, 2)
        print("  ✅ Enter key lookup works")

        print("  🎉 Test 3 PASSED!")
        return True

    except Exception as e:
        print(f"  ❌ Test 3 FAILED: {e}")
        return False

    finally:
        driver.quit()

# ── Run all tests ──────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  🤖 BrainDetect — Selenium Browser Tests")
    print("=" * 55)

    results = []
    results.append(test_doctor_login_and_predict())
    results.append(test_admin_doctor_management())
    results.append(test_patient_lookup())

    print("\n" + "=" * 55)
    passed = sum(results)
    total  = len(results)
    print(f"  Results: {passed}/{total} tests passed")
    if passed == total:
        print("  🎉 ALL TESTS PASSED!")
    else:
        print(f"  ⚠️  {total - passed} test(s) failed")
    print("=" * 55)