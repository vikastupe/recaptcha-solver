import os
from PIL import Image
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from bs4 import BeautifulSoup

class ReCaptcha:
    def __init__(self):
        # Threshold for accepting a choice other than the predicted, adjust if many false positive results. default is 0.6
        self._threshold = 0.40

        # suppress tensorflow logs
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
        # check if data dir is present
        if os.path.exists(os.path.join(os.getcwd(), "data")):
            self.__data_dir = os.path.join(os.getcwd(), "data")
        else:
            os.mkdir(os.path.join(os.getcwd(), "data"))
            self.__data_dir = os.path.join(os.getcwd(), "data")

        # A list of possible classes

        self.__CLASSES = [
            "bicycle",
            "bridge",
            "bus",
            "car",
            "chimney",
            "crosswalk",
            "hydrant",
            "motorcycle",
            "other",
            "palm",
            "stairs",
            "traffic"
        ]
        self.__model = self.__get_model()

    def __get_model(self):
        f = tf.keras.utils.get_file(
            fname="model.h5",
            origin="",
            cache_dir=self.__data_dir,
            cache_subdir='model')
        model = keras.models.load_model(f, compile=False)
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=["accuracy"])
        return model

    def __slice_image(self, path):
        from PIL import Image
        img = np.array(Image.open(path))

        ys = img.shape[0] // 3
        xs = img.shape[1] // 3

        topLeft = img[0:ys, 0:xs]
        topMid = img[0:ys, xs:xs * 2]
        topRight = img[0:ys, xs * 2:xs * 3]
        midLeft = img[ys:ys * 2, 0:xs]
        midmid = img[ys:ys * 2, xs:xs * 2]
        midRight = img[ys:ys * 2, xs * 2:xs * 3]
        bottomLeft = img[ys * 2:ys * 3, 0:xs]
        bottomMid = img[ys * 2:ys * 3, xs:xs * 2]
        bottomRight = img[ys * 2:ys * 3, xs * 2:xs * 3]

        return [topLeft, topMid, topRight, midLeft, midmid, midRight, bottomLeft, bottomMid, bottomRight]

    def __predict_tile(self, tile):
        # resize the image
        i = img_to_array(tile)
        to_predict = i.reshape((-1, 224, 224, 3))
        prediction = self.__model.predict(to_predict)
        # return a list of the prediction array, the class name with highest probability and its index
        return [prediction, self.__CLASSES[np.argmax(prediction)], np.argmax(prediction)]
        
    def recaptcha_solver(self, page):
        """
        :param page:
        :return: Bool
        if you want to use this method then you should pass page object of playwright browser
        This method contains playwright expath.
        """
        if page.frame_locator('//iframe[@title="recaptcha challenge expires in two minutes"]').locator(
                selector='//button[@id="recaptcha-reload-button"]').count() > 0:
            page.frame_locator('//iframe[@title="recaptcha challenge expires in two minutes"]').locator(
                selector='//button[@id="recaptcha-reload-button"]').click(timeout=2000)
            page.wait_for_timeout(2000)
        else:
            frame_ele = page.frame_locator('//iframe[@title="reCAPTCHA"]')
            frame_ele.locator(selector='//div[@class="recaptcha-checkbox-border"]').click(timeout=10000)

        frame_image = page.frame_locator('//iframe[@title="recaptcha challenge expires in two minutes"]')
        reload_images = frame_image.locator(selector='//button[@id="recaptcha-reload-button"]')

        while True:
            # title_wrapper = frame_image.locator(selector='id="rc-imageselect"]')
            element_string = frame_image.locator("tr:nth-child(3) > td").first.inner_html()
            soup = BeautifulSoup(element_string, "html.parser")
            image_class = soup.find('img')['class']
            num_of_split = image_class[0].split("-")[-1]
            if num_of_split == "33":
                # dynamic_captcha = True
                pass
            else:
                reload_images.click()
                page.wait_for_timeout(2000)
                continue

            # Get the object of the captcha where we suppose to look for
            captcha_object = frame_image.locator("strong").inner_html()
            print("recaptcha type::::::::::", frame_image.locator('[id="rc-imageselect"]').text_content())
            recaptcha_type = "dynamic" if "verify once there are none left" in frame_image.locator(
                '[id="rc-imageselect"]').text_content() else "static"
            print("recaptcha type::::::::::", frame_image.locator('[id="rc-imageselect"]').text_content(),
                  "---------------------", recaptcha_type)

            for i in self.__CLASSES:
                if i in captcha_object:
                    class_index = self.__CLASSES.index(i)
                    class_object = i
                    print("class index is ", str(class_index), i)

            # first run of solving the captcha
            while True:
                check = []
                for i in range(9):
                    xpath = "//td[contains(@tabindex, '" + str(i + 4) + "')]"
                    matched_tile = frame_image.locator(xpath)
                    matched_tile.screenshot(path=os.path.join(self.__data_dir, "tile.jpg"))
                    img = Image.open(os.path.join(self.__data_dir, "tile.jpg")).convert('RGB')
                    img = img.resize(size=(224, 224))
                    result = self.__predict_tile(img)
                    current_object_probability = result[0][0][class_index]
                    compare_probability = result[2] * self._threshold
                    print("The predicted tile to be ", result[1], "and probability is", current_object_probability)

                    '''
                            Two methods for predictioin here, The simple matching of the text was first implemented but false negative/positive results
                            was seen.
                            To compromise getting the probability of the current captcha object and assigning a thresold seems to yeild a better results
                            '''

                    if result[1] in class_object:
                        print("found a match clicking tile ", str(i + 1))
                        # tabindex="4"
                        matched_tile.click()
                        check.append("found")
                        page.wait_for_timeout(3000)
                    elif current_object_probability > compare_probability:
                        print("found a match clicking tile ", str(i + 1))
                        # tabindex="4"
                        matched_tile.click()
                        check.append("found")
                        page.wait_for_timeout(3000)
                    else:
                        print(" not a match .. skipping!")
                        page.wait_for_timeout(100)
                        continue
                if len(check) < 1:
                    frame_image.locator(selector='//button[@id="recaptcha-verify-button"]').click(timeout=2000)
                    page.set_default_timeout(2000)
                    try:
                        try_again = frame_image.locator(
                            selector='//div[contains(text(), "Please try again.") and @tabindex="0"]').inner_html()
                        print("static      ", try_again)
                        if try_again == 'Please try again.':
                            return False
                    except: pass
                    try:
                        try_again = frame_image.locator(
                            selector='//div[contains(text(), "Please also check the new images.") and @tabindex="0"]').inner_html()
                        if try_again == "Please also check the new images.":
                            return False
                    except: pass
                    try:
                        try_again = frame_image.locator(
                            selector='//div[contains(text(), "Please select all matching images.") and @tabindex="0"]').inner_html()
                        if try_again == 'Please select all matching images.':
                            return False
                    except: pass
                    return True

                else:
                    if recaptcha_type == "static":
                        page.set_default_timeout(2000)
                        frame_image.locator(selector='//button[@id="recaptcha-verify-button"]').click(timeout=2000)
                        try:

                            try_again = frame_image.locator(
                                selector='//div[contains(text(), "Please try again.") and @tabindex="0"]').inner_html()
                            print("static      ", try_again)
                            if try_again == 'Please try again.':
                                return False
                        except:
                            pass
                        try:
                            try_again = frame_image.locator(
                                selector='//div[contains(text(), "Please also check the new images.") and @tabindex="0"]').inner_html()
                            if try_again == "Please also check the new images.":
                                return False
                        except:
                            pass
                        try:
                            try_again = frame_image.locator(
                                selector='//div[contains(text(), "Please select all matching images.") and @tabindex="0"]').inner_html()
                            if try_again == 'Please select all matching images.':
                                return False
                        except:
                            pass
                        return True
                    else:
                        page.set_default_timeout(30000)
                        continue


if __name__ == "__main__":
    # Demo
    from playwright.sync_api import sync_playwright
    captcha_url = "https://www.google.com/recaptcha/api2/demo"
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto(captcha_url)
    Captcha_obj = ReCaptcha()
    # actual implementation inside code
    count = 6
    while count:
        print(f"trying to solve {7 - count}")
        captcha_solved = Captcha_obj.recaptcha_solver(page)
        print("recaptcha___________", captcha_solved)
        if captcha_solved:
            # captcha_solved = True
            print("recaptcha solved...")
            break
        else:
            print('recaptcha count:::::::::::::::', count)
            count -= 1
            page.wait_for_timeout(5000)
    else:
        print("recaptcha not solved......")
    status = Captcha_obj.recaptcha_solver(page)
    print(status)
    context.close()
    browser.close()
