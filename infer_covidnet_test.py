import logging
import cv2

logger = logging.getLogger(__name__)


def test(t, data_dict):
    logger.info("===== Test::infer covidnet =====")
    logger.info("----- Use default parameters")
    img = cv2.imread(data_dict["images"]["detection"]["coco"], cv2.IMREAD_GRAYSCALE)
    input_img_0 = t.getInput(0)
    input_img_0.setImage(img)
    t.run()