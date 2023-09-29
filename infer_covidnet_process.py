from ikomia import utils, core, dataprocess
import copy
import os
import cv2
from infer_covidnet.covidnet import CovidNet


# --------------------
# - Class to handle the process parameters
# - Inherits core.CProtocolTaskParam from Ikomia API
# --------------------
class CovidNetParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.model_path = os.path.dirname(os.path.realpath(__file__)) + "/models/covid-net.pb"

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        # Example : self.windowSize = int(paramMap["windowSize"])
        pass

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        # Example : paramMap["windowSize"] = str(self.windowSize)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits core.CProtocolTask or derived from Ikomia API
# --------------------
class CovidNetProcess(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add graphics output
        self.add_output(dataprocess.CGraphicsOutput())
        # Add numeric output
        self.add_output(dataprocess.CNumericIO())

        # Create parameters class
        if param is None:
            self.set_param_object(CovidNetParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        param = self.get_param_object()

        if not os.path.exists(param.model_path):
            print("Downloading model, please wait...")
            model_url = utils.get_model_hub_url() + "/" + self.name + "/covid-net.pb"
            self.download(model_url, param.model_path)

        self.covid_model = CovidNet(model_path=param.model_path)

        # Load class names
        self.class_names = []
        class_names_path = os.path.dirname(os.path.realpath(__file__)) + "/models/class_names"

        with open(class_names_path) as f:
            for row in f:
                self.class_names.append(row[:-1])

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 3

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()

        # Forward input image
        self.forward_input_image(0, 0)

        # Get input :
        input_img = self.get_input(0)
        src_image = input_img.get_image()

        if src_image.ndim == 2:
            color_image = cv2.cvtColor(src_image, cv2.COLOR_GRAY2RGB)
        else:
            color_image = src_image

        h = color_image.shape[0]
        w = color_image.shape[1]

        # Step progress bar:
        self.emit_step_progress()
        
        # Run prediction
        prediction = self.covid_model.predict(color_image)

        # Step progress bar:
        self.emit_step_progress()

        # Set graphics output
        graphics_output = self.get_output(1)
        graphics_output.set_new_layer("CovidNet")
        graphics_output.set_image_index(0)
        class_index = prediction.argmax(axis=1)[0]
        msg = self.class_names[class_index] + ": {:.3f}".format(prediction[0][class_index]) 
        graphics_output.add_text(msg, 0.05*w, 0.05*h)

        # Init numeric output
        numeric_ouput = self.get_output(2)
        numeric_ouput.clear_data()
        numeric_ouput.set_output_type(dataprocess.NumericOutputType.TABLE)
        numeric_ouput.add_value_list(prediction.flatten().tolist(), "Probability", self.class_names)
        
        # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits dataprocess.CProcessFactory from Ikomia API
# --------------------
class CovidNetProcessFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_covidnet"
        self.info.short_description = "A tailored Deep Convolutional Neural Network Design " \
                                      "for detection of COVID-19 cases from chest radiography images."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Classification"
        self.info.version = "1.2.0"
        self.info.icon_path = "icon/icon.png"
        self.info.authors = "Linda Wang, Alexander Wong"
        self.info.article = "COVID-Net: A Tailored Deep Convolutional Neural Network Design for Detection " \
                            "of COVID-19 Cases from Chest Radiography Images"
        self.info.journal = ""
        self.info.year = 2020
        self.info.license = "GNU Affero General Public License 3.0."
        self.info.documentation_link = "https://arxiv.org/pdf/2003.09871.pdf"
        self.info.repository = "https://github.com/Ikomia-hub/infer_covidnet"
        self.info.original_repository = "https://github.com/lindawangg/COVID-Net"
        self.info.keywords = "covid-19,coronavirus,x-ray,radiography,chest,lung,dnn"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "CLASSIFICATION"

    def create(self, param=None):
        # Create process object
        return CovidNetProcess(self.info.name, param)
