from ikomia import core, dataprocess
import copy
import os
import cv2
from CovidNet.CovidNet_inference import Covidnet


# --------------------
# - Class to handle the process parameters
# - Inherits core.CProtocolTaskParam from Ikomia API
# --------------------
class CovidNetParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.model_path = os.path.dirname(os.path.realpath(__file__)) + "/models/covid-net.pb"

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        # Example : self.windowSize = int(paramMap["windowSize"])
        pass

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
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
        self.addOutput(dataprocess.CGraphicsOutput())
        # Add numeric output
        self.addOutput(dataprocess.CNumericIO())

        # Create parameters class
        if param is None:
            self.setParam(CovidNetParam())
        else:
            self.setParam(copy.deepcopy(param))

        param = self.getParam()
        self.covid_model = Covidnet(model_path=param.model_path)

        # Load class names
        self.class_names = []
        class_names_path = os.path.dirname(os.path.realpath(__file__)) + "/models/class_names"

        with open(class_names_path) as f:
            for row in f:
                self.class_names.append(row[:-1])

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 3

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        # Forward input image
        self.forwardInputImage(0, 0)

        # Get input :
        input_img = self.getInput(0)
        src_image = input_img.getImage()

        if src_image.ndim == 2:
            color_image = cv2.cvtColor(src_image, cv2.COLOR_GRAY2RGB)
        else:
            color_image = src_image

        h = color_image.shape[0]
        w = color_image.shape[1]

        # Step progress bar:
        self.emitStepProgress()
        
        # Run prediction
        prediction = self.covid_model.predict(color_image)

        # Step progress bar:
        self.emitStepProgress()

        # Set graphics output
        graphics_output = self.getOutput(1)
        graphics_output.setNewLayer("CovidNet")
        graphics_output.setImageIndex(0)
        class_index = prediction.argmax(axis=1)[0]
        msg = self.class_names[class_index] + ": {:.3f}".format(prediction[0][class_index]) 
        graphics_output.addText(msg, 0.05*w, 0.05*h)

        # Init numeric output
        numeric_ouput = self.getOutput(2)
        numeric_ouput.clearData()
        numeric_ouput.setOutputType(dataprocess.NumericOutputType.TABLE)
        numeric_ouput.addValueList(prediction.flatten().tolist(), "Probability", self.class_names)
        
        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits dataprocess.CProcessFactory from Ikomia API
# --------------------
class CovidNetProcessFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "CovidNet"
        self.info.shortDescription = "A tailored Deep Convolutional Neural Network Design " \
                                    "for detection of COVID-19 cases from chest radiography images."
        self.info.description = "The COVID-19 pandemic continues to have a devastating effect on the health " \
                                "and well-being of the global population. A critical step in the fight against " \
                                "COVID-19 is effective screening of infected patients, with one of the key screening " \
                                "approaches being radiological imaging using chest radiography. It was found in early " \
                                "studies that patients present abnormalities in chest radiography images that are characteristic " \
                                "of those infected with COVID-19. Motivated by this, a number of artificial intelligence (AI) " \
                                "systems based on deep learning have been proposed and results have been shown to be quite promising " \
                                "in terms of accuracy in detecting patients infected with COVID-19 using chest radiography images. " \
                                "However, to the best of the authors’ knowledge, these developed AI systems have been closed source " \
                                "and unavailable to the research community for deeper understanding and extension, and unavailable " \
                                "for public access and use. Therefore, in this study we introduce COVID-Net, a deep convolutional " \
                                "neural network design tailored for the detection of COVID-19 cases from chest  radiography  images " \
                                "that is open source and available to the general public. We also describe the chest radiography dataset " \
                                "leveraged to train COVID-Net, which we will refer to as COVIDx and is comprised of 16,756 chest " \
                                "radiography images across 13,645 patient cases from two open access data repositories. " \
                                "Furthermore, we investigate how COVID-Net makes predictions using an explainability method " \
                                "in an attempt to gain deeper insights into critical factors associated with COVID cases, " \
                                "which can aid clinicians in improved screening. By no means a production-ready solution, " \
                                "the hope is that the open access COVID-Net, along with the description on constructing " \
                                "the open source COVIDx dataset, will be leveraged and build upon by both researchers and " \
                                "citizen data scientists alike to accelerate the development of highly accurate yet practical " \
                                "deep learning solutions for detecting COVID-19 cases and accelerate treatment of those " \
                                "who need it the most."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Classification"
        self.info.version = "1.0.0"
        self.info.iconPath = "icon/icon.png"
        self.info.authors = "Linda Wang, Alexander Wong"
        self.info.article = "COVID-Net: A Tailored Deep Convolutional Neural Network Design for Detection " \
                            "of COVID-19 Cases from Chest Radiography Images"
        self.info.journal = ""
        self.info.year = 2020
        self.info.license = "GNU Affero General Public License 3.0."
        self.info.documentationLink = "https://arxiv.org/pdf/2003.09871.pdf"
        self.info.repository = "https://github.com/lindawangg/COVID-Net"
        self.info.keywords = "covid-19,coronavirus,x-ray,radiography,chest,lung,dnn"


    def create(self, param=None):
        # Create process object
        return CovidNetProcess(self.info.name, param)
