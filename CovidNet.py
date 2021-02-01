from ikomia import dataprocess
import CovidNet_process as processMod
import CovidNet_widget as widgetMod


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class CovidNet(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        # Instantiate process object
        return processMod.CovidNetProcessFactory()

    def getWidgetFactory(self):
        # Instantiate associated widget object
        return widgetMod.CovidNetWidgetFactory()
