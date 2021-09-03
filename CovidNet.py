from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class CovidNet(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        from CovidNet.CovidNet_process import CovidNetProcessFactory
        # Instantiate process object
        return CovidNetProcessFactory()

    def getWidgetFactory(self):
        from CovidNet.CovidNet_widget import CovidNetWidgetFactory
        # Instantiate associated widget object
        return CovidNetWidgetFactory()
