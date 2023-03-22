from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        from infer_covidnet.infer_covidnet_process import CovidNetProcessFactory
        # Instantiate process object
        return CovidNetProcessFactory()

    def get_widget_factory(self):
        from infer_covidnet.infer_covidnet_widget import CovidNetWidgetFactory
        # Instantiate associated widget object
        return CovidNetWidgetFactory()
