import warnings
warnings.warn("CozeAPIClient is deprecated. Use DeepseekAPIClient instead.", DeprecationWarning)

from deepseek_client import DeepseekAPIClient as CozeAPIClient

class CozeBotManager:
    def __init__(self, api_client):
        self.client = api_client
    
    def getBot(self, bot_name):
        return "deepseek-chat"
    
    def listDeployedBots(self):
        return [{"name": "Deepseek", "bot_id": "deepseek-chat"}]