class Application:
    def __init__(self, title, url, imageUrl):
        self._title = title
        self._url = url
        self._imageUrl = imageUrl

    #Getters
    def getTitle(self):
        return self._title

    def getUrl(self):
        return self._url

    def getImageUrl(self):
        return self.imageUrl

    #Setters
    def setTitle(self, title):
        self._title = title

    def setUrl(self, url):
        self._url = url

    def setImageUrl(self, imageUrl):
        self._imageUrl = imageUrl
