from flask import Flask
from flask_restful import Api
from controllers.applications import Applications

app = Flask(__name__)

api = Api(app)
api.add_resource(Applications, '/api/applications')

if __name__ == '__main__':
    app.run(debug=True)
